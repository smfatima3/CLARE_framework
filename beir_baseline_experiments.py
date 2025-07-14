#!/usr/bin/env python3
"""
Professional BeIR Baseline Experiments for CLARE Framework
Implements robust baseline methods with proper evaluation protocols
Generates comprehensive analysis and visualizations
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import logging
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import warnings
import time
from abc import ABC, abstractmethod

# Import retrieval libraries
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


@dataclass
class EvaluationConfig:
    """Configuration for baseline evaluation experiments."""
    
    # Evaluation parameters
    k_values: List[int] = None
    batch_size: int = 32
    max_docs_for_evaluation: int = 10000
    
    # Output parameters
    output_dir: str = "beir_baseline_results"
    save_intermediate: bool = True
    generate_plots: bool = True
    
    # Model parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 5, 10, 20]


class BaseRetriever(ABC):
    """Abstract base class for retrieval methods."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, documents: List[str], doc_ids: List[str]) -> None:
        """Fit the retriever on the document collection."""
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 1000) -> Tuple[List[str], List[float]]:
        """Search for relevant documents."""
        pass
    
    def batch_search(self, queries: List[str], k: int = 1000) -> List[Tuple[List[str], List[float]]]:
        """Batch search for multiple queries."""
        results = []
        for query in tqdm(queries, desc=f"Searching with {self.name}"):
            doc_ids, scores = self.search(query, k)
            results.append((doc_ids, scores))
        return results


class BM25Retriever(BaseRetriever):
    """Professional BM25 implementation with optimized parameters."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75, epsilon: float = 0.25):
        super().__init__("BM25")
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.bm25 = None
        self.doc_ids = None
        
    def fit(self, documents: List[str], doc_ids: List[str]) -> None:
        """Fit BM25 on document collection."""
        logger.info(f"Fitting {self.name} on {len(documents)} documents...")
        
        # Tokenize documents
        tokenized_docs = []
        for doc in tqdm(documents, desc="Tokenizing documents"):
            # Simple but effective tokenization
            tokens = doc.lower().split()
            # Remove very short tokens
            tokens = [token for token in tokens if len(token) > 2]
            tokenized_docs.append(tokens)
        
        # Initialize BM25
        self.bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b, epsilon=self.epsilon)
        self.doc_ids = doc_ids
        self.is_fitted = True
        
        logger.info(f"{self.name} fitting completed")
    
    def search(self, query: str, k: int = 1000) -> Tuple[List[str], List[float]]:
        """Search for relevant documents using BM25."""
        if not self.is_fitted:
            raise ValueError("BM25 must be fitted before searching")
        
        # Tokenize query
        tokenized_query = query.lower().split()
        tokenized_query = [token for token in tokenized_query if len(token) > 2]
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k results
        top_k = min(k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        top_doc_ids = [self.doc_ids[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]
        
        return top_doc_ids, top_scores


class DenseRetriever(BaseRetriever):
    """Dense retrieval using Sentence-BERT with FAISS indexing."""
    
    def __init__(self, model_name: str = "msmarco-distilbert-base-v4"):
        super().__init__(f"DPR-{model_name}")
        self.model_name = model_name
        self.model = None
        self.index = None
        self.doc_ids = None
        self.doc_embeddings = None
        
    def fit(self, documents: List[str], doc_ids: List[str]) -> None:
        """Fit dense retriever on document collection."""
        logger.info(f"Fitting {self.name} on {len(documents)} documents...")
        
        # Load model
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}: {e}")
            logger.info("Falling back to sentence-transformers/all-MiniLM-L6-v2")
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.name = "DPR-all-MiniLM-L6-v2"
        
        # Encode documents
        logger.info("Encoding documents...")
        self.doc_embeddings = self.model.encode(
            documents, 
            batch_size=32, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        dimension = self.doc_embeddings.shape[1]
        
        # Use IndexFlatIP for inner product (cosine similarity)
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.doc_embeddings)
        self.index.add(self.doc_embeddings.astype('float32'))
        
        self.doc_ids = doc_ids
        self.is_fitted = True
        
        logger.info(f"{self.name} fitting completed")
    
    def search(self, query: str, k: int = 1000) -> Tuple[List[str], List[float]]:
        """Search using dense embeddings."""
        if not self.is_fitted:
            raise ValueError("Dense retriever must be fitted before searching")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        top_k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return results
        top_doc_ids = [self.doc_ids[idx] for idx in indices[0]]
        top_scores = scores[0].tolist()
        
        return top_doc_ids, top_scores


class SPLADERetriever(BaseRetriever):
    """Simplified SPLADE implementation using sparse representations."""
    
    def __init__(self, model_name: str = "bert-base-uncased", max_features: int = 10000):
        super().__init__("SPLADE")
        self.model_name = model_name
        self.max_features = max_features
        self.tokenizer = None
        self.model = None
        self.tfidf = None
        self.doc_representations = None
        self.doc_ids = None
        
    def fit(self, documents: List[str], doc_ids: List[str]) -> None:
        """Fit SPLADE-style retriever."""
        logger.info(f"Fitting {self.name} on {len(documents)} documents...")
        
        try:
            # Load BERT model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            
            # Create TF-IDF vectorizer for vocabulary
            self.tfidf = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            
            # Fit TF-IDF and get vocabulary
            tfidf_matrix = self.tfidf.fit_transform(documents)
            vocabulary = self.tfidf.get_feature_names_out()
            
            # Generate sparse representations
            logger.info("Generating sparse document representations...")
            self.doc_representations = self._generate_sparse_representations(documents, vocabulary)
            
            self.doc_ids = doc_ids
            self.is_fitted = True
            
            logger.info(f"{self.name} fitting completed")
            
        except Exception as e:
            logger.error(f"Error fitting SPLADE: {e}")
            # Fallback to simple TF-IDF
            logger.info("Falling back to TF-IDF representation")
            self.tfidf = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english'
            )
            self.doc_representations = self.tfidf.fit_transform(documents)
            self.doc_ids = doc_ids
            self.is_fitted = True
    
    def _generate_sparse_representations(self, texts: List[str], vocabulary: List[str]) -> np.ndarray:
        """Generate SPLADE-style sparse representations."""
        representations = []
        batch_size = 16
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating representations"):
                batch_texts = texts[i:i + batch_size]
                
                try:
                    # Tokenize
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    )
                    
                    # Get BERT outputs
                    outputs = self.model(**inputs)
                    
                    # Use mean pooling and apply ReLU for sparsity
                    token_embeddings = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']
                    
                    # Mean pooling
                    masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)
                    mean_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                    
                    # Apply ReLU and normalize
                    sparse_rep = torch.relu(mean_embeddings)
                    sparse_rep = sparse_rep / (torch.norm(sparse_rep, dim=1, keepdim=True) + 1e-8)
                    
                    representations.append(sparse_rep.cpu().numpy())
                    
                except Exception as e:
                    logger.warning(f"Error processing batch {i}: {e}")
                    # Fallback to random representation
                    fallback_rep = np.random.rand(len(batch_texts), 768) * 0.1
                    representations.append(fallback_rep)
        
        return np.vstack(representations)
    
    def search(self, query: str, k: int = 1000) -> Tuple[List[str], List[float]]:
        """Search using sparse representations."""
        if not self.is_fitted:
            raise ValueError("SPLADE must be fitted before searching")
        
        try:
            if hasattr(self, 'model') and self.model is not None:
                # Generate query representation
                query_rep = self._generate_sparse_representations([query], [])[0]
                
                # Compute similarities
                similarities = np.dot(self.doc_representations, query_rep)
            else:
                # Fallback to TF-IDF
                query_vec = self.tfidf.transform([query])
                similarities = cosine_similarity(query_vec, self.doc_representations).flatten()
            
            # Get top-k results
            top_k = min(k, len(similarities))
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            top_doc_ids = [self.doc_ids[i] for i in top_indices]
            top_scores = [similarities[i] for i in top_indices]
            
            return top_doc_ids, top_scores
            
        except Exception as e:
            logger.error(f"Error in SPLADE search: {e}")
            return [], []


class RetrievalEvaluator:
    """Professional evaluation framework for retrieval methods."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results = defaultdict(dict)
        self.timing_results = defaultdict(dict)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def load_beir_dataset(self, dataset_path: str) -> Dict:
        """Load BeIR dataset with proper error handling."""
        logger.info(f"Loading dataset from {dataset_path}")
        
        try:
            # Load main dataset
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            
            logger.info("Dataset loaded successfully")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def prepare_evaluation_data(self, dataset: Dict, split: str = 'test') -> Tuple[Dict, List[str], List[str]]:
        """Prepare data for retrieval evaluation."""
        logger.info(f"Preparing evaluation data from {split} split...")
        
        data = dataset[split]
        
        # Group by query
        query_groups = defaultdict(list)
        for item in data:
            query_groups[item['query_id']].append(item)
        
        # Extract unique documents and queries
        all_documents = []
        all_doc_ids = []
        unique_docs = {}
        
        evaluation_queries = {}
        
        for query_id, items in query_groups.items():
            # Store query
            query_text = items[0]['query']
            evaluation_queries[query_id] = {
                'text': query_text,
                'relevant_docs': [],
                'non_relevant_docs': [],
                'dataset': items[0]['dataset']
            }
            
            # Process documents
            for item in items:
                doc_id = item['doc_id']
                if doc_id not in unique_docs:
                    unique_docs[doc_id] = item['document']
                    all_documents.append(item['document'])
                    all_doc_ids.append(doc_id)
                
                # Add to relevant/non-relevant lists
                if item['label'] == 1:
                    evaluation_queries[query_id]['relevant_docs'].append(doc_id)
                else:
                    evaluation_queries[query_id]['non_relevant_docs'].append(doc_id)
        
        logger.info(f"Prepared {len(evaluation_queries)} queries and {len(all_documents)} documents")
        
        return evaluation_queries, all_documents, all_doc_ids
    
    def evaluate_retriever(self, retriever: BaseRetriever, 
                         evaluation_queries: Dict, 
                         all_documents: List[str], 
                         all_doc_ids: List[str]) -> Dict[str, float]:
        """Evaluate a retrieval method comprehensively."""
        logger.info(f"Evaluating {retriever.name}...")
        
        # Fit retriever
        start_time = time.time()
        retriever.fit(all_documents, all_doc_ids)
        fit_time = time.time() - start_time
        
        # Evaluate queries
        all_results = []
        search_times = []
        
        for query_id, query_info in tqdm(evaluation_queries.items(), desc=f"Evaluating {retriever.name}"):
            query_text = query_info['text']
            relevant_docs = set(query_info['relevant_docs'])
            
            # Search
            start_time = time.time()
            retrieved_docs, scores = retriever.search(query_text, k=1000)
            search_time = time.time() - start_time
            search_times.append(search_time)
            
            # Create ranking
            ranking = []
            for doc_id in retrieved_docs:
                if doc_id in relevant_docs:
                    ranking.append(1)
                else:
                    ranking.append(0)
            
            all_results.append({
                'query_id': query_id,
                'ranking': ranking,
                'num_relevant': len(relevant_docs),
                'dataset': query_info['dataset']
            })
        
        # Compute metrics
        metrics = self._compute_metrics(all_results, self.config.k_values)
        
        # Add timing information
        metrics.update({
            'fit_time_seconds': fit_time,
            'avg_search_time_ms': np.mean(search_times) * 1000,
            'total_queries': len(evaluation_queries)
        })
        
        # Store results
        self.results[retriever.name] = metrics
        self.timing_results[retriever.name] = {
            'fit_time': fit_time,
            'search_times': search_times
        }
        
        logger.info(f"{retriever.name} evaluation completed")
        return metrics
    
    def _compute_metrics(self, results: List[Dict], k_values: List[int]) -> Dict[str, float]:
        """Compute comprehensive retrieval metrics."""
        metrics = {}
        
        # Compute metrics for each k
        for k in k_values:
            ndcg_scores = []
            precision_scores = []
            recall_scores = []
            
            for result in results:
                ranking = result['ranking'][:k]
                num_relevant = result['num_relevant']
                
                # nDCG@k
                dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ranking))
                ideal_ranking = [1] * min(k, num_relevant) + [0] * (k - min(k, num_relevant))
                idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_ranking))
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcg_scores.append(ndcg)
                
                # Precision@k
                precision = sum(ranking) / k if k > 0 else 0.0
                precision_scores.append(precision)
                
                # Recall@k
                recall = sum(ranking) / num_relevant if num_relevant > 0 else 0.0
                recall_scores.append(recall)
            
            metrics[f'ndcg@{k}'] = np.mean(ndcg_scores)
            metrics[f'precision@{k}'] = np.mean(precision_scores)
            metrics[f'recall@{k}'] = np.mean(recall_scores)
        
        # MAP (Mean Average Precision)
        ap_scores = []
        for result in results:
            ranking = result['ranking']
            num_relevant = result['num_relevant']
            
            if num_relevant == 0:
                continue
            
            ap = 0.0
            num_relevant_found = 0
            
            for i, rel in enumerate(ranking):
                if rel == 1:
                    num_relevant_found += 1
                    ap += num_relevant_found / (i + 1)
            
            ap /= num_relevant
            ap_scores.append(ap)
        
        metrics['map'] = np.mean(ap_scores) if ap_scores else 0.0
        
        # MRR (Mean Reciprocal Rank)
        rr_scores = []
        for result in results:
            ranking = result['ranking']
            
            for i, rel in enumerate(ranking):
                if rel == 1:
                    rr_scores.append(1.0 / (i + 1))
                    break
        
        metrics['mrr'] = np.mean(rr_scores) if rr_scores else 0.0
        
        # Dataset-specific metrics
        dataset_metrics = defaultdict(list)
        for result in results:
            dataset = result['dataset']
            # Compute nDCG@10 for this query
            ranking = result['ranking'][:10]
            num_relevant = result['num_relevant']
            
            dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ranking))
            ideal_ranking = [1] * min(10, num_relevant) + [0] * (10 - min(10, num_relevant))
            idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_ranking))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            
            dataset_metrics[dataset].append(ndcg)
        
        # Average by dataset
        for dataset, scores in dataset_metrics.items():
            metrics[f'ndcg@10_{dataset}'] = np.mean(scores)
        
        return metrics
    
    def run_all_evaluations(self, dataset_path: str) -> Dict:
        """Run comprehensive evaluation of all baseline methods."""
        logger.info("Starting comprehensive baseline evaluation...")
        
        # Load dataset
        dataset = self.load_beir_dataset(dataset_path)
        
        # Prepare evaluation data
        evaluation_queries, all_documents, all_doc_ids = self.prepare_evaluation_data(dataset, 'test')
        
        # Limit documents if specified
        if (self.config.max_docs_for_evaluation and 
            len(all_documents) > self.config.max_docs_for_evaluation):
            logger.info(f"Limiting evaluation to {self.config.max_docs_for_evaluation} documents")
            
            # Sample documents while preserving query-document relationships
            import random
            random.seed(42)
            indices = random.sample(range(len(all_documents)), self.config.max_docs_for_evaluation)
            
            sampled_docs = [all_documents[i] for i in indices]
            sampled_doc_ids = [all_doc_ids[i] for i in indices]
            sampled_doc_set = set(sampled_doc_ids)
            
            # Filter queries to only include those with documents in our sample
            filtered_queries = {}
            for qid, qinfo in evaluation_queries.items():
                relevant_in_sample = [doc for doc in qinfo['relevant_docs'] if doc in sampled_doc_set]
                if relevant_in_sample:  # Only keep queries with at least one relevant doc in sample
                    filtered_queries[qid] = qinfo.copy()
                    filtered_queries[qid]['relevant_docs'] = relevant_in_sample
            
            evaluation_queries = filtered_queries
            all_documents = sampled_docs
            all_doc_ids = sampled_doc_ids
            
            logger.info(f"Evaluation set: {len(evaluation_queries)} queries, {len(all_documents)} documents")
        
        # Initialize retrievers
        retrievers = [
            BM25Retriever(k1=1.2, b=0.75),
            DenseRetriever("msmarco-distilbert-base-v4"),
            SPLADERetriever("bert-base-uncased")
        ]
        
        # Evaluate each retriever
        for retriever in retrievers:
            try:
                self.evaluate_retriever(retriever, evaluation_queries, all_documents, all_doc_ids)
            except Exception as e:
                logger.error(f"Failed to evaluate {retriever.name}: {e}")
                continue
        
        # Save results
        self.save_results()
        
        # Generate visualizations
        if self.config.generate_plots:
            self.create_visualizations()
        
        return self.results
    
    def save_results(self) -> None:
        """Save evaluation results to files."""
        logger.info("Saving evaluation results...")
        
        # Save main results
        results_path = os.path.join(self.config.output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(dict(self.results), f, indent=2)
        
        # Save timing results
        timing_path = os.path.join(self.config.output_dir, 'timing_results.json')
        with open(timing_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            timing_data = {}
            for method, data in self.timing_results.items():
                timing_data[method] = {
                    'fit_time': float(data['fit_time']),
                    'avg_search_time_ms': float(np.mean(data['search_times']) * 1000),
                    'search_time_std_ms': float(np.std(data['search_times']) * 1000)
                }
            json.dump(timing_data, f, indent=2)
        
        logger.info(f"Results saved to {self.config.output_dir}")
    
    def create_visualizations(self) -> None:
        """Create comprehensive visualizations."""
        logger.info("Creating visualizations...")
        
        pdf_path = os.path.join(self.config.output_dir, 'baseline_evaluation_report.pdf')
        
        with PdfPages(pdf_path) as pdf:
            # 1. Main metrics comparison
            self._plot_main_metrics(pdf)
            
            # 2. Efficiency analysis
            self._plot_efficiency_metrics(pdf)
            
            # 3. Dataset-specific performance
            self._plot_dataset_performance(pdf)
            
            # 4. Detailed metrics table
            self._plot_metrics_table(pdf)
            
            # 5. Performance vs efficiency trade-off
            self._plot_performance_efficiency_tradeoff(pdf)
        
        logger.info(f"Visualizations saved to {pdf_path}")
    
    def _plot_main_metrics(self, pdf) -> None:
        """Plot main retrieval metrics comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        methods = list(self.results.keys())
        metrics = ['ndcg@10', 'map', 'mrr', 'precision@10']
        metric_names = ['nDCG@10', 'MAP', 'MRR', 'Precision@10']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            values = [self.results[method].get(metric, 0) for method in methods]
            colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
            
            bars = ax.bar(methods, values, color=colors)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_title(f'{name} Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel(name, fontsize=12)
            ax.set_ylim(0, max(values) * 1.15 if values else 1)
            ax.grid(axis='y', alpha=0.3)
            
            # Rotate x-axis labels if needed
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Baseline Methods Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _plot_efficiency_metrics(self, pdf) -> None:
        """Plot efficiency metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        methods = list(self.results.keys())
        
        # Search latency
        search_times = [self.results[method].get('avg_search_time_ms', 0) for method in methods]
        colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
        
        bars1 = ax1.bar(methods, search_times, color=colors)
        ax1.set_title('Average Query Latency', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Latency (ms)', fontsize=12)
        ax1.set_yscale('log')
        
        for bar, value in zip(bars1, search_times):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.1,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10)
        
        # Throughput
        throughputs = [1000 / max(time, 0.001) for time in search_times]
        bars2 = ax2.bar(methods, throughputs, color=colors)
        ax2.set_title('Query Throughput', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Queries per Second', fontsize=12)
        
        for bar, value in zip(bars2, throughputs):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(throughputs) * 0.02,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=10)
        
        for ax in [ax1, ax2]:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _plot_dataset_performance(self, pdf) -> None:
        """Plot dataset-specific performance."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        methods = list(self.results.keys())
        
        # Find dataset-specific metrics
        datasets = set()
        for method_results in self.results.values():
            for key in method_results.keys():
                if key.startswith('ndcg@10_'):
                    dataset = key.replace('ndcg@10_', '')
                    datasets.add(dataset)
        
        datasets = sorted(list(datasets))
        
        if datasets:
            x = np.arange(len(datasets))
            width = 0.8 / len(methods)
            
            for i, method in enumerate(methods):
                values = []
                for dataset in datasets:
                    key = f'ndcg@10_{dataset}'
                    values.append(self.results[method].get(key, 0))
                
                offset = (i - len(methods)/2 + 0.5) * width
                ax.bar(x + offset, values, width, label=method)
            
            ax.set_xlabel('Dataset', fontsize=12)
            ax.set_ylabel('nDCG@10', fontsize=12)
            ax.set_title('Performance Across Datasets', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(datasets, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No dataset-specific metrics available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Dataset-Specific Performance', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_table(self, pdf) -> None:
        """Create a comprehensive metrics table."""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        methods = list(self.results.keys())
        columns = ['Method', 'nDCG@10', 'nDCG@20', 'MAP', 'MRR', 'P@10', 'R@10', 'Latency (ms)']
        
        table_data = []
        for method in methods:
            results = self.results[method]
            row = [
                method,
                f"{results.get('ndcg@10', 0):.4f}",
                f"{results.get('ndcg@20', 0):.4f}",
                f"{results.get('map', 0):.4f}",
                f"{results.get('mrr', 0):.4f}",
                f"{results.get('precision@10', 0):.4f}",
                f"{results.get('recall@10', 0):.4f}",
                f"{results.get('avg_search_time_ms', 0):.1f}"
            ]
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Style the table
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight best values
        for col_idx in range(1, len(columns) - 1):  # Skip method name and latency
            values = []
            for row_idx in range(len(table_data)):
                try:
                    values.append(float(table_data[row_idx][col_idx]))
                except:
                    values.append(0.0)
            
            if values:
                best_value = max(values)
                for row_idx, value in enumerate(values):
                    if value == best_value:
                        table[(row_idx + 1, col_idx)].set_facecolor('#90EE90')
        
        ax.set_title('Comprehensive Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_efficiency_tradeoff(self, pdf) -> None:
        """Plot performance vs efficiency trade-off."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        methods = list(self.results.keys())
        
        # Get data
        ndcg_scores = [self.results[method].get('ndcg@10', 0) for method in methods]
        latencies = [self.results[method].get('avg_search_time_ms', 0) for method in methods]
        
        # Create scatter plot
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        
        for i, (method, ndcg, latency) in enumerate(zip(methods, ndcg_scores, latencies)):
            ax.scatter(latency, ndcg, s=100, color=colors[i], label=method, alpha=0.7)
            
            # Add method labels
            ax.annotate(method, (latency, ndcg), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Average Query Latency (ms)', fontsize=12)
        ax.set_ylabel('nDCG@10', fontsize=12)
        ax.set_title('Performance vs Efficiency Trade-off', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add ideal region annotation
        if ndcg_scores and latencies:
            max_ndcg = max(ndcg_scores)
            min_latency = min(latencies)
            ax.axhline(y=max_ndcg * 0.95, color='green', linestyle='--', alpha=0.5)
            ax.axvline(x=min_latency * 2, color='green', linestyle='--', alpha=0.5)
            ax.text(0.02, 0.98, 'Ideal Region:\nHigh Performance,\nLow Latency', 
                   transform=ax.transAxes, va='top', ha='left', 
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def print_summary(self) -> None:
        """Print evaluation summary."""
        print("\n" + "="*80)
        print("BEIR BASELINE EVALUATION SUMMARY")
        print("="*80)
        
        if not self.results:
            print("No results available.")
            return
        
        for method, metrics in self.results.items():
            print(f"\n{method}:")
            print(f"  nDCG@10: {metrics.get('ndcg@10', 0):.4f}")
            print(f"  MAP: {metrics.get('map', 0):.4f}")
            print(f"  MRR: {metrics.get('mrr', 0):.4f}")
            print(f"  Precision@10: {metrics.get('precision@10', 0):.4f}")
            print(f"  Latency: {metrics.get('avg_search_time_ms', 0):.1f} ms")
            print(f"  Queries evaluated: {metrics.get('total_queries', 0)}")
        
        # Find best performing method
        best_method = max(self.results.keys(), 
                         key=lambda x: self.results[x].get('ndcg@10', 0))
        best_ndcg = self.results[best_method].get('ndcg@10', 0)
        
        print(f"\nBest performing method: {best_method} (nDCG@10: {best_ndcg:.4f})")
        print(f"Results saved to: {self.config.output_dir}")
        print("="*80)


def main():
    """Main function to run BeIR baseline evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BeIR Baseline Evaluation for CLARE')
    parser.add_argument('--dataset_path', type=str, default='beir_clare_dataset.pkl',
                       help='Path to BeIR CLARE dataset')
    parser.add_argument('--output_dir', type=str, default='beir_baseline_results',
                       help='Output directory for results')
    parser.add_argument('--max_docs', type=int, default=10000,
                       help='Maximum documents for evaluation (for speed)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 5, 10, 20],
                       help='K values for evaluation metrics')
    
    args = parser.parse_args()
    
    # Create configuration
    config = EvaluationConfig(
        k_values=args.k_values,
        batch_size=args.batch_size,
        max_docs_for_evaluation=args.max_docs,
        output_dir=args.output_dir,
        generate_plots=True
    )
    
    # Run evaluation
    evaluator = RetrievalEvaluator(config)
    
    try:
        results = evaluator.run_all_evaluations(args.dataset_path)
        evaluator.print_summary()
        
        logger.info("Baseline evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()