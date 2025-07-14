#!/usr/bin/env python3
"""
BeIR Dataset Integration for CLARE Framework
Professional implementation following software engineering best practices
Supports BeIR benchmark datasets with proper evaluation protocols
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset as HFDataset
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Set
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, vstack
import re
import random
from tqdm import tqdm
import pickle
import os
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
import hashlib
from abc import ABC, abstractmethod
import warnings
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")


@dataclass
class BeIRConfig:
    """Configuration for BeIR dataset processing aligned with CLARE paper specifications."""
    
    # Model parameters
    max_query_length: int = 128
    max_doc_length: int = 512
    negative_sampling_ratio: int = 4
    min_query_length: int = 3
    min_doc_length: int = 10
    
    # Clustering parameters
    vocab_size: int = 30000
    min_df: int = 5
    max_df: float = 0.5
    n_clusters: int = 100
    
    # Dataset parameters
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    
    # Processing parameters
    use_stemming: bool = True
    remove_stopwords: bool = True
    max_samples_per_dataset: Optional[int] = None
    
    # Hard negative mining
    use_hard_negatives: bool = True
    hard_negative_ratio: float = 0.5  # Ratio of hard vs random negatives
    
    # Supported datasets
    supported_datasets: List[str] = field(default_factory=lambda: [
        "BeIR/trec-news-generated-queries",
        "BeIR/cqadupstack-generated-queries"
    ])


class BaseDatasetProcessor(ABC):
    """Abstract base class for dataset processors."""
    
    def __init__(self, config: BeIRConfig):
        self.config = config
        self.stop_words = set(stopwords.words('english')) if config.remove_stopwords else set()
        
    @abstractmethod
    def process_dataset(self, dataset_name: str) -> Dict[str, Union[List, Dict]]:
        """Process a specific dataset and return structured data."""
        pass
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Remove very short texts
        if len(text.split()) < self.config.min_query_length:
            return ""
            
        return text
    
    def _generate_doc_id(self, text: str, prefix: str = "doc") -> str:
        """Generate unique document ID."""
        hash_obj = hashlib.md5(text.encode('utf-8'))
        return f"{prefix}_{hash_obj.hexdigest()[:12]}"
    
    def _is_valid_pair(self, query: str, document: str) -> bool:
        """Check if query-document pair meets quality criteria."""
        return (
            len(query.split()) >= self.config.min_query_length and
            len(document.split()) >= self.config.min_doc_length and
            query.strip() != document.strip()  # Avoid identical query-doc pairs
        )


class BeIRDatasetProcessor(BaseDatasetProcessor):
    """Processor for BeIR benchmark datasets."""
    
    def __init__(self, config: BeIRConfig):
        super().__init__(config)
        self.tfidf_vectorizer = None
        
    def process_dataset(self, dataset_name: str) -> Dict[str, Union[List, Dict]]:
        """
        Process BeIR dataset with proper train/validation/test splits.
        
        Args:
            dataset_name: Name of the BeIR dataset
            
        Returns:
            Dictionary containing processed data and metadata
        """
        logger.info(f"Processing BeIR dataset: {dataset_name}")
        
        if dataset_name not in self.config.supported_datasets:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        try:
            # Load dataset
            dataset = load_dataset(dataset_name, split="train")
            logger.info(f"Loaded {len(dataset)} samples from {dataset_name}")
            
            # Sample if specified
            if (self.config.max_samples_per_dataset and 
                len(dataset) > self.config.max_samples_per_dataset):
                indices = random.sample(
                    range(len(dataset)), 
                    self.config.max_samples_per_dataset
                )
                dataset = dataset.select(indices)
                logger.info(f"Sampled {len(dataset)} examples")
            
            # Process dataset based on type
            if "trec-news" in dataset_name:
                return self._process_trec_news(dataset, dataset_name)
            elif "cqadupstack" in dataset_name:
                return self._process_cqadupstack(dataset, dataset_name)
            else:
                return self._process_generic_beir(dataset, dataset_name)
                
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {e}")
            return self._empty_result()
    
    def _process_trec_news(self, dataset, dataset_name: str) -> Dict[str, Union[List, Dict]]:
        """Process TREC News dataset."""
        logger.info("Processing TREC News dataset...")
        
        processed_data = []
        documents_for_matrix = []
        unique_docs = {}
        
        for idx, item in enumerate(tqdm(dataset, desc="Processing TREC News")):
            # Extract fields
            query = self._clean_text(item.get('query', ''))
            doc_id = item.get('_id', f"trec_news_{idx}")
            title = self._clean_text(item.get('title', ''))
            text = self._clean_text(item.get('text', ''))
            
            # Combine title and text for document
            document = f"{title} {text}".strip() if title and text else (title or text)
            document = self._clean_text(document)
            
            if not self._is_valid_pair(query, document):
                continue
                
            # Generate consistent doc_id
            doc_hash_id = self._generate_doc_id(document, "trec_news")
            
            # Store unique documents
            if doc_hash_id not in unique_docs:
                unique_docs[doc_hash_id] = document
                documents_for_matrix.append(document)
            
            # Create positive pair
            processed_data.append({
                'query_id': f"trec_news_q_{idx}",
                'query': query,
                'doc_id': doc_hash_id,
                'document': document,
                'label': 1,
                'dataset': 'trec_news',
                'original_id': doc_id
            })
        
        # Generate negative pairs
        negative_pairs = self._generate_negative_pairs(
            processed_data, unique_docs, 'trec_news'
        )
        processed_data.extend(negative_pairs)
        
        return {
            'data': processed_data,
            'documents': documents_for_matrix,
            'unique_docs': unique_docs,
            'dataset_name': dataset_name
        }
    
    def _process_cqadupstack(self, dataset, dataset_name: str) -> Dict[str, Union[List, Dict]]:
        """Process CQADupStack dataset."""
        logger.info("Processing CQADupStack dataset...")
        
        processed_data = []
        documents_for_matrix = []
        unique_docs = {}
        
        for idx, item in enumerate(tqdm(dataset, desc="Processing CQADupStack")):
            # Extract fields
            query = self._clean_text(item.get('query', ''))
            doc_id = item.get('_id', f"cqa_{idx}")
            title = self._clean_text(item.get('title', ''))
            text = self._clean_text(item.get('text', ''))
            
            # For CQA, treat title as question and text as answer
            if title and text:
                document = f"Question: {title} Answer: {text}"
            else:
                document = title or text
            
            document = self._clean_text(document)
            
            if not self._is_valid_pair(query, document):
                continue
                
            # Generate consistent doc_id
            doc_hash_id = self._generate_doc_id(document, "cqa")
            
            # Store unique documents
            if doc_hash_id not in unique_docs:
                unique_docs[doc_hash_id] = document
                documents_for_matrix.append(document)
            
            # Create positive pair
            processed_data.append({
                'query_id': f"cqa_q_{idx}",
                'query': query,
                'doc_id': doc_hash_id,
                'document': document,
                'label': 1,
                'dataset': 'cqadupstack',
                'original_id': doc_id
            })
        
        # Generate negative pairs
        negative_pairs = self._generate_negative_pairs(
            processed_data, unique_docs, 'cqadupstack'
        )
        processed_data.extend(negative_pairs)
        
        return {
            'data': processed_data,
            'documents': documents_for_matrix,
            'unique_docs': unique_docs,
            'dataset_name': dataset_name
        }
    
    def _process_generic_beir(self, dataset, dataset_name: str) -> Dict[str, Union[List, Dict]]:
        """Process generic BeIR dataset."""
        logger.info(f"Processing generic BeIR dataset: {dataset_name}")
        
        processed_data = []
        documents_for_matrix = []
        unique_docs = {}
        
        for idx, item in enumerate(tqdm(dataset, desc=f"Processing {dataset_name}")):
            # Extract fields with fallbacks
            query = self._clean_text(item.get('query', ''))
            doc_id = item.get('_id', f"generic_{idx}")
            title = self._clean_text(item.get('title', ''))
            text = self._clean_text(item.get('text', ''))
            
            # Combine title and text
            document = f"{title} {text}".strip() if title and text else (title or text)
            document = self._clean_text(document)
            
            if not self._is_valid_pair(query, document):
                continue
                
            # Generate consistent doc_id
            dataset_prefix = dataset_name.split('/')[-1].replace('-', '_')
            doc_hash_id = self._generate_doc_id(document, dataset_prefix)
            
            # Store unique documents
            if doc_hash_id not in unique_docs:
                unique_docs[doc_hash_id] = document
                documents_for_matrix.append(document)
            
            # Create positive pair
            processed_data.append({
                'query_id': f"{dataset_prefix}_q_{idx}",
                'query': query,
                'doc_id': doc_hash_id,
                'document': document,
                'label': 1,
                'dataset': dataset_prefix,
                'original_id': doc_id
            })
        
        # Generate negative pairs
        negative_pairs = self._generate_negative_pairs(
            processed_data, unique_docs, dataset_prefix
        )
        processed_data.extend(negative_pairs)
        
        return {
            'data': processed_data,
            'documents': documents_for_matrix,
            'unique_docs': unique_docs,
            'dataset_name': dataset_name
        }
    
    def _generate_negative_pairs(self, positive_data: List[Dict], 
                                unique_docs: Dict[str, str], 
                                dataset_prefix: str) -> List[Dict]:
        """Generate negative pairs using both random and hard negative sampling."""
        logger.info(f"Generating negative pairs for {dataset_prefix}...")
        
        negative_pairs = []
        
        # Group by query
        query_groups = defaultdict(list)
        for item in positive_data:
            query_groups[item['query_id']].append(item)
        
        all_doc_ids = list(unique_docs.keys())
        
        # Initialize TF-IDF for hard negatives if enabled
        if self.config.use_hard_negatives and not self.tfidf_vectorizer:
            logger.info("Building TF-IDF index for hard negative mining...")
            doc_texts = list(unique_docs.values())
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=min(10000, len(doc_texts) * 2),
                stop_words='english',
                ngram_range=(1, 2)
            )
            try:
                self.doc_tfidf_matrix = self.tfidf_vectorizer.fit_transform(doc_texts)
                self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(all_doc_ids)}
            except Exception as e:
                logger.warning(f"Failed to build TF-IDF index: {e}. Using random negatives only.")
                self.config.use_hard_negatives = False
        
        for query_id, positive_items in tqdm(query_groups.items(), desc="Creating negatives"):
            positive_doc_ids = {item['doc_id'] for item in positive_items}
            query_text = positive_items[0]['query']
            
            # Available negative documents
            available_negatives = [doc_id for doc_id in all_doc_ids 
                                 if doc_id not in positive_doc_ids]
            
            if not available_negatives:
                continue
            
            num_negatives = min(
                self.config.negative_sampling_ratio * len(positive_items),
                len(available_negatives)
            )
            
            selected_negatives = []
            
            if self.config.use_hard_negatives and hasattr(self, 'doc_tfidf_matrix'):
                # Mix of hard and random negatives
                num_hard = int(num_negatives * self.config.hard_negative_ratio)
                num_random = num_negatives - num_hard
                
                # Get hard negatives using TF-IDF similarity
                try:
                    query_vector = self.tfidf_vectorizer.transform([query_text])
                    similarities = (query_vector * self.doc_tfidf_matrix.T).toarray().flatten()
                    
                    # Get indices of available documents
                    available_indices = [self.doc_id_to_idx[doc_id] 
                                       for doc_id in available_negatives 
                                       if doc_id in self.doc_id_to_idx]
                    
                    if available_indices:
                        # Sort by similarity and take top candidates as hard negatives
                        available_sims = [(idx, similarities[idx]) for idx in available_indices]
                        available_sims.sort(key=lambda x: x[1], reverse=True)
                        
                        hard_indices = [idx for idx, _ in available_sims[:num_hard]]
                        hard_negatives = [all_doc_ids[idx] for idx in hard_indices]
                        selected_negatives.extend(hard_negatives)
                        
                        # Remove hard negatives from available pool
                        remaining_negatives = [doc_id for doc_id in available_negatives 
                                             if doc_id not in hard_negatives]
                    else:
                        remaining_negatives = available_negatives
                except Exception as e:
                    logger.warning(f"Hard negative mining failed: {e}. Using random sampling.")
                    remaining_negatives = available_negatives
                    num_random = num_negatives
            else:
                remaining_negatives = available_negatives
                num_random = num_negatives
            
            # Add random negatives
            if remaining_negatives and num_random > 0:
                random_negatives = random.sample(
                    remaining_negatives, 
                    min(num_random, len(remaining_negatives))
                )
                selected_negatives.extend(random_negatives)
            
            # Create negative pairs
            for neg_doc_id in selected_negatives:
                negative_pairs.append({
                    'query_id': query_id,
                    'query': query_text,
                    'doc_id': neg_doc_id,
                    'document': unique_docs[neg_doc_id],
                    'label': 0,
                    'dataset': dataset_prefix
                })
        
        logger.info(f"Generated {len(negative_pairs)} negative pairs for {dataset_prefix}")
        return negative_pairs
    
    def _empty_result(self) -> Dict[str, Union[List, Dict]]:
        """Return empty result structure."""
        return {
            'data': [],
            'documents': [],
            'unique_docs': {},
            'dataset_name': 'empty'
        }


class TermDocumentMatrixBuilder:
    """Enhanced term-document matrix builder for CLARE's semantic cluster discovery."""
    
    def __init__(self, config: BeIRConfig):
        self.config = config
        self.vectorizer = None
        self.vocabulary = None
        
    def build_matrix(self, documents: List[str], 
                    fit_vectorizer: bool = True) -> Tuple[csr_matrix, List[str]]:
        """
        Build term-document matrix optimized for semantic cluster discovery.
        
        Args:
            documents: List of document texts
            fit_vectorizer: Whether to fit the vectorizer
            
        Returns:
            Tuple of (term_doc_matrix, vocabulary)
        """
        logger.info(f"Building term-document matrix for {len(documents)} documents...")
        
        if fit_vectorizer or self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=self.config.vocab_size,
                min_df=self.config.min_df,
                max_df=self.config.max_df,
                stop_words='english' if self.config.remove_stopwords else None,
                ngram_range=(1, 2),  # Include bigrams
                sublinear_tf=True,  # Log normalization
                dtype=np.float32,
                token_pattern=r'(?u)\b[A-Za-z][A-Za-z]+\b'  # Only alphabetic tokens
            )
            
            try:
                term_doc_matrix = self.vectorizer.fit_transform(documents)
                self.vocabulary = self.vectorizer.get_feature_names_out().tolist()
            except ValueError as e:
                logger.error(f"Error building term-document matrix: {e}")
                # Fallback with simpler parameters
                self.vectorizer = TfidfVectorizer(
                    max_features=min(5000, len(documents)),
                    min_df=1,
                    max_df=0.95,
                    stop_words='english'
                )
                term_doc_matrix = self.vectorizer.fit_transform(documents)
                self.vocabulary = self.vectorizer.get_feature_names_out().tolist()
        else:
            term_doc_matrix = self.vectorizer.transform(documents)
        
        logger.info(f"Term-document matrix shape: {term_doc_matrix.shape}")
        logger.info(f"Matrix density: {term_doc_matrix.nnz / np.prod(term_doc_matrix.shape):.4f}")
        
        return term_doc_matrix, self.vocabulary
    
    def get_word_embeddings(self, vocabulary: List[str], 
                           model_name: str = "bert-base-uncased") -> np.ndarray:
        """Extract word embeddings for vocabulary using BERT."""
        logger.info("Extracting BERT embeddings for vocabulary...")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            
            embeddings = []
            batch_size = 64
            
            with torch.no_grad():
                for i in tqdm(range(0, len(vocabulary), batch_size), 
                             desc="Extracting embeddings"):
                    batch_words = vocabulary[i:i + batch_size]
                    
                    # Handle multi-word terms (bigrams)
                    processed_words = []
                    for word in batch_words:
                        if ' ' in word:  # Bigram
                            processed_words.append(word.replace(' ', ' [SEP] '))
                        else:
                            processed_words.append(word)
                    
                    try:
                        inputs = tokenizer(
                            processed_words, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True, 
                            max_length=20
                        )
                        
                        outputs = model(**inputs)
                        # Use mean pooling over tokens
                        attention_mask = inputs['attention_mask']
                        token_embeddings = outputs.last_hidden_state
                        
                        # Mask padding tokens
                        masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)
                        # Mean pooling
                        word_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                        
                        embeddings.append(word_embeddings.cpu().numpy())
                    except Exception as e:
                        logger.warning(f"Error processing batch {i}: {e}")
                        # Fallback: random embeddings
                        fallback_embeddings = np.random.randn(len(batch_words), 768) * 0.1
                        embeddings.append(fallback_embeddings)
            
            embeddings = np.vstack(embeddings)
            logger.info(f"Word embeddings shape: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error extracting embeddings: {e}")
            # Fallback: random embeddings
            logger.info("Using random embeddings as fallback")
            return np.random.randn(len(vocabulary), 768) * 0.1


class BeIRDatasetBuilder:
    """Main dataset builder for CLARE framework using BeIR benchmarks."""
    
    def __init__(self, config: BeIRConfig):
        self.config = config
        self.processor = BeIRDatasetProcessor(config)
        self.matrix_builder = TermDocumentMatrixBuilder(config)
        
    def build_complete_dataset(self) -> Dict[str, Union[List, Dict, np.ndarray, csr_matrix]]:
        """
        Build complete dataset from BeIR benchmarks.
        
        Returns:
            Dictionary containing all processed data and matrices
        """
        logger.info("Building complete CLARE dataset from BeIR benchmarks...")
        
        all_data = []
        all_documents = []
        all_unique_docs = {}
        dataset_stats = {}
        
        # Process each supported dataset
        for dataset_name in self.config.supported_datasets:
            logger.info(f"\nProcessing {dataset_name}...")
            
            try:
                result = self.processor.process_dataset(dataset_name)
                
                if result['data']:
                    all_data.extend(result['data'])
                    all_documents.extend(result['documents'])
                    all_unique_docs.update(result['unique_docs'])
                    
                    # Calculate stats
                    dataset_key = dataset_name.split('/')[-1]
                    dataset_stats[dataset_key] = {
                        'total': len(result['data']),
                        'positive': sum(1 for x in result['data'] if x['label'] == 1),
                        'negative': sum(1 for x in result['data'] if x['label'] == 0),
                        'unique_docs': len(result['unique_docs'])
                    }
                    
                    logger.info(f"Successfully processed {dataset_name}")
                else:
                    logger.warning(f"No data extracted from {dataset_name}")
                    
            except Exception as e:
                logger.error(f"Failed to process {dataset_name}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data was successfully processed from any dataset")
        
        logger.info(f"\nTotal samples collected: {len(all_data)}")
        logger.info(f"Total unique documents: {len(all_unique_docs)}")
        logger.info(f"Dataset distribution: {dataset_stats}")
        
        # Build term-document matrix
        logger.info("\nBuilding term-document matrix for semantic cluster discovery...")
        
        # Use unique documents for matrix
        document_texts = list(all_unique_docs.values())
        
        # Sample if too many documents (for memory efficiency)
        max_docs_for_matrix = 50000
        if len(document_texts) > max_docs_for_matrix:
            logger.info(f"Sampling {max_docs_for_matrix} documents from {len(document_texts)} for matrix construction")
            sampled_indices = random.sample(range(len(document_texts)), max_docs_for_matrix)
            document_texts = [document_texts[i] for i in sampled_indices]
        
        term_doc_matrix, vocabulary = self.matrix_builder.build_matrix(document_texts)
        
        # Extract word embeddings
        logger.info("\nExtracting word embeddings for vocabulary...")
        word_embeddings = self.matrix_builder.get_word_embeddings(vocabulary)
        
        # Create final dataset structure
        dataset = {
            'train_data': all_data,
            'term_doc_matrix': term_doc_matrix,
            'vocabulary': vocabulary,
            'word_embeddings': word_embeddings,
            'dataset_stats': dataset_stats,
            'unique_docs': all_unique_docs,
            'config': self.config
        }
        
        logger.info("Dataset building completed successfully!")
        return dataset
    
    def create_train_val_test_splits(self, dataset: Dict) -> Dict:
        """
        Create stratified train/validation/test splits maintaining query groups.
        
        Args:
            dataset: Complete dataset dictionary
            
        Returns:
            Dictionary with train/val/test splits
        """
        logger.info("\nCreating train/validation/test splits...")
        
        all_data = dataset['train_data']
        
        # Group by query to avoid data leakage
        query_groups = defaultdict(list)
        for item in all_data:
            query_groups[item['query_id']].append(item)
        
        # Stratify by dataset to ensure balanced representation
        dataset_queries = defaultdict(list)
        for query_id, items in query_groups.items():
            dataset_name = items[0]['dataset']
            dataset_queries[dataset_name].append(query_id)
        
        # Create splits for each dataset
        train_queries, val_queries, test_queries = [], [], []
        
        for dataset_name, query_ids in dataset_queries.items():
            random.seed(self.config.random_state)
            random.shuffle(query_ids)
            
            n_queries = len(query_ids)
            n_train = int(n_queries * self.config.train_ratio)
            n_val = int(n_queries * self.config.val_ratio)
            
            train_queries.extend(query_ids[:n_train])
            val_queries.extend(query_ids[n_train:n_train + n_val])
            test_queries.extend(query_ids[n_train + n_val:])
            
            logger.info(f"{dataset_name}: {n_train} train, {n_val} val, "
                       f"{n_queries - n_train - n_val} test queries")
        
        # Create data splits
        splits = {split: [] for split in ['train', 'validation', 'test']}
        query_splits = {
            'train': train_queries,
            'validation': val_queries,
            'test': test_queries
        }
        
        for split, queries in query_splits.items():
            for qid in queries:
                splits[split].extend(query_groups[qid])
            
            # Shuffle within splits
            random.seed(self.config.random_state)
            random.shuffle(splits[split])
        
        # Add matrix and embeddings to splits
        final_splits = {
            **splits,
            'term_doc_matrix': dataset['term_doc_matrix'],
            'vocabulary': dataset['vocabulary'],
            'word_embeddings': dataset['word_embeddings'],
            'dataset_stats': dataset['dataset_stats'],
            'unique_docs': dataset['unique_docs'],
            'config': dataset['config']
        }
        
        # Log split statistics
        logger.info(f"\nFinal splits:")
        for split in ['train', 'validation', 'test']:
            data = splits[split]
            n_queries = len(set(item['query_id'] for item in data))
            n_positive = sum(1 for item in data if item['label'] == 1)
            n_negative = sum(1 for item in data if item['label'] == 0)
            
            logger.info(f"{split.capitalize()}: {len(data)} pairs, {n_queries} queries, "
                       f"{n_positive} positive, {n_negative} negative")
        
        return final_splits
    
    def save_dataset(self, dataset: Dict, filepath: str) -> None:
        """
        Save dataset with proper error handling and metadata.
        
        Args:
            dataset: Dataset dictionary to save
            filepath: Path to save the dataset
        """
        logger.info(f"Saving dataset to {filepath}")
        
        try:
            # Create directory
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            
            # Save main data (excluding large matrices)
            main_data = {
                'train': dataset['train'],
                'validation': dataset['validation'],
                'test': dataset['test'],
                'vocabulary': dataset['vocabulary'],
                'dataset_stats': dataset['dataset_stats'],
                'config': dataset.get('config').__dict__ if hasattr(dataset.get('config'), '__dict__') else dataset.get('config')
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(main_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save term-document matrix separately
            matrix_path = filepath.replace('.pkl', '_term_doc_matrix.npz')
            if 'term_doc_matrix' in dataset:
                from scipy.sparse import save_npz
                save_npz(matrix_path, dataset['term_doc_matrix'])
            
            # Save word embeddings
            embeddings_path = filepath.replace('.pkl', '_word_embeddings.npy')
            if 'word_embeddings' in dataset:
                np.save(embeddings_path, dataset['word_embeddings'])
            
            logger.info(f"Dataset saved successfully:")
            logger.info(f"  Main data: {filepath}")
            logger.info(f"  Term-doc matrix: {matrix_path}")
            logger.info(f"  Word embeddings: {embeddings_path}")
            
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            raise
    
    def load_dataset(self, filepath: str) -> Dict:
        """
        Load dataset with all components.
        
        Args:
            filepath: Path to the main dataset file
            
        Returns:
            Complete dataset dictionary
        """
        logger.info(f"Loading dataset from {filepath}")
        
        try:
            # Load main data
            with open(filepath, 'rb') as f:
                dataset = pickle.load(f)
            
            # Load term-document matrix
            matrix_path = filepath.replace('.pkl', '_term_doc_matrix.npz')
            if os.path.exists(matrix_path):
                from scipy.sparse import load_npz
                dataset['term_doc_matrix'] = load_npz(matrix_path)
                logger.info(f"Loaded term-document matrix: {dataset['term_doc_matrix'].shape}")
            
            # Load word embeddings
            embeddings_path = filepath.replace('.pkl', '_word_embeddings.npy')
            if os.path.exists(embeddings_path):
                dataset['word_embeddings'] = np.load(embeddings_path)
                logger.info(f"Loaded word embeddings: {dataset['word_embeddings'].shape}")
            
            logger.info("Dataset loaded successfully")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise


class BeIRTorchDataset(Dataset):
    """
    PyTorch Dataset for CLARE training with BeIR data.
    Handles proper tokenization and negative sampling.
    """
    
    def __init__(self, data: List[Dict], tokenizer, config: BeIRConfig):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        
        # Group by query for efficient batch processing
        self.query_groups = defaultdict(list)
        for item in data:
            self.query_groups[item['query_id']].append(item)
        
        self.queries = list(self.query_groups.keys())
        logger.info(f"Created dataset with {len(self.queries)} unique queries, "
                   f"{len(data)} total pairs")
    
    def __len__(self) -> int:
        return len(self.queries)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training example with query, positive document, and negatives.
        
        Args:
            idx: Index of the query
            
        Returns:
            Dictionary with tokenized inputs and metadata
        """
        query_id = self.queries[idx]
        group = self.query_groups[query_id]
        
        # Get query text (same for all items in group)
        query_text = group[0]['query']
        
        # Separate positive and negative documents
        positive_docs = [item for item in group if item['label'] == 1]
        negative_docs = [item for item in group if item['label'] == 0]
        
        # Sample one positive document
        if not positive_docs:
            # Fallback: use next query if current has no positives
            return self.__getitem__((idx + 1) % len(self))
        
        positive_doc = random.choice(positive_docs)
        
        # Sample negative documents
        num_negatives = min(len(negative_docs), self.config.negative_sampling_ratio)
        if num_negatives > 0:
            sampled_negatives = random.sample(negative_docs, num_negatives)
        else:
            sampled_negatives = []
        
        # Tokenize query
        query_encoding = self.tokenizer(
            query_text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_query_length,
            return_tensors='pt'
        )
        
        # Tokenize positive document
        pos_encoding = self.tokenizer(
            positive_doc['document'],
            truncation=True,
            padding='max_length',
            max_length=self.config.max_doc_length,
            return_tensors='pt'
        )
        
        # Tokenize negative documents
        neg_input_ids_list = []
        neg_attention_mask_list = []
        
        for i in range(self.config.negative_sampling_ratio):
            if i < len(sampled_negatives):
                neg_encoding = self.tokenizer(
                    sampled_negatives[i]['document'],
                    truncation=True,
                    padding='max_length',
                    max_length=self.config.max_doc_length,
                    return_tensors='pt'
                )
                neg_input_ids_list.append(neg_encoding['input_ids'].squeeze())
                neg_attention_mask_list.append(neg_encoding['attention_mask'].squeeze())
            else:
                # Pad with zeros for missing negatives
                neg_input_ids_list.append(
                    torch.zeros(self.config.max_doc_length, dtype=torch.long)
                )
                neg_attention_mask_list.append(
                    torch.zeros(self.config.max_doc_length, dtype=torch.long)
                )
        
        # Stack negative encodings
        neg_input_ids = torch.stack(neg_input_ids_list)
        neg_attention_mask = torch.stack(neg_attention_mask_list)
        
        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(),
            'pos_doc_input_ids': pos_encoding['input_ids'].squeeze(),
            'pos_doc_attention_mask': pos_encoding['attention_mask'].squeeze(),
            'neg_doc_input_ids': neg_input_ids,
            'neg_doc_attention_mask': neg_attention_mask,
            'query_id': query_id,
            'dataset': positive_doc['dataset'],
            'num_valid_negatives': min(len(sampled_negatives), self.config.negative_sampling_ratio)
        }


class BeIRDatasetAnalyzer:
    """Comprehensive analyzer for BeIR dataset statistics and quality metrics."""
    
    def __init__(self, dataset: Dict):
        self.dataset = dataset
        
    def analyze_dataset(self) -> Dict:
        """
        Perform comprehensive analysis of the dataset.
        
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Analyzing BeIR dataset...")
        
        analysis = {
            'overview': self._analyze_overview(),
            'query_statistics': self._analyze_queries(),
            'document_statistics': self._analyze_documents(),
            'dataset_distribution': self._analyze_dataset_distribution(),
            'vocabulary_statistics': self._analyze_vocabulary(),
            'quality_metrics': self._analyze_quality(),
            'matrix_statistics': self._analyze_matrix()
        }
        
        return analysis
    
    def _analyze_overview(self) -> Dict:
        """Analyze overall dataset statistics."""
        overview = {}
        
        for split in ['train', 'validation', 'test']:
            if split in self.dataset:
                data = self.dataset[split]
                
                unique_queries = set(item['query_id'] for item in data)
                unique_docs = set(item['doc_id'] for item in data)
                positive_pairs = sum(1 for item in data if item['label'] == 1)
                negative_pairs = sum(1 for item in data if item['label'] == 0)
                
                overview[split] = {
                    'total_pairs': len(data),
                    'unique_queries': len(unique_queries),
                    'unique_documents': len(unique_docs),
                    'positive_pairs': positive_pairs,
                    'negative_pairs': negative_pairs,
                    'positive_ratio': positive_pairs / len(data) if data else 0
                }
        
        return overview
    
    def _analyze_queries(self) -> Dict:
        """Analyze query characteristics."""
        query_stats = {}
        
        for split in ['train', 'validation', 'test']:
            if split in self.dataset:
                data = self.dataset[split]
                
                query_lengths = []
                unique_queries = {}
                
                for item in data:
                    if item['query_id'] not in unique_queries:
                        unique_queries[item['query_id']] = item['query']
                        query_lengths.append(len(item['query'].split()))
                
                if query_lengths:
                    query_stats[split] = {
                        'count': len(query_lengths),
                        'avg_length': np.mean(query_lengths),
                        'std_length': np.std(query_lengths),
                        'min_length': np.min(query_lengths),
                        'max_length': np.max(query_lengths),
                        'median_length': np.median(query_lengths),
                        'percentiles': {
                            '25th': np.percentile(query_lengths, 25),
                            '75th': np.percentile(query_lengths, 75),
                            '95th': np.percentile(query_lengths, 95)
                        }
                    }
        
        return query_stats
    
    def _analyze_documents(self) -> Dict:
        """Analyze document characteristics."""
        doc_stats = {}
        
        for split in ['train', 'validation', 'test']:
            if split in self.dataset:
                data = self.dataset[split]
                
                doc_lengths = []
                unique_docs = {}
                
                for item in data:
                    if item['doc_id'] not in unique_docs:
                        unique_docs[item['doc_id']] = item['document']
                        doc_lengths.append(len(item['document'].split()))
                
                if doc_lengths:
                    doc_stats[split] = {
                        'count': len(doc_lengths),
                        'avg_length': np.mean(doc_lengths),
                        'std_length': np.std(doc_lengths),
                        'min_length': np.min(doc_lengths),
                        'max_length': np.max(doc_lengths),
                        'median_length': np.median(doc_lengths),
                        'percentiles': {
                            '25th': np.percentile(doc_lengths, 25),
                            '75th': np.percentile(doc_lengths, 75),
                            '95th': np.percentile(doc_lengths, 95)
                        }
                    }
        
        return doc_stats
    
    def _analyze_dataset_distribution(self) -> Dict:
        """Analyze distribution across different source datasets."""
        distribution = {}
        
        for split in ['train', 'validation', 'test']:
            if split in self.dataset:
                data = self.dataset[split]
                
                dataset_counts = defaultdict(int)
                dataset_positives = defaultdict(int)
                
                for item in data:
                    dataset_name = item['dataset']
                    dataset_counts[dataset_name] += 1
                    if item['label'] == 1:
                        dataset_positives[dataset_name] += 1
                
                distribution[split] = {
                    'total_by_dataset': dict(dataset_counts),
                    'positives_by_dataset': dict(dataset_positives),
                    'ratios_by_dataset': {
                        dataset: dataset_positives[dataset] / dataset_counts[dataset]
                        for dataset in dataset_counts
                    }
                }
        
        return distribution
    
    def _analyze_vocabulary(self) -> Dict:
        """Analyze vocabulary statistics."""
        vocab_stats = {}
        
        if 'vocabulary' in self.dataset:
            vocab = self.dataset['vocabulary']
            
            # Basic statistics
            vocab_stats['size'] = len(vocab)
            vocab_stats['sample_terms'] = vocab[:20]
            
            # Term length analysis
            term_lengths = [len(term.split()) for term in vocab]
            vocab_stats['term_lengths'] = {
                'unigrams': sum(1 for length in term_lengths if length == 1),
                'bigrams': sum(1 for length in term_lengths if length == 2),
                'avg_length': np.mean(term_lengths)
            }
        
        return vocab_stats
    
    def _analyze_quality(self) -> Dict:
        """Analyze dataset quality metrics."""
        quality = {}
        
        # Check for potential issues
        issues = []
        
        for split in ['train', 'validation', 'test']:
            if split in self.dataset:
                data = self.dataset[split]
                
                # Check query-document overlap
                identical_pairs = sum(
                    1 for item in data 
                    if item['query'].lower().strip() == item['document'].lower().strip()
                )
                
                if identical_pairs > 0:
                    issues.append(f"{split}: {identical_pairs} identical query-document pairs")
                
                # Check for very short queries/documents
                short_queries = sum(
                    1 for item in data 
                    if len(item['query'].split()) < 3
                )
                short_docs = sum(
                    1 for item in data 
                    if len(item['document'].split()) < 10
                )
                
                if short_queries > len(data) * 0.1:
                    issues.append(f"{split}: {short_queries} very short queries (>10%)")
                
                if short_docs > len(data) * 0.1:
                    issues.append(f"{split}: {short_docs} very short documents (>10%)")
        
        quality['issues'] = issues
        quality['overall_quality'] = 'Good' if len(issues) == 0 else 'Issues detected'
        
        return quality
    
    def _analyze_matrix(self) -> Dict:
        """Analyze term-document matrix statistics."""
        matrix_stats = {}
        
        if 'term_doc_matrix' in self.dataset:
            matrix = self.dataset['term_doc_matrix']
            
            matrix_stats['shape'] = matrix.shape
            matrix_stats['density'] = matrix.nnz / np.prod(matrix.shape)
            matrix_stats['avg_terms_per_doc'] = matrix.nnz / matrix.shape[1]
            matrix_stats['total_non_zero'] = matrix.nnz
            matrix_stats['memory_usage_mb'] = matrix.data.nbytes / (1024 * 1024)
        
        return matrix_stats
    
    def print_analysis(self, analysis: Dict) -> None:
        """Print comprehensive analysis in a readable format."""
        print("\n" + "="*80)
        print("BeIR DATASET ANALYSIS FOR CLARE FRAMEWORK")
        print("="*80)
        
        # Overview
        print("\nOVERALL STATISTICS:")
        for split, stats in analysis['overview'].items():
            print(f"\n{split.upper()}:")
            print(f"  Total pairs: {stats['total_pairs']:,}")
            print(f"  Unique queries: {stats['unique_queries']:,}")
            print(f"  Unique documents: {stats['unique_documents']:,}")
            print(f"  Positive pairs: {stats['positive_pairs']:,} ({stats['positive_ratio']:.1%})")
            print(f"  Negative pairs: {stats['negative_pairs']:,}")
        
        # Query statistics
        print(f"\nQUERY STATISTICS:")
        print(f"{'Split':<12} {'Count':<8} {'Avg Len':<8} {'Std Dev':<8} {'Min':<5} {'Max':<5} {'Median':<7}")
        print("-" * 70)
        for split, stats in analysis['query_statistics'].items():
            print(f"{split:<12} {stats['count']:<8} {stats['avg_length']:<8.1f} "
                  f"{stats['std_length']:<8.1f} {stats['min_length']:<5} "
                  f"{stats['max_length']:<5} {stats['median_length']:<7.1f}")
        
        # Document statistics
        print(f"\nDOCUMENT STATISTICS:")
        print(f"{'Split':<12} {'Count':<8} {'Avg Len':<8} {'Std Dev':<8} {'Min':<5} {'Max':<6} {'Median':<7}")
        print("-" * 70)
        for split, stats in analysis['document_statistics'].items():
            print(f"{split:<12} {stats['count']:<8} {stats['avg_length']:<8.1f} "
                  f"{stats['std_length']:<8.1f} {stats['min_length']:<5} "
                  f"{stats['max_length']:<6} {stats['median_length']:<7.1f}")
        
        # Dataset distribution
        print(f"\nDATASET DISTRIBUTION:")
        for split, dist in analysis['dataset_distribution'].items():
            print(f"\n{split.upper()}:")
            for dataset, count in dist['total_by_dataset'].items():
                positives = dist['positives_by_dataset'][dataset]
                ratio = dist['ratios_by_dataset'][dataset]
                print(f"  {dataset}: {count:,} pairs ({positives:,} positive, {ratio:.1%})")
        
        # Vocabulary
        if analysis['vocabulary_statistics']:
            vocab_stats = analysis['vocabulary_statistics']
            print(f"\nVOCABULARY:")
            print(f"  Size: {vocab_stats['size']:,}")
            print(f"  Unigrams: {vocab_stats['term_lengths']['unigrams']:,}")
            print(f"  Bigrams: {vocab_stats['term_lengths']['bigrams']:,}")
            print(f"  Sample terms: {', '.join(vocab_stats['sample_terms'][:10])}")
        
        # Matrix statistics
        if analysis['matrix_statistics']:
            matrix_stats = analysis['matrix_statistics']
            print(f"\nTERM-DOCUMENT MATRIX:")
            print(f"  Shape: {matrix_stats['shape'][0]:,} terms × {matrix_stats['shape'][1]:,} documents")
            print(f"  Density: {matrix_stats['density']:.6f}")
            print(f"  Non-zero entries: {matrix_stats['total_non_zero']:,}")
            print(f"  Avg terms per document: {matrix_stats['avg_terms_per_doc']:.1f}")
            print(f"  Memory usage: {matrix_stats['memory_usage_mb']:.1f} MB")
        
        # Quality assessment
        print(f"\nQUALITY ASSESSMENT:")
        quality = analysis['quality_metrics']
        print(f"  Overall quality: {quality['overall_quality']}")
        if quality['issues']:
            print("  Issues detected:")
            for issue in quality['issues']:
                print(f"    - {issue}")
        else:
            print("  No major issues detected")
        
        print("\n" + "="*80)


# Example usage and main execution
def main():
    """Main function demonstrating BeIR dataset integration."""
    
    # Configuration
    config = BeIRConfig(
        max_query_length=128,
        max_doc_length=512,
        negative_sampling_ratio=4,
        min_query_length=3,
        min_doc_length=10,
        vocab_size=30000,
        n_clusters=100,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        use_hard_negatives=True,
        hard_negative_ratio=0.5,
        max_samples_per_dataset=20000  # Limit for demonstration
    )
    
    # Build dataset
    dataset_builder = BeIRDatasetBuilder(config)
    
    print("Building CLARE dataset from BeIR benchmarks...")
    print("Supported datasets:")
    for dataset in config.supported_datasets:
        print(f"  - {dataset}")
    print()
    
    try:
        # Build complete dataset
        complete_dataset = dataset_builder.build_complete_dataset()
        
        # Create splits
        dataset_splits = dataset_builder.create_train_val_test_splits(complete_dataset)
        
        # Analyze dataset
        analyzer = BeIRDatasetAnalyzer(dataset_splits)
        analysis = analyzer.analyze_dataset()
        analyzer.print_analysis(analysis)
        
        # Save dataset
        output_path = 'beir_clare_dataset.pkl'
        dataset_builder.save_dataset(dataset_splits, output_path)
        print(f"\nDataset saved to: {output_path}")
        
        # Test PyTorch dataset
        print("\nTesting PyTorch Dataset integration...")
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Create training dataset
        train_dataset = BeIRTorchDataset(
            dataset_splits['train'][:100],  # Test with subset
            tokenizer,
            config
        )
        
        # Create DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        
        # Test data loading
        print("Testing data loading...")
        for i, batch in enumerate(train_loader):
            if i == 0:  # Just test first batch
                print(f"Batch keys: {list(batch.keys())}")
                print(f"Query shape: {batch['query_input_ids'].shape}")
                print(f"Positive doc shape: {batch['pos_doc_input_ids'].shape}")
                print(f"Negative docs shape: {batch['neg_doc_input_ids'].shape}")
                print(f"Datasets in batch: {batch['dataset']}")
                print(f"Valid negatives per sample: {batch['num_valid_negatives']}")
            break
        
        print("\nBeIR dataset integration completed successfully!")
        
    except Exception as e:
        logger.error(f"Dataset integration failed: {e}")
        raise


if __name__ == "__main__":
    main()