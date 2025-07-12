#!/usr/bin/env python3
"""
Baseline Experiments for CLARE Paper
Implements and evaluates baseline methods: BM25, DPR, Sentence-BERT, SPLADE
Generates tables and visualizations in PDF format
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import logging

# Import retrieval libraries
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


class BaselineEvaluator:
    """Evaluates baseline retrieval methods"""
    
    def __init__(self, dataset_path: str, output_dir: str = "baseline_results"):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
        
        # Initialize results storage
        self.results = {
            'method_results': {},
            'dataset_breakdown': {},
            'query_analysis': {},
            'efficiency_metrics': {}
        }
    
    def prepare_data(self, split: str = 'test'):
        """Prepare data for baseline evaluation"""
        data = self.dataset[split]
        
        # Group by query
        query_groups = {}
        for item in data:
            qid = item['query_id']
            if qid not in query_groups:
                query_groups[qid] = {
                    'query': item['query'],
                    'positive_docs': [],
                    'negative_docs': [],
                    'dataset': item['dataset']
                }
            
            if item['label'] == 1:
                query_groups[qid]['positive_docs'].append(item['document'])
            else:
                query_groups[qid]['negative_docs'].append(item['document'])
        
        # Create corpus
        all_docs = []
        doc_ids = []
        for qid, group in query_groups.items():
            all_docs.extend(group['positive_docs'])
            all_docs.extend(group['negative_docs'])
            doc_ids.extend([f"{qid}_pos_{i}" for i in range(len(group['positive_docs']))])
            doc_ids.extend([f"{qid}_neg_{i}" for i in range(len(group['negative_docs']))])
        
        return query_groups, all_docs, doc_ids
    
    def evaluate_bm25(self, query_groups: Dict, all_docs: List[str], doc_ids: List[str]):
        """Evaluate BM25 baseline"""
        logger.info("Evaluating BM25...")
        
        # Tokenize documents for BM25
        tokenized_docs = [doc.lower().split() for doc in all_docs]
        
        # Initialize BM25
        bm25 = BM25Okapi(tokenized_docs)
        
        results = []
        times = []
        
        for qid, group in tqdm(query_groups.items(), desc="BM25 evaluation"):
            query = group['query']
            tokenized_query = query.lower().split()
            
            # Measure retrieval time
            import time
            start_time = time.time()
            scores = bm25.get_scores(tokenized_query)
            retrieval_time = time.time() - start_time
            times.append(retrieval_time)
            
            # Get rankings
            doc_score_pairs = [(doc_ids[i], scores[i]) for i in range(len(scores))]
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Evaluate
            relevant_docs = {f"{qid}_pos_{i}" for i in range(len(group['positive_docs']))}
            rankings = [1 if doc_id in relevant_docs else 0 for doc_id, _ in doc_score_pairs]
            
            results.append({
                'query_id': qid,
                'rankings': rankings,
                'dataset': group['dataset']
            })
        
        metrics = self._compute_metrics(results)
        metrics['avg_latency_ms'] = np.mean(times) * 1000
        
        self.results['method_results']['BM25'] = metrics
        self.results['efficiency_metrics']['BM25'] = {
            'avg_latency_ms': metrics['avg_latency_ms'],
            'throughput_qps': 1000 / metrics['avg_latency_ms']
        }
        
        return metrics
    
    def evaluate_dpr(self, query_groups: Dict, all_docs: List[str], doc_ids: List[str]):
        """Evaluate Dense Passage Retrieval (using Sentence-BERT as approximation)"""
        logger.info("Evaluating DPR (Sentence-BERT)...")
        
        # Load model
        model = SentenceTransformer('msmarco-distilbert-base-v4')
        
        # Encode documents
        logger.info("Encoding documents...")
        doc_embeddings = model.encode(all_docs, batch_size=32, show_progress_bar=True)
        
        # Build FAISS index
        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(doc_embeddings.astype('float32'))
        
        results = []
        times = []
        
        for qid, group in tqdm(query_groups.items(), desc="DPR evaluation"):
            query = group['query']
            
            # Encode query
            import time
            start_time = time.time()
            query_embedding = model.encode([query])
            
            # Search
            k = len(all_docs)
            distances, indices = index.search(query_embedding.astype('float32'), k)
            retrieval_time = time.time() - start_time
            times.append(retrieval_time)
            
            # Get rankings
            relevant_docs = {f"{qid}_pos_{i}" for i in range(len(group['positive_docs']))}
            rankings = [1 if doc_ids[idx] in relevant_docs else 0 for idx in indices[0]]
            
            results.append({
                'query_id': qid,
                'rankings': rankings,
                'dataset': group['dataset']
            })
        
        metrics = self._compute_metrics(results)
        metrics['avg_latency_ms'] = np.mean(times) * 1000
        
        self.results['method_results']['DPR'] = metrics
        self.results['efficiency_metrics']['DPR'] = {
            'avg_latency_ms': metrics['avg_latency_ms'],
            'throughput_qps': 1000 / metrics['avg_latency_ms']
        }
        
        return metrics
    
    def evaluate_splade(self, query_groups: Dict, all_docs: List[str], doc_ids: List[str]):
        """Evaluate SPLADE (simplified version using sparse BERT representations)"""
        logger.info("Evaluating SPLADE (simplified)...")
        
        # Load BERT tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased')
        model.eval()
        
        def get_splade_representation(texts, batch_size=16):
            """Get sparse SPLADE-like representations"""
            all_reps = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True, 
                                 max_length=512, return_tensors='pt')
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use max pooling over tokens and ReLU for sparsity
                    token_embeddings = outputs.last_hidden_state
                    sparse_rep = torch.max(torch.relu(token_embeddings), dim=1)[0]
                    all_reps.append(sparse_rep)
            
            return torch.cat(all_reps, dim=0).numpy()
        
        # Encode documents
        logger.info("Encoding documents with SPLADE...")
        doc_reps = get_splade_representation(all_docs)
        
        results = []
        times = []
        
        for qid, group in tqdm(query_groups.items(), desc="SPLADE evaluation"):
            query = group['query']
            
            import time
            start_time = time.time()
            query_rep = get_splade_representation([query])
            
            # Compute similarities
            similarities = np.dot(doc_reps, query_rep.T).squeeze()
            sorted_indices = np.argsort(similarities)[::-1]
            retrieval_time = time.time() - start_time
            times.append(retrieval_time)
            
            # Get rankings
            relevant_docs = {f"{qid}_pos_{i}" for i in range(len(group['positive_docs']))}
            rankings = [1 if doc_ids[idx] in relevant_docs else 0 for idx in sorted_indices]
            
            results.append({
                'query_id': qid,
                'rankings': rankings,
                'dataset': group['dataset']
            })
        
        metrics = self._compute_metrics(results)
        metrics['avg_latency_ms'] = np.mean(times) * 1000
        
        self.results['method_results']['SPLADE'] = metrics
        self.results['efficiency_metrics']['SPLADE'] = {
            'avg_latency_ms': metrics['avg_latency_ms'],
            'throughput_qps': 1000 / metrics['avg_latency_ms']
        }
        
        return metrics
    
    def _compute_metrics(self, results: List[Dict], k_values: List[int] = [10, 20]):
        """Compute retrieval metrics"""
        metrics = {}
        
        for k in k_values:
            ndcg_scores = []
            precision_scores = []
            recall_scores = []
            
            for result in results:
                rankings = result['rankings'][:k]
                
                # nDCG@k
                dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(rankings))
                ideal_rankings = sorted(result['rankings'], reverse=True)[:k]
                idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_rankings))
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_scores.append(ndcg)
                
                # Precision@k
                precision = sum(rankings) / k
                precision_scores.append(precision)
                
                # Recall@k
                total_relevant = sum(result['rankings'])
                recall = sum(rankings) / total_relevant if total_relevant > 0 else 0
                recall_scores.append(recall)
            
            metrics[f'ndcg@{k}'] = np.mean(ndcg_scores)
            metrics[f'precision@{k}'] = np.mean(precision_scores)
            metrics[f'recall@{k}'] = np.mean(recall_scores)
        
        # MAP
        ap_scores = []
        for result in results:
            rankings = result['rankings']
            ap = 0
            num_relevant = 0
            
            for i, rel in enumerate(rankings):
                if rel == 1:
                    num_relevant += 1
                    ap += num_relevant / (i + 1)
            
            total_relevant = sum(rankings)
            if total_relevant > 0:
                ap_scores.append(ap / total_relevant)
        
        metrics['map'] = np.mean(ap_scores) if ap_scores else 0
        
        # MRR
        rr_scores = []
        for result in results:
            for i, rel in enumerate(result['rankings']):
                if rel == 1:
                    rr_scores.append(1.0 / (i + 1))
                    break
        
        metrics['mrr'] = np.mean(rr_scores) if rr_scores else 0
        
        # Dataset breakdown
        dataset_metrics = {}
        for dataset in set(r['dataset'] for r in results):
            dataset_results = [r for r in results if r['dataset'] == dataset]
            dataset_ndcg = np.mean([
                self._compute_single_ndcg(r['rankings'], 10) for r in dataset_results
            ])
            dataset_metrics[dataset] = dataset_ndcg
        
        metrics['dataset_breakdown'] = dataset_metrics
        
        return metrics
    
    def _compute_single_ndcg(self, rankings: List[int], k: int) -> float:
        """Compute nDCG for a single query"""
        rankings_k = rankings[:k]
        dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(rankings_k))
        ideal_rankings = sorted(rankings, reverse=True)[:k]
        idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_rankings))
        return dcg / idcg if idcg > 0 else 0
    
    def create_comparison_table(self):
        """Create comparison table of all methods"""
        # Prepare data for table
        methods = list(self.results['method_results'].keys())
        metrics_data = []
        
        for method in methods:
            method_metrics = self.results['method_results'][method]
            row = {
                'Method': method,
                'nDCG@10': f"{method_metrics['ndcg@10']:.4f}",
                'nDCG@20': f"{method_metrics['ndcg@20']:.4f}",
                'MAP': f"{method_metrics['map']:.4f}",
                'MRR': f"{method_metrics['mrr']:.4f}",
                'P@10': f"{method_metrics['precision@10']:.4f}",
                'R@10': f"{method_metrics['recall@10']:.4f}",
                'Latency (ms)': f"{method_metrics['avg_latency_ms']:.1f}"
            }
            metrics_data.append(row)
        
        df = pd.DataFrame(metrics_data)
        
        # Style the dataframe
        styled_df = df.style.set_properties(**{
            'text-align': 'center',
            'font-size': '12pt',
            'border-color': 'black',
            'border-style': 'solid',
            'border-width': '1px'
        })
        
        # Highlight best values
        numeric_cols = ['nDCG@10', 'nDCG@20', 'MAP', 'MRR', 'P@10', 'R@10']
        for col in numeric_cols:
            df[col] = df[col].astype(float)
            max_val = df[col].max()
            styled_df = styled_df.applymap(
                lambda x: 'background-color: lightgreen' if x == max_val else '',
                subset=[col]
            )
        
        return df, styled_df
    
    def create_visualizations(self, pdf_path: str):
        """Create all visualizations and save to PDF"""
        with PdfPages(pdf_path) as pdf:
            # 1. Overall Performance Comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Bar plot of main metrics
            methods = list(self.results['method_results'].keys())
            metrics = ['ndcg@10', 'map', 'mrr', 'precision@10']
            metric_names = ['nDCG@10', 'MAP', 'MRR', 'P@10']
            
            for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
                ax = axes[idx // 2, idx % 2]
                values = [self.results['method_results'][m][metric] for m in methods]
                bars = ax.bar(methods, values)
                
                # Color bars
                colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                ax.set_title(f'{name} Comparison', fontsize=14, fontweight='bold')
                ax.set_ylabel(name, fontsize=12)
                ax.set_ylim(0, max(values) * 1.1)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 2. Efficiency Comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Latency comparison
            latencies = [self.results['efficiency_metrics'][m]['avg_latency_ms'] for m in methods]
            bars = ax1.bar(methods, latencies, color='coral')
            ax1.set_title('Average Query Latency', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Latency (ms)', fontsize=12)
            ax1.set_yscale('log')
            
            for bar, value in zip(bars, latencies):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                        f'{value:.1f}', ha='center', va='bottom', fontsize=10)
            
            # Throughput comparison
            throughputs = [self.results['efficiency_metrics'][m]['throughput_qps'] for m in methods]
            bars = ax2.bar(methods, throughputs, color='lightblue')
            ax2.set_title('Query Throughput', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Queries per Second', fontsize=12)
            
            for bar, value in zip(bars, throughputs):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.0f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 3. Dataset-wise Performance
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Prepare data
            datasets = []
            for method in methods:
                if 'dataset_breakdown' in self.results['method_results'][method]:
                    datasets = list(self.results['method_results'][method]['dataset_breakdown'].keys())
                    break
            
            if datasets:
                dataset_performance = []
                for dataset in datasets:
                    row = [dataset]
                    for method in methods:
                        if 'dataset_breakdown' in self.results['method_results'][method]:
                            value = self.results['method_results'][method]['dataset_breakdown'].get(dataset, 0)
                            row.append(value)
                        else:
                            row.append(0)
                    dataset_performance.append(row)
                
                # Create grouped bar plot
                x = np.arange(len(datasets))
                width = 0.8 / len(methods)
                
                for i, method in enumerate(methods):
                    values = [row[i+1] for row in dataset_performance]
                    offset = (i - len(methods)/2 + 0.5) * width
                    ax.bar(x + offset, values, width, label=method)
                
                ax.set_xlabel('Dataset', fontsize=12)
                ax.set_ylabel('nDCG@10', fontsize=12)
                ax.set_title('Performance Across Datasets', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(datasets, rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 4. Radar Chart for Multi-metric Comparison
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            metrics = ['nDCG@10', 'nDCG@20', 'MAP', 'MRR', 'P@10', 'R@10']
            num_vars = len(metrics)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]
            
            for method in methods:
                values = []
                for metric in ['ndcg@10', 'ndcg@20', 'map', 'mrr', 'precision@10', 'recall@10']:
                    values.append(self.results['method_results'][method][metric])
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, label=method)
                ax.fill(angles, values, alpha=0.15)
            
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, fontsize=12)
            ax.set_ylim(0, 1)
            ax.set_title('Multi-Metric Comparison', fontsize=16, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            ax.grid(True)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 5. Comparison Table
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.axis('tight')
            ax.axis('off')
            
            df, _ = self.create_comparison_table()
            
            # Create table
            table_data = df.values.tolist()
            col_labels = df.columns.tolist()
            
            table = ax.table(cellText=table_data, colLabels=col_labels,
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Style the table
            for i in range(len(col_labels)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Highlight best values
            for i in range(1, len(table_data) + 1):
                for j in range(1, len(col_labels) - 1):  # Skip Method and Latency columns
                    try:
                        val = float(table_data[i-1][j])
                        col_vals = [float(row[j]) for row in table_data]
                        if val == max(col_vals):
                            table[(i, j)].set_facecolor('#90EE90')
                    except:
                        pass
            
            ax.set_title('Baseline Methods Performance Comparison', 
                        fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    def run_all_baselines(self):
        """Run all baseline evaluations"""
        # Prepare data
        query_groups, all_docs, doc_ids = self.prepare_data('test')
        
        # Run evaluations
        logger.info("Running baseline evaluations...")
        self.evaluate_bm25(query_groups, all_docs, doc_ids)
        self.evaluate_dpr(query_groups, all_docs, doc_ids)
        self.evaluate_splade(query_groups, all_docs, doc_ids)
        
        # Save results
        results_path = os.path.join(self.output_dir, 'baseline_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create PDF report
        pdf_path = os.path.join(self.output_dir, 'baseline_experiments.pdf')
        self.create_visualizations(pdf_path)
        
        logger.info(f"Results saved to {results_path}")
        logger.info(f"PDF report saved to {pdf_path}")
        
        return self.results


def main():
    """Main function to run baseline experiments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run baseline experiments for CLARE')
    parser.add_argument('--dataset_path', type=str, default='clare_dataset_complete.pkl',
                       help='Path to CLARE dataset')
    parser.add_argument('--output_dir', type=str, default='baseline_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run evaluations
    evaluator = BaselineEvaluator(args.dataset_path, args.output_dir)
    results = evaluator.run_all_baselines()
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE EXPERIMENTS SUMMARY")
    print("="*60)
    
    for method, metrics in results['method_results'].items():
        print(f"\n{method}:")
        print(f"  nDCG@10: {metrics['ndcg@10']:.4f}")
        print(f"  MAP: {metrics['map']:.4f}")
        print(f"  MRR: {metrics['mrr']:.4f}")
        print(f"  Latency: {metrics['avg_latency_ms']:.1f} ms")
    
    print(f"\nPDF report saved to: {os.path.join(args.output_dir, 'baseline_experiments.pdf')}")


if __name__ == "__main__":
    main()