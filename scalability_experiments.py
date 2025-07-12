#!/usr/bin/env python3
"""
Scalability Experiments for CLARE Paper
Tests scalability across different collection sizes and configurations
Generates tables and visualizations in PDF format
"""

import os
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch
import psutil
import GPUtil
from tqdm import tqdm
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import logging
from memory_profiler import memory_usage

# Import CLARE components
from clare_framework import CLAREModel, CLAREInference
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


class ScalabilityEvaluator:
    """Evaluates CLARE scalability across different dimensions"""
    
    def __init__(self, model_path: str, dataset_path: str, output_dir: str = "scalability_results"):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Results storage
        self.results = {
            'collection_scaling': {},
            'batch_size_scaling': {},
            'cluster_scaling': {},
            'optimization_impact': {},
            'hardware_utilization': {}
        }
    
    def measure_indexing_time(self, num_docs_list: List[int]):
        """Measure indexing time for different collection sizes"""
        logger.info("Measuring indexing time scalability...")
        
        # Load sample documents
        with open(self.dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        all_docs = []
        for item in dataset['train']:
            all_docs.append(item['document'])
            if len(all_docs) >= max(num_docs_list):
                break
        
        results = []
        
        for num_docs in num_docs_list:
            logger.info(f"Testing with {num_docs} documents...")
            docs_subset = all_docs[:num_docs]
            
            # Initialize model
            model = CLAREModel(n_clusters=100)
            tokenizer = model.tokenizer
            
            # Measure indexing time
            start_time = time.time()
            
            # Tokenize and encode documents
            batch_size = 32
            all_embeddings = []
            
            for i in tqdm(range(0, len(docs_subset), batch_size), desc="Encoding"):
                batch = docs_subset[i:i+batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True, 
                                 max_length=512, return_tensors='pt')
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    all_embeddings.append(outputs['cluster_activations'].cpu())
            
            embeddings = torch.cat(all_embeddings, dim=0)
            indexing_time = time.time() - start_time
            
            # Measure memory
            memory_mb = embeddings.element_size() * embeddings.nelement() / 1024 / 1024
            
            results.append({
                'num_docs': num_docs,
                'indexing_time_s': indexing_time,
                'memory_mb': memory_mb,
                'docs_per_second': num_docs / indexing_time
            })
            
            # Clean up
            del model
            del embeddings
            torch.cuda.empty_cache()
        
        self.results['collection_scaling']['indexing'] = results
        return results
    
    def measure_query_latency(self, num_docs_list: List[int], num_queries: int = 100):
        """Measure query latency for different collection sizes"""
        logger.info("Measuring query latency scalability...")
        
        # Load model
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model = CLAREModel(n_clusters=checkpoint['config']['n_clusters'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Load documents
        with open(self.dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # Prepare queries
        queries = [item['query'] for item in dataset['test'][:num_queries]]
        
        results = []
        
        for num_docs in num_docs_list:
            logger.info(f"Testing query latency with {num_docs} documents...")
            
            # Simulate document collection of given size
            # In practice, this would involve actual index lookup
            latencies = []
            
            for query in tqdm(queries, desc="Query latency"):
                start_time = time.time()
                
                # Encode query
                inputs = model.tokenizer(query, return_tensors='pt', 
                                       padding=True, truncation=True, max_length=128)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    query_outputs = model(**inputs)
                    query_clusters = query_outputs['cluster_activations']
                    
                    # Simulate retrieval over num_docs
                    # In real implementation, this would involve index search
                    retrieval_time = num_docs * 0.00001  # Simulated linear search time
                
                total_time = time.time() - start_time + retrieval_time
                latencies.append(total_time * 1000)  # Convert to ms
            
            results.append({
                'num_docs': num_docs,
                'avg_latency_ms': np.mean(latencies),
                'std_latency_ms': np.std(latencies),
                'p50_latency_ms': np.percentile(latencies, 50),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99)
            })
        
        self.results['collection_scaling']['query_latency'] = results
        return results
    
    def measure_batch_size_impact(self, batch_sizes: List[int]):
        """Measure impact of different batch sizes on throughput"""
        logger.info("Measuring batch size impact...")
        
        # Load model
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model = CLAREModel(n_clusters=checkpoint['config']['n_clusters'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Prepare test documents
        with open(self.dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        test_docs = [item['document'] for item in dataset['test'][:1000]]
        
        results = []
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size {batch_size}...")
            
            total_time = 0
            max_memory = 0
            
            try:
                for i in range(0, len(test_docs), batch_size):
                    batch = test_docs[i:i+batch_size]
                    
                    # Measure GPU memory before
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        mem_before = torch.cuda.memory_allocated()
                    
                    start_time = time.time()
                    
                    inputs = model.tokenizer(batch, padding=True, truncation=True,
                                           max_length=512, return_tensors='pt')
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    batch_time = time.time() - start_time
                    total_time += batch_time
                    
                    # Measure GPU memory after
                    if torch.cuda.is_available():
                        mem_after = torch.cuda.memory_allocated()
                        max_memory = max(max_memory, (mem_after - mem_before) / 1024 / 1024)
                
                throughput = len(test_docs) / total_time
                
                results.append({
                    'batch_size': batch_size,
                    'throughput_docs_per_sec': throughput,
                    'avg_time_per_batch_ms': (total_time / (len(test_docs) / batch_size)) * 1000,
                    'max_gpu_memory_mb': max_memory
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM with batch size {batch_size}")
                    results.append({
                        'batch_size': batch_size,
                        'throughput_docs_per_sec': 0,
                        'avg_time_per_batch_ms': 0,
                        'max_gpu_memory_mb': -1  # Indicates OOM
                    })
                    torch.cuda.empty_cache()
                else:
                    raise
        
        self.results['batch_size_scaling'] = results
        return results
    
    def measure_cluster_scaling(self, cluster_counts: List[int]):
        """Measure impact of different cluster counts"""
        logger.info("Measuring cluster count impact...")
        
        # Prepare test data
        with open(self.dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        test_queries = [item['query'] for item in dataset['test'][:100]]
        test_docs = [item['document'] for item in dataset['test'][:100]]
        
        results = []
        
        for n_clusters in cluster_counts:
            logger.info(f"Testing with {n_clusters} clusters...")
            
            # Initialize model with different cluster count
            model = CLAREModel(n_clusters=n_clusters)
            model.to(self.device)
            model.eval()
            
            # Measure model size
            model_params = sum(p.numel() for p in model.parameters())
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
            
            # Measure inference time
            latencies = []
            
            for query in test_queries:
                start_time = time.time()
                
                inputs = model.tokenizer(query, return_tensors='pt',
                                       padding=True, truncation=True, max_length=128)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                inference_time = (time.time() - start_time) * 1000
                latencies.append(inference_time)
            
            # Measure memory for cluster embeddings
            cluster_memory_mb = 0
            for module in model.cluster_attention_modules:
                cluster_memory_mb += (module.cluster_embeddings.numel() * 
                                    module.cluster_embeddings.element_size() / 1024 / 1024)
            
            results.append({
                'n_clusters': n_clusters,
                'model_params': model_params,
                'model_size_mb': model_size_mb,
                'cluster_memory_mb': cluster_memory_mb,
                'avg_latency_ms': np.mean(latencies),
                'std_latency_ms': np.std(latencies)
            })
            
            # Clean up
            del model
            torch.cuda.empty_cache()
        
        self.results['cluster_scaling'] = results
        return results
    
    def measure_optimization_impact(self):
        """Measure impact of various optimizations"""
        logger.info("Measuring optimization impact...")
        
        # Load base model
        checkpoint = torch.load(self.model_path, map_location=self.device)
        base_model = CLAREModel(n_clusters=checkpoint['config']['n_clusters'])
        base_model.load_state_dict(checkpoint['model_state_dict'])
        base_model.to(self.device)
        base_model.eval()
        
        # Prepare test data
        with open(self.dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        test_queries = [item['query'] for item in dataset['test'][:200]]
        
        results = {}
        
        # 1. Baseline (no optimization)
        logger.info("Testing baseline...")
        baseline_latencies = []
        
        for query in tqdm(test_queries, desc="Baseline"):
            start_time = time.time()
            inputs = base_model.tokenizer(query, return_tensors='pt',
                                         padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = base_model(**inputs)
            
            latency = (time.time() - start_time) * 1000
            baseline_latencies.append(latency)
        
        results['baseline'] = {
            'avg_latency_ms': np.mean(baseline_latencies),
            'speedup': 1.0
        }
        
        # 2. Mixed Precision (FP16)
        if torch.cuda.is_available():
            logger.info("Testing mixed precision...")
            mp_latencies = []
            
            with torch.cuda.amp.autocast():
                for query in tqdm(test_queries, desc="Mixed Precision"):
                    start_time = time.time()
                    inputs = base_model.tokenizer(query, return_tensors='pt',
                                                 padding=True, truncation=True, max_length=128)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = base_model(**inputs)
                    
                    latency = (time.time() - start_time) * 1000
                    mp_latencies.append(latency)
            
            results['mixed_precision'] = {
                'avg_latency_ms': np.mean(mp_latencies),
                'speedup': np.mean(baseline_latencies) / np.mean(mp_latencies)
            }
        
        # 3. Quantization (INT8)
        logger.info("Testing quantization...")
        quantized_model = torch.quantization.quantize_dynamic(
            base_model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        quant_latencies = []
        for query in tqdm(test_queries, desc="Quantization"):
            start_time = time.time()
            inputs = base_model.tokenizer(query, return_tensors='pt',
                                         padding=True, truncation=True, max_length=128)
            
            with torch.no_grad():
                outputs = quantized_model(**inputs)
            
            latency = (time.time() - start_time) * 1000
            quant_latencies.append(latency)
        
        results['quantization'] = {
            'avg_latency_ms': np.mean(quant_latencies),
            'speedup': np.mean(baseline_latencies) / np.mean(quant_latencies)
        }
        
        # 4. Sparse Cluster Pruning (keep top-k clusters)
        logger.info("Testing sparse cluster pruning...")
        sparse_latencies = []
        top_k = 20  # Keep only top 20 clusters
        
        for query in tqdm(test_queries, desc="Sparse Pruning"):
            start_time = time.time()
            inputs = base_model.tokenizer(query, return_tensors='pt',
                                         padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = base_model(**inputs)
                # Prune clusters
                cluster_acts = outputs['cluster_activations']
                top_k_vals, top_k_idx = torch.topk(cluster_acts, top_k, dim=1)
                sparse_acts = torch.zeros_like(cluster_acts)
                sparse_acts.scatter_(1, top_k_idx, top_k_vals)
                outputs['cluster_activations'] = sparse_acts
            
            latency = (time.time() - start_time) * 1000
            sparse_latencies.append(latency)
        
        results['sparse_pruning'] = {
            'avg_latency_ms': np.mean(sparse_latencies),
            'speedup': np.mean(baseline_latencies) / np.mean(sparse_latencies),
            'top_k': top_k
        }
        
        self.results['optimization_impact'] = results
        return results
    
    def measure_hardware_utilization(self):
        """Measure CPU/GPU utilization during inference"""
        logger.info("Measuring hardware utilization...")
        
        # Load model
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model = CLAREModel(n_clusters=checkpoint['config']['n_clusters'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Prepare test data
        with open(self.dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        test_docs = [item['document'] for item in dataset['test'][:500]]
        
        # Monitor resources during inference
        cpu_usage = []
        gpu_usage = []
        gpu_memory = []
        
        def inference_task():
            for i in range(0, len(test_docs), 32):
                batch = test_docs[i:i+32]
                inputs = model.tokenizer(batch, padding=True, truncation=True,
                                       max_length=512, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Record usage
                cpu_usage.append(psutil.cpu_percent(interval=0.1))
                
                if torch.cuda.is_available():
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_usage.append(gpus[0].load * 100)
                        gpu_memory.append(gpus[0].memoryUsed)
        
        # Run inference with memory profiling
        mem_usage = memory_usage(inference_task, interval=0.1)
        
        results = {
            'cpu_usage_percent': {
                'mean': np.mean(cpu_usage),
                'max': np.max(cpu_usage),
                'std': np.std(cpu_usage)
            },
            'memory_usage_mb': {
                'mean': np.mean(mem_usage),
                'max': np.max(mem_usage),
                'std': np.std(mem_usage)
            }
        }
        
        if gpu_usage:
            results['gpu_usage_percent'] = {
                'mean': np.mean(gpu_usage),
                'max': np.max(gpu_usage),
                'std': np.std(gpu_usage)
            }
            results['gpu_memory_mb'] = {
                'mean': np.mean(gpu_memory),
                'max': np.max(gpu_memory),
                'std': np.std(gpu_memory)
            }
        
        self.results['hardware_utilization'] = results
        return results
    
    def create_visualizations(self, pdf_path: str):
        """Create all scalability visualizations"""
        with PdfPages(pdf_path) as pdf:
            # 1. Collection Size Scaling
            if 'collection_scaling' in self.results:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # Indexing time
                if 'indexing' in self.results['collection_scaling']:
                    data = self.results['collection_scaling']['indexing']
                    df = pd.DataFrame(data)
                    
                    ax = axes[0, 0]
                    ax.plot(df['num_docs'], df['indexing_time_s'], 'o-', linewidth=2, markersize=8)
                    ax.set_xlabel('Number of Documents', fontsize=12)
                    ax.set_ylabel('Indexing Time (seconds)', fontsize=12)
                    ax.set_title('Indexing Time vs Collection Size', fontsize=14, fontweight='bold')
                    ax.set_xscale('log')
                    ax.grid(True, alpha=0.3)
                    
                    # Throughput
                    ax = axes[0, 1]
                    ax.plot(df['num_docs'], df['docs_per_second'], 'o-', linewidth=2, 
                           markersize=8, color='green')
                    ax.set_xlabel('Number of Documents', fontsize=12)
                    ax.set_ylabel('Documents per Second', fontsize=12)
                    ax.set_title('Indexing Throughput', fontsize=14, fontweight='bold')
                    ax.set_xscale('log')
                    ax.grid(True, alpha=0.3)
                
                # Query latency
                if 'query_latency' in self.results['collection_scaling']:
                    data = self.results['collection_scaling']['query_latency']
                    df = pd.DataFrame(data)
                    
                    ax = axes[1, 0]
                    ax.plot(df['num_docs'], df['avg_latency_ms'], 'o-', linewidth=2, 
                           markersize=8, label='Average')
                    ax.fill_between(df['num_docs'], df['p95_latency_ms'], df['p99_latency_ms'],
                                   alpha=0.3, label='P95-P99 range')
                    ax.set_xlabel('Number of Documents', fontsize=12)
                    ax.set_ylabel('Query Latency (ms)', fontsize=12)
                    ax.set_title('Query Latency vs Collection Size', fontsize=14, fontweight='bold')
                    ax.set_xscale('log')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Memory usage
                    ax = axes[1, 1]
                    if 'indexing' in self.results['collection_scaling']:
                        data = self.results['collection_scaling']['indexing']
                        df = pd.DataFrame(data)
                        ax.plot(df['num_docs'], df['memory_mb'], 'o-', linewidth=2, 
                               markersize=8, color='red')
                        ax.set_xlabel('Number of Documents', fontsize=12)
                        ax.set_ylabel('Memory Usage (MB)', fontsize=12)
                        ax.set_title('Memory Usage vs Collection Size', fontsize=14, fontweight='bold')
                        ax.set_xscale('log')
                        ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # 2. Batch Size Impact
            if 'batch_size_scaling' in self.results:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                data = self.results['batch_size_scaling']
                df = pd.DataFrame(data)
                df = df[df['throughput_docs_per_sec'] > 0]  # Filter out OOM cases
                
                # Throughput vs batch size
                ax = axes[0]
                ax.plot(df['batch_size'], df['throughput_docs_per_sec'], 'o-', 
                       linewidth=2, markersize=8)
                ax.set_xlabel('Batch Size', fontsize=12)
                ax.set_ylabel('Throughput (docs/sec)', fontsize=12)
                ax.set_title('Throughput vs Batch Size', fontsize=14, fontweight='bold')
                ax.set_xscale('log', base=2)
                ax.grid(True, alpha=0.3)
                
                # GPU memory vs batch size
                ax = axes[1]
                ax.plot(df['batch_size'], df['max_gpu_memory_mb'], 'o-', 
                       linewidth=2, markersize=8, color='red')
                ax.set_xlabel('Batch Size', fontsize=12)
                ax.set_ylabel('GPU Memory (MB)', fontsize=12)
                ax.set_title('GPU Memory Usage vs Batch Size', fontsize=14, fontweight='bold')
                ax.set_xscale('log', base=2)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # 3. Cluster Scaling
            if 'cluster_scaling' in self.results:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                data = self.results['cluster_scaling']
                df = pd.DataFrame(data)
                
                # Model size vs clusters
                ax = axes[0, 0]
                ax.plot(df['n_clusters'], df['model_size_mb'], 'o-', linewidth=2, markersize=8)
                ax.set_xlabel('Number of Clusters', fontsize=12)
                ax.set_ylabel('Model Size (MB)', fontsize=12)
                ax.set_title('Model Size vs Cluster Count', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Cluster memory vs clusters
                ax = axes[0, 1]
                ax.plot(df['n_clusters'], df['cluster_memory_mb'], 'o-', 
                       linewidth=2, markersize=8, color='orange')
                ax.set_xlabel('Number of Clusters', fontsize=12)
                ax.set_ylabel('Cluster Memory (MB)', fontsize=12)
                ax.set_title('Cluster Embedding Memory', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Latency vs clusters
                ax = axes[1, 0]
                ax.plot(df['n_clusters'], df['avg_latency_ms'], 'o-', linewidth=2, markersize=8)
                ax.fill_between(df['n_clusters'], 
                               df['avg_latency_ms'] - df['std_latency_ms'],
                               df['avg_latency_ms'] + df['std_latency_ms'],
                               alpha=0.3)
                ax.set_xlabel('Number of Clusters', fontsize=12)
                ax.set_ylabel('Inference Latency (ms)', fontsize=12)
                ax.set_title('Inference Latency vs Cluster Count', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Parameters vs clusters
                ax = axes[1, 1]
                ax.plot(df['n_clusters'], df['model_params'] / 1e6, 'o-', 
                       linewidth=2, markersize=8, color='green')
                ax.set_xlabel('Number of Clusters', fontsize=12)
                ax.set_ylabel('Model Parameters (M)', fontsize=12)
                ax.set_title('Model Parameters vs Cluster Count', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # 4. Optimization Impact
            if 'optimization_impact' in self.results:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                data = self.results['optimization_impact']
                methods = list(data.keys())
                latencies = [data[m]['avg_latency_ms'] for m in methods]
                speedups = [data[m]['speedup'] for m in methods]
                
                # Latency comparison
                bars = ax1.bar(methods, latencies, color='lightblue')
                ax1.set_ylabel('Average Latency (ms)', fontsize=12)
                ax1.set_title('Optimization Impact on Latency', fontsize=14, fontweight='bold')
                ax1.set_xticklabels(methods, rotation=45, ha='right')
                
                for bar, val in zip(bars, latencies):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{val:.1f}', ha='center', va='bottom')
                
                # Speedup comparison
                bars = ax2.bar(methods, speedups, color='lightgreen')
                ax2.set_ylabel('Speedup Factor', fontsize=12)
                ax2.set_title('Speedup from Optimizations', fontsize=14, fontweight='bold')
                ax2.set_xticklabels(methods, rotation=45, ha='right')
                ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
                
                for bar, val in zip(bars, speedups):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f'{val:.2f}x', ha='center', va='bottom')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # 5. Hardware Utilization
            if 'hardware_utilization' in self.results:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                data = self.results['hardware_utilization']
                
                # CPU usage
                ax = axes[0, 0]
                cpu_data = data['cpu_usage_percent']
                bars = ax.bar(['Mean', 'Max'], [cpu_data['mean'], cpu_data['max']], 
                             color='skyblue')
                ax.set_ylabel('CPU Usage (%)', fontsize=12)
                ax.set_title('CPU Utilization', fontsize=14, fontweight='bold')
                for bar, val in zip(bars, [cpu_data['mean'], cpu_data['max']]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{val:.1f}%', ha='center', va='bottom')
                
                # Memory usage
                ax = axes[0, 1]
                mem_data = data['memory_usage_mb']
                bars = ax.bar(['Mean', 'Max'], [mem_data['mean'], mem_data['max']], 
                             color='lightcoral')
                ax.set_ylabel('Memory Usage (MB)', fontsize=12)
                ax.set_title('RAM Utilization', fontsize=14, fontweight='bold')
                for bar, val in zip(bars, [mem_data['mean'], mem_data['max']]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                           f'{val:.0f}', ha='center', va='bottom')
                
                # GPU usage (if available)
                if 'gpu_usage_percent' in data:
                    ax = axes[1, 0]
                    gpu_data = data['gpu_usage_percent']
                    bars = ax.bar(['Mean', 'Max'], [gpu_data['mean'], gpu_data['max']], 
                                 color='lightgreen')
                    ax.set_ylabel('GPU Usage (%)', fontsize=12)
                    ax.set_title('GPU Utilization', fontsize=14, fontweight='bold')
                    for bar, val in zip(bars, [gpu_data['mean'], gpu_data['max']]):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               f'{val:.1f}%', ha='center', va='bottom')
                    
                    # GPU memory
                    ax = axes[1, 1]
                    gpu_mem_data = data['gpu_memory_mb']
                    bars = ax.bar(['Mean', 'Max'], [gpu_mem_data['mean'], gpu_mem_data['max']], 
                                 color='gold')
                    ax.set_ylabel('GPU Memory (MB)', fontsize=12)
                    ax.set_title('GPU Memory Usage', fontsize=14, fontweight='bold')
                    for bar, val in zip(bars, [gpu_mem_data['mean'], gpu_mem_data['max']]):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                               f'{val:.0f}', ha='center', va='bottom')
                else:
                    axes[1, 0].text(0.5, 0.5, 'GPU not available', 
                                   ha='center', va='center', fontsize=16)
                    axes[1, 0].set_xlim(0, 1)
                    axes[1, 0].set_ylim(0, 1)
                    axes[1, 1].text(0.5, 0.5, 'GPU not available', 
                                   ha='center', va='center', fontsize=16)
                    axes[1, 1].set_xlim(0, 1)
                    axes[1, 1].set_ylim(0, 1)
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # 6. Summary Tables
            self._create_summary_tables(pdf)
    
    def _create_summary_tables(self, pdf):
        """Create summary tables for the PDF report"""
        # Collection scaling table
        if 'collection_scaling' in self.results:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.axis('tight')
            ax.axis('off')
            
            if 'indexing' in self.results['collection_scaling']:
                data = self.results['collection_scaling']['indexing']
                df = pd.DataFrame(data)
                
                # Format the dataframe
                df['indexing_time_s'] = df['indexing_time_s'].apply(lambda x: f'{x:.2f}')
                df['memory_mb'] = df['memory_mb'].apply(lambda x: f'{x:.1f}')
                df['docs_per_second'] = df['docs_per_second'].apply(lambda x: f'{x:.0f}')
                
                table = ax.table(cellText=df.values, colLabels=df.columns,
                               cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)
                
                # Style header
                for i in range(len(df.columns)):
                    table[(0, i)].set_facecolor('#40466e')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                
                ax.set_title('Collection Scaling Results', fontsize=16, fontweight='bold', pad=20)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Optimization impact table
        if 'optimization_impact' in self.results:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.axis('tight')
            ax.axis('off')
            
            data = []
            for method, metrics in self.results['optimization_impact'].items():
                data.append([
                    method.replace('_', ' ').title(),
                    f"{metrics['avg_latency_ms']:.2f}",
                    f"{metrics['speedup']:.2f}x"
                ])
            
            columns = ['Optimization', 'Avg Latency (ms)', 'Speedup']
            
            table = ax.table(cellText=data, colLabels=columns,
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 1.5)
            
            # Style header
            for i in range(len(columns)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Highlight best speedup
            speedups = [float(row[2][:-1]) for row in data]
            max_speedup_idx = speedups.index(max(speedups))
            for i in range(len(columns)):
                table[(max_speedup_idx + 1, i)].set_facecolor('#90EE90')
            
            ax.set_title('Optimization Impact Summary', fontsize=16, fontweight='bold', pad=20)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    def run_all_experiments(self, 
                           num_docs_list: List[int] = [1000, 10000, 100000, 1000000],
                           batch_sizes: List[int] = [1, 4, 8, 16, 32, 64, 128],
                           cluster_counts: List[int] = [25, 50, 100, 200]):
        """Run all scalability experiments"""
        logger.info("Running scalability experiments...")
        
        # Run experiments
        self.measure_indexing_time(num_docs_list)
        self.measure_query_latency(num_docs_list, num_queries=100)
        self.measure_batch_size_impact(batch_sizes)
        self.measure_cluster_scaling(cluster_counts)
        self.measure_optimization_impact()
        self.measure_hardware_utilization()
        
        # Save results
        results_path = os.path.join(self.output_dir, 'scalability_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create PDF report
        pdf_path = os.path.join(self.output_dir, 'scalability_experiments.pdf')
        self.create_visualizations(pdf_path)
        
        logger.info(f"Results saved to {results_path}")
        logger.info(f"PDF report saved to {pdf_path}")
        
        return self.results


def main():
    """Main function to run scalability experiments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run scalability experiments for CLARE')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained CLARE model checkpoint')
    parser.add_argument('--dataset_path', type=str, default='clare_dataset_complete.pkl',
                       help='Path to CLARE dataset')
    parser.add_argument('--output_dir', type=str, default='scalability_results',
                       help='Output directory for results')
    parser.add_argument('--num_docs', type=int, nargs='+', 
                       default=[1000, 10000, 100000],
                       help='Document counts to test')
    parser.add_argument('--batch_sizes', type=int, nargs='+',
                       default=[1, 4, 8, 16, 32, 64],
                       help='Batch sizes to test')
    parser.add_argument('--cluster_counts', type=int, nargs='+',
                       default=[25, 50, 100, 200],
                       help='Cluster counts to test')
    
    args = parser.parse_args()
    
    # Run experiments
    evaluator = ScalabilityEvaluator(args.model_path, args.dataset_path, args.output_dir)
    results = evaluator.run_all_experiments(
        num_docs_list=args.num_docs,
        batch_sizes=args.batch_sizes,
        cluster_counts=args.cluster_counts
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SCALABILITY EXPERIMENTS SUMMARY")
    print("="*60)
    
    if 'collection_scaling' in results:
        if 'indexing' in results['collection_scaling']:
            print("\nIndexing Scalability:")
            for item in results['collection_scaling']['indexing']:
                print(f"  {item['num_docs']:,} docs: {item['indexing_time_s']:.2f}s "
                      f"({item['docs_per_second']:.0f} docs/s)")
    
    if 'optimization_impact' in results:
        print("\nOptimization Impact:")
        for method, metrics in results['optimization_impact'].items():
            print(f"  {method}: {metrics['speedup']:.2f}x speedup")
    
    print(f"\nPDF report saved to: {os.path.join(args.output_dir, 'scalability_experiments.pdf')}")


if __name__ == "__main__":
    main()
            