#!/usr/bin/env python3
"""
Domain Adaptation Experiments for CLARE Paper
Tests cross-domain performance and domain-specific fine-tuning
Generates results and visualizations in PDF format
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import logging
from sklearn.metrics import confusion_matrix
from collections import defaultdict

# Import CLARE components
from clare_framework import CLAREModel, CLARETrainer
from clare_dataset_integration import CLARETorchDataset, DatasetConfig
from transformers import get_linear_schedule_with_warmup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


class DomainAdaptationEvaluator:
    """Evaluates domain adaptation capabilities of CLARE"""
    
    def __init__(self, base_model_path: str, dataset_path: str, output_dir: str = "domain_adaptation_results"):
        self.base_model_path = base_model_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load dataset
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
        
        # Results storage
        self.results = {
            'zero_shot_transfer': {},
            'domain_fine_tuning': {},
            'few_shot_adaptation': {},
            'cross_domain_similarity': {},
            'cluster_analysis': {}
        }
    
    def get_domain_data(self, domain: str, split: str = 'test'):
        """Get data for a specific domain"""
        domain_data = []
        for item in self.dataset[split]:
            if item['dataset'] == domain:
                domain_data.append(item)
        return domain_data
    
    def get_all_domains(self):
        """Get list of all domains in dataset"""
        domains = set()
        for split in ['train', 'validation', 'test']:
            for item in self.dataset[split]:
                domains.add(item['dataset'])
        return list(domains)
    
    def evaluate_zero_shot_transfer(self):
        """Evaluate zero-shot transfer across domains"""
        logger.info("Evaluating zero-shot transfer...")
        
        # Load base model
        checkpoint = torch.load(self.base_model_path, map_location=self.device)
        model = CLAREModel(
            n_clusters=checkpoint['config']['n_clusters'],
            cluster_layers=checkpoint['config'].get('cluster_layers', [3, 6, 9])
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Get all domains
        domains = self.get_all_domains()
        results = {}
        
        for target_domain in domains:
            logger.info(f"Testing on {target_domain}...")
            
            # Get test data for target domain
            test_data = self.get_domain_data(target_domain, 'test')
            
            if not test_data:
                continue
            
            # Create dataloader
            dataset_config = DatasetConfig(
                max_query_length=128,
                max_doc_length=512,
                negative_sampling_ratio=4
            )
            
            test_dataset = CLARETorchDataset(test_data, model.tokenizer, dataset_config)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            
            # Evaluate
            metrics = self._evaluate_model(model, test_loader)
            results[target_domain] = metrics
        
        self.results['zero_shot_transfer'] = results
        return results
    
    def evaluate_domain_fine_tuning(self, target_domains: List[str] = None, 
                                   fine_tune_epochs: int = 3):
        """Evaluate performance after domain-specific fine-tuning"""
        logger.info("Evaluating domain fine-tuning...")
        
        if target_domains is None:
            target_domains = self.get_all_domains()
        
        results = {}
        
        for target_domain in target_domains:
            logger.info(f"Fine-tuning for {target_domain}...")
            
            # Load base model
            checkpoint = torch.load(self.base_model_path, map_location=self.device)
            model = CLAREModel(
                n_clusters=checkpoint['config']['n_clusters'],
                cluster_layers=checkpoint['config'].get('cluster_layers', [3, 6, 9])
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            # Get domain data
            train_data = self.get_domain_data(target_domain, 'train')
            val_data = self.get_domain_data(target_domain, 'validation')
            test_data = self.get_domain_data(target_domain, 'test')
            
            if not train_data or not test_data:
                continue
            
            # Create datasets
            dataset_config = DatasetConfig(
                max_query_length=128,
                max_doc_length=512,
                negative_sampling_ratio=4
            )
            
            train_dataset = CLARETorchDataset(train_data, model.tokenizer, dataset_config)
            val_dataset = CLARETorchDataset(val_data, model.tokenizer, dataset_config) if val_data else None
            test_dataset = CLARETorchDataset(test_data, model.tokenizer, dataset_config)
            
            # Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False) if val_dataset else None
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            
            # Fine-tune model
            fine_tuned_model = self._fine_tune_model(
                model, train_loader, val_loader, 
                epochs=fine_tune_epochs, 
                learning_rate=5e-6
            )
            
            # Evaluate fine-tuned model
            metrics = self._evaluate_model(fine_tuned_model, test_loader)
            
            # Compare with zero-shot
            zero_shot_metrics = self.results['zero_shot_transfer'].get(target_domain, {})
            
            results[target_domain] = {
                'fine_tuned': metrics,
                'zero_shot': zero_shot_metrics,
                'improvement': {
                    'ndcg@10': metrics.get('ndcg@10', 0) - zero_shot_metrics.get('ndcg@10', 0),
                    'accuracy': metrics.get('accuracy', 0) - zero_shot_metrics.get('accuracy', 0)
                }
            }
            
            # Save fine-tuned model
            model_path = os.path.join(self.output_dir, f'clare_{target_domain}_finetuned.pt')
            torch.save({
                'model_state_dict': fine_tuned_model.state_dict(),
                'config': checkpoint['config'],
                'domain': target_domain
            }, model_path)
        
        self.results['domain_fine_tuning'] = results
        return results
    
    def evaluate_few_shot_adaptation(self, shot_counts: List[int] = [1, 5, 10, 20]):
        """Evaluate few-shot adaptation performance"""
        logger.info("Evaluating few-shot adaptation...")
        
        domains = self.get_all_domains()
        results = {}
        
        for target_domain in domains:
            logger.info(f"Few-shot adaptation for {target_domain}...")
            domain_results = {}
            
            # Get domain data
            train_data = self.get_domain_data(target_domain, 'train')
            test_data = self.get_domain_data(target_domain, 'test')
            
            if not train_data or not test_data:
                continue
            
            for n_shots in shot_counts:
                logger.info(f"Testing {n_shots}-shot adaptation...")
                
                # Load base model
                checkpoint = torch.load(self.base_model_path, map_location=self.device)
                model = CLAREModel(
                    n_clusters=checkpoint['config']['n_clusters'],
                    cluster_layers=checkpoint['config'].get('cluster_layers', [3, 6, 9])
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                
                # Sample n-shot examples
                shot_data = train_data[:n_shots] if len(train_data) >= n_shots else train_data
                
                # Create datasets
                dataset_config = DatasetConfig(
                    max_query_length=128,
                    max_doc_length=512,
                    negative_sampling_ratio=1  # Less negatives for few-shot
                )
                
                shot_dataset = CLARETorchDataset(shot_data, model.tokenizer, dataset_config)
                test_dataset = CLARETorchDataset(test_data, model.tokenizer, dataset_config)
                
                shot_loader = DataLoader(shot_dataset, batch_size=min(n_shots, 4), shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
                
                # Few-shot fine-tune
                adapted_model = self._fine_tune_model(
                    model, shot_loader, None, 
                    epochs=10,  # More epochs for few examples
                    learning_rate=1e-5
                )
                
                # Evaluate
                metrics = self._evaluate_model(adapted_model, test_loader)
                domain_results[n_shots] = metrics
            
            results[target_domain] = domain_results
        
        self.results['few_shot_adaptation'] = results
        return results
    
    def analyze_cross_domain_similarity(self):
        """Analyze cluster activation similarity across domains"""
        logger.info("Analyzing cross-domain cluster similarity...")
        
        # Load model
        checkpoint = torch.load(self.base_model_path, map_location=self.device)
        model = CLAREModel(
            n_clusters=checkpoint['config']['n_clusters'],
            cluster_layers=checkpoint['config'].get('cluster_layers', [3, 6, 9])
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Get cluster activations for each domain
        domains = self.get_all_domains()
        domain_activations = {}
        
        for domain in domains:
            logger.info(f"Extracting activations for {domain}...")
            
            test_data = self.get_domain_data(domain, 'test')[:100]  # Sample
            if not test_data:
                continue
            
            activations = []
            
            for item in test_data:
                # Encode query
                inputs = model.tokenizer(
                    item['query'], 
                    return_tensors='pt',
                    padding=True, 
                    truncation=True, 
                    max_length=128
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    activations.append(outputs['cluster_activations'].cpu().numpy())
            
            if activations:
                domain_activations[domain] = np.vstack(activations)
        
        # Compute cross-domain similarity matrix
        similarity_matrix = np.zeros((len(domains), len(domains)))
        
        for i, domain1 in enumerate(domains):
            for j, domain2 in enumerate(domains):
                if domain1 in domain_activations and domain2 in domain_activations:
                    # Average cosine similarity between domain activations
                    acts1 = domain_activations[domain1]
                    acts2 = domain_activations[domain2]
                    
                    # Compute mean activation patterns
                    mean_acts1 = np.mean(acts1, axis=0)
                    mean_acts2 = np.mean(acts2, axis=0)
                    
                    # Cosine similarity
                    similarity = np.dot(mean_acts1, mean_acts2) / (
                        np.linalg.norm(mean_acts1) * np.linalg.norm(mean_acts2)
                    )
                    similarity_matrix[i, j] = similarity
        
        self.results['cross_domain_similarity'] = {
            'domains': domains,
            'similarity_matrix': similarity_matrix.tolist(),
            'domain_activations': {k: v.mean(axis=0).tolist() for k, v in domain_activations.items()}
        }
        
        return similarity_matrix, domains
    
    def analyze_domain_specific_clusters(self):
        """Analyze which clusters are most active for each domain"""
        logger.info("Analyzing domain-specific cluster patterns...")
        
        # Load model
        checkpoint = torch.load(self.base_model_path, map_location=self.device)
        model = CLAREModel(
            n_clusters=checkpoint['config']['n_clusters'],
            cluster_layers=checkpoint['config'].get('cluster_layers', [3, 6, 9])
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Get cluster discovery info if available
        cluster_words = None
        if 'cluster_discovery' in checkpoint:
            from clare_framework import SemanticClusterDiscovery
            cluster_discovery = SemanticClusterDiscovery(k=checkpoint['config']['n_clusters'])
            cluster_discovery.W = checkpoint['cluster_discovery']['W']
            cluster_discovery.H = checkpoint['cluster_discovery']['H']
            cluster_discovery.vocab = checkpoint['vocabulary']
            cluster_words = cluster_discovery.get_cluster_words(top_k=10)
        
        domains = self.get_all_domains()
        domain_cluster_stats = {}
        
        for domain in domains:
            logger.info(f"Analyzing clusters for {domain}...")
            
            test_data = self.get_domain_data(domain, 'test')[:200]
            if not test_data:
                continue
            
            cluster_activations = []
            
            for item in test_data:
                # Encode query
                inputs = model.tokenizer(
                    item['query'],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    cluster_activations.append(outputs['cluster_activations'][0].cpu().numpy())
            
            if cluster_activations:
                cluster_activations = np.vstack(cluster_activations)
                
                # Compute statistics
                mean_activation = np.mean(cluster_activations, axis=0)
                std_activation = np.std(cluster_activations, axis=0)
                
                # Find top clusters
                top_cluster_indices = np.argsort(mean_activation)[-10:][::-1]
                
                top_clusters = []
                for idx in top_cluster_indices:
                    cluster_info = {
                        'cluster_id': int(idx),
                        'mean_activation': float(mean_activation[idx]),
                        'std_activation': float(std_activation[idx])
                    }
                    
                    if cluster_words:
                        cluster_info['words'] = cluster_words.get(idx, [])[:5]
                    
                    top_clusters.append(cluster_info)
                
                domain_cluster_stats[domain] = {
                    'mean_activation': mean_activation.tolist(),
                    'std_activation': std_activation.tolist(),
                    'top_clusters': top_clusters,
                    'sparsity': float((cluster_activations > 0.01).mean())
                }
        
        self.results['cluster_analysis'] = domain_cluster_stats
        return domain_cluster_stats
    
    def _fine_tune_model(self, model, train_loader, val_loader, epochs, learning_rate):
        """Fine-tune model on domain data"""
        # Initialize trainer
        trainer = CLARETrainer(model, self.device)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            
            for batch in train_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                if batch['neg_doc_input_ids'].numel() == 0:
                    continue
                
                optimizer.zero_grad()
                
                # Forward pass
                query_outputs = model(
                    input_ids=batch['query_input_ids'],
                    attention_mask=batch['query_attention_mask']
                )
                
                pos_doc_outputs = model(
                    input_ids=batch['pos_doc_input_ids'],
                    attention_mask=batch['pos_doc_attention_mask']
                )
                
                neg_doc_outputs_list = []
                if batch['neg_doc_input_ids'].dim() == 3:
                    for i in range(batch['neg_doc_input_ids'].size(1)):
                        neg_outputs = model(
                            input_ids=batch['neg_doc_input_ids'][:, i, :],
                            attention_mask=batch['neg_doc_attention_mask'][:, i, :]
                        )
                        neg_doc_outputs_list.append(neg_outputs)
                
                # Compute loss
                loss, _ = trainer.compute_loss(
                    query_outputs, pos_doc_outputs, neg_doc_outputs_list
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
        
        return model
    
    def _evaluate_model(self, model, test_loader):
        """Evaluate model on test data"""
        model.eval()
        
        all_scores = []
        all_labels = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                if batch['neg_doc_input_ids'].numel() == 0:
                    continue
                
                # Forward pass
                query_outputs = model(
                    input_ids=batch['query_input_ids'],
                    attention_mask=batch['query_attention_mask']
                )
                
                pos_doc_outputs = model(
                    input_ids=batch['pos_doc_input_ids'],
                    attention_mask=batch['pos_doc_attention_mask']
                )
                
                # Positive scores
                pos_scores = torch.sum(
                    query_outputs['cluster_activations'] * pos_doc_outputs['cluster_activations'],
                    dim=1
                )
                
                # Negative scores
                neg_scores_list = []
                if batch['neg_doc_input_ids'].dim() == 3:
                    for i in range(batch['neg_doc_input_ids'].size(1)):
                        neg_outputs = model(
                            input_ids=batch['neg_doc_input_ids'][:, i, :],
                            attention_mask=batch['neg_doc_attention_mask'][:, i, :]
                        )
                        neg_score = torch.sum(
                            query_outputs['cluster_activations'] * neg_outputs['cluster_activations'],
                            dim=1
                        )
                        neg_scores_list.append(neg_score)
                
                # Collect for metrics
                batch_size = pos_scores.size(0)
                for i in range(batch_size):
                    all_scores.append(pos_scores[i].item())
                    all_labels.append(1)
                    
                    for neg_score in neg_scores_list:
                        all_scores.append(neg_score[i].item())
                        all_labels.append(0)
                    
                    # Accuracy
                    if neg_scores_list:
                        neg_scores_batch = torch.stack([ns[i] for ns in neg_scores_list])
                        if pos_scores[i] > neg_scores_batch.max():
                            correct += 1
                        total += 1
        
        # Compute metrics
        metrics = self._compute_ranking_metrics(all_scores, all_labels)
        metrics['accuracy'] = correct / total if total > 0 else 0
        
        return metrics
    
    def _compute_ranking_metrics(self, scores, labels):
        """Compute ranking metrics"""
        # Group into queries (assuming fixed number per query)
        metrics = {}
        
        # Simple nDCG@10 calculation
        query_groups = []
        current_group = {'scores': [], 'labels': []}
        
        for score, label in zip(scores, labels):
            current_group['scores'].append(score)
            current_group['labels'].append(label)
            
            if len(current_group['scores']) >= 5:  # 1 positive + 4 negatives
                query_groups.append(current_group)
                current_group = {'scores': [], 'labels': []}
        
        ndcg_scores = []
        for group in query_groups:
            sorted_indices = np.argsort(group['scores'])[::-1]
            sorted_labels = [group['labels'][i] for i in sorted_indices]
            
            # DCG@10
            dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(sorted_labels[:10]))
            ideal_labels = sorted(group['labels'], reverse=True)[:10]
            idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_labels))
            
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
        
        metrics['ndcg@10'] = np.mean(ndcg_scores) if ndcg_scores else 0
        
        return metrics
    
    def create_visualizations(self, pdf_path: str):
        """Create domain adaptation visualizations"""
        with PdfPages(pdf_path) as pdf:
            # 1. Zero-shot transfer performance
            if 'zero_shot_transfer' in self.results:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                data = self.results['zero_shot_transfer']
                domains = list(data.keys())
                ndcg_scores = [data[d].get('ndcg@10', 0) for d in domains]
                accuracies = [data[d].get('accuracy', 0) for d in domains]
                
                x = np.arange(len(domains))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, ndcg_scores, width, label='nDCG@10')
                bars2 = ax.bar(x + width/2, accuracies, width, label='Accuracy')
                
                ax.set_xlabel('Domain', fontsize=12)
                ax.set_ylabel('Score', fontsize=12)
                ax.set_title('Zero-Shot Transfer Performance Across Domains', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(domains, rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # 2. Fine-tuning improvements
            if 'domain_fine_tuning' in self.results:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                data = self.results['domain_fine_tuning']
                domains = list(data.keys())
                
                # nDCG improvements
                ax = axes[0]
                zero_shot_ndcg = [data[d]['zero_shot'].get('ndcg@10', 0) for d in domains]
                fine_tuned_ndcg = [data[d]['fine_tuned'].get('ndcg@10', 0) for d in domains]
                improvements = [data[d]['improvement']['ndcg@10'] for d in domains]
                
                x = np.arange(len(domains))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, zero_shot_ndcg, width, label='Zero-shot', alpha=0.7)
                bars2 = ax.bar(x + width/2, fine_tuned_ndcg, width, label='Fine-tuned', alpha=0.7)
                
                # Add improvement arrows
                for i, (zs, ft, imp) in enumerate(zip(zero_shot_ndcg, fine_tuned_ndcg, improvements)):
                    if imp > 0:
                        ax.annotate('', xy=(i + width/2, ft), xytext=(i - width/2, zs),
                                   arrowprops=dict(arrowstyle='->', color='green', lw=2))
                        ax.text(i, max(zs, ft) + 0.02, f'+{imp:.3f}', 
                               ha='center', va='bottom', fontsize=9, color='green')
                
                ax.set_xlabel('Domain', fontsize=12)
                ax.set_ylabel('nDCG@10', fontsize=12)
                ax.set_title('Domain Fine-tuning: nDCG@10 Improvements', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(domains, rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                
                # Accuracy improvements
                ax = axes[1]
                zero_shot_acc = [data[d]['zero_shot'].get('accuracy', 0) for d in domains]
                fine_tuned_acc = [data[d]['fine_tuned'].get('accuracy', 0) for d in domains]
                
                bars1 = ax.bar(x - width/2, zero_shot_acc, width, label='Zero-shot', alpha=0.7)
                bars2 = ax.bar(x + width/2, fine_tuned_acc, width, label='Fine-tuned', alpha=0.7)
                
                ax.set_xlabel('Domain', fontsize=12)
                ax.set_ylabel('Accuracy', fontsize=12)
                ax.set_title('Domain Fine-tuning: Accuracy Improvements', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(domains, rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # 3. Few-shot adaptation curves
            if 'few_shot_adaptation' in self.results:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                data = self.results['few_shot_adaptation']
                
                for idx, metric in enumerate(['ndcg@10', 'accuracy']):
                    ax = axes[idx]
                    
                    for domain, shot_results in data.items():
                        shot_counts = sorted(shot_results.keys())
                        scores = [shot_results[n].get(metric, 0) for n in shot_counts]
                        
                        ax.plot(shot_counts, scores, 'o-', label=domain, linewidth=2, markersize=8)
                    
                    ax.set_xlabel('Number of Examples', fontsize=12)
                    ax.set_ylabel(metric.upper() if metric == 'ndcg@10' else metric.capitalize(), fontsize=12)
                    ax.set_title(f'Few-Shot Adaptation: {metric.upper()}', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_xscale('log')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # 4. Cross-domain similarity heatmap
            if 'cross_domain_similarity' in self.results:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                data = self.results['cross_domain_similarity']
                similarity_matrix = np.array(data['similarity_matrix'])
                domains = data['domains']
                
                # Create heatmap
                im = ax.imshow(similarity_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
                
                # Set ticks
                ax.set_xticks(np.arange(len(domains)))
                ax.set_yticks(np.arange(len(domains)))
                ax.set_xticklabels(domains, rotation=45, ha='right')
                ax.set_yticklabels(domains)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Cosine Similarity', fontsize=12)
                
                # Add text annotations
                for i in range(len(domains)):
                    for j in range(len(domains)):
                        text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                                     ha='center', va='center', color='black' if similarity_matrix[i, j] < 0.5 else 'white')
                
                ax.set_title('Cross-Domain Cluster Activation Similarity', fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # 5. Domain-specific cluster analysis
            if 'cluster_analysis' in self.results:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                axes = axes.ravel()
                
                data = self.results['cluster_analysis']
                
                for idx, (domain, stats) in enumerate(data.items()):
                    if idx >= 4:
                        break
                    
                    ax = axes[idx]
                    
                    # Plot top cluster activations
                    top_clusters = stats['top_clusters']
                    cluster_ids = [c['cluster_id'] for c in top_clusters[:10]]
                    activations = [c['mean_activation'] for c in top_clusters[:10]]
                    stds = [c['std_activation'] for c in top_clusters[:10]]
                    
                    x = np.arange(len(cluster_ids))
                    bars = ax.bar(x, activations, yerr=stds, capsize=5)
                    
                    # Color bars by activation strength
                    colors = plt.cm.viridis(np.array(activations) / max(activations))
                    for bar, color in zip(bars, colors):
                        bar.set_color(color)
                    
                    ax.set_xlabel('Cluster ID', fontsize=10)
                    ax.set_ylabel('Mean Activation', fontsize=10)
                    ax.set_title(f'{domain} - Top Activated Clusters', fontsize=12, fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels(cluster_ids)
                    
                    # Add sparsity info
                    ax.text(0.95, 0.95, f"Sparsity: {stats['sparsity']:.3f}", 
                           transform=ax.transAxes, ha='right', va='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # 6. Summary table
            self._create_summary_table(pdf)
    
    def _create_summary_table(self, pdf):
        """Create summary table for domain adaptation results"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data
        if 'domain_fine_tuning' in self.results:
            data = []
            domains = list(self.results['domain_fine_tuning'].keys())
            
            for domain in domains:
                results = self.results['domain_fine_tuning'][domain]
                zero_shot = results['zero_shot']
                fine_tuned = results['fine_tuned']
                
                data.append([
                    domain,
                    f"{zero_shot.get('ndcg@10', 0):.4f}",
                    f"{fine_tuned.get('ndcg@10', 0):.4f}",
                    f"{results['improvement']['ndcg@10']:.4f}",
                    f"{zero_shot.get('accuracy', 0):.4f}",
                    f"{fine_tuned.get('accuracy', 0):.4f}",
                    f"{results['improvement']['accuracy']:.4f}"
                ])
            
            columns = ['Domain', 'Zero-shot\nnDCG@10', 'Fine-tuned\nnDCG@10', 'nDCG\nImprovement',
                      'Zero-shot\nAccuracy', 'Fine-tuned\nAccuracy', 'Accuracy\nImprovement']
            
            table = ax.table(cellText=data, colLabels=columns,
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Style header
            for i in range(len(columns)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Highlight improvements
            for i in range(1, len(data) + 1):
                # nDCG improvement
                if float(data[i-1][3]) > 0:
                    table[(i, 3)].set_facecolor('#90EE90')
                # Accuracy improvement  
                if float(data[i-1][6]) > 0:
                    table[(i, 6)].set_facecolor('#90EE90')
            
            ax.set_title('Domain Adaptation Results Summary', fontsize=16, fontweight='bold', pad=20)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def run_all_experiments(self, fine_tune_epochs: int = 3,
                           shot_counts: List[int] = [1, 5, 10, 20]):
        """Run all domain adaptation experiments"""
        logger.info("Running domain adaptation experiments...")
        
        # Run experiments
        self.evaluate_zero_shot_transfer()
        self.evaluate_domain_fine_tuning(fine_tune_epochs=fine_tune_epochs)
        self.evaluate_few_shot_adaptation(shot_counts=shot_counts)
        self.analyze_cross_domain_similarity()
        self.analyze_domain_specific_clusters()
        
        # Save results
        results_path = os.path.join(self.output_dir, 'domain_adaptation_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create PDF report
        pdf_path = os.path.join(self.output_dir, 'domain_adaptation_experiments.pdf')
        self.create_visualizations(pdf_path)
        
        logger.info(f"Results saved to {results_path}")
        logger.info(f"PDF report saved to {pdf_path}")
        
        return self.results


def main():
    """Main function to run domain adaptation experiments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run domain adaptation experiments for CLARE')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained CLARE model checkpoint')
    parser.add_argument('--dataset_path', type=str, default='clare_dataset_complete.pkl',
                       help='Path to CLARE dataset')
    parser.add_argument('--output_dir', type=str, default='domain_adaptation_results',
                       help='Output directory for results')
    parser.add_argument('--fine_tune_epochs', type=int, default=3,
                       help='Number of epochs for domain fine-tuning')
    parser.add_argument('--shot_counts', type=int, nargs='+', default=[1, 5, 10, 20],
                       help='Number of examples for few-shot adaptation')
    
    args = parser.parse_args()
    
    # Run experiments
    evaluator = DomainAdaptationEvaluator(args.model_path, args.dataset_path, args.output_dir)
    results = evaluator.run_all_experiments(
        fine_tune_epochs=args.fine_tune_epochs,
        shot_counts=args.shot_counts
    )
    
    # Print summary
    print("\n" + "="*60)
    print("DOMAIN ADAPTATION EXPERIMENTS SUMMARY")
    print("="*60)
    
    if 'zero_shot_transfer' in results:
        print("\nZero-shot Transfer:")
        for domain, metrics in results['zero_shot_transfer'].items():
            print(f"  {domain}: nDCG@10={metrics.get('ndcg@10', 0):.4f}, "
                  f"Accuracy={metrics.get('accuracy', 0):.4f}")
    
    if 'domain_fine_tuning' in results:
        print("\nDomain Fine-tuning Improvements:")
        for domain, data in results['domain_fine_tuning'].items():
            print(f"  {domain}: nDCG@10 +{data['improvement']['ndcg@10']:.4f}, "
                  f"Accuracy +{data['improvement']['accuracy']:.4f}")
    
    print(f"\nPDF report saved to: {os.path.join(args.output_dir, 'domain_adaptation_experiments.pdf')}")


if __name__ == "__main__":
    main()