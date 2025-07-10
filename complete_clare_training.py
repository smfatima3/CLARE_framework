#!/usr/bin/env python3
"""
CLARE Training Script - Paper Aligned Implementation
Implements the two-phase training procedure from the paper:
Phase 1: Semantic Cluster Discovery
Phase 2: Cluster-Aware Model Training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import logging
import os
import json
import wandb
from datetime import datetime
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import time
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import scipy.sparse
from collections import defaultdict
import math

# Import our custom modules
from clare_framework import CLAREModel, SemanticClusterDiscovery, CLAREInference
from clare_dataset_integration import (
    CLAREDatasetBuilder, CLARETorchDataset, DatasetConfig, 
    DatasetAnalyzer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLARETrainer:
    """Trainer for CLARE model following paper's training procedure"""
    
    def __init__(self, model: CLAREModel, device: torch.device = torch.device('cuda')):
        self.model = model.to(device)
        self.device = device
        
    def compute_loss(self, query_outputs: Dict, pos_doc_outputs: Dict, 
                    neg_doc_outputs_list: List[Dict], lambda3: float = 0.1, 
                    lambda4: float = 0.01) -> Tuple[torch.Tensor, Dict]:
        """
        Compute CLARE loss function as specified in paper:
        L_total = L_rank + λ₃L_cluster + λ₄L_sparsity
        """
        # Extract cluster activations
        query_clusters = query_outputs['cluster_activations']  # [batch_size, n_clusters]
        pos_doc_clusters = pos_doc_outputs['cluster_activations']  # [batch_size, n_clusters]
        
        batch_size = query_clusters.size(0)
        
        # 1. Ranking Loss (Contrastive)
        # Paper uses contrastive loss with multiple negatives
        positive_scores = torch.sum(query_clusters * pos_doc_clusters, dim=1)  # [batch_size]
        
        negative_scores = []
        for neg_doc_outputs in neg_doc_outputs_list:
            neg_clusters = neg_doc_outputs['cluster_activations']  # [batch_size, n_clusters]
            neg_score = torch.sum(query_clusters * neg_clusters, dim=1)  # [batch_size]
            negative_scores.append(neg_score)
        
        if negative_scores:
            negative_scores = torch.stack(negative_scores, dim=1)  # [batch_size, num_negatives]
            
            # Compute contrastive loss as in paper
            # L_rank = -log(exp(score(q,d+)) / Σ exp(score(q,d-)))
            pos_exp = torch.exp(positive_scores).unsqueeze(1)  # [batch_size, 1]
            neg_exp = torch.exp(negative_scores)  # [batch_size, num_negatives]
            
            denominator = pos_exp + torch.sum(neg_exp, dim=1, keepdim=True)
            ranking_loss = -torch.log(pos_exp / denominator).mean()
        else:
            ranking_loss = torch.tensor(0.0, device=self.device)
        
        # 2. Cluster Consistency Loss (KL Divergence)
        # L_cluster = KL(α_q || α_d+) as specified in paper
        cluster_loss = nn.functional.kl_div(
            torch.log(query_clusters + 1e-8),  # Add small epsilon for numerical stability
            pos_doc_clusters,
            reduction='batchmean'
        )
        
        # 3. Sparsity Loss
        # Encourage sparse cluster activations
        sparsity_loss = (torch.mean(torch.sum(query_clusters, dim=1)) + 
                        torch.mean(torch.sum(pos_doc_clusters, dim=1)))
        
        # Total loss
        total_loss = ranking_loss + lambda3 * cluster_loss + lambda4 * sparsity_loss
        
        # Compute accuracy for monitoring
        with torch.no_grad():
            if negative_scores:
                # Check if positive score is higher than all negatives
                predictions = (positive_scores.unsqueeze(1) > negative_scores).all(dim=1)
                accuracy = predictions.float().mean().item()
            else:
                accuracy = 0.0
        
        return total_loss, {
            'ranking_loss': ranking_loss.item(),
            'cluster_loss': cluster_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'total_loss': total_loss.item(),
            'accuracy': accuracy
        }

class CLARETrainingManager:
    """Manager for CLARE training process - Paper Aligned"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize dataset configuration
        self.dataset_config = DatasetConfig(
            max_query_length=config['max_query_length'],
            max_doc_length=config['max_doc_length'],
            negative_sampling_ratio=config['negative_sampling_ratio'],
            min_query_length=config['min_query_length'],
            min_doc_length=config['min_doc_length'],
            vocab_size=config.get('vocab_size', 30000),
            n_clusters=config['n_clusters']
        )
        
        # Initialize components
        self.model = None
        self.trainer = None
        self.cluster_discovery = None
        self.tokenizer = None
        self.term_doc_matrix = None
        self.vocabulary = None
        self.word_embeddings = None
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_ndcg = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_ndcg': [],
            'val_ndcg': [],
            'learning_rate': []
        }
        
    def setup_wandb(self):
        """Initialize Weights & Biases logging"""
        if self.config.get('use_wandb', False):
            wandb.init(
                project="clare-paper-aligned",
                config=self.config,
                name=f"clare_{self.config['n_clusters']}clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def load_dataset(self) -> Dict:
        """Load dataset including term-document matrix"""
        logger.info("Loading CLARE dataset...")
        
        dataset_path = self.config.get('dataset_path', 'clare_dataset_complete.pkl')
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please run dataset integration first.")
        
        # Load main dataset
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # Load term-document matrix
        matrix_path = dataset_path.replace('.pkl', '_term_doc_matrix.npz')
        if os.path.exists(matrix_path):
            self.term_doc_matrix = scipy.sparse.load_npz(matrix_path)
            logger.info(f"Loaded term-document matrix: {self.term_doc_matrix.shape}")
        else:
            raise FileNotFoundError(f"Term-document matrix not found at {matrix_path}")
        
        # Load word embeddings
        embeddings_path = dataset_path.replace('.pkl', '_word_embeddings.npy')
        if os.path.exists(embeddings_path):
            self.word_embeddings = np.load(embeddings_path)
            logger.info(f"Loaded word embeddings: {self.word_embeddings.shape}")
        else:
            raise FileNotFoundError(f"Word embeddings not found at {embeddings_path}")
        
        self.vocabulary = dataset['vocabulary']
        logger.info(f"Vocabulary size: {len(self.vocabulary)}")
        
        return dataset
    
    def phase1_semantic_cluster_discovery(self):
        """
        Phase 1: Semantic Cluster Discovery
        Implements Algorithm 1 from the paper
        """
        logger.info("="*60)
        logger.info("PHASE 1: Semantic Cluster Discovery")
        logger.info("="*60)
        
        # Initialize Semantic Cluster Discovery
        self.cluster_discovery = SemanticClusterDiscovery(
            k=self.config['n_clusters'],
            lambda1=self.config.get('lambda1', 0.01),
            lambda2=self.config.get('lambda2', 0.1),
            max_iter=self.config.get('cluster_max_iter', 100),
            random_state=42
        )
        
        # Fit clusters on term-document matrix
        logger.info(f"Discovering {self.config['n_clusters']} semantic clusters...")
        
        # Convert sparse matrix to dense for processing (may need optimization for large matrices)
        if self.term_doc_matrix.shape[1] > 10000:
            logger.info("Sampling documents for cluster discovery...")
            # Sample columns (documents) for feasibility
            sample_indices = np.random.choice(self.term_doc_matrix.shape[1], 10000, replace=False)
            term_doc_sample = self.term_doc_matrix[:, sample_indices].toarray()
        else:
            term_doc_sample = self.term_doc_matrix.toarray()
        
        # Fit clusters
        self.cluster_discovery.fit(
            term_doc_sample,
            self.word_embeddings,
            self.vocabulary
        )
        
        # Get cluster information
        cluster_words = self.cluster_discovery.get_cluster_words(top_k=20)
        
        # Log cluster information
        logger.info("\nDiscovered Clusters:")
        for i in range(min(10, self.config['n_clusters'])):  # Show first 10 clusters
            logger.info(f"Cluster {i}: {', '.join(cluster_words[i][:10])}")
        
        # Save cluster discovery results
        cluster_path = os.path.join(
            self.config.get('checkpoint_dir', 'checkpoints'),
            'cluster_discovery.pkl'
        )
        os.makedirs(os.path.dirname(cluster_path), exist_ok=True)
        
        with open(cluster_path, 'wb') as f:
            pickle.dump({
                'W': self.cluster_discovery.W,
                'H': self.cluster_discovery.H,
                'vocabulary': self.vocabulary,
                'cluster_words': cluster_words
            }, f)
        
        logger.info(f"Cluster discovery results saved to {cluster_path}")
    
    def phase2_setup_model(self):
        """
        Phase 2: Initialize CLARE model with discovered clusters
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: Model Initialization")
        logger.info("="*60)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.get('model_name', 'bert-base-uncased')
        )
        
        # Initialize CLARE model with correct parameters from paper
        self.model = CLAREModel(
            model_name=self.config.get('model_name', 'bert-base-uncased'),
            n_clusters=self.config['n_clusters'],
            n_heads=self.config.get('n_heads', 8),
            cluster_layers=self.config.get('cluster_layers', [3, 6, 9])
        ).to(self.device)
        
        # Initialize cluster embeddings with discovered clusters
        if hasattr(self.cluster_discovery, 'W') and self.word_embeddings is not None:
            # Use cluster-term matrix to initialize cluster embeddings
            # Average word embeddings weighted by cluster membership
            cluster_embeddings = []
            
            # Get the vocabulary size from the cluster-term matrix
            vocab_size = self.cluster_discovery.W.shape[0]
            
            # Make sure we have matching dimensions
            if vocab_size != self.word_embeddings.shape[0]:
                logger.warning(f"Vocabulary size mismatch: W has {vocab_size} terms, "
                             f"embeddings have {self.word_embeddings.shape[0]} terms")
                # Use the minimum size to avoid index errors
                vocab_size = min(vocab_size, self.word_embeddings.shape[0])
            
            for i in range(self.config['n_clusters']):
                # Get cluster weights for the available vocabulary
                cluster_weights = self.cluster_discovery.W[:vocab_size, i]
                
                # Normalize weights
                weight_sum = np.sum(cluster_weights)
                if weight_sum > 0:
                    cluster_weights = cluster_weights / weight_sum
                else:
                    # If no weights, use uniform distribution
                    cluster_weights = np.ones(vocab_size) / vocab_size
                
                # Weighted average of word embeddings (only for available vocab)
                weighted_embedding = np.sum(
                    self.word_embeddings[:vocab_size] * cluster_weights[:, np.newaxis], 
                    axis=0
                )
                cluster_embeddings.append(weighted_embedding)
            
            cluster_embeddings = np.vstack(cluster_embeddings)
            
            # Initialize model's cluster embeddings
            with torch.no_grad():
                # Find cluster attention modules and initialize
                for module in self.model.cluster_attention_modules:
                    # Project to model dimension if needed
                    if cluster_embeddings.shape[1] != module.d_model:
                        # Use a linear projection
                        projection = nn.Linear(cluster_embeddings.shape[1], module.d_model)
                        projected_embeddings = projection(
                            torch.tensor(cluster_embeddings, dtype=torch.float32)
                        ).detach().numpy()
                    else:
                        projected_embeddings = cluster_embeddings
                    
                    module.cluster_embeddings.data = torch.tensor(
                        projected_embeddings, 
                        dtype=torch.float32
                    ).to(self.device)
            
            logger.info(f"Initialized cluster embeddings from discovered clusters")
        else:
            logger.warning("No cluster discovery results found, using random initialization")
        
        # Initialize trainer
        self.trainer = CLARETrainer(self.model, self.device)
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def create_data_loaders(self, dataset: Dict) -> Dict:
        """Create PyTorch DataLoaders for training"""
        logger.info("Creating data loaders...")
        
        data_loaders = {}
        
        for split in ['train', 'validation', 'test']:
            dataset_split = CLARETorchDataset(
                dataset[split],
                self.tokenizer,
                self.dataset_config
            )
            
            shuffle = (split == 'train')
            batch_size = self.config.get('batch_size', 16) if split == 'train' else 32
            
            data_loaders[split] = DataLoader(
                dataset_split,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.config.get('num_workers', 4),
                pin_memory=True,
                drop_last=(split == 'train')  # Drop last batch for training stability
            )
            
            logger.info(f"{split.capitalize()} dataloader: {len(dataset_split)} queries, "
                       f"{len(data_loaders[split])} batches")
        
        return data_loaders
    
    def setup_optimizer_scheduler(self, train_loader):
        """Setup optimizer and learning rate scheduler"""
        logger.info("Setting up optimizer and scheduler...")
        
        # Separate parameters for different learning rates
        transformer_params = []
        cluster_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'transformer' in name:
                transformer_params.append(param)
            elif 'cluster' in name:
                cluster_params.append(param)
            else:
                other_params.append(param)
        
        # Create optimizer with different learning rates
        optimizer = optim.AdamW([
            {'params': transformer_params, 'lr': self.config.get('learning_rate', 2e-5)},
            {'params': cluster_params, 'lr': self.config.get('cluster_lr', 1e-4)},
            {'params': other_params, 'lr': self.config.get('learning_rate', 2e-5)}
        ], weight_decay=self.config.get('weight_decay', 0.01))
        
        # Calculate total training steps
        total_steps = len(train_loader) * self.config.get('num_epochs', 3)
        warmup_steps = int(total_steps * self.config.get('warmup_ratio', 0.1))
        
        # Create scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Training steps: {total_steps}, Warmup steps: {warmup_steps}")
        
        return optimizer, scheduler
    
    def train_epoch(self, train_loader, optimizer, scheduler, epoch):
        """Train for one epoch"""
        self.model.train()
        
        epoch_metrics = defaultdict(float)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Skip if no negatives
            if batch['neg_doc_input_ids'].numel() == 0:
                continue
            
            optimizer.zero_grad()
            
            # Forward pass for query
            query_outputs = self.model(
                input_ids=batch['query_input_ids'],
                attention_mask=batch['query_attention_mask']
            )
            
            # Forward pass for positive document
            pos_doc_outputs = self.model(
                input_ids=batch['pos_doc_input_ids'],
                attention_mask=batch['pos_doc_attention_mask']
            )
            
            # Forward pass for negative documents
            neg_doc_outputs_list = []
            if batch['neg_doc_input_ids'].dim() == 3:  # [batch_size, num_negatives, seq_len]
                batch_size, num_negatives, seq_len = batch['neg_doc_input_ids'].shape
                
                for i in range(num_negatives):
                    neg_outputs = self.model(
                        input_ids=batch['neg_doc_input_ids'][:, i, :],
                        attention_mask=batch['neg_doc_attention_mask'][:, i, :]
                    )
                    neg_doc_outputs_list.append(neg_outputs)
            
            # Compute loss
            loss, loss_components = self.trainer.compute_loss(
                query_outputs, 
                pos_doc_outputs, 
                neg_doc_outputs_list,
                lambda3=self.config.get('lambda3', 0.1),
                lambda4=self.config.get('lambda4', 0.01)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.get('max_grad_norm', 1.0)
            )
            
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            for key, value in loss_components.items():
                epoch_metrics[key] += value
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'Loss': f"{loss_components['total_loss']:.4f}",
                'Acc': f"{loss_components['accuracy']:.3f}",
                'LR': f"{current_lr:.2e}"
            })
            
            # Log to wandb
            if self.config.get('use_wandb', False) and batch_idx % 100 == 0:
                wandb.log({
                    'train_loss_step': loss_components['total_loss'],
                    'train_ranking_loss': loss_components['ranking_loss'],
                    'train_cluster_loss': loss_components['cluster_loss'],
                    'train_sparsity_loss': loss_components['sparsity_loss'],
                    'train_accuracy_step': loss_components['accuracy'],
                    'learning_rate': current_lr,
                    'epoch': epoch,
                    'step': epoch * len(train_loader) + batch_idx
                })
        
        # Average metrics
        num_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def validate_epoch(self, val_loader, epoch):
        """Validate for one epoch with nDCG computation"""
        self.model.eval()
        
        epoch_metrics = defaultdict(float)
        all_scores = []
        all_labels = []
        dataset_scores = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Skip if no negatives
                if batch['neg_doc_input_ids'].numel() == 0:
                    continue
                
                # Forward pass
                query_outputs = self.model(
                    input_ids=batch['query_input_ids'],
                    attention_mask=batch['query_attention_mask']
                )
                
                pos_doc_outputs = self.model(
                    input_ids=batch['pos_doc_input_ids'],
                    attention_mask=batch['pos_doc_attention_mask']
                )
                
                neg_doc_outputs_list = []
                if batch['neg_doc_input_ids'].dim() == 3:
                    batch_size, num_negatives, seq_len = batch['neg_doc_input_ids'].shape
                    
                    for i in range(num_negatives):
                        neg_outputs = self.model(
                            input_ids=batch['neg_doc_input_ids'][:, i, :],
                            attention_mask=batch['neg_doc_attention_mask'][:, i, :]
                        )
                        neg_doc_outputs_list.append(neg_outputs)
                
                # Compute loss
                loss, loss_components = self.trainer.compute_loss(
                    query_outputs, 
                    pos_doc_outputs, 
                    neg_doc_outputs_list,
                    lambda3=self.config.get('lambda3', 0.1),
                    lambda4=self.config.get('lambda4', 0.01)
                )
                
                # Update metrics
                for key, value in loss_components.items():
                    epoch_metrics[key] += value
                
                # Collect scores for nDCG calculation
                query_clusters = query_outputs['cluster_activations']
                pos_clusters = pos_doc_outputs['cluster_activations']
                
                # Positive scores
                pos_scores = torch.sum(query_clusters * pos_clusters, dim=1)
                
                # Negative scores
                neg_scores = []
                for neg_outputs in neg_doc_outputs_list:
                    neg_clusters = neg_outputs['cluster_activations']
                    neg_score = torch.sum(query_clusters * neg_clusters, dim=1)
                    neg_scores.append(neg_score)
                
                # Store for nDCG calculation
                batch_size = pos_scores.size(0)
                for i in range(batch_size):
                    # Add positive score with label 1
                    all_scores.append(pos_scores[i].item())
                    all_labels.append(1)
                    
                    # Add negative scores with label 0
                    for neg_score in neg_scores:
                        all_scores.append(neg_score[i].item())
                        all_labels.append(0)
                    
                    # Track by dataset
                    dataset_name = batch['dataset'][i]
                    dataset_scores[dataset_name].append({
                        'pos_score': pos_scores[i].item(),
                        'neg_scores': [ns[i].item() for ns in neg_scores]
                    })
        
        # Average metrics
        num_batches = len(val_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # Compute nDCG@10
        ndcg_10 = self.compute_ndcg(all_scores, all_labels, k=10)
        epoch_metrics['ndcg@10'] = ndcg_10
        
        # Compute per-dataset metrics
        dataset_metrics = {}
        for dataset_name, scores in dataset_scores.items():
            dataset_accuracy = np.mean([
                s['pos_score'] > max(s['neg_scores']) if s['neg_scores'] else True
                for s in scores
            ])
            dataset_metrics[dataset_name] = dataset_accuracy
        
        logger.info(f"Validation - Loss: {epoch_metrics['total_loss']:.4f}, "
                   f"Accuracy: {epoch_metrics['accuracy']:.4f}, "
                   f"nDCG@10: {epoch_metrics['ndcg@10']:.4f}")
        logger.info(f"Dataset accuracies: {dataset_metrics}")
        
        return epoch_metrics, dataset_metrics
    
    def compute_ndcg(self, scores: List[float], labels: List[int], k: int = 10) -> float:
        """Compute nDCG@k metric"""
        if not scores:
            return 0.0
        
        # Group scores by query (assuming sequential order)
        query_groups = []
        current_group = {'scores': [], 'labels': []}
        
        for score, label in zip(scores, labels):
            current_group['scores'].append(score)
            current_group['labels'].append(label)
            
            # Simple grouping - assumes each query has same number of docs
            if len(current_group['scores']) >= 5:  # 1 positive + 4 negatives
                query_groups.append(current_group)
                current_group = {'scores': [], 'labels': []}
        
        # Compute nDCG for each query
        ndcg_scores = []
        
        for group in query_groups:
            # Sort by scores
            sorted_indices = np.argsort(group['scores'])[::-1]
            sorted_labels = [group['labels'][i] for i in sorted_indices]
            
            # Compute DCG@k
            dcg = 0.0
            for i in range(min(k, len(sorted_labels))):
                rel = sorted_labels[i]
                dcg += (2**rel - 1) / np.log2(i + 2)
            
            # Compute IDCG@k
            ideal_labels = sorted(group['labels'], reverse=True)
            idcg = 0.0
            for i in range(min(k, len(ideal_labels))):
                rel = ideal_labels[i]
                idcg += (2**rel - 1) / np.log2(i + 2)
            
            # Compute nDCG
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def save_checkpoint(self, epoch, optimizer, scheduler, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'training_history': self.training_history,
            'vocabulary': self.vocabulary,
            'cluster_discovery': {
                'W': self.cluster_discovery.W,
                'H': self.cluster_discovery.H
            }
        }
        
        # Save current checkpoint
        checkpoint_path = os.path.join(
            self.config.get('checkpoint_dir', 'checkpoints'),
            f'clare_epoch_{epoch}.pt'
        )
        
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model based on nDCG
        if metrics.get('ndcg@10', 0) > self.best_val_ndcg:
            self.best_val_ndcg = metrics.get('ndcg@10', 0)
            best_path = os.path.join(
                self.config.get('checkpoint_dir', 'checkpoints'),
                'clare_best_model.pt'
            )
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with nDCG@10: {self.best_val_ndcg:.4f}")
    
    def evaluate_interpretability(self, test_loader, num_examples: int = 50):
        """Evaluate interpretability with cluster analysis"""
        logger.info("\nEvaluating interpretability...")
        
        self.model.eval()
        interpretability_examples = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if len(interpretability_examples) >= num_examples:
                    break
                
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get query outputs
                query_outputs = self.model(
                    input_ids=batch['query_input_ids'],
                    attention_mask=batch['query_attention_mask']
                )
                
                # Get cluster activations
                cluster_activations = query_outputs['cluster_activations'][0].cpu().numpy()
                
                # Get top clusters
                top_k = 5
                top_clusters = np.argsort(cluster_activations)[-top_k:][::-1]
                
                # Get cluster words
                cluster_words = self.cluster_discovery.get_cluster_words(top_k=10)
                
                # Decode query
                query_text = self.tokenizer.decode(
                    batch['query_input_ids'][0], 
                    skip_special_tokens=True
                )
                
                example = {
                    'query': query_text,
                    'top_clusters': top_clusters.tolist(),
                    'cluster_activations': cluster_activations[top_clusters].tolist(),
                    'cluster_descriptions': [
                        cluster_words.get(int(c), [])[:5] for c in top_clusters
                    ]
                }
                
                interpretability_examples.append(example)
        
        # Log examples
        logger.info("\nInterpretability Examples:")
        for i, example in enumerate(interpretability_examples[:5]):
            logger.info(f"\nExample {i+1}:")
            logger.info(f"Query: {example['query']}")
            logger.info("Top Clusters:")
            for j, (cluster_id, activation, words) in enumerate(zip(
                example['top_clusters'], 
                example['cluster_activations'],
                example['cluster_descriptions']
            )):
                logger.info(f"  Cluster {cluster_id} (activation={activation:.3f}): {', '.join(words)}")
        
        # Save interpretability results
        interp_path = os.path.join(
            self.config.get('checkpoint_dir', 'checkpoints'),
            'interpretability_examples.json'
        )
        with open(interp_path, 'w') as f:
            json.dump(interpretability_examples, f, indent=2)
        
        return interpretability_examples
    
    def plot_training_history(self):
        """Plot training history with paper-style formatting"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss plots
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plots
        axes[0, 1].plot(self.training_history['train_accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(self.training_history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # nDCG plots
        axes[0, 2].plot(self.training_history['val_ndcg'], label='Validation nDCG@10', 
                       linewidth=2, color='green')
        axes[0, 2].set_title('Validation nDCG@10', fontsize=14)
        axes[0, 2].set_xlabel('Epoch', fontsize=12)
        axes[0, 2].set_ylabel('nDCG@10', fontsize=12)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 0].plot(self.training_history['learning_rate'], linewidth=2, color='orange')
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss components plot
        if 'ranking_loss' in self.training_history:
            axes[1, 1].plot(self.training_history['ranking_loss'], label='Ranking Loss', linewidth=2)
            axes[1, 1].plot(self.training_history['cluster_loss'], label='Cluster Loss', linewidth=2)
            axes[1, 1].plot(self.training_history['sparsity_loss'], label='Sparsity Loss', linewidth=2)
            axes[1, 1].set_title('Loss Components', fontsize=14)
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Loss', fontsize=12)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Remove empty subplot
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        
        # Save plot
        plots_dir = self.config.get('plots_dir', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        
        if self.config.get('use_wandb', False):
            wandb.log({"training_history": wandb.Image(plt)})
        
        plt.close()
    
    def evaluate_model(self, test_loader):
        """Comprehensive evaluation on test set matching paper metrics"""
        logger.info("\nEvaluating model on test set...")
        
        self.model.eval()
        
        # Metrics storage
        all_scores = []
        all_labels = []
        dataset_metrics = defaultdict(lambda: {'scores': [], 'labels': []})
        
        # Cluster activation statistics
        cluster_activation_stats = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                if batch['neg_doc_input_ids'].numel() == 0:
                    continue
                
                # Forward pass
                query_outputs = self.model(
                    input_ids=batch['query_input_ids'],
                    attention_mask=batch['query_attention_mask']
                )
                
                pos_doc_outputs = self.model(
                    input_ids=batch['pos_doc_input_ids'],
                    attention_mask=batch['pos_doc_attention_mask']
                )
                
                # Get cluster activations for analysis
                query_clusters = query_outputs['cluster_activations']
                pos_clusters = pos_doc_outputs['cluster_activations']
                
                # Compute cluster statistics
                cluster_sparsity = (query_clusters > 0.01).float().mean(dim=1)
                cluster_activation_stats['sparsity'].extend(cluster_sparsity.cpu().numpy())
                
                # Positive scores
                pos_scores = torch.sum(query_clusters * pos_clusters, dim=1)
                
                # Process negatives
                neg_scores_list = []
                if batch['neg_doc_input_ids'].dim() == 3:
                    batch_size, num_negatives, seq_len = batch['neg_doc_input_ids'].shape
                    
                    for i in range(num_negatives):
                        neg_outputs = self.model(
                            input_ids=batch['neg_doc_input_ids'][:, i, :],
                            attention_mask=batch['neg_doc_attention_mask'][:, i, :]
                        )
                        neg_clusters = neg_outputs['cluster_activations']
                        neg_score = torch.sum(query_clusters * neg_clusters, dim=1)
                        neg_scores_list.append(neg_score)
                
                # Collect scores
                batch_size = pos_scores.size(0)
                for i in range(batch_size):
                    # Overall metrics
                    all_scores.append(pos_scores[i].item())
                    all_labels.append(1)
                    
                    for neg_score in neg_scores_list:
                        all_scores.append(neg_score[i].item())
                        all_labels.append(0)
                    
                    # Dataset-specific metrics
                    dataset_name = batch['dataset'][i]
                    dataset_metrics[dataset_name]['scores'].append(pos_scores[i].item())
                    dataset_metrics[dataset_name]['labels'].append(1)
                    
                    for neg_score in neg_scores_list:
                        dataset_metrics[dataset_name]['scores'].append(neg_score[i].item())
                        dataset_metrics[dataset_name]['labels'].append(0)
        
        # Compute metrics matching paper
        results = {}
        
        # Overall metrics
        results['ndcg@10'] = self.compute_ndcg(all_scores, all_labels, k=10)
        results['ndcg@20'] = self.compute_ndcg(all_scores, all_labels, k=20)
        results['map'] = self.compute_map(all_scores, all_labels)
        results['mrr'] = self.compute_mrr(all_scores, all_labels)
        
        # Accuracy (positive score > all negative scores)
        accuracy = self.compute_accuracy(all_scores, all_labels)
        results['accuracy'] = accuracy
        
        # Dataset-specific metrics
        results['dataset_metrics'] = {}
        for dataset_name, data in dataset_metrics.items():
            results['dataset_metrics'][dataset_name] = {
                'ndcg@10': self.compute_ndcg(data['scores'], data['labels'], k=10),
                'accuracy': self.compute_accuracy(data['scores'], data['labels']),
                'num_queries': sum(1 for l in data['labels'] if l == 1)
            }
        
        # Cluster statistics
        results['cluster_stats'] = {
            'avg_sparsity': np.mean(cluster_activation_stats['sparsity']),
            'std_sparsity': np.std(cluster_activation_stats['sparsity'])
        }
        
        # Log results
        logger.info("\n" + "="*60)
        logger.info("TEST SET RESULTS (Paper Metrics)")
        logger.info("="*60)
        logger.info(f"nDCG@10: {results['ndcg@10']:.4f}")
        logger.info(f"nDCG@20: {results['ndcg@20']:.4f}")
        logger.info(f"MAP: {results['map']:.4f}")
        logger.info(f"MRR: {results['mrr']:.4f}")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        
        logger.info("\nDataset-Specific Results:")
        for dataset, metrics in results['dataset_metrics'].items():
            logger.info(f"{dataset}: nDCG@10={metrics['ndcg@10']:.4f}, "
                       f"Acc={metrics['accuracy']:.4f}, n={metrics['num_queries']}")
        
        logger.info(f"\nCluster Sparsity: {results['cluster_stats']['avg_sparsity']:.3f} "
                   f"(±{results['cluster_stats']['std_sparsity']:.3f})")
        
        # Log to wandb
        if self.config.get('use_wandb', False):
            wandb.log({
                'test_ndcg@10': results['ndcg@10'],
                'test_ndcg@20': results['ndcg@20'],
                'test_map': results['map'],
                'test_mrr': results['mrr'],
                'test_accuracy': results['accuracy'],
                **{f'test_{ds}_ndcg@10': m['ndcg@10'] 
                   for ds, m in results['dataset_metrics'].items()}
            })
        
        return results
    
    def compute_map(self, scores: List[float], labels: List[int]) -> float:
        """Compute Mean Average Precision"""
        # Group into queries
        query_groups = []
        current_group = {'scores': [], 'labels': []}
        
        for score, label in zip(scores, labels):
            current_group['scores'].append(score)
            current_group['labels'].append(label)
            
            if len(current_group['scores']) >= 5:  # 1 positive + 4 negatives
                query_groups.append(current_group)
                current_group = {'scores': [], 'labels': []}
        
        ap_scores = []
        for group in query_groups:
            # Sort by scores
            sorted_indices = np.argsort(group['scores'])[::-1]
            sorted_labels = [group['labels'][i] for i in sorted_indices]
            
            # Compute AP
            ap = 0.0
            relevant_docs = 0
            
            for i, label in enumerate(sorted_labels):
                if label == 1:
                    relevant_docs += 1
                    ap += relevant_docs / (i + 1)
            
            if relevant_docs > 0:
                ap_scores.append(ap / relevant_docs)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def compute_mrr(self, scores: List[float], labels: List[int]) -> float:
        """Compute Mean Reciprocal Rank"""
        # Group into queries
        query_groups = []
        current_group = {'scores': [], 'labels': []}
        
        for score, label in zip(scores, labels):
            current_group['scores'].append(score)
            current_group['labels'].append(label)
            
            if len(current_group['scores']) >= 5:
                query_groups.append(current_group)
                current_group = {'scores': [], 'labels': []}
        
        rr_scores = []
        for group in query_groups:
            # Sort by scores
            sorted_indices = np.argsort(group['scores'])[::-1]
            sorted_labels = [group['labels'][i] for i in sorted_indices]
            
            # Find first relevant
            for i, label in enumerate(sorted_labels):
                if label == 1:
                    rr_scores.append(1.0 / (i + 1))
                    break
        
        return np.mean(rr_scores) if rr_scores else 0.0
    
    def compute_accuracy(self, scores: List[float], labels: List[int]) -> float:
        """Compute accuracy (positive score > all negatives)"""
        # Group into queries
        query_groups = []
        current_group = {'scores': [], 'labels': []}
        
        for score, label in zip(scores, labels):
            current_group['scores'].append(score)
            current_group['labels'].append(label)
            
            if len(current_group['scores']) >= 5:
                query_groups.append(current_group)
                current_group = {'scores': [], 'labels': []}
        
        correct = 0
        total = 0
        
        for group in query_groups:
            pos_scores = [s for s, l in zip(group['scores'], group['labels']) if l == 1]
            neg_scores = [s for s, l in zip(group['scores'], group['labels']) if l == 0]
            
            if pos_scores and neg_scores:
                if max(pos_scores) > max(neg_scores):
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def train(self):
        """Main training loop implementing paper's two-phase procedure"""
        logger.info("Starting CLARE training (Paper-Aligned Implementation)...")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        # Setup
        self.setup_wandb()
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Analyze dataset
        analyzer = DatasetAnalyzer(dataset)
        analysis = analyzer.analyze_dataset()
        analyzer.print_analysis(analysis)
        
        # Phase 1: Semantic Cluster Discovery
        self.phase1_semantic_cluster_discovery()
        
        # Phase 2: Model Training
        self.phase2_setup_model()
        
        # Create data loaders
        data_loaders = self.create_data_loaders(dataset)
        
        # Setup optimizer and scheduler
        optimizer, scheduler = self.setup_optimizer_scheduler(data_loaders['train'])
        
        # Training loop
        num_epochs = self.config.get('num_epochs', 3)
        
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            # Training phase
            train_metrics = self.train_epoch(
                data_loaders['train'], optimizer, scheduler, epoch
            )
            
            # Validation phase
            val_metrics, dataset_metrics = self.validate_epoch(
                data_loaders['validation'], epoch
            )
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['total_loss'])
            self.training_history['val_loss'].append(val_metrics['total_loss'])
            self.training_history['train_accuracy'].append(train_metrics['accuracy'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['val_ndcg'].append(val_metrics.get('ndcg@10', 0))
            self.training_history['learning_rate'].append(scheduler.get_last_lr()[0])
            
            # Store loss components
            for key in ['ranking_loss', 'cluster_loss', 'sparsity_loss']:
                if key not in self.training_history:
                    self.training_history[key] = []
                self.training_history[key].append(train_metrics.get(key, 0))
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['total_loss'],
                    'val_loss': val_metrics['total_loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_ndcg@10': val_metrics.get('ndcg@10', 0),
                    **{f'val_{ds}_accuracy': acc for ds, acc in dataset_metrics.items()}
                })
            
            # Save checkpoint
            self.save_checkpoint(epoch, optimizer, scheduler, val_metrics)
            
            # Early stopping based on nDCG
            if self.config.get('early_stopping', False):
                if val_metrics.get('ndcg@10', 0) > self.best_val_ndcg:
                    self.best_val_ndcg = val_metrics.get('ndcg@10', 0)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.get('patience', 3):
                        logger.info("Early stopping triggered")
                        break
        
        # Final evaluation
        logger.info("\n" + "="*60)
        logger.info("FINAL EVALUATION")
        logger.info("="*60)
        
        # Test set evaluation
        test_results = self.evaluate_model(data_loaders['test'])
        
        # Interpretability evaluation
        interpretability_examples = self.evaluate_interpretability(data_loaders['test'])
        
        # Plot training history
        self.plot_training_history()
        
        # Save final results
        results_path = os.path.join(
            self.config.get('checkpoint_dir', 'checkpoints'),
            'final_results.json'
        )
        
        final_results = {
            'test_results': test_results,
            'best_val_ndcg': self.best_val_ndcg,
            'training_history': self.training_history,
            'config': self.config,
            'num_clusters': self.config['n_clusters'],
            'interpretability_examples': interpretability_examples[:10]
        }
        
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Final logging
        logger.info("\n" + "="*60)
        logger.info("Training completed!")
        logger.info(f"Best validation nDCG@10: {self.best_val_ndcg:.4f}")
        logger.info(f"Test nDCG@10: {test_results['ndcg@10']:.4f}")
        logger.info(f"Test MAP: {test_results['map']:.4f}")
        logger.info(f"Test MRR: {test_results['mrr']:.4f}")
        logger.info(f"Results saved to {results_path}")
        logger.info("="*60)
        
        return final_results


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CLARE Training Script - Paper Aligned')
    
    # Model parameters (from paper)
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='Pre-trained transformer model')
    parser.add_argument('--n_clusters', type=int, default=100,
                       help='Number of semantic clusters (k in paper)')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads for cluster attention')
    parser.add_argument('--cluster_layers', type=int, nargs='+', default=[3, 6, 9],
                       help='Transformer layers to apply cluster attention')
    
    # Cluster discovery parameters
    parser.add_argument('--lambda1', type=float, default=0.01,
                       help='Sparsity regularization for cluster discovery')
    parser.add_argument('--lambda2', type=float, default=0.1,
                       help='Semantic coherence regularization')
    parser.add_argument('--cluster_max_iter', type=int, default=100,
                       help='Max iterations for cluster discovery')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate for transformer')
    parser.add_argument('--cluster_lr', type=float, default=1e-4,
                       help='Learning rate for cluster parameters')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio for learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Max gradient norm for clipping')
    
    # Loss weights (from paper)
    parser.add_argument('--lambda3', type=float, default=0.1,
                       help='Weight for cluster consistency loss')
    parser.add_argument('--lambda4', type=float, default=0.01,
                       help='Weight for sparsity loss')
    
    # Dataset parameters
    parser.add_argument('--max_query_length', type=int, default=128,
                       help='Maximum query length')
    parser.add_argument('--max_doc_length', type=int, default=512,
                       help='Maximum document length')
    parser.add_argument('--negative_sampling_ratio', type=int, default=4,
                       help='Number of negatives per positive')
    parser.add_argument('--min_query_length', type=int, default=3,
                       help='Minimum query length')
    parser.add_argument('--min_doc_length', type=int, default=10,
                       help='Minimum document length')
    parser.add_argument('--vocab_size', type=int, default=30000,
                       help='Vocabulary size for term-document matrix')
    
    # Training control
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping based on nDCG')
    parser.add_argument('--patience', type=int, default=3,
                       help='Patience for early stopping')
    
    # Paths
    parser.add_argument('--dataset_path', type=str, default='clare_dataset_complete.pkl',
                       help='Path to preprocessed dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory for saving checkpoints')
    parser.add_argument('--plots_dir', type=str, default='plots',
                       help='Directory for saving plots')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Resume training
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume training from checkpoint')
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    config = vars(args)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create training manager
    trainer = CLARETrainingManager(config)
    
    # Resume from checkpoint if specified
    if config.get('resume_from'):
        logger.info(f"Resuming from checkpoint: {config['resume_from']}")
        checkpoint = torch.load(config['resume_from'])
        # Load model state, training history, etc.
        trainer.training_history = checkpoint.get('training_history', trainer.training_history)
        trainer.best_val_ndcg = checkpoint.get('metrics', {}).get('ndcg@10', 0)
    
    # Start training
    try:
        results = trainer.train()
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        # Clean up wandb
        if config.get('use_wandb', False):
            wandb.finish()


if __name__ == "__main__":
    main()
