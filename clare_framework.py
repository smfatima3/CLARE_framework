import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Dict, List, Tuple, Optional
import json
import pickle
from tqdm import tqdm
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticClusterDiscovery:
    """
    Semantic Cluster Discovery using regularized NMF with semantic constraints
    Implements Algorithm 1 from the paper
    """
    def __init__(self, k: int = 100, lambda1: float = 0.01, lambda2: float = 0.1, 
                 max_iter: int = 100, random_state: int = 42):
        self.k = k  # Number of clusters
        self.lambda1 = lambda1  # Sparsity regularization
        self.lambda2 = lambda2  # Semantic coherence regularization
        self.max_iter = max_iter
        self.random_state = random_state
        self.W = None  # Cluster-term matrix [|V| × k]
        self.H = None  # Document-cluster matrix [k × n]
        self.vocab = None
        
    def _semantic_coherence_regularizer(self, H: np.ndarray, word_embeddings: np.ndarray) -> float:
        """
        Compute semantic coherence regularizer from paper
        Ω_semantic(H) = Σᵢ Σⱼ sim(wᵢ, wⱼ) × ||hᵢ - hⱼ||²₂
        """
        coherence_loss = 0.0
        
        # Compute cluster centroids in embedding space
        cluster_centroids = []
        for i in range(self.k):
            # Get top terms for cluster i
            top_term_indices = np.argsort(self.W[:, i])[-20:]  # Top 20 terms
            cluster_weights = self.W[top_term_indices, i]
            cluster_weights = cluster_weights / (np.sum(cluster_weights) + 1e-8)
            
            # Weighted average of embeddings
            cluster_embedding = np.sum(
                word_embeddings[top_term_indices] * cluster_weights[:, np.newaxis], 
                axis=0
            )
            cluster_centroids.append(cluster_embedding)
        
        cluster_centroids = np.vstack(cluster_centroids)
        
        # Compute pairwise similarities
        similarities = cosine_similarity(cluster_centroids)
        
        # Compute regularizer
        for i in range(self.k):
            for j in range(i+1, self.k):
                sim_ij = similarities[i, j]
                h_diff = np.linalg.norm(H[i] - H[j], ord=2)**2
                coherence_loss += sim_ij * h_diff
        
        return coherence_loss
    
    def _compute_semantic_gradient(self, H: np.ndarray, word_embeddings: np.ndarray) -> np.ndarray:
        """Compute gradient of semantic coherence regularizer"""
        grad = np.zeros_like(H)
        
        # Compute cluster centroids
        cluster_centroids = []
        for i in range(self.k):
            top_term_indices = np.argsort(self.W[:, i])[-20:]
            cluster_weights = self.W[top_term_indices, i]
            cluster_weights = cluster_weights / (np.sum(cluster_weights) + 1e-8)
            cluster_embedding = np.sum(
                word_embeddings[top_term_indices] * cluster_weights[:, np.newaxis], 
                axis=0
            )
            cluster_centroids.append(cluster_embedding)
        
        cluster_centroids = np.vstack(cluster_centroids)
        similarities = cosine_similarity(cluster_centroids)
        
        # Compute gradient
        for i in range(self.k):
            for j in range(i+1, self.k):
                sim_ij = similarities[i, j]
                grad[i] += 2 * sim_ij * (H[i] - H[j])
                grad[j] += 2 * sim_ij * (H[j] - H[i])
        
        return grad
    
    def fit(self, X: np.ndarray, word_embeddings: np.ndarray, vocab: List[str]):
        """
        Fit the Semantic Cluster Discovery model
        
        Args:
            X: Term-document matrix [|V| × n]
            word_embeddings: Pre-trained word embeddings [|V| × d]
            vocab: Vocabulary list
        """
        self.vocab = vocab
        n_features, n_samples = X.shape
        
        logger.info(f"Fitting Semantic Cluster Discovery with k={self.k} clusters")
        logger.info(f"Term-document matrix shape: {X.shape}")
        
        # Initialize W and H using NMF initialization
        np.random.seed(self.random_state)
        
        # Better initialization using SVD
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.W = np.abs(U[:, :self.k])
        self.H = np.abs(S[:self.k, np.newaxis] * Vt[:self.k, :])
        
        # Normalize
        self.W = self.W / (np.linalg.norm(self.W, axis=0, keepdims=True) + 1e-8)
        self.H = self.H / (np.linalg.norm(self.H, axis=1, keepdims=True) + 1e-8)
        
        prev_loss = float('inf')
        alpha = 0.01  # Learning rate
        
        for iteration in range(self.max_iter):
            # Update W (cluster-term matrix)
            # Gradient: ∇W = -2(X - WH)H^T + λ₁
            reconstruction = self.W @ self.H
            grad_W = -2 * (X - reconstruction) @ self.H.T + self.lambda1
            
            # Projected gradient update
            self.W = self.W - alpha * grad_W
            self.W = np.maximum(self.W, 0)  # Non-negative constraint
            
            # Normalize columns
            col_norms = np.linalg.norm(self.W, axis=0, keepdims=True)
            self.W = self.W / (col_norms + 1e-8)
            
            # Update H (document-cluster matrix)
            # Gradient includes semantic regularizer
            reconstruction = self.W @ self.H
            grad_H = -2 * self.W.T @ (X - reconstruction)
            
            # Add semantic coherence gradient
            if word_embeddings is not None:
                semantic_grad = self._compute_semantic_gradient(self.H, word_embeddings)
                grad_H += self.lambda2 * semantic_grad
            
            # Projected gradient update
            self.H = self.H - alpha * grad_H
            self.H = np.maximum(self.H, 0)  # Non-negative constraint
            
            # Normalize rows
            row_norms = np.linalg.norm(self.H, axis=1, keepdims=True)
            self.H = self.H / (row_norms + 1e-8)
            
            # Compute loss
            reconstruction_loss = np.linalg.norm(X - self.W @ self.H, 'fro')**2
            sparsity_loss = self.lambda1 * np.sum(self.W)
            
            if word_embeddings is not None:
                semantic_loss = self.lambda2 * self._semantic_coherence_regularizer(self.H, word_embeddings)
            else:
                semantic_loss = 0
            
            total_loss = reconstruction_loss + sparsity_loss + semantic_loss
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Loss = {total_loss:.4f} "
                           f"(Recon: {reconstruction_loss:.4f}, "
                           f"Sparse: {sparsity_loss:.4f}, "
                           f"Semantic: {semantic_loss:.4f})")
            
            # Check convergence
            if abs(prev_loss - total_loss) < 1e-6:
                logger.info(f"Converged at iteration {iteration}")
                break
            
            prev_loss = total_loss
            
            # Decay learning rate
            if iteration % 20 == 0:
                alpha *= 0.9
        
        logger.info("Semantic Cluster Discovery completed")
        return self
    
    def get_cluster_words(self, top_k: int = 10) -> Dict[int, List[str]]:
        """Get top words for each cluster"""
        cluster_words = {}
        
        for i in range(self.k):
            # Get top term indices for cluster i
            top_indices = np.argsort(self.W[:, i])[-top_k:][::-1]
            cluster_words[i] = [self.vocab[idx] for idx in top_indices]
        
        return cluster_words
    
    def get_cluster_coherence(self) -> Dict[int, float]:
        """Compute coherence score for each cluster"""
        cluster_coherence = {}
        
        for i in range(self.k):
            # Get cluster term weights
            cluster_weights = self.W[:, i]
            
            # Compute entropy as inverse coherence measure
            # Lower entropy = higher coherence
            normalized_weights = cluster_weights / (np.sum(cluster_weights) + 1e-8)
            entropy = -np.sum(normalized_weights * np.log(normalized_weights + 1e-8))
            
            # Convert to coherence score (higher is better)
            cluster_coherence[i] = 1.0 / (1.0 + entropy)
        
        return cluster_coherence

class ClusterAttention(nn.Module):
    """
    Cluster Attention Module for computing attention over semantic clusters
    Implements the cluster attention mechanism from the paper
    """
    def __init__(self, d_model: int, n_clusters: int, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_clusters = n_clusters
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Cluster embeddings initialized later with discovered clusters
        self.cluster_embeddings = nn.Parameter(torch.randn(n_clusters, d_model))
        
        # Multi-head projections
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for module in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of cluster attention
        
        Args:
            hidden_states: [batch_size, seq_len, d_model]
        
        Returns:
            cluster_enhanced_states: [batch_size, seq_len, d_model]
            cluster_attention_weights: [batch_size, n_heads, seq_len, n_clusters]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project inputs to Q, K, V
        Q = self.W_Q(hidden_states)  # [batch_size, seq_len, d_model]
        K = self.W_K(self.cluster_embeddings).unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, n_clusters, d_model]
        V = self.W_V(self.cluster_embeddings).unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, n_clusters, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        K = K.view(batch_size, self.n_clusters, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, n_clusters, d_k]
        V = V.view(batch_size, self.n_clusters, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, n_clusters, d_k]
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [batch_size, n_heads, seq_len, n_clusters]
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # [batch_size, n_heads, seq_len, d_k]
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        cluster_enhanced = self.W_O(context)
        cluster_enhanced = self.dropout(cluster_enhanced)
        
        # Residual connection and layer norm
        output = self.layer_norm(hidden_states + cluster_enhanced)
        
        return output, attention_weights

class CLAREModel(nn.Module):
    """
    CLARE: CLuster-Aligned Retrieval Embeddings
    Paper-aligned implementation with cluster attention at specific layers
    """
    def __init__(self, model_name: str = "bert-base-uncased", n_clusters: int = 100, 
                 n_heads: int = 8, cluster_layers: List[int] = [3, 6, 9]):
        super().__init__()
        
        self.model_name = model_name
        self.n_clusters = n_clusters
        self.n_heads = n_heads
        self.cluster_layers = cluster_layers
        
        # Load pre-trained transformer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.d_model = self.transformer.config.hidden_size
        
        # Add cluster attention modules for specified layers
        self.cluster_attention_modules = nn.ModuleList([
            ClusterAttention(self.d_model, n_clusters, n_heads)
            for _ in cluster_layers
        ])
        
        # Cluster projection head for final cluster activations
        self.cluster_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, n_clusters)
        )
        
        # Initialize projection head
        for layer in self.cluster_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CLARE model
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            Dictionary containing:
                - pooled_hidden: Final pooled representation [batch_size, d_model]
                - cluster_activations: Cluster activation scores [batch_size, n_clusters]
                - cluster_attention_weights: Attention weights from each cluster layer
        """
        # Get transformer outputs with all hidden states
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        all_hidden_states = outputs.hidden_states  # Tuple of [batch_size, seq_len, d_model]
        
        # Apply cluster attention at specified layers
        cluster_enhanced_hidden = all_hidden_states[-1]  # Start with final layer
        cluster_attention_weights = []
        
        # Apply cluster attention modules
        cluster_module_idx = 0
        for i, layer_idx in enumerate(self.cluster_layers):
            if layer_idx < len(all_hidden_states):
                # Get hidden states at this layer
                layer_hidden = all_hidden_states[layer_idx]
                
                # Apply cluster attention
                enhanced_hidden, attention_weights = self.cluster_attention_modules[cluster_module_idx](layer_hidden)
                
                # Combine with current hidden states
                if i == 0:
                    cluster_enhanced_hidden = enhanced_hidden
                else:
                    # Residual connection across cluster attention layers
                    cluster_enhanced_hidden = cluster_enhanced_hidden + enhanced_hidden
                
                cluster_attention_weights.append(attention_weights)
                cluster_module_idx += 1
        
        # Pool hidden states (mean pooling with attention mask)
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(cluster_enhanced_hidden).float()
        sum_hidden = torch.sum(cluster_enhanced_hidden * mask_expanded, dim=1)
        sum_mask = torch.sum(mask_expanded, dim=1)
        pooled_hidden = sum_hidden / (sum_mask + 1e-9)
        
        # Project to cluster space
        cluster_logits = self.cluster_projection(pooled_hidden)  # [batch_size, n_clusters]
        
        # Apply softmax to get cluster activations
        cluster_activations = F.softmax(cluster_logits, dim=-1)
        
        return {
            'pooled_hidden': pooled_hidden,
            'cluster_activations': cluster_activations,
            'cluster_attention_weights': cluster_attention_weights,
            'cluster_logits': cluster_logits
        }
    
    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode query to cluster activations"""
        outputs = self.forward(input_ids, attention_mask)
        return outputs['cluster_activations']
    
    def encode_document(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode document to cluster activations"""
        outputs = self.forward(input_ids, attention_mask)
        return outputs['cluster_activations']
    
    def compute_similarity(self, query_activations: torch.Tensor, 
                         doc_activations: torch.Tensor, 
                         method: str = 'dot') -> torch.Tensor:
        """
        Compute similarity between query and document using cluster activations
        
        Args:
            query_activations: [batch_size, n_clusters]
            doc_activations: [batch_size, n_clusters]
            method: Similarity method ('dot', 'cosine', 'weighted')
        
        Returns:
            similarity_scores: [batch_size]
        """
        if method == 'dot':
            # Dot product similarity (default in paper)
            scores = torch.sum(query_activations * doc_activations, dim=-1)
        
        elif method == 'cosine':
            # Cosine similarity
            query_norm = query_activations / (torch.norm(query_activations, dim=-1, keepdim=True) + 1e-9)
            doc_norm = doc_activations / (torch.norm(doc_activations, dim=-1, keepdim=True) + 1e-9)
            scores = torch.sum(query_norm * doc_norm, dim=-1)
        
        elif method == 'weighted':
            # Weighted similarity (learnable weights per cluster)
            if not hasattr(self, 'cluster_weights'):
                self.cluster_weights = nn.Parameter(torch.ones(self.n_clusters))
            scores = torch.sum(self.cluster_weights * query_activations * doc_activations, dim=-1)
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        return scores

class CLAREInference:
    """
    Inference utilities for CLARE model
    Provides interpretable explanations for retrieval decisions
    """
    def __init__(self, model: CLAREModel, cluster_discovery: SemanticClusterDiscovery, 
                 device: torch.device = torch.device('cuda')):
        self.model = model.to(device)
        self.cluster_discovery = cluster_discovery
        self.device = device
        self.model.eval()
        
        # Cache cluster information
        self.cluster_words = cluster_discovery.get_cluster_words(top_k=20)
        self.cluster_coherence = cluster_discovery.get_cluster_coherence()
    
    def encode_text(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Encode text and return cluster activations with attention"""
        encoding = self.model.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            outputs = self.model(**encoding)
        
        return outputs
    
    def compute_similarity_with_explanation(self, query: str, document: str) -> Tuple[float, Dict]:
        """
        Compute similarity between query and document with detailed explanations
        
        Returns:
            similarity: Similarity score
            explanation: Dictionary containing cluster-based explanation
        """
        # Encode texts
        query_outputs = self.encode_text(query)
        doc_outputs = self.encode_text(document)
        
        # Get cluster activations
        query_clusters = query_outputs['cluster_activations'][0].cpu()
        doc_clusters = doc_outputs['cluster_activations'][0].cpu()
        
        # Compute similarity
        similarity = torch.sum(query_clusters * doc_clusters).item()
        
        # Generate detailed explanation
        explanation = self.generate_explanation(
            query_clusters.numpy(), 
            doc_clusters.numpy(),
            query, 
            document
        )
        explanation['similarity_score'] = similarity
        
        return similarity, explanation
    
    def generate_explanation(self, query_clusters: np.ndarray, 
                           doc_clusters: np.ndarray,
                           query_text: str,
                           doc_text: str,
                           top_k: int = 5) -> Dict:
        """
        Generate interpretable explanation for retrieval decision
        
        Args:
            query_clusters: Query cluster activations
            doc_clusters: Document cluster activations
            query_text: Original query text
            doc_text: Original document text
            top_k: Number of top clusters to include
        
        Returns:
            Detailed explanation dictionary
        """
        # Get top activated clusters for query
        query_top_indices = np.argsort(query_clusters)[-top_k:][::-1]
        query_top_clusters = []
        
        for idx in query_top_indices:
            cluster_info = {
                'cluster_id': int(idx),
                'activation': float(query_clusters[idx]),
                'words': self.cluster_words.get(idx, [])[:10],
                'coherence': float(self.cluster_coherence.get(idx, 0))
            }
            query_top_clusters.append(cluster_info)
        
        # Get top activated clusters for document
        doc_top_indices = np.argsort(doc_clusters)[-top_k:][::-1]
        doc_top_clusters = []
        
        for idx in doc_top_indices:
            cluster_info = {
                'cluster_id': int(idx),
                'activation': float(doc_clusters[idx]),
                'words': self.cluster_words.get(idx, [])[:10],
                'coherence': float(self.cluster_coherence.get(idx, 0))
            }
            doc_top_clusters.append(cluster_info)
        
        # Find shared clusters (intersection of top clusters)
        query_cluster_set = set(query_top_indices)
        doc_cluster_set = set(doc_top_indices)
        shared_cluster_ids = query_cluster_set.intersection(doc_cluster_set)
        
        shared_clusters = []
        for cluster_id in shared_cluster_ids:
            cluster_info = {
                'cluster_id': int(cluster_id),
                'query_activation': float(query_clusters[cluster_id]),
                'doc_activation': float(doc_clusters[cluster_id]),
                'contribution': float(query_clusters[cluster_id] * doc_clusters[cluster_id]),
                'words': self.cluster_words.get(cluster_id, [])[:10],
                'coherence': float(self.cluster_coherence.get(cluster_id, 0))
            }
            shared_clusters.append(cluster_info)
        
        # Sort shared clusters by contribution
        shared_clusters.sort(key=lambda x: x['contribution'], reverse=True)
        
        # Compute cluster-based similarity breakdown
        total_similarity = np.sum(query_clusters * doc_clusters)
        top_cluster_contribution = sum(c['contribution'] for c in shared_clusters[:top_k])
        
        explanation = {
            'query_text': query_text,
            'doc_text': doc_text[:200] + '...' if len(doc_text) > 200 else doc_text,
            'query_top_clusters': query_top_clusters,
            'doc_top_clusters': doc_top_clusters,
            'shared_clusters': shared_clusters,
            'num_shared_clusters': len(shared_clusters),
            'cluster_overlap_ratio': len(shared_clusters) / top_k,
            'top_cluster_contribution_ratio': top_cluster_contribution / total_similarity if total_similarity > 0 else 0,
            'interpretation': self._generate_natural_language_explanation(
                query_top_clusters, doc_top_clusters, shared_clusters
            )
        }
        
        return explanation
    
    def _generate_natural_language_explanation(self, query_clusters: List[Dict], 
                                              doc_clusters: List[Dict], 
                                              shared_clusters: List[Dict]) -> str:
        """Generate human-readable explanation"""
        explanation_parts = []
        
        # Describe query focus
        if query_clusters:
            top_query_words = query_clusters[0]['words'][:5]
            explanation_parts.append(
                f"The query focuses on topics related to: {', '.join(top_query_words)}"
            )
        
        # Describe document content
        if doc_clusters:
            top_doc_words = doc_clusters[0]['words'][:5]
            explanation_parts.append(
                f"The document primarily discusses: {', '.join(top_doc_words)}"
            )
        
        # Describe matching
        if shared_clusters:
            explanation_parts.append(
                f"Strong topical overlap found in {len(shared_clusters)} clusters, "
                f"particularly around: {', '.join(shared_clusters[0]['words'][:3])}"
            )
        else:
            explanation_parts.append(
                "Limited topical overlap between query and document"
            )
        
        return " ".join(explanation_parts)
    
    def batch_encode(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Batch encode multiple texts"""
        all_activations = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encodings = self.model.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                encodings = {k: v.to(self.device) for k, v in encodings.items()}
                outputs = self.model(**encodings)
                all_activations.append(outputs['cluster_activations'].cpu())
        
        return torch.cat(all_activations, dim=0)

# Utility functions for evaluation
def compute_ranking_metrics(query_activations: torch.Tensor, 
                          doc_activations: torch.Tensor, 
                          relevance_labels: torch.Tensor,
                          k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
    """
    Compute ranking metrics for evaluation
    
    Args:
        query_activations: [n_queries, n_clusters]
        doc_activations: [n_docs, n_clusters]
        relevance_labels: [n_queries, n_docs] binary relevance
        k_values: List of k values for metrics@k
    
    Returns:
        Dictionary of metric values
    """
    metrics = {}
    n_queries = query_activations.shape[0]
    
    # Compute all pairwise similarities
    similarities = torch.matmul(query_activations, doc_activations.t())  # [n_queries, n_docs]
    
    for k in k_values:
        ndcg_scores = []
        precision_scores = []
        recall_scores = []
        
        for q_idx in range(n_queries):
            # Get query similarities and labels
            q_sims = similarities[q_idx]
            q_labels = relevance_labels[q_idx]
            
            # Get top-k documents
            top_k_indices = torch.argsort(q_sims, descending=True)[:k]
            top_k_labels = q_labels[top_k_indices]
            
            # Compute nDCG@k
            dcg = torch.sum((2**top_k_labels - 1) / torch.log2(torch.arange(1, k+1).float() + 1))
            ideal_labels = torch.sort(q_labels, descending=True)[0][:k]
            idcg = torch.sum((2**ideal_labels - 1) / torch.log2(torch.arange(1, k+1).float() + 1))
            ndcg = (dcg / idcg).item() if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
            
            # Compute Precision@k
            num_relevant = torch.sum(top_k_labels).item()
            precision = num_relevant / k
            precision_scores.append(precision)
            
            # Compute Recall@k
            total_relevant = torch.sum(q_labels).item()
            recall = num_relevant / total_relevant if total_relevant > 0 else 0.0
            recall_scores.append(recall)
        
        metrics[f'ndcg@{k}'] = np.mean(ndcg_scores)
        metrics[f'precision@{k}'] = np.mean(precision_scores)
        metrics[f'recall@{k}'] = np.mean(recall_scores)
    
    # Compute MAP (Mean Average Precision)
    ap_scores = []
    for q_idx in range(n_queries):
        q_sims = similarities[q_idx]
        q_labels = relevance_labels[q_idx]
        
        # Sort by similarity
        sorted_indices = torch.argsort(q_sims, descending=True)
        sorted_labels = q_labels[sorted_indices]
        
        # Compute AP
        ap = 0.0
        num_relevant = 0
        
        for i, label in enumerate(sorted_labels):
            if label == 1:
                num_relevant += 1
                ap += num_relevant / (i + 1)
        
        total_relevant = torch.sum(q_labels).item()
        if total_relevant > 0:
            ap_scores.append(ap / total_relevant)
    
    metrics['map'] = np.mean(ap_scores) if ap_scores else 0.0
    
    # Compute MRR (Mean Reciprocal Rank)
    rr_scores = []
    for q_idx in range(n_queries):
        q_sims = similarities[q_idx]
        q_labels = relevance_labels[q_idx]
        
        sorted_indices = torch.argsort(q_sims, descending=True)
        sorted_labels = q_labels[sorted_indices]
        
        # Find first relevant document
        for i, label in enumerate(sorted_labels):
            if label == 1:
                rr_scores.append(1.0 / (i + 1))
                break
    
    metrics['mrr'] = np.mean(rr_scores) if rr_scores else 0.0
    
    return metrics

class CLAREDataset(Dataset):
    """
    Dataset class for CLARE training - simplified version for testing
    """
    def __init__(self, queries: List[str], documents: List[str], 
                 labels: List[int], tokenizer, max_length: int = 512):
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = self.queries[idx]
        document = self.documents[idx]
        label = self.labels[idx]
        
        # Tokenize query
        query_encoding = self.tokenizer(
            query,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize document
        doc_encoding = self.tokenizer(
            document,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(),
            'doc_input_ids': doc_encoding['input_ids'].squeeze(),
            'doc_attention_mask': doc_encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = CLAREModel(
        model_name="bert-base-uncased",
        n_clusters=50,  # Smaller for testing
        cluster_layers=[3, 6, 9]
    )
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    logger.info("\nTesting forward pass...")
    
    # Create dummy input
    tokenizer = model.tokenizer
    test_query = "What is machine learning?"
    test_doc = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
    
    # Tokenize
    query_encoding = tokenizer(test_query, return_tensors='pt', padding=True, truncation=True)
    doc_encoding = tokenizer(test_doc, return_tensors='pt', padding=True, truncation=True)
    
    # Forward pass
    with torch.no_grad():
        query_outputs = model(**query_encoding)
        doc_outputs = model(**doc_encoding)
    
    logger.info(f"Query cluster activations shape: {query_outputs['cluster_activations'].shape}")
    logger.info(f"Top 5 activated clusters for query: {torch.topk(query_outputs['cluster_activations'][0], 5).indices.tolist()}")
    
    # Compute similarity
    similarity = model.compute_similarity(
        query_outputs['cluster_activations'],
        doc_outputs['cluster_activations']
    )
    logger.info(f"Query-Document similarity: {similarity.item():.4f}")
    
    # Test Semantic Cluster Discovery
    logger.info("\nTesting Semantic Cluster Discovery...")
    
    # Create dummy term-document matrix
    vocab_size = 1000
    num_docs = 100
    X = np.random.rand(vocab_size, num_docs)
    X = (X > 0.8).astype(float)  # Make it sparse
    
    # Create dummy embeddings
    embedding_dim = 768
    word_embeddings = np.random.randn(vocab_size, embedding_dim)
    vocab = [f"word_{i}" for i in range(vocab_size)]
    
    # Fit cluster discovery
    cluster_discovery = SemanticClusterDiscovery(k=10, max_iter=20)
    cluster_discovery.fit(X, word_embeddings, vocab)
    
    # Get cluster words
    cluster_words = cluster_discovery.get_cluster_words(top_k=5)
    logger.info("\nDiscovered clusters:")
    for i in range(min(5, len(cluster_words))):
        logger.info(f"Cluster {i}: {cluster_words[i]}")
    
    logger.info("\nCLARE framework initialized successfully!")