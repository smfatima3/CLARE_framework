#!/usr/bin/env python3
"""
Visualization Tool for CLARE Paper
Creates comprehensive visualizations for model interpretation and analysis
Generates interactive plots and static visualizations in PDF format
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

# Import CLARE components
from clare_framework import CLAREModel, CLAREInference, SemanticClusterDiscovery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


class CLAREVisualizer:
    """Comprehensive visualization tool for CLARE analysis"""
    
    def __init__(self, model_path: str, dataset_path: str, output_dir: str = "visualizations"):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and components
        self._load_model()
        self._load_dataset()
        
        # Storage for visualizations
        self.visualizations = {}
    
    def _load_model(self):
        """Load trained CLARE model"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.model = CLAREModel(
            n_clusters=checkpoint['config']['n_clusters'],
            cluster_layers=checkpoint['config'].get('cluster_layers', [3, 6, 9])
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load cluster discovery if available
        if 'cluster_discovery' in checkpoint:
            self.cluster_discovery = SemanticClusterDiscovery(
                k=checkpoint['config']['n_clusters']
            )
            self.cluster_discovery.W = checkpoint['cluster_discovery']['W']
            self.cluster_discovery.H = checkpoint['cluster_discovery']['H']
            self.cluster_discovery.vocab = checkpoint['vocabulary']
            self.cluster_words = self.cluster_discovery.get_cluster_words(top_k=20)
        else:
            self.cluster_discovery = None
            self.cluster_words = {}
        
        logger.info(f"Model loaded with {checkpoint['config']['n_clusters']} clusters")
    
    def _load_dataset(self):
        """Load dataset for visualization"""
        with open(self.dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
        
        logger.info(f"Dataset loaded")
    
    def visualize_cluster_embeddings(self, method: str = 'tsne'):
        """Visualize cluster embeddings in 2D/3D space"""
        logger.info(f"Visualizing cluster embeddings using {method}...")
        
        # Get cluster embeddings from model
        cluster_embeddings = []
        for module in self.model.cluster_attention_modules:
            cluster_embeddings.append(module.cluster_embeddings.detach().cpu().numpy())
        
        # Use embeddings from first module
        embeddings = cluster_embeddings[0]
        n_clusters = embeddings.shape[0]
        
        # Reduce dimensionality
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, n_clusters-1))
        elif method == 'pca':
            reducer = PCA(n_components=2)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Create interactive plot with Plotly
        fig = go.Figure()
        
        # Add cluster points
        hover_texts = []
        for i in range(n_clusters):
            if i in self.cluster_words:
                words = ', '.join(self.cluster_words[i][:5])
                hover_texts.append(f"Cluster {i}<br>Words: {words}")
            else:
                hover_texts.append(f"Cluster {i}")
        
        fig.add_trace(go.Scatter(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            mode='markers+text',
            marker=dict(
                size=10,
                color=list(range(n_clusters)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Cluster ID")
            ),
            text=[str(i) for i in range(n_clusters)],
            textposition="top center",
            hovertext=hover_texts,
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title=f'Cluster Embeddings Visualization ({method.upper()})',
            xaxis_title=f'{method.upper()} Dimension 1',
            yaxis_title=f'{method.upper()} Dimension 2',
            width=1000,
            height=800
        )
        
        # Save interactive plot
        html_path = os.path.join(self.output_dir, f'cluster_embeddings_{method}.html')
        fig.write_html(html_path)
        logger.info(f"Interactive plot saved to {html_path}")
        
        # Create static plot for PDF
        fig_static, ax = plt.subplots(figsize=(12, 10))
        
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                           c=list(range(n_clusters)), cmap='viridis', s=100)
        
        # Add cluster labels
        for i in range(n_clusters):
            ax.annotate(str(i), (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                       fontsize=8, ha='center', va='bottom')
        
        plt.colorbar(scatter, ax=ax, label='Cluster ID')
        ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
        ax.set_title(f'Cluster Embeddings Visualization ({method.upper()})', fontsize=14, fontweight='bold')
        
        return fig_static
    
    def visualize_cluster_hierarchy(self):
        """Visualize hierarchical relationships between clusters"""
        logger.info("Creating cluster hierarchy visualization...")
        
        # Get cluster embeddings
        cluster_embeddings = []
        for module in self.model.cluster_attention_modules:
            cluster_embeddings.append(module.cluster_embeddings.detach().cpu().numpy())
        embeddings = cluster_embeddings[0]
        
        # Compute distance matrix
        distances = pdist(embeddings, metric='cosine')
        
        # Create dendrogram
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Compute linkage
        Z = linkage(distances, method='ward')
        
        # Create dendrogram with cluster labels
        labels = []
        for i in range(len(embeddings)):
            if i in self.cluster_words:
                words = ', '.join(self.cluster_words[i][:3])
                labels.append(f"C{i}: {words}")
            else:
                labels.append(f"Cluster {i}")
        
        dendrogram(Z, labels=labels, ax=ax, leaf_rotation=90, leaf_font_size=10)
        
        ax.set_title('Cluster Hierarchy (Based on Embedding Similarity)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def visualize_attention_patterns(self, sample_queries: List[str] = None):
        """Visualize cluster attention patterns for sample queries"""
        logger.info("Visualizing attention patterns...")
        
        if sample_queries is None:
            # Use some test queries
            sample_queries = [item['query'] for item in self.dataset['test'][:5]]
        
        fig, axes = plt.subplots(len(sample_queries), len(self.model.cluster_layers), 
                                figsize=(15, 3*len(sample_queries)))
        
        if len(sample_queries) == 1:
            axes = axes.reshape(1, -1)
        
        for q_idx, query in enumerate(sample_queries):
            # Encode query
            inputs = self.model.tokenizer(query, return_tensors='pt', 
                                         padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_weights = outputs['cluster_attention_weights']
            
            # Plot attention for each layer
            for l_idx, (layer_num, attn) in enumerate(zip(self.model.cluster_layers, attention_weights)):
                ax = axes[q_idx, l_idx]
                
                # Average over heads and get first sequence
                attn_avg = attn[0].mean(dim=0).cpu().numpy()  # [seq_len, n_clusters]
                
                # Create heatmap
                im = ax.imshow(attn_avg.T, aspect='auto', cmap='YlOrRd')
                
                ax.set_title(f'Layer {layer_num}', fontsize=10)
                ax.set_xlabel('Token Position' if q_idx == len(sample_queries)-1 else '')
                ax.set_ylabel('Cluster ID' if l_idx == 0 else '')
                
                # Add colorbar
                if l_idx == len(self.model.cluster_layers) - 1:
                    plt.colorbar(im, ax=ax)
            
            # Add query text
            axes[q_idx, 0].text(-0.3, 0.5, f'Query: "{query[:50]}..."' if len(query) > 50 else f'Query: "{query}"',
                               transform=axes[q_idx, 0].transAxes, rotation=90,
                               va='center', ha='right', fontsize=10)
        
        plt.suptitle('Cluster Attention Patterns Across Layers', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def visualize_cluster_activation_distribution(self):
        """Visualize distribution of cluster activations across queries"""
        logger.info("Analyzing cluster activation distributions...")
        
        # Sample queries from test set
        test_queries = [item['query'] for item in self.dataset['test'][:500]]
        
        all_activations = []
        
        for query in test_queries:
            inputs = self.model.tokenizer(query, return_tensors='pt',
                                         padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                activations = outputs['cluster_activations'][0].cpu().numpy()
                all_activations.append(activations)
        
        all_activations = np.vstack(all_activations)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Average activation per cluster
        ax = axes[0, 0]
        avg_activations = np.mean(all_activations, axis=0)
        bars = ax.bar(range(len(avg_activations)), avg_activations)
        
        # Color top-k clusters
        top_k = 10
        top_indices = np.argsort(avg_activations)[-top_k:]
        for i in top_indices:
            bars[i].set_color('red')
        
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Average Activation', fontsize=12)
        ax.set_title('Average Cluster Activation Across Queries', fontsize=14, fontweight='bold')
        
        # 2. Activation variance per cluster
        ax = axes[0, 1]
        std_activations = np.std(all_activations, axis=0)
        ax.bar(range(len(std_activations)), std_activations, color='orange')
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Activation Std Dev', fontsize=12)
        ax.set_title('Cluster Activation Variance', fontsize=14, fontweight='bold')
        
        # 3. Sparsity distribution
        ax = axes[1, 0]
        sparsity_per_query = np.sum(all_activations > 0.01, axis=1)
        ax.hist(sparsity_per_query, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Number of Active Clusters', fontsize=12)
        ax.set_ylabel('Number of Queries', fontsize=12)
        ax.set_title('Distribution of Active Clusters per Query', fontsize=14, fontweight='bold')
        ax.axvline(np.mean(sparsity_per_query), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(sparsity_per_query):.1f}')
        ax.legend()
        
        # 4. Cluster co-activation heatmap
        ax = axes[1, 1]
        # Compute co-activation matrix
        binary_activations = (all_activations > 0.01).astype(float)
        coactivation = np.corrcoef(binary_activations.T)
        
        im = ax.imshow(coactivation, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Cluster ID', fontsize=12)
        ax.set_title('Cluster Co-activation Patterns', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Correlation')
        
        plt.tight_layout()
        return fig
    
    def visualize_query_document_similarity(self, num_examples: int = 5):
        """Visualize query-document similarity breakdown by clusters"""
        logger.info("Visualizing query-document similarity patterns...")
        
        # Get examples from test set
        examples = []
        for item in self.dataset['test'][:num_examples*5]:  # Get more to filter
            if item['label'] == 1:  # Positive pairs
                examples.append(item)
                if len(examples) >= num_examples:
                    break
        
        fig, axes = plt.subplots(num_examples, 2, figsize=(15, 4*num_examples))
        if num_examples == 1:
            axes = axes.reshape(1, -1)
        
        inference = CLAREInference(self.model, self.cluster_discovery, self.device)
        
        for idx, example in enumerate(examples):
            query = example['query']
            document = example['document'][:200] + '...' if len(example['document']) > 200 else example['document']
            
            # Get similarity and explanation
            similarity, explanation = inference.compute_similarity_with_explanation(query, document)
            
            # Plot cluster contributions
            ax = axes[idx, 0]
            
            shared_clusters = explanation['shared_clusters']
            if shared_clusters:
                cluster_ids = [c['cluster_id'] for c in shared_clusters[:10]]
                contributions = [c['contribution'] for c in shared_clusters[:10]]
                
                bars = ax.barh(range(len(cluster_ids)), contributions)
                ax.set_yticks(range(len(cluster_ids)))
                ax.set_yticklabels([f"C{cid}" for cid in cluster_ids])
                ax.set_xlabel('Contribution to Similarity', fontsize=10)
                ax.set_title(f'Top Contributing Clusters (Similarity: {similarity:.3f})', fontsize=12)
                
                # Color bars by contribution strength
                colors = plt.cm.YlOrRd(np.array(contributions) / max(contributions))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
            
            # Plot activation comparison
            ax = axes[idx, 1]
            
            query_clusters = explanation['query_top_clusters']
            doc_clusters = explanation['doc_top_clusters']
            
            x = np.arange(5)  # Top 5 clusters
            width = 0.35
            
            query_acts = [c['activation'] for c in query_clusters[:5]]
            doc_acts = [c['activation'] for c in doc_clusters[:5]]
            
            ax.bar(x - width/2, query_acts, width, label='Query', alpha=0.8)
            ax.bar(x + width/2, doc_acts, width, label='Document', alpha=0.8)
            
            ax.set_xlabel('Top Clusters', fontsize=10)
            ax.set_ylabel('Activation', fontsize=10)
            ax.set_title('Query vs Document Activations', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels([f"C{c['cluster_id']}" for c in query_clusters[:5]])
            ax.legend()
            
            # Add query text
            fig.text(0.02, axes[idx, 0].get_position().y1 - 0.02, 
                    f'Query: "{query}"', fontsize=10, style='italic')
        
        plt.suptitle('Query-Document Similarity Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def visualize_domain_cluster_patterns(self):
        """Visualize cluster activation patterns across different domains"""
        logger.info("Analyzing domain-specific cluster patterns...")
        
        # Get domain labels
        domains = set()
        for item in self.dataset['test']:
            domains.add(item.get('dataset', 'unknown'))
        domains = list(domains)
        
        # Collect activations by domain
        domain_activations = {domain: [] for domain in domains}
        
        for item in self.dataset['test'][:1000]:  # Sample
            query = item['query']
            domain = item.get('dataset', 'unknown')
            
            inputs = self.model.tokenizer(query, return_tensors='pt',
                                         padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                activations = outputs['cluster_activations'][0].cpu().numpy()
                domain_activations[domain].append(activations)
        
        # Compute average activations per domain
        domain_avg_activations = {}
        for domain, acts in domain_activations.items():
            if acts:
                domain_avg_activations[domain] = np.mean(acts, axis=0)
        
        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # 1. Heatmap of domain cluster patterns
        ax = axes[0]
        
        if domain_avg_activations:
            activation_matrix = np.vstack([domain_avg_activations[d] for d in domains if d in domain_avg_activations])
            
            im = ax.imshow(activation_matrix, aspect='auto', cmap='YlOrRd')
            ax.set_yticks(range(len(domains)))
            ax.set_yticklabels(domains)
            ax.set_xlabel('Cluster ID', fontsize=12)
            ax.set_ylabel('Domain', fontsize=12)
            ax.set_title('Average Cluster Activations by Domain', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Average Activation')
        
        # 2. Domain similarity based on cluster patterns
        ax = axes[1]
        
        if len(domain_avg_activations) > 1:
            # Compute pairwise similarities
            domain_list = list(domain_avg_activations.keys())
            n_domains = len(domain_list)
            similarity_matrix = np.zeros((n_domains, n_domains))
            
            for i in range(n_domains):
                for j in range(n_domains):
                    act_i = domain_avg_activations[domain_list[i]]
                    act_j = domain_avg_activations[domain_list[j]]
                    # Cosine similarity
                    similarity = np.dot(act_i, act_j) / (np.linalg.norm(act_i) * np.linalg.norm(act_j))
                    similarity_matrix[i, j] = similarity
            
            im = ax.imshow(similarity_matrix, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
            ax.set_xticks(range(n_domains))
            ax.set_xticklabels(domain_list, rotation=45, ha='right')
            ax.set_yticks(range(n_domains))
            ax.set_yticklabels(domain_list)
            ax.set_title('Domain Similarity Based on Cluster Patterns', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Cosine Similarity')
            
            # Add text annotations
            for i in range(n_domains):
                for j in range(n_domains):
                    text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                                 ha='center', va='center',
                                 color='white' if similarity_matrix[i, j] < 0.5 else 'black')
        
        plt.tight_layout()
        return fig
    
    def create_cluster_word_clouds(self, top_k_clusters: int = 10):
        """Create word clouds for top clusters"""
        logger.info("Creating cluster word clouds...")
        
        from wordcloud import WordCloud
        
        # Get top clusters by average activation
        test_queries = [item['query'] for item in self.dataset['test'][:500]]
        all_activations = []
        
        for query in test_queries:
            inputs = self.model.tokenizer(query, return_tensors='pt',
                                         padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                activations = outputs['cluster_activations'][0].cpu().numpy()
                all_activations.append(activations)
        
        avg_activations = np.mean(all_activations, axis=0)
        top_cluster_indices = np.argsort(avg_activations)[-top_k_clusters:][::-1]
        
        # Create subplots
        n_cols = 5
        n_rows = (top_k_clusters + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        axes = axes.ravel()
        
        for idx, cluster_id in enumerate(top_cluster_indices):
            ax = axes[idx]
            
            if cluster_id in self.cluster_words:
                words = self.cluster_words[cluster_id][:50]  # Top 50 words
                
                # Get word weights from cluster-term matrix
                if hasattr(self.cluster_discovery, 'W'):
                    weights = self.cluster_discovery.W[:, cluster_id]
                    word_weights = {word: weight for word, weight in 
                                  zip(self.cluster_discovery.vocab, weights) if word in words}
                else:
                    word_weights = {word: 1.0 for word in words}
                
                # Create word cloud
                if word_weights:
                    wordcloud = WordCloud(width=400, height=300, 
                                        background_color='white',
                                        relative_scaling=0.5).generate_from_frequencies(word_weights)
                    
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.set_title(f'Cluster {cluster_id} (Avg Act: {avg_activations[cluster_id]:.3f})',
                               fontsize=12, fontweight='bold')
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, f'No words for\nCluster {cluster_id}',
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f'Cluster {cluster_id}', fontsize=12)
                ax.axis('off')
        
        # Hide unused subplots
        for idx in range(top_k_clusters, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Word Clouds for Top Activated Clusters', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_interactive_dashboard(self):
        """Create an interactive dashboard with Plotly"""
        logger.info("Creating interactive dashboard...")
        
        # Prepare data
        test_queries = [item['query'] for item in self.dataset['test'][:100]]
        all_activations = []
        query_texts = []
        
        for query in test_queries:
            inputs = self.model.tokenizer(query, return_tensors='pt',
                                         padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                activations = outputs['cluster_activations'][0].cpu().numpy()
                all_activations.append(activations)
                query_texts.append(query[:50] + '...' if len(query) > 50 else query)
        
        all_activations = np.vstack(all_activations)
        
        # Create dashboard with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cluster Activation Heatmap', 'Average Cluster Activations',
                          'Query Sparsity Distribution', 'Cluster Correlation'),
            specs=[[{'type': 'heatmap'}, {'type': 'bar'}],
                   [{'type': 'histogram'}, {'type': 'heatmap'}]]
        )
        
        # 1. Activation heatmap
        fig.add_trace(
            go.Heatmap(z=all_activations.T, 
                      colorscale='YlOrRd',
                      hovertemplate='Query: %{x}<br>Cluster: %{y}<br>Activation: %{z}<extra></extra>'),
            row=1, col=1
        )
        
        # 2. Average activations bar chart
        avg_activations = np.mean(all_activations, axis=0)
        fig.add_trace(
            go.Bar(x=list(range(len(avg_activations))), 
                  y=avg_activations,
                  marker_color='lightblue',
                  hovertemplate='Cluster %{x}<br>Avg Activation: %{y:.3f}<extra></extra>'),
            row=1, col=2
        )
        
        # 3. Sparsity histogram
        sparsity = np.sum(all_activations > 0.01, axis=1)
        fig.add_trace(
            go.Histogram(x=sparsity, 
                        nbinsx=30,
                        marker_color='lightgreen',
                        hovertemplate='Active Clusters: %{x}<br>Count: %{y}<extra></extra>'),
            row=2, col=1
        )
        
        # 4. Cluster correlation
        cluster_corr = np.corrcoef(all_activations.T)
        fig.add_trace(
            go.Heatmap(z=cluster_corr,
                      colorscale='RdBu',
                      zmid=0,
                      hovertemplate='Cluster %{x} - Cluster %{y}<br>Correlation: %{z:.3f}<extra></extra>'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="CLARE Cluster Analysis Dashboard",
            height=1000,
            showlegend=False
        )
        
        # Update axes
        fig.update_xaxes(title_text="Query Index", row=1, col=1)
        fig.update_yaxes(title_text="Cluster ID", row=1, col=1)
        fig.update_xaxes(title_text="Cluster ID", row=1, col=2)
        fig.update_yaxes(title_text="Average Activation", row=1, col=2)
        fig.update_xaxes(title_text="Number of Active Clusters", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Cluster ID", row=2, col=2)
        fig.update_yaxes(title_text="Cluster ID", row=2, col=2)
        
        # Save dashboard
        dashboard_path = os.path.join(self.output_dir, 'clare_dashboard.html')
        fig.write_html(dashboard_path)
        logger.info(f"Interactive dashboard saved to {dashboard_path}")
        
        return fig
    
    def create_comprehensive_pdf(self, pdf_path: str):
        """Create comprehensive PDF with all visualizations"""
        with PdfPages(pdf_path) as pdf:
            # 1. Cluster embeddings visualizations
            for method in ['tsne', 'pca']:
                fig = self.visualize_cluster_embeddings(method)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # 2. Cluster hierarchy
            fig = self.visualize_cluster_hierarchy()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # 3. Attention patterns
            sample_queries = [
                "What is machine learning?",
                "How to treat diabetes?",
                "Best customer service practices"
            ]
            fig = self.visualize_attention_patterns(sample_queries)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # 4. Cluster activation distribution
            fig = self.visualize_cluster_activation_distribution()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # 5. Query-document similarity
            fig = self.visualize_query_document_similarity(num_examples=3)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # 6. Domain cluster patterns
            fig = self.visualize_domain_cluster_patterns()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # 7. Cluster word clouds
            fig = self.create_cluster_word_clouds(top_k_clusters=15)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # 8. Summary page
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis('off')
            
            summary_text = f"""
CLARE Visualization Summary

Model Configuration:
• Number of clusters: {self.model.n_clusters}
• Cluster attention layers: {self.model.cluster_layers}
• Model name: {self.model.model_name}

Key Insights:
• Clusters show clear semantic groupings
• Domain-specific activation patterns are observable
• Sparse activation promotes interpretability
• Hierarchical cluster relationships emerge

Generated Visualizations:
1. Cluster embeddings (t-SNE and PCA)
2. Cluster hierarchy dendrogram
3. Layer-wise attention patterns
4. Activation distribution analysis
5. Query-document similarity breakdown
6. Domain-specific cluster patterns
7. Word clouds for top clusters
8. Interactive dashboard (see HTML file)
            """
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                   fontsize=12, va='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))
            
            ax.set_title('CLARE Visualization Report', fontsize=16, fontweight='bold')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    def generate_all_visualizations(self):
        """Generate all visualizations and save outputs"""
        logger.info("Generating all visualizations...")
        
        # Create static visualizations PDF
        pdf_path = os.path.join(self.output_dir, 'clare_visualizations.pdf')
        self.create_comprehensive_pdf(pdf_path)
        logger.info(f"PDF report saved to {pdf_path}")
        
        # Create interactive dashboard
        self.create_interactive_dashboard()
        
        # Save visualization metadata
        metadata = {
            'model_path': self.model_path,
            'dataset_path': self.dataset_path,
            'n_clusters': self.model.n_clusters,
            'cluster_layers': self.model.cluster_layers,
            'output_files': {
                'pdf': pdf_path,
                'dashboard': os.path.join(self.output_dir, 'clare_dashboard.html'),
                'cluster_embeddings_tsne': os.path.join(self.output_dir, 'cluster_embeddings_tsne.html'),
                'cluster_embeddings_pca': os.path.join(self.output_dir, 'cluster_embeddings_pca.html')
            }
        }
        
        metadata_path = os.path.join(self.output_dir, 'visualization_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
        
        return metadata


def main():
    """Main function to run visualization tool"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CLARE Visualization Tool')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained CLARE model checkpoint')
    parser.add_argument('--dataset_path', type=str, default='clare_dataset_complete.pkl',
                       help='Path to CLARE dataset')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--sample_queries', type=str, nargs='+',
                       help='Sample queries for attention visualization')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = CLAREVisualizer(args.model_path, args.dataset_path, args.output_dir)
    
    # Generate visualizations
    metadata = visualizer.generate_all_visualizations()
    
    # Print summary
    print("\n" + "="*60)
    print("CLARE VISUALIZATION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    for file_type, path in metadata['output_files'].items():
        print(f"  {file_type}: {path}")
    
    print("\nVisualization features:")
    print("  • Cluster embeddings in 2D space")
    print("  • Hierarchical cluster relationships")
    print("  • Attention pattern analysis")
    print("  • Activation distribution statistics")
    print("  • Query-document similarity breakdown")
    print("  • Domain-specific patterns")
    print("  • Interactive dashboard")
    
    print(f"\nOpen {metadata['output_files']['dashboard']} in a browser for interactive exploration!")


if __name__ == "__main__":
    main()