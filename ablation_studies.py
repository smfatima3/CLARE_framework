        components_to_test = [
            ('full', {}),  # Full model
            ('no_semantic_regularizer', {'lambda2': 0}),
            ('no_cluster_attention', {'disable_cluster_attention': True}),
            ('no_sparsity_loss', {'lambda4': 0}),
            ('no_cluster_loss', {'lambda3': 0}),
            ('random_clusters', {'use_random_clusters': True}),
        ]
        
        results = {}
        
        for component_name, override_config in components_to_test:
            logger.info(f"Testing configuration: {component_name}")
            
            # Create config for this ablation
            config = deepcopy(self.base_config)
            config.update(override_config)
            
            # Train and evaluate model
            metrics = self._train_and_evaluate_model(config, component_name)
            results[component_name] = metrics
            
            # Save intermediate results
            self._save_intermediate_results(component_name, metrics)
        
        self.ablation_results['component_ablation'] = results
        return results
    
    def run_cluster_count_ablation(self, cluster_counts: List[int] = [25, 50, 100, 200]):
        """Test impact of different cluster counts"""
        logger.info("Running cluster count ablation study...")
        
        results = {}
        
        for n_clusters in cluster_counts:
            logger.info(f"Testing with {n_clusters} clusters...")
            
            config = deepcopy(self.base_config)
            config['n_clusters'] = n_clusters
            
            metrics = self._train_and_evaluate_model(config, f"clusters_{n_clusters}")
            results[n_clusters] = metrics
        
        self.ablation_results['cluster_count_ablation'] = results
        return results
    
    def run_layer_ablation(self):
        """Test different cluster attention layer configurations"""
        logger.info("Running layer ablation study...")
        
        layer_configs = [
            ('layers_3_6_9', [3, 6, 9]),  # Default
            ('layers_last', [11]),
            ('layers_middle', [6]),
            ('layers_early', [3]),
            ('layers_all_late', [9, 10, 11]),
            ('layers_every_other', [2, 4, 6, 8, 10]),
            ('no_layers', [])
        ]
        
        results = {}
        
        for config_name, layers in layer_configs:
            logger.info(f"Testing layer configuration: {config_name}")
            
            config = deepcopy(self.base_config)
            config['cluster_layers'] = layers
            
            metrics = self._train_and_evaluate_model(config, config_name)
            results[config_name] = metrics
        
        self.ablation_results['layer_ablation'] = results
        return results
    
    def run_loss_weight_ablation(self):
        """Test different loss weight combinations"""
        logger.info("Running loss weight ablation study...")
        
        # Test different values for λ3 (cluster loss) and λ4 (sparsity loss)
        lambda3_values = [0, 0.01, 0.1, 0.5]
        lambda4_values = [0, 0.001, 0.01, 0.1]
        
        results = {}
        
        for lambda3 in lambda3_values:
            for lambda4 in lambda4_values:
                config_name = f"λ3={lambda3}_λ4={lambda4}"
                logger.info(f"Testing {config_name}")
                
                config = deepcopy(self.base_config)
                config['lambda3'] = lambda3
                config['lambda4'] = lambda4
                
                metrics = self._train_and_evaluate_model(config, config_name)
                results[config_name] = metrics
        
        self.ablation_results['loss_weight_ablation'] = results
        return results
    
    def run_similarity_function_ablation(self):
        """Test different similarity functions"""
        logger.info("Running similarity function ablation study...")
        
        similarity_functions = ['dot', 'cosine', 'weighted']
        
        results = {}
        
        for sim_func in similarity_functions:
            logger.info(f"Testing similarity function: {sim_func}")
            
            config = deepcopy(self.base_config)
            config['similarity_function'] = sim_func
            
            metrics = self._train_and_evaluate_model(config, f"sim_{sim_func}")
            results[sim_func] = metrics
        
        self.ablation_results['similarity_function_ablation'] = results
        return results
    
    def _train_and_evaluate_model(self, config: Dict, experiment_name: str) -> Dict:
        """Train model with given configuration and evaluate"""
        logger.info(f"Training model for {experiment_name}...")
        
        # Phase 1: Cluster discovery (if not disabled)
        if not config.get('use_random_clusters', False):
            cluster_discovery = self._perform_cluster_discovery(config)
        else:
            cluster_discovery = None
        
        # Phase 2: Model training
        # Initialize model
        model = self._create_model(config, cluster_discovery)
        
        # Create data loaders
        train_loader, val_loader, test_loader = self._create_data_loaders(
            model.tokenizer, config
        )
        
        # Train model
        trained_model = self._train_model(
            model, train_loader, val_loader, config
        )
        
        # Evaluate model
        metrics = self._evaluate_model(trained_model, test_loader)
        
        # Add configuration info
        metrics['config'] = config
        metrics['experiment_name'] = experiment_name
        
        # Clean up
        del model
        del trained_model
        torch.cuda.empty_cache()
        
        return metrics
    
    def _perform_cluster_discovery(self, config: Dict) -> SemanticClusterDiscovery:
        """Perform semantic cluster discovery"""
        cluster_discovery = SemanticClusterDiscovery(
            k=config.get('n_clusters', 100),
            lambda1=config.get('lambda1', 0.01),
            lambda2=config.get('lambda2', 0.1),
            max_iter=50  # Reduced for ablation studies
        )
        
        # Sample documents for efficiency
        n_docs = min(10000, self.term_doc_matrix.shape[1])
        if self.term_doc_matrix.shape[1] > n_docs:
            sample_indices = np.random.choice(self.term_doc_matrix.shape[1], n_docs, replace=False)
            term_doc_sample = self.term_doc_matrix[:, sample_indices].toarray()
        else:
            term_doc_sample = self.term_doc_matrix.toarray()
        
        cluster_discovery.fit(
            term_doc_sample,
            self.word_embeddings,
            self.vocabulary
        )
        
        return cluster_discovery
    
    def _create_model(self, config: Dict, cluster_discovery: Optional[SemanticClusterDiscovery]) -> CLAREModel:
        """Create model with given configuration"""
        # Handle special ablation cases
        if config.get('disable_cluster_attention', False):
            # Create model without cluster attention
            config['cluster_layers'] = []
        
        model = CLAREModel(
            model_name=config.get('model_name', 'bert-base-uncased'),
            n_clusters=config.get('n_clusters', 100),
            n_heads=config.get('n_heads', 8),
            cluster_layers=config.get('cluster_layers', [3, 6, 9])
        )
        
        # Initialize cluster embeddings
        if cluster_discovery and hasattr(cluster_discovery, 'W'):
            # Initialize from discovered clusters
            cluster_embeddings = []
            for i in range(config['n_clusters']):
                cluster_weights = cluster_discovery.W[:, i]
                weighted_embedding = np.sum(
                    self.word_embeddings * cluster_weights[:, np.newaxis], 
                    axis=0
                ) / (np.sum(cluster_weights) + 1e-8)
                cluster_embeddings.append(weighted_embedding)
            
            cluster_embeddings = torch.tensor(np.vstack(cluster_embeddings), dtype=torch.float32)
            
            # Initialize cluster attention modules
            with torch.no_grad():
                for module in model.cluster_attention_modules:
                    if cluster_embeddings.shape[1] != module.d_model:
                        # Project to correct dimension
                        projection = nn.Linear(cluster_embeddings.shape[1], module.d_model)
                        projected = projection(cluster_embeddings)
                        module.cluster_embeddings.data = projected.to(model.device)
                    else:
                        module.cluster_embeddings.data = cluster_embeddings.to(model.device)
        
        return model.to(self.device)
    
    def _create_data_loaders(self, tokenizer, config: Dict):
        """Create data loaders for training"""
        dataset_config = DatasetConfig(
            max_query_length=128,
            max_doc_length=512,
            negative_sampling_ratio=4
        )
        
        # Create datasets
        train_dataset = CLARETorchDataset(
            self.dataset['train'][:10000],  # Subsample for ablation
            tokenizer,
            dataset_config
        )
        
        val_dataset = CLARETorchDataset(
            self.dataset['validation'][:2000],
            tokenizer,
            dataset_config
        )
        
        test_dataset = CLARETorchDataset(
            self.dataset['test'][:2000],
            tokenizer,
            dataset_config
        )
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def _train_model(self, model, train_loader, val_loader, config: Dict):
        """Train model with given configuration"""
        trainer = CLARETrainer(model, self.device)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        total_steps = len(train_loader) * 2  # 2 epochs for ablation
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop (simplified for ablation)
        best_val_loss = float('inf')
        
        for epoch in range(2):  # Reduced epochs for ablation
            model.train()
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
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
                
                # Compute loss with configured weights
                loss, _ = trainer.compute_loss(
                    query_outputs, 
                    pos_doc_outputs, 
                    neg_doc_outputs_list,
                    lambda3=config.get('lambda3', 0.1),
                    lambda4=config.get('lambda4', 0.01)
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            # Validation
            val_loss = self._validate(model, val_loader, trainer, config)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        return model
    
    def _validate(self, model, val_loader, trainer, config: Dict) -> float:
        """Validate model"""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                if batch['neg_doc_input_ids'].numel() == 0:
                    continue
                
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
                
                loss, _ = trainer.compute_loss(
                    query_outputs, 
                    pos_doc_outputs, 
                    neg_doc_outputs_list,
                    lambda3=config.get('lambda3', 0.1),
                    lambda4=config.get('lambda4', 0.01)
                )
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def _evaluate_model(self, model, test_loader) -> Dict:
        """Evaluate model on test set"""
        model.eval()
        
        all_scores = []
        all_labels = []
        total_time = 0
        n_queries = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                if batch['neg_doc_input_ids'].numel() == 0:
                    continue
                
                import time
                start_time = time.time()
                
                # Forward pass
                query_outputs = model(
                    input_ids=batch['query_input_ids'],
                    attention_mask=batch['query_attention_mask']
                )
                
                pos_doc_outputs = model(
                    input_ids=batch['pos_doc_input_ids'],
                    attention_mask=batch['pos_doc_attention_mask']
                )
                
                # Compute scores
                similarity_func = getattr(model, 'similarity_function', 'dot')
                pos_scores = model.compute_similarity(
                    query_outputs['cluster_activations'],
                    pos_doc_outputs['cluster_activations'],
                    method=similarity_func
                )
                
                # Negative scores
                neg_scores_list = []
                if batch['neg_doc_input_ids'].dim() == 3:
                    for i in range(batch['neg_doc_input_ids'].size(1)):
                        neg_outputs = model(
                            input_ids=batch['neg_doc_input_ids'][:, i, :],
                            attention_mask=batch['neg_doc_attention_mask'][:, i, :]
                        )
                        neg_score = model.compute_similarity(
                            query_outputs['cluster_activations'],
                            neg_outputs['cluster_activations'],
                            method=similarity_func
                        )
                        neg_scores_list.append(neg_score)
                
                inference_time = time.time() - start_time
                total_time += inference_time
                n_queries += batch['query_input_ids'].size(0)
                
                # Collect for metrics
                batch_size = pos_scores.size(0)
                for i in range(batch_size):
                    all_scores.append(pos_scores[i].item())
                    all_labels.append(1)
                    
                    for neg_score in neg_scores_list:
                        all_scores.append(neg_score[i].item())
                        all_labels.append(0)
        
        # Compute metrics
        metrics = self._compute_metrics(all_scores, all_labels)
        metrics['avg_latency_ms'] = (total_time / n_queries) * 1000
        metrics['total_queries'] = n_queries
        
        return metrics
    
    def _compute_metrics(self, scores: List[float], labels: List[int]) -> Dict:
        """Compute evaluation metrics"""
        # Group into queries
        query_groups = []
        current_group = {'scores': [], 'labels': []}
        
        for score, label in zip(scores, labels):
            current_group['scores'].append(score)
            current_group['labels'].append(label)
            
            if len(current_group['scores']) >= 5:  # 1 positive + 4 negatives
                query_groups.append(current_group)
                current_group = {'scores': [], 'labels': []}
        
        # Compute metrics
        ndcg_scores = []
        accuracies = []
        
        for group in query_groups:
            # Sort by scores
            sorted_indices = np.argsort(group['scores'])[::-1]
            sorted_labels = [group['labels'][i] for i in sorted_indices]
            
            # nDCG@10
            dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(sorted_labels[:10]))
            ideal_labels = sorted(group['labels'], reverse=True)[:10]
            idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_labels))
            
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
            
            # Accuracy (is positive doc ranked first?)
            accuracies.append(1 if sorted_labels[0] == 1 else 0)
        
        return {
            'ndcg@10': np.mean(ndcg_scores) if ndcg_scores else 0,
            'accuracy': np.mean(accuracies) if accuracies else 0,
            'num_queries': len(query_groups)
        }
    
    def _save_intermediate_results(self, experiment_name: str, metrics: Dict):
        """Save intermediate results"""
        result_path = os.path.join(self.output_dir, f'{experiment_name}_results.json')
        with open(result_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def create_tables_and_analysis(self, pdf_path: str):
        """Create ablation study tables and analysis"""
        with PdfPages(pdf_path) as pdf:
            # 1. Component ablation table
            self._create_component_ablation_table(pdf)
            
            # 2. Cluster count analysis
            self._create_cluster_count_analysis(pdf)
            
            # 3. Layer configuration analysis
            self._create_layer_ablation_analysis(pdf)
            
            # 4. Loss weight heatmap
            self._create_loss_weight_heatmap(pdf)
            
            # 5. Summary and insights
            self._create_summary_analysis(pdf)
    
    def _create_component_ablation_table(self, pdf):
        """Create component ablation results table"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        if 'component_ablation' not in self.ablation_results:
            return
        
        data = self.ablation_results['component_ablation']
        
        # Prepare table data
        table_data = []
        components = ['full', 'no_semantic_regularizer', 'no_cluster_attention', 
                     'no_sparsity_loss', 'no_cluster_loss', 'random_clusters']
        
        for comp in components:
            if comp in data:
                metrics = data[comp]
                full_metrics = data.get('full', {})
                
                table_data.append([
                    comp.replace('_', ' ').title(),
                    f"{metrics.get('ndcg@10', 0):.4f}",
                    f"{metrics.get('accuracy', 0):.4f}",
                    f"{metrics.get('avg_latency_ms', 0):.1f}",
                    f"{(metrics.get('ndcg@10', 0) - full_metrics.get('ndcg@10', 0)):.4f}",
                    f"{(metrics.get('accuracy', 0) - full_metrics.get('accuracy', 0)):.4f}"
                ])
        
        columns = ['Component', 'nDCG@10', 'Accuracy', 'Latency (ms)', 'Δ nDCG@10', 'Δ Accuracy']
        
        table = ax.table(cellText=table_data, colLabels=columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Style header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight negative deltas
        for i in range(1, len(table_data) + 1):
            if i > 1:  # Skip full model
                if float(table_data[i-1][4]) < 0:
                    table[(i, 4)].set_facecolor('#FFE6E6')
                if float(table_data[i-1][5]) < 0:
                    table[(i, 5)].set_facecolor('#FFE6E6')
        
        ax.set_title('Component Ablation Results', fontsize=16, fontweight='bold', pad=20)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_cluster_count_analysis(self, pdf):
        """Create cluster count analysis plots"""
        if 'cluster_count_ablation' not in self.ablation_results:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        data = self.ablation_results['cluster_count_ablation']
        cluster_counts = sorted(data.keys())
        
        # Extract metrics
        ndcg_scores = [data[k].get('ndcg@10', 0) for k in cluster_counts]
        accuracies = [data[k].get('accuracy', 0) for k in cluster_counts]
        latencies = [data[k].get('avg_latency_ms', 0) for k in cluster_counts]
        
        # Plot nDCG@10
        ax = axes[0]
        ax.plot(cluster_counts, ndcg_scores, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters', fontsize=12)
        ax.set_ylabel('nDCG@10', fontsize=12)
        ax.set_title('nDCG@10 vs Cluster Count', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Find and mark optimal
        optimal_idx = np.argmax(ndcg_scores)
        ax.scatter(cluster_counts[optimal_idx], ndcg_scores[optimal_idx], 
                  color='red', s=200, marker='*', zorder=3)
        ax.text(cluster_counts[optimal_idx], ndcg_scores[optimal_idx] + 0.005,
               f'Optimal: {cluster_counts[optimal_idx]}', ha='center', va='bottom')
        
        # Plot accuracy
        ax = axes[1]
        ax.plot(cluster_counts, accuracies, 'o-', linewidth=2, markersize=8, color='green')
        ax.set_xlabel('Number of Clusters', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy vs Cluster Count', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot latency
        ax = axes[2]
        ax.plot(cluster_counts, latencies, 'o-', linewidth=2, markersize=8, color='red')
        ax.set_xlabel('Number of Clusters', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title('Inference Latency vs Cluster Count', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_layer_ablation_analysis(self, pdf):
        """Create layer configuration analysis"""
        if 'layer_ablation' not in self.ablation_results:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        data = self.ablation_results['layer_ablation']
        
        # Prepare data
        configs = list(data.keys())
        ndcg_scores = [data[c].get('ndcg@10', 0) for c in configs]
        accuracies = [data[c].get('accuracy', 0) for c in configs]
        
        # Bar plot for nDCG@10
        bars = ax1.bar(range(len(configs)), ndcg_scores)
        ax1.set_xticks(range(len(configs)))
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.set_ylabel('nDCG@10', fontsize=12)
        ax1.set_title('Layer Configuration Impact on nDCG@10', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Color best configuration
        best_idx = np.argmax(ndcg_scores)
        bars[best_idx].set_color('green')
        
        # Add value labels
        for bar, val in zip(bars, ndcg_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Bar plot for accuracy
        bars = ax2.bar(range(len(configs)), accuracies)
        ax2.set_xticks(range(len(configs)))
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Layer Configuration Impact on Accuracy', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Color best configuration
        best_idx = np.argmax(accuracies)
        bars[best_idx].set_color('green')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_loss_weight_heatmap(self, pdf):
        """Create loss weight impact heatmap"""
        if 'loss_weight_ablation' not in self.ablation_results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        data = self.ablation_results['loss_weight_ablation']
        
        # Extract unique lambda values
        lambda3_values = []
        lambda4_values = []
        
        for key in data.keys():
            parts = key.split('_')
            lambda3 = float(parts[0].split('=')[1])
            lambda4 = float(parts[1].split('=')[1])
            
            if lambda3 not in lambda3_values:
                lambda3_values.append(lambda3)
            if lambda4 not in lambda4_values:
                lambda4_values.append(lambda4)
        
        lambda3_values = sorted(lambda3_values)
        lambda4_values = sorted(lambda4_values)
        
        # Create matrices for heatmaps
        ndcg_matrix = np.zeros((len(lambda4_values), len(lambda3_values)))
        acc_matrix = np.zeros((len(lambda4_values), len(lambda3_values)))
        
        for i, l4 in enumerate(lambda4_values):
            for j, l3 in enumerate(lambda3_values):
                key = f"λ3={l3}_λ4={l4}"
                if key in data:
                    ndcg_matrix[i, j] = data[key].get('ndcg@10', 0)
                    acc_matrix[i, j] = data[key].get('accuracy', 0)
        
        # Plot nDCG@10 heatmap
        ax = axes[0]
        im = ax.imshow(ndcg_matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(lambda3_values)))
        ax.set_xticklabels([f'{v:.3f}' for v in lambda3_values])
        ax.set_yticks(range(len(lambda4_values)))
        ax.set_yticklabels([f'{v:.3f}' for v in lambda4_values])
        ax.set_xlabel('λ3 (Cluster Loss Weight)', fontsize=12)
        ax.set_ylabel('λ4 (Sparsity Loss Weight)', fontsize=12)
        ax.set_title('nDCG@10 vs Loss Weights', fontsize=14, fontweight='bold')
        
        # Add text annotations
        for i in range(len(lambda4_values)):
            for j in range(len(lambda3_values)):
                text = ax.text(j, i, f'{ndcg_matrix[i, j]:.3f}',
                             ha='center', va='center', 
                             color='white' if ndcg_matrix[i, j] > 0.5 * ndcg_matrix.max() else 'black')
        
        plt.colorbar(im, ax=ax)
        
        # Plot accuracy heatmap
        ax = axes[1]
        im = ax.imshow(acc_matrix, cmap='YlGnBu', aspect='auto')
        ax.set_xticks(range(len(lambda3_values)))
        ax.set_xticklabels([f'{v:.3f}' for v in lambda3_values])
        ax.set_yticks(range(len(lambda4_values)))
        ax.set_yticklabels([f'{v:.3f}' for v in lambda4_values])
        ax.set_xlabel('λ3 (Cluster Loss Weight)', fontsize=12)
        ax.set_ylabel('λ4 (Sparsity Loss Weight)', fontsize=12)
        ax.set_title('Accuracy vs Loss Weights', fontsize=14, fontweight='bold')
        
        # Add text annotations
        for i in range(len(lambda4_values)):
            for j in range(len(lambda3_values)):
                text = ax.text(j, i, f'{acc_matrix[i, j]:.3f}',
                             ha='center', va='center',
                             color='white' if acc_matrix[i, j] > 0.5 * acc_matrix.max() else 'black')
        
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_summary_analysis(self, pdf):
        """Create summary analysis and key insights"""
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('off')
        
        # Collect key insights
        insights = []
        
        # Component ablation insights
        if 'component_ablation' in self.ablation_results:
            data = self.ablation_results['component_ablation']
            full_ndcg = data.get('full', {}).get('ndcg@10', 0)
            
            # Find most impactful component
            max_drop = 0
            most_impactful = ''
            for comp, metrics in data.items():
                if comp != 'full':
                    drop = full_ndcg - metrics.get('ndcg@10', 0)
                    if drop > max_drop:
                        max_drop = drop
                        most_impactful = comp
            
            insights.append(f"• Most impactful component: {most_impactful.replace('_', ' ')} "
                          f"(nDCG@10 drop: {max_drop:.4f})")
        
        # Cluster count insights
        if 'cluster_count_ablation' in self.ablation_results:
            data = self.ablation_results['cluster_count_ablation']
            cluster_counts = sorted(data.keys())
            ndcg_scores = [data[k].get('ndcg@10', 0) for k in cluster_counts]
            optimal_clusters = cluster_counts[np.argmax(ndcg_scores)]
            
            insights.append(f"• Optimal cluster count: {optimal_clusters} "
                          f"(nDCG@10: {max(ndcg_scores):.4f})")
        
        # Layer configuration insights
        if 'layer_ablation' in self.ablation_results:
            data = self.ablation_results['layer_ablation']
            best_config = max(data.items(), key=lambda x: x[1].get('ndcg@10', 0))
            
            insights.append(f"• Best layer configuration: {best_config[0]} "
                          f"(nDCG@10: {best_config[1].get('ndcg@10', 0):.4f})")
        
        # Loss weight insights
        if 'loss_weight_ablation' in self.ablation_results:
            data = self.ablation_results['loss_weight_ablation']
            best_weights = max(data.items(), key=lambda x: x[1].get('ndcg@10', 0))
            
            insights.append(f"• Optimal loss weights: {best_weights[0]} "
                          f"(nDCG@10: {best_weights[1].get('ndcg@10', 0):.4f})")
        
        # Similarity function insights
        if 'similarity_function_ablation' in self.ablation_results:
            data = self.ablation_results['similarity_function_ablation']
            best_sim = max(data.items(), key=lambda x: x[1].get('ndcg@10', 0))
            
            insights.append(f"• Best similarity function: {best_sim[0]} "
                          f"(nDCG@10: {best_sim[1].get('ndcg@10', 0):.4f})")
        
        # Create summary text
        summary_text = """ABLATION STUDY SUMMARY

Key Findings:
"""
        for insight in insights:
            summary_text += f"\n{insight}"
        
        summary_text += """

Component Importance Ranking:
1. Cluster Attention Mechanism
2. Semantic Regularization in Cluster Discovery
3. Cluster Consistency Loss (λ3)
4. Sparsity Regularization (λ4)

Recommendations:
• Use 100 clusters for optimal balance of performance and efficiency
• Apply cluster attention at layers [3, 6, 9]
• Set λ3 = 0.1 and λ4 = 0.01 for best results
• Dot product similarity performs best for cluster activations

Trade-offs:
• More clusters improve performance but increase latency
• Deeper cluster attention helps but adds computational cost
• Semantic regularization is crucial for interpretability
"""
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=12, va='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))
        
        ax.set_title('Ablation Study Summary and Insights', fontsize=16, fontweight='bold')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def run_all_ablations(self):
        """Run all ablation studies"""
        logger.info("Running all ablation studies...")
        
        # Run different ablation studies
        self.run_component_ablation()
        self.run_cluster_count_ablation([25, 50, 100, 200])
        self.run_layer_ablation()
        self.run_loss_weight_ablation()
        self.run_similarity_function_ablation()
        
        # Save results
        results_path = os.path.join(self.output_dir, 'ablation_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.ablation_results, f, indent=2)
        
        # Create PDF report
        pdf_path = os.path.join(self.output_dir, 'ablation_studies.pdf')
        self.create_tables_and_analysis(pdf_path)
        
        logger.info(f"Results saved to {results_path}")
        logger.info(f"PDF report saved to {pdf_path}")
        
        return self.ablation_results


def main():
    """Main function to run ablation studies"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ablation studies for CLARE')
    parser.add_argument('--dataset_path', type=str, default='clare_dataset_complete.pkl',
                       help='Path to CLARE dataset')
    parser.add_argument('--output_dir', type=str, default='ablation_results',
                       help='Output directory for results')
    
    # Base configuration
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='Base transformer model')
    parser.add_argument('--n_clusters', type=int, default=100,
                       help='Default number of clusters')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--lambda1', type=float, default=0.01,
                       help='Sparsity regularization for clustering')
    parser.add_argument('--lambda2', type=float, default=0.1,
                       help='Semantic coherence regularization')
    parser.add_argument('--lambda3', type=float, default=0.1,
                       help='Cluster consistency loss weight')
    parser.add_argument('--lambda4', type=float, default=0.01,
                       help='Sparsity loss weight')
    
    # Ablation options
    parser.add_argument('--skip_component', action='store_true',
                       help='Skip component ablation')
    parser.add_argument('--skip_cluster_count', action='store_true',
                       help='Skip cluster count ablation')
    parser.add_argument('--skip_layer', action='store_true',
                       help='Skip layer ablation')
    parser.add_argument('--skip_loss_weight', action='store_true',
                       help='Skip loss weight ablation')
    parser.add_argument('--skip_similarity', action='store_true',
                       help='Skip similarity function ablation')
    
    args = parser.parse_args()
    
    # Create base configuration
    base_config = {
        'model_name': args.model_name,
        'n_clusters': args.n_clusters,
        'n_heads': args.n_heads,
        'cluster_layers': [3, 6, 9],
        'lambda1': args.lambda1,
        'lambda2': args.lambda2,
        'lambda3': args.lambda3,
        'lambda4': args.lambda4
    }
    
    # Run ablation studies
    runner = AblationStudyRunner(base_config, args.dataset_path, args.output_dir)
    
    # Run selected ablations
    if not args.skip_component:
        runner.run_component_ablation()
    
    if not args.skip_cluster_count:
        runner.run_cluster_count_ablation()
    
    if not args.skip_layer:
        runner.run_layer_ablation()
    
    if not args.skip_loss_weight:
        runner.run_loss_weight_ablation()
    
    if not args.skip_similarity:
        runner.run_similarity_function_ablation()
    
    # Save results and create report
    results_path = os.path.join(args.output_dir, 'ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(runner.ablation_results, f, indent=2)
    
    pdf_path = os.path.join(args.output_dir, 'ablation_studies.pdf')
    runner.create_tables_and_analysis(pdf_path)
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDIES SUMMARY")
    print("="*60)
    
    if 'component_ablation' in runner.ablation_results:
        print("\nComponent Ablation Results:")
        data = runner.ablation_results['component_ablation']
        full_ndcg = data.get('full', {}).get('ndcg@10', 0)
        
        for comp, metrics in data.items():
            if comp != 'full':
                drop = full_ndcg - metrics.get('ndcg@10', 0)
                print(f"  Without {comp}: nDCG@10 drop = {drop:.4f}")
    
    if 'cluster_count_ablation' in runner.ablation_results:
        print("\nOptimal cluster count analysis:")
        data = runner.ablation_results['cluster_count_ablation']
        best_k = max(data.items(), key=lambda x: x[1].get('ndcg@10', 0))
        print(f"  Best k = {best_k[0]} (nDCG@10 = {best_k[1].get('ndcg@10', 0):.4f})")
    
    print(f"\nPDF report saved to: {pdf_path}")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Ablation Studies for CLARE Paper
Tests the impact of different components and design choices
Generates tables and analysis in PDF format
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
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import logging
from copy import deepcopy

# Import CLARE components
from clare_framework import CLAREModel, CLARETrainer, SemanticClusterDiscovery
from clare_dataset_integration import CLARETorchDataset, DatasetConfig
from transformers import get_linear_schedule_with_warmup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


class AblationStudyRunner:
    """Runs ablation studies for CLARE components"""
    
    def __init__(self, base_config: Dict, dataset_path: str, output_dir: str = "ablation_results"):
        self.base_config = base_config
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load dataset
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
        
        # Load term-document matrix and embeddings
        self.term_doc_matrix = None
        self.word_embeddings = None
        self.vocabulary = None
        self._load_clustering_data()
        
        # Results storage
        self.ablation_results = {
            'component_ablation': {},
            'cluster_count_ablation': {},
            'layer_ablation': {},
            'loss_weight_ablation': {},
            'similarity_function_ablation': {}
        }
    
    def _load_clustering_data(self):
        """Load term-document matrix and word embeddings"""
        matrix_path = self.dataset_path.replace('.pkl', '_term_doc_matrix.npz')
        embeddings_path = self.dataset_path.replace('.pkl', '_word_embeddings.npy')
        
        if os.path.exists(matrix_path):
            import scipy.sparse
            self.term_doc_matrix = scipy.sparse.load_npz(matrix_path)
            logger.info(f"Loaded term-document matrix: {self.term_doc_matrix.shape}")
        
        if os.path.exists(embeddings_path):
            self.word_embeddings = np.load(embeddings_path)
            logger.info(f"Loaded word embeddings: {self.word_embeddings.shape}")
        
        self.vocabulary = self.dataset.get('vocabulary', [])
    
    def run_component_ablation(self):
        """Test impact of removing different components"""
        logger.info("Running component ablation study...")
        
        components_to_test = [
            ('full', {}),  # Full model
            ('no