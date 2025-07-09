import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from datasets import load_dataset, DatasetDict
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, vstack
import re
import random
from tqdm import tqdm
import pickle
import os
from dataclasses import dataclass
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
import hashlib

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """Configuration for dataset processing aligned with paper specifications"""
    max_query_length: int = 128
    max_doc_length: int = 512
    negative_sampling_ratio: int = 4  # Paper mentions multiple negatives
    min_query_length: int = 3  # Shorter to match MS MARCO avg
    min_doc_length: int = 10
    vocab_size: int = 30000  # For term-document matrix
    min_df: int = 5  # Minimum document frequency
    max_df: float = 0.5  # Maximum document frequency
    use_stemming: bool = True
    remove_stopwords: bool = True
    
    # Dataset-specific configs
    msmarco_sample_size: Optional[int] = None  # None for full dataset
    bioasq_sample_size: Optional[int] = None
    granola_sample_size: Optional[int] = None
    
    # Cluster discovery parameters
    n_clusters: int = 100  # As per paper
    
class TermDocumentMatrixBuilder:
    """Build term-document matrices for Semantic Cluster Discovery"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.vectorizer = None
        self.vocabulary = None
        self.stop_words = set(stopwords.words('english')) if config.remove_stopwords else None
        
    def build_matrix(self, documents: List[str], 
                    fit_vectorizer: bool = True) -> Tuple[csr_matrix, List[str]]:
        """
        Build term-document matrix as specified in paper
        Returns: (term_doc_matrix, vocabulary)
        """
        logger.info(f"Building term-document matrix for {len(documents)} documents...")
        
        if fit_vectorizer or self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=self.config.vocab_size,
                min_df=self.config.min_df,
                max_df=self.config.max_df,
                stop_words='english' if self.config.remove_stopwords else None,
                ngram_range=(1, 2),  # Include bigrams for better semantic capture
                sublinear_tf=True,  # Use log normalization
                dtype=np.float32
            )
            term_doc_matrix = self.vectorizer.fit_transform(documents)
            self.vocabulary = self.vectorizer.get_feature_names_out()
        else:
            term_doc_matrix = self.vectorizer.transform(documents)
        
        logger.info(f"Term-document matrix shape: {term_doc_matrix.shape}")
        logger.info(f"Vocabulary size: {len(self.vocabulary)}")
        
        return term_doc_matrix, list(self.vocabulary)
    
    def get_word_embeddings(self, vocabulary: List[str], 
                           model_name: str = "bert-base-uncased") -> np.ndarray:
        """
        Get pre-trained word embeddings for vocabulary
        Used in Semantic Cluster Discovery
        """
        logger.info("Extracting word embeddings for vocabulary...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        
        embeddings = []
        batch_size = 64
        
        with torch.no_grad():
            for i in tqdm(range(0, len(vocabulary), batch_size), 
                         desc="Extracting embeddings"):
                batch_words = vocabulary[i:i + batch_size]
                
                # Tokenize words
                inputs = tokenizer(batch_words, return_tensors="pt", 
                                 padding=True, truncation=True, max_length=10)
                
                # Get embeddings
                outputs = model(**inputs)
                # Use mean pooling over tokens
                word_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(word_embeddings.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        logger.info(f"Word embeddings shape: {embeddings.shape}")
        
        return embeddings

class Query2DocMSMARCOProcessor:
    """Process intfloat/query2doc_msmarco dataset"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        
    def process_dataset(self) -> Dict[str, List]:
        """
        Process query2doc MS MARCO dataset
        This dataset contains query-document pairs with relevance labels
        """
        logger.info("Processing intfloat/query2doc_msmarco dataset...")
        
        try:
            # Load the dataset
            dataset = load_dataset("intfloat/query2doc_msmarco", split="train")
            
            # Sample if needed
            if self.config.msmarco_sample_size:
                dataset = dataset.shuffle(seed=42).select(range(min(self.config.msmarco_sample_size, len(dataset))))
            
            processed_data = []
            documents_for_matrix = []
            unique_docs = {}
            
            # Process each example
            for idx, item in enumerate(tqdm(dataset, desc="Processing query2doc_msmarco")):
                query = item.get('query', '').strip()
                doc = item.get('doc', '').strip()
                
                # Filter by length requirements
                if (len(query.split()) >= self.config.min_query_length and
                    len(doc.split()) >= self.config.min_doc_length):
                    
                    # Generate document ID
                    doc_id = f"msmarco_d_{hashlib.md5(doc.encode()).hexdigest()[:8]}"
                    
                    # Store document for term-document matrix
                    if doc_id not in unique_docs:
                        unique_docs[doc_id] = doc
                        documents_for_matrix.append(doc)
                    
                    # Create positive pair (query2doc pairs are positive by default)
                    processed_data.append({
                        'query_id': f"msmarco_q_{idx}",
                        'query': query,
                        'doc_id': doc_id,
                        'document': doc,
                        'label': 1,  # Positive pair
                        'dataset': 'query2doc_msmarco'
                    })
            
            # Create negative pairs by random sampling
            logger.info("Creating negative pairs for query2doc_msmarco...")
            negative_pairs = []
            
            # Group by query
            query_groups = defaultdict(list)
            for item in processed_data:
                query_groups[item['query_id']].append(item)
            
            # For each query, sample negative documents
            all_doc_ids = list(unique_docs.keys())
            
            for query_id, positive_items in tqdm(query_groups.items(), desc="Creating negatives"):
                positive_doc_ids = {item['doc_id'] for item in positive_items}
                query_text = positive_items[0]['query']
                
                # Sample negative documents
                negative_doc_ids = [doc_id for doc_id in all_doc_ids if doc_id not in positive_doc_ids]
                
                if negative_doc_ids:
                    num_negatives = min(self.config.negative_sampling_ratio * len(positive_items), 
                                      len(negative_doc_ids))
                    sampled_negative_ids = random.sample(negative_doc_ids, num_negatives)
                    
                    for neg_doc_id in sampled_negative_ids:
                        negative_pairs.append({
                            'query_id': query_id,
                            'query': query_text,
                            'doc_id': neg_doc_id,
                            'document': unique_docs[neg_doc_id],
                            'label': 0,  # Negative pair
                            'dataset': 'query2doc_msmarco'
                        })
            
            # Combine positive and negative pairs
            processed_data.extend(negative_pairs)
            
            logger.info(f"Processed {len(processed_data)} query-document pairs from query2doc_msmarco")
            logger.info(f"Positive pairs: {sum(1 for x in processed_data if x['label'] == 1)}")
            logger.info(f"Negative pairs: {sum(1 for x in processed_data if x['label'] == 0)}")
            logger.info(f"Unique documents: {len(unique_docs)}")
            
            return {
                'data': processed_data,
                'documents': documents_for_matrix,
                'unique_docs': unique_docs
            }
            
        except Exception as e:
            logger.error(f"Error processing query2doc_msmarco: {e}")
            return {'data': [], 'documents': [], 'unique_docs': {}}

class BioASQProcessor:
    """Process BeIR/bioasq-generated-queries dataset"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        
    def process_dataset(self) -> Dict[str, List]:
        """
        Process BioASQ generated queries dataset
        This dataset contains biomedical queries and documents
        """
        logger.info("Processing BeIR/bioasq-generated-queries dataset...")
        
        try:
            # Load the dataset
            dataset = load_dataset("BeIR/bioasq-generated-queries", split="train")
            
            # Sample if needed
            if self.config.bioasq_sample_size:
                dataset = dataset.shuffle(seed=42).select(range(min(self.config.bioasq_sample_size, len(dataset))))
            
            processed_data = []
            documents_for_matrix = []
            unique_docs = {}
            
            # Process each example
            for idx, item in enumerate(tqdm(dataset, desc="Processing BioASQ")):
                query = item.get('query', '').strip()
                text = item.get('text', '').strip()
                
                # Filter by length requirements
                if (len(query.split()) >= self.config.min_query_length and
                    len(text.split()) >= self.config.min_doc_length):
                    
                    # Generate document ID
                    doc_id = f"bioasq_d_{hashlib.md5(text.encode()).hexdigest()[:8]}"
                    
                    # Store document for term-document matrix
                    if doc_id not in unique_docs:
                        unique_docs[doc_id] = text
                        documents_for_matrix.append(text)
                    
                    # Create positive pair
                    processed_data.append({
                        'query_id': f"bioasq_q_{idx}",
                        'query': query,
                        'doc_id': doc_id,
                        'document': text,
                        'label': 1,  # Positive pair
                        'dataset': 'bioasq'
                    })
            
            # Create negative pairs
            logger.info("Creating negative pairs for BioASQ...")
            negative_pairs = []
            
            # Group by query
            query_groups = defaultdict(list)
            for item in processed_data:
                query_groups[item['query_id']].append(item)
            
            # For each query, sample negative documents
            all_doc_ids = list(unique_docs.keys())
            
            for query_id, positive_items in tqdm(query_groups.items(), desc="Creating negatives"):
                positive_doc_ids = {item['doc_id'] for item in positive_items}
                query_text = positive_items[0]['query']
                
                # Sample negative documents
                negative_doc_ids = [doc_id for doc_id in all_doc_ids if doc_id not in positive_doc_ids]
                
                if negative_doc_ids:
                    num_negatives = min(self.config.negative_sampling_ratio * len(positive_items), 
                                      len(negative_doc_ids))
                    sampled_negative_ids = random.sample(negative_doc_ids, num_negatives)
                    
                    for neg_doc_id in sampled_negative_ids:
                        negative_pairs.append({
                            'query_id': query_id,
                            'query': query_text,
                            'doc_id': neg_doc_id,
                            'document': unique_docs[neg_doc_id],
                            'label': 0,  # Negative pair
                            'dataset': 'bioasq'
                        })
            
            # Combine positive and negative pairs
            processed_data.extend(negative_pairs)
            
            logger.info(f"Processed {len(processed_data)} query-document pairs from BioASQ")
            logger.info(f"Positive pairs: {sum(1 for x in processed_data if x['label'] == 1)}")
            logger.info(f"Negative pairs: {sum(1 for x in processed_data if x['label'] == 0)}")
            logger.info(f"Unique documents: {len(unique_docs)}")
            
            return {
                'data': processed_data,
                'documents': documents_for_matrix,
                'unique_docs': unique_docs
            }
            
        except Exception as e:
            logger.error(f"Error processing BioASQ: {e}")
            return {'data': [], 'documents': [], 'unique_docs': {}}

class GranolaEntityProcessor:
    """Process google/granola-entity-questions dataset"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        
    def process_dataset(self) -> Dict[str, List]:
        """
        Process Granola entity questions dataset
        This dataset contains entity-focused questions
        """
        logger.info("Processing google/granola-entity-questions dataset...")
        
        try:
            # Load the dataset
            dataset = load_dataset("google/granola-entity-questions", split="train")
            
            # Sample if needed
            if self.config.granola_sample_size:
                dataset = dataset.shuffle(seed=42).select(range(min(self.config.granola_sample_size, len(dataset))))
            
            processed_data = []
            documents_for_matrix = []
            unique_docs = {}
            
            # Process each example
            for idx, item in enumerate(tqdm(dataset, desc="Processing Granola")):
                question = item.get('question', '').strip()
                
                # Create document from entity information
                entity_name = item.get('entity_name', '').strip()
                entity_description = item.get('entity_description', '').strip()
                
                # Combine entity information as document
                if entity_description:
                    document = f"{entity_name}: {entity_description}"
                else:
                    document = entity_name
                
                # Filter by length requirements
                if (len(question.split()) >= self.config.min_query_length and
                    len(document.split()) >= self.config.min_doc_length):
                    
                    # Generate document ID
                    doc_id = f"granola_d_{hashlib.md5(document.encode()).hexdigest()[:8]}"
                    
                    # Store document for term-document matrix
                    if doc_id not in unique_docs:
                        unique_docs[doc_id] = document
                        documents_for_matrix.append(document)
                    
                    # Create positive pair
                    processed_data.append({
                        'query_id': f"granola_q_{idx}",
                        'query': question,
                        'doc_id': doc_id,
                        'document': document,
                        'label': 1,  # Positive pair
                        'dataset': 'granola',
                        'entity_name': entity_name
                    })
            
            # Create negative pairs
            logger.info("Creating negative pairs for Granola...")
            negative_pairs = []
            
            # Group by query
            query_groups = defaultdict(list)
            for item in processed_data:
                query_groups[item['query_id']].append(item)
            
            # For each query, sample negative documents
            all_doc_ids = list(unique_docs.keys())
            
            for query_id, positive_items in tqdm(query_groups.items(), desc="Creating negatives"):
                positive_doc_ids = {item['doc_id'] for item in positive_items}
                query_text = positive_items[0]['query']
                
                # Sample negative documents (preferably from different entities)
                negative_doc_ids = [doc_id for doc_id in all_doc_ids if doc_id not in positive_doc_ids]
                
                if negative_doc_ids:
                    num_negatives = min(self.config.negative_sampling_ratio * len(positive_items), 
                                      len(negative_doc_ids))
                    sampled_negative_ids = random.sample(negative_doc_ids, num_negatives)
                    
                    for neg_doc_id in sampled_negative_ids:
                        negative_pairs.append({
                            'query_id': query_id,
                            'query': query_text,
                            'doc_id': neg_doc_id,
                            'document': unique_docs[neg_doc_id],
                            'label': 0,  # Negative pair
                            'dataset': 'granola'
                        })
            
            # Combine positive and negative pairs
            processed_data.extend(negative_pairs)
            
            logger.info(f"Processed {len(processed_data)} query-document pairs from Granola")
            logger.info(f"Positive pairs: {sum(1 for x in processed_data if x['label'] == 1)}")
            logger.info(f"Negative pairs: {sum(1 for x in processed_data if x['label'] == 0)}")
            logger.info(f"Unique documents: {len(unique_docs)}")
            
            return {
                'data': processed_data,
                'documents': documents_for_matrix,
                'unique_docs': unique_docs
            }
            
        except Exception as e:
            logger.error(f"Error processing Granola: {e}")
            return {'data': [], 'documents': [], 'unique_docs': {}}

class CLAREDatasetBuilder:
    """Main dataset builder for CLARE framework - Paper Aligned Version"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.matrix_builder = TermDocumentMatrixBuilder(config)
        self.query2doc_processor = Query2DocMSMARCOProcessor(config)
        self.bioasq_processor = BioASQProcessor(config)
        self.granola_processor = GranolaEntityProcessor(config)
        
    def build_complete_dataset(self) -> Dict:
        """
        Build complete dataset using the three specified datasets:
        - intfloat/query2doc_msmarco
        - BeIR/bioasq-generated-queries
        - google/granola-entity-questions
        """
        logger.info("Building complete CLARE dataset...")
        
        all_data = []
        all_documents = []
        all_unique_docs = {}
        dataset_stats = {}
        
        # 1. Process query2doc_msmarco
        logger.info("\n[1/3] Processing query2doc_msmarco...")
        msmarco_result = self.query2doc_processor.process_dataset()
        if msmarco_result['data']:
            all_data.extend(msmarco_result['data'])
            all_documents.extend(msmarco_result['documents'])
            all_unique_docs.update(msmarco_result['unique_docs'])
            dataset_stats['query2doc_msmarco'] = {
                'total': len(msmarco_result['data']),
                'positive': sum(1 for x in msmarco_result['data'] if x['label'] == 1),
                'negative': sum(1 for x in msmarco_result['data'] if x['label'] == 0)
            }
        
        # 2. Process BioASQ
        logger.info("\n[2/3] Processing BioASQ...")
        bioasq_result = self.bioasq_processor.process_dataset()
        if bioasq_result['data']:
            all_data.extend(bioasq_result['data'])
            all_documents.extend(bioasq_result['documents'])
            all_unique_docs.update(bioasq_result['unique_docs'])
            dataset_stats['bioasq'] = {
                'total': len(bioasq_result['data']),
                'positive': sum(1 for x in bioasq_result['data'] if x['label'] == 1),
                'negative': sum(1 for x in bioasq_result['data'] if x['label'] == 0)
            }
        
        # 3. Process Granola
        logger.info("\n[3/3] Processing Granola Entity Questions...")
        granola_result = self.granola_processor.process_dataset()
        if granola_result['data']:
            all_data.extend(granola_result['data'])
            all_documents.extend(granola_result['documents'])
            all_unique_docs.update(granola_result['unique_docs'])
            dataset_stats['granola'] = {
                'total': len(granola_result['data']),
                'positive': sum(1 for x in granola_result['data'] if x['label'] == 1),
                'negative': sum(1 for x in granola_result['data'] if x['label'] == 0)
            }
        
        logger.info(f"\nTotal samples collected: {len(all_data)}")
        logger.info(f"Total unique documents: {len(all_unique_docs)}")
        logger.info(f"Dataset distribution: {dataset_stats}")
        
        # 4. Build term-document matrix for Semantic Cluster Discovery
        logger.info("\nBuilding term-document matrix...")
        
        # Use all unique documents
        document_texts = list(all_unique_docs.values())
        
        # Sample if too many documents
        if len(document_texts) > 100000:
            logger.info(f"Sampling 100,000 documents from {len(document_texts)} for term-document matrix")
            document_texts = random.sample(document_texts, 100000)
        
        term_doc_matrix, vocabulary = self.matrix_builder.build_matrix(document_texts)
        
        # 5. Get word embeddings for vocabulary
        logger.info("\nExtracting word embeddings for vocabulary...")
        word_embeddings = self.matrix_builder.get_word_embeddings(vocabulary)
        
        # Create final dataset structure
        dataset = {
            'train_data': all_data,
            'term_doc_matrix': term_doc_matrix,
            'vocabulary': vocabulary,
            'word_embeddings': word_embeddings,
            'dataset_stats': dataset_stats,
            'unique_docs': all_unique_docs
        }
        
        return dataset
    
    def create_train_val_test_splits(self, dataset: Dict, 
                                   train_ratio: float = 0.8,
                                   val_ratio: float = 0.1,
                                   test_ratio: float = 0.1) -> Dict:
        """
        Create train/validation/test splits maintaining query groups
        Paper mentions using specific splits for MS MARCO
        """
        logger.info("\nCreating train/validation/test splits...")
        
        all_data = dataset['train_data']
        
        # Group by query to avoid leakage
        query_groups = defaultdict(list)
        for item in all_data:
            query_groups[item['query_id']].append(item)
        
        # Separate by dataset for stratified splitting
        dataset_queries = defaultdict(list)
        for query_id, items in query_groups.items():
            dataset_name = items[0]['dataset']
            dataset_queries[dataset_name].append(query_id)
        
        # Split each dataset proportionally
        train_queries = []
        val_queries = []
        test_queries = []
        
        for dataset_name, query_ids in dataset_queries.items():
            random.shuffle(query_ids)
            
            n_queries = len(query_ids)
            n_train = int(n_queries * train_ratio)
            n_val = int(n_queries * val_ratio)
            
            train_queries.extend(query_ids[:n_train])
            val_queries.extend(query_ids[n_train:n_train + n_val])
            test_queries.extend(query_ids[n_train + n_val:])
            
            logger.info(f"{dataset_name}: {n_train} train, {n_val} val, {n_queries - n_train - n_val} test queries")
        
        # Create splits
        train_data = []
        val_data = []
        test_data = []
        
        for qid in train_queries:
            train_data.extend(query_groups[qid])
        for qid in val_queries:
            val_data.extend(query_groups[qid])
        for qid in test_queries:
            test_data.extend(query_groups[qid])
        
        # Shuffle within splits
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
        # Maintain matrix and embeddings
        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data,
            'term_doc_matrix': dataset['term_doc_matrix'],
            'vocabulary': dataset['vocabulary'],
            'word_embeddings': dataset['word_embeddings'],
            'dataset_stats': dataset['dataset_stats'],
            'unique_docs': dataset['unique_docs']
        }
        
        logger.info(f"\nFinal splits:")
        logger.info(f"Train: {len(train_data)} pairs ({len(train_queries)} queries)")
        logger.info(f"Val: {len(val_data)} pairs ({len(val_queries)} queries)")
        logger.info(f"Test: {len(test_data)} pairs ({len(test_queries)} queries)")
        
        return splits
    
    def save_dataset(self, dataset: Dict, filepath: str):
        """Save dataset including term-document matrix"""
        logger.info(f"\nSaving dataset to {filepath}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save main data as pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'train': dataset['train'],
                'validation': dataset['validation'],
                'test': dataset['test'],
                'vocabulary': dataset['vocabulary'],
                'dataset_stats': dataset['dataset_stats']
            }, f)
        
        # Save term-document matrix separately (scipy sparse format)
        import scipy.sparse
        matrix_path = filepath.replace('.pkl', '_term_doc_matrix.npz')
        scipy.sparse.save_npz(matrix_path, dataset['term_doc_matrix'])
        
        # Save word embeddings
        embeddings_path = filepath.replace('.pkl', '_word_embeddings.npy')
        np.save(embeddings_path, dataset['word_embeddings'])
        
        logger.info(f"Dataset saved successfully")
        logger.info(f"Term-document matrix saved to {matrix_path}")
        logger.info(f"Word embeddings saved to {embeddings_path}")
    
    def load_dataset(self, filepath: str) -> Dict:
        """Load dataset including term-document matrix"""
        logger.info(f"Loading dataset from {filepath}")
        
        # Load main data
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        
        # Load term-document matrix
        import scipy.sparse
        matrix_path = filepath.replace('.pkl', '_term_doc_matrix.npz')
        if os.path.exists(matrix_path):
            dataset['term_doc_matrix'] = scipy.sparse.load_npz(matrix_path)
        
        # Load word embeddings
        embeddings_path = filepath.replace('.pkl', '_word_embeddings.npy')
        if os.path.exists(embeddings_path):
            dataset['word_embeddings'] = np.load(embeddings_path)
        
        logger.info("Dataset loaded successfully")
        return dataset

class CLARETorchDataset(Dataset):
    """PyTorch Dataset for CLARE training - Paper Aligned"""
    
    def __init__(self, data: List[Dict], tokenizer, config: DatasetConfig):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        
        # Group by query for efficient batch negative sampling
        self.query_groups = defaultdict(list)
        for item in data:
            self.query_groups[item['query_id']].append(item)
        
        self.queries = list(self.query_groups.keys())
        
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        """
        Return a query with its positive and negative documents
        Following paper's training strategy
        """
        query_id = self.queries[idx]
        group = self.query_groups[query_id]
        
        # Get query text (same for all items in group)
        query_text = group[0]['query']
        
        # Separate positive and negative documents
        positive_docs = [item for item in group if item['label'] == 1]
        negative_docs = [item for item in group if item['label'] == 0]
        
        # Sample one positive (if available)
        if positive_docs:
            positive_doc = random.choice(positive_docs)
        else:
            # Skip queries without positive documents
            return self.__getitem__((idx + 1) % len(self))
        
        # Sample negatives (paper uses multiple negatives per positive)
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
        neg_encodings = []
        for neg_doc in sampled_negatives:
            neg_encoding = self.tokenizer(
                neg_doc['document'],
                truncation=True,
                padding='max_length',
                max_length=self.config.max_doc_length,
                return_tensors='pt'
            )
            neg_encodings.append(neg_encoding)
        
        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(),
            'pos_doc_input_ids': pos_encoding['input_ids'].squeeze(),
            'pos_doc_attention_mask': pos_encoding['attention_mask'].squeeze(),
            'neg_doc_input_ids': torch.stack([neg['input_ids'].squeeze() 
                                             for neg in neg_encodings]) if neg_encodings else torch.tensor([]),
            'neg_doc_attention_mask': torch.stack([neg['attention_mask'].squeeze() 
                                                  for neg in neg_encodings]) if neg_encodings else torch.tensor([]),
            'query_id': query_id,
            'dataset': positive_doc['dataset']
        }

class DatasetAnalyzer:
    """Analyzer for dataset statistics and quality - Paper Aligned"""
    
    def __init__(self, dataset: Dict):
        self.dataset = dataset
        
    def analyze_dataset(self) -> Dict:
        """Analyze dataset statistics as per paper's evaluation"""
        logger.info("Analyzing dataset...")
        
        analysis = {
            'total_queries': 0,
            'total_documents': 0,
            'total_pairs': 0,
            'positive_pairs': 0,
            'negative_pairs': 0,
            'datasets': {},
            'query_stats': {},
            'document_stats': {},
            'vocabulary_stats': {}
        }
        
        # Analyze train/val/test splits
        for split in ['train', 'validation', 'test']:
            if split not in self.dataset:
                continue
                
            data = self.dataset[split]
            
            # Unique queries and documents
            unique_queries = set()
            unique_docs = set()
            dataset_distribution = defaultdict(int)
            
            query_lengths = []
            doc_lengths = []
            
            positive_count = 0
            negative_count = 0
            
            for item in data:
                unique_queries.add(item['query_id'])
                unique_docs.add(item['doc_id'])
                dataset_distribution[item['dataset']] += 1
                
                query_lengths.append(len(item['query'].split()))
                doc_lengths.append(len(item['document'].split()))
                
                if item['label'] == 1:
                    positive_count += 1
                else:
                    negative_count += 1
            
            analysis[f'{split}_queries'] = len(unique_queries)
            analysis[f'{split}_documents'] = len(unique_docs)
            analysis[f'{split}_pairs'] = len(data)
            analysis[f'{split}_positive'] = positive_count
            analysis[f'{split}_negative'] = negative_count
            analysis[f'{split}_dataset_dist'] = dict(dataset_distribution)
            
            # Query and document statistics
            if query_lengths:
                analysis['query_stats'][split] = {
                    'avg_length': np.mean(query_lengths),
                    'std_length': np.std(query_lengths),
                    'min_length': np.min(query_lengths),
                    'max_length': np.max(query_lengths),
                    'median_length': np.median(query_lengths)
                }
            
            if doc_lengths:
                analysis['document_stats'][split] = {
                    'avg_length': np.mean(doc_lengths),
                    'std_length': np.std(doc_lengths),
                    'min_length': np.min(doc_lengths),
                    'max_length': np.max(doc_lengths),
                    'median_length': np.median(doc_lengths)
                }
            
            analysis['positive_pairs'] += positive_count
            analysis['negative_pairs'] += negative_count
        
        analysis['total_pairs'] = analysis['positive_pairs'] + analysis['negative_pairs']
        
        # Vocabulary statistics
        if 'vocabulary' in self.dataset:
            vocab = self.dataset['vocabulary']
            analysis['vocabulary_stats'] = {
                'size': len(vocab),
                'sample_terms': vocab[:20]  # First 20 terms
            }
        
        # Term-document matrix statistics
        if 'term_doc_matrix' in self.dataset:
            matrix = self.dataset['term_doc_matrix']
            analysis['term_doc_matrix_stats'] = {
                'shape': matrix.shape,
                'density': matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
                'avg_terms_per_doc': matrix.nnz / matrix.shape[1],
                'total_non_zero': matrix.nnz
            }
        
        # Dataset-specific statistics
        if 'dataset_stats' in self.dataset:
            analysis['dataset_breakdown'] = self.dataset['dataset_stats']
        
        return analysis
    
    def print_analysis(self, analysis: Dict):
        """Print dataset analysis in paper format"""
        print("\n" + "="*80)
        print("CLARE DATASET ANALYSIS (Paper-Aligned)")
        print("="*80)
        
        # Overall statistics
        print("\nOVERALL STATISTICS:")
        print(f"Total query-document pairs: {analysis['total_pairs']:,}")
        print(f"Positive pairs: {analysis['positive_pairs']:,} ({analysis['positive_pairs']/analysis['total_pairs']*100:.1f}%)")
        print(f"Negative pairs: {analysis['negative_pairs']:,} ({analysis['negative_pairs']/analysis['total_pairs']*100:.1f}%)")
        
        # Dataset breakdown
        if 'dataset_breakdown' in analysis:
            print("\nDATASET BREAKDOWN:")
            for dataset, stats in analysis['dataset_breakdown'].items():
                print(f"\n{dataset}:")
                print(f"  Total pairs: {stats['total']:,}")
                print(f"  Positive: {stats['positive']:,} ({stats['positive']/stats['total']*100:.1f}%)")
                print(f"  Negative: {stats['negative']:,} ({stats['negative']/stats['total']*100:.1f}%)")
        
        # Split statistics
        print("\nSPLIT STATISTICS:")
        for split in ['train', 'validation', 'test']:
            if f'{split}_pairs' in analysis:
                print(f"\n{split.upper()}:")
                print(f"  Queries: {analysis.get(f'{split}_queries', 0):,}")
                print(f"  Documents: {analysis.get(f'{split}_documents', 0):,}")
                print(f"  Query-Doc Pairs: {analysis.get(f'{split}_pairs', 0):,}")
                print(f"  Positive: {analysis.get(f'{split}_positive', 0):,}")
                print(f"  Negative: {analysis.get(f'{split}_negative', 0):,}")
                
                # Dataset distribution within split
                if f'{split}_dataset_dist' in analysis:
                    print(f"  Dataset Distribution:")
                    for dataset, count in analysis[f'{split}_dataset_dist'].items():
                        percentage = count / analysis[f'{split}_pairs'] * 100
                        print(f"    {dataset}: {count:,} ({percentage:.1f}%)")
        
        # Query statistics (matching paper format)
        print("\nQUERY STATISTICS:")
        print("Split      Avg Length  Std Dev   Min   Max   Median")
        print("-" * 55)
        for split, stats in analysis.get('query_stats', {}).items():
            print(f"{split:<11} {stats['avg_length']:>8.1f} {stats['std_length']:>8.1f} "
                  f"{stats['min_length']:>5} {stats['max_length']:>5} {stats['median_length']:>7.1f}")
        
        # Document statistics  
        print("\nDOCUMENT STATISTICS:")
        print("Split      Avg Length  Std Dev   Min    Max    Median")
        print("-" * 58)
        for split, stats in analysis.get('document_stats', {}).items():
            print(f"{split:<11} {stats['avg_length']:>8.1f} {stats['std_length']:>8.1f} "
                  f"{stats['min_length']:>5} {stats['max_length']:>6} {stats['median_length']:>8.1f}")
        
        # Vocabulary statistics
        if 'vocabulary_stats' in analysis:
            print(f"\nVOCABULARY:")
            print(f"  Size: {analysis['vocabulary_stats']['size']:,}")
            print(f"  Sample terms: {', '.join(analysis['vocabulary_stats']['sample_terms'][:10])}")
        
        # Term-document matrix (for Semantic Cluster Discovery)
        if 'term_doc_matrix_stats' in analysis:
            stats = analysis['term_doc_matrix_stats']
            print(f"\nTERM-DOCUMENT MATRIX (for Semantic Cluster Discovery):")
            print(f"  Shape: {stats['shape'][0]:,} terms × {stats['shape'][1]:,} documents")
            print(f"  Density: {stats['density']:.4f}")
            print(f"  Non-zero entries: {stats['total_non_zero']:,}")
            print(f"  Avg terms per document: {stats['avg_terms_per_doc']:.1f}")
        
        print("\n" + "="*80)

# Example usage and testing
if __name__ == "__main__":
    # Configure dataset
    config = DatasetConfig(
        max_query_length=128,
        max_doc_length=512,
        negative_sampling_ratio=4,  # As per paper
        min_query_length=3,
        min_doc_length=10,
        vocab_size=30000,
        n_clusters=100,  # Paper specification
        # Set sample sizes for testing (remove for full datasets)
        msmarco_sample_size=10000,
        bioasq_sample_size=5000,
        granola_sample_size=5000
    )
    
    # Build dataset
    dataset_builder = CLAREDatasetBuilder(config)
    
    # Build complete dataset from the three sources
    print("Building CLARE dataset from specified sources...")
    print("- intfloat/query2doc_msmarco")
    print("- BeIR/bioasq-generated-queries")
    print("- google/granola-entity-questions")
    print()
    
    complete_dataset = dataset_builder.build_complete_dataset()
    
    # Create train/val/test splits
    dataset_splits = dataset_builder.create_train_val_test_splits(
        complete_dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    # Analyze dataset
    analyzer = DatasetAnalyzer(dataset_splits)
    analysis = analyzer.analyze_dataset()
    analyzer.print_analysis(analysis)
    
    # Save dataset
    output_path = 'clare_dataset_complete.pkl'
    dataset_builder.save_dataset(dataset_splits, output_path)
    print(f"\nDataset saved to: {output_path}")
    
    # Test PyTorch dataset
    print("\nTesting PyTorch Dataset...")
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Create training dataset
    train_dataset = CLARETorchDataset(
        dataset_splits['train'][:100],  # Test with first 100 samples
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
    print("\nTesting data loading...")
    for i, batch in enumerate(train_loader):
        if i == 0:  # Just test first batch
            print(f"Batch keys: {batch.keys()}")
            print(f"Query shape: {batch['query_input_ids'].shape}")
            print(f"Positive doc shape: {batch['pos_doc_input_ids'].shape}")
            if batch['neg_doc_input_ids'].numel() > 0:
                print(f"Negative docs shape: {batch['neg_doc_input_ids'].shape}")
            print(f"Datasets in batch: {batch['dataset']}")
        break
    
    print("\nDataset integration completed successfully!")