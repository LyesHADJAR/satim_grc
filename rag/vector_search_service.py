"""
Enhanced Vector Search Service with FAISS Integration for GRC Analysis
Current Date: 2025-06-13 20:26:29 UTC
Current User: LyesHADJAR
"""
import json
import logging
import os
import numpy as np
import faiss
import pickle
from typing import Dict, Any, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import re

logger = logging.getLogger(__name__)

class EnhancedVectorSearchService:
    """Enhanced vector search service with FAISS integration for GRC compliance analysis."""
    
    def __init__(self, data_paths: Dict[str, str], 
                 tfidf_weight: float = 0.3, 
                 bm25_weight: float = 0.3, 
                 embedding_weight: float = 0.4):
        """
        Initialize the enhanced search service with hybrid vectorization.
        
        Args:
            data_paths: Dictionary mapping document types to file paths
            tfidf_weight: Weight for TF-IDF scores in hybrid search
            bm25_weight: Weight for BM25 scores in hybrid search  
            embedding_weight: Weight for embedding similarity in hybrid search
        """
        self.data_paths = data_paths
        self.weights = {
            'tfidf': tfidf_weight,
            'bm25': bm25_weight,
            'embedding': embedding_weight
        }
        
        # Initialize storage
        self.chunks = {}  # {doc_type: [chunks]}
        self.vector_indices = {}  # {doc_type: faiss_index}
        self.tfidf_vectorizers = {}  # {doc_type: vectorizer}
        self.bm25_models = {}  # {doc_type: bm25_model}
        self.metadata = {}  # {doc_type: metadata}
        
        # Load data and build indices
        self._load_all_data()
        
    def _load_all_data(self):
        """Load all document data and build/load vector indices."""
        for doc_type, file_path in self.data_paths.items():
            logger.info(f"Loading data for {doc_type} from {file_path}")
            
            if not os.path.exists(file_path):
                logger.warning(f"Data file {file_path} does not exist for {doc_type}")
                continue
                
            try:
                # Load document chunks
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                
                if not chunks:
                    logger.warning(f"No chunks found in {file_path}")
                    continue
                    
                self.chunks[doc_type] = chunks
                logger.info(f"Loaded {len(chunks)} chunks for {doc_type}")
                
                # Check for existing vector database
                vector_db_path = file_path.replace('.json', '.faiss')
                metadata_path = f"{vector_db_path}.meta"
                
                if os.path.exists(vector_db_path) and os.path.exists(metadata_path):
                    self._load_vector_db(doc_type, vector_db_path, metadata_path)
                else:
                    self._build_vector_db(doc_type)
                    
            except Exception as e:
                logger.error(f"Error loading data for {doc_type}: {e}")
                continue
    
    def _load_vector_db(self, doc_type: str, vector_path: str, metadata_path: str):
        """Load existing vector database and metadata."""
        try:
            # Load FAISS index
            self.vector_indices[doc_type] = faiss.read_index(vector_path)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                
            self.tfidf_vectorizers[doc_type] = metadata.get('tfidf_vectorizer')
            self.bm25_models[doc_type] = metadata.get('bm25_model')
            self.metadata[doc_type] = metadata
            
            # Verify integrity
            if self.vector_indices[doc_type].ntotal != len(self.chunks[doc_type]):
                logger.warning(f"Vector index size mismatch for {doc_type}. Rebuilding.")
                self._build_vector_db(doc_type)
            else:
                logger.info(f"Successfully loaded vector database for {doc_type}")
                
        except Exception as e:
            logger.error(f"Error loading vector database for {doc_type}: {e}")
            self._build_vector_db(doc_type)
    
    def _build_vector_db(self, doc_type: str):
        """Build vector database from scratch for a document type."""
        logger.info(f"Building vector database for {doc_type}")
        
        chunks = self.chunks[doc_type]
        if not chunks:
            logger.warning(f"No chunks available for {doc_type}")
            return
            
        try:
            # Preprocess chunks
            texts = []
            for chunk in chunks:
                if isinstance(chunk, dict):
                    text = chunk.get('text', '') or chunk.get('content', '')
                else:
                    text = str(chunk)
                texts.append(self._preprocess_text(text))
            
            # Build TF-IDF vectorizer
            tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                max_df=0.8,
                min_df=2
            )
            tfidf_vectors = tfidf_vectorizer.fit_transform(texts)
            self.tfidf_vectorizers[doc_type] = tfidf_vectorizer
            
            # Build BM25 model
            tokenized_texts = [text.split() for text in texts]
            bm25_model = BM25Okapi(tokenized_texts)
            self.bm25_models[doc_type] = bm25_model
            
            # Create simple TF-IDF based embeddings for FAISS
            # Convert sparse matrix to dense for FAISS
            dense_vectors = tfidf_vectors.toarray().astype('float32')
            
            # Create FAISS index
            embedding_dim = dense_vectors.shape[1]
            index = faiss.IndexFlatL2(embedding_dim)
            index.add(dense_vectors)
            self.vector_indices[doc_type] = index
            
            # Save vector database
            self._save_vector_db(doc_type)
            
            logger.info(f"Built vector database for {doc_type} with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error building vector database for {doc_type}: {e}")
    
    def _save_vector_db(self, doc_type: str):
        """Save vector database and metadata to disk."""
        try:
            base_path = self.data_paths[doc_type].replace('.json', '')
            vector_path = f"{base_path}.faiss"
            metadata_path = f"{base_path}.faiss.meta"
            
            # Save FAISS index
            faiss.write_index(self.vector_indices[doc_type], vector_path)
            
            # Save metadata
            metadata = {
                'tfidf_vectorizer': self.tfidf_vectorizers[doc_type],
                'bm25_model': self.bm25_models[doc_type],
                'doc_type': doc_type,
                'chunk_count': len(self.chunks[doc_type]),
                'last_updated': str(np.datetime64('now'))
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.info(f"Saved vector database for {doc_type}")
            
        except Exception as e:
            logger.error(f"Error saving vector database for {doc_type}: {e}")
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for indexing and search."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\-\(\)]', ' ', text)
        
        return text
    
    def semantic_search(self, query: str, doc_types: Optional[List[str]] = None, 
                       top_k: int = 5, method: str = 'hybrid') -> List[Dict[str, Any]]:
        """
        Perform semantic search across document types.
        
        Args:
            query: Search query
            doc_types: Document types to search (if None, search all)
            top_k: Number of results to return
            method: Search method ('hybrid', 'tfidf', 'bm25', 'vector')
            
        Returns:
            List of search results with scores and metadata
        """
        if doc_types is None:
            doc_types = list(self.chunks.keys())
            
        all_results = []
        
        for doc_type in doc_types:
            if doc_type not in self.chunks:
                continue
                
            try:
                results = self._search_document_type(query, doc_type, top_k * 2, method)
                
                # Add document type to results
                for result in results:
                    result['document_type'] = doc_type
                    
                all_results.extend(results)
                
            except Exception as e:
                logger.error(f"Error searching {doc_type}: {e}")
                continue
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        return all_results[:top_k]
    
    def _search_document_type(self, query: str, doc_type: str, top_k: int, method: str) -> List[Dict[str, Any]]:
        """Search within a specific document type."""
        preprocessed_query = self._preprocess_text(query)
        chunks = self.chunks[doc_type]
        
        if method == 'hybrid':
            # Combine multiple search methods
            tfidf_scores = self._tfidf_search(preprocessed_query, doc_type, len(chunks))
            bm25_scores = self._bm25_search(preprocessed_query, doc_type, len(chunks))
            vector_scores = self._vector_search(preprocessed_query, doc_type, len(chunks))
            
            # Combine scores
            combined_scores = []
            for i in range(len(chunks)):
                combined_score = (
                    self.weights['tfidf'] * tfidf_scores.get(i, 0) +
                    self.weights['bm25'] * bm25_scores.get(i, 0) +
                    self.weights['embedding'] * vector_scores.get(i, 0)
                )
                combined_scores.append((i, combined_score))
                
        elif method == 'tfidf':
            scores = self._tfidf_search(preprocessed_query, doc_type, len(chunks))
            combined_scores = [(i, score) for i, score in scores.items()]
            
        elif method == 'bm25':
            scores = self._bm25_search(preprocessed_query, doc_type, len(chunks))
            combined_scores = [(i, score) for i, score in scores.items()]
            
        elif method == 'vector':
            scores = self._vector_search(preprocessed_query, doc_type, len(chunks))
            combined_scores = [(i, score) for i, score in scores.items()]
            
        else:
            logger.warning(f"Unknown search method: {method}. Using hybrid.")
            return self._search_document_type(query, doc_type, top_k, 'hybrid')
        
        # Sort and format results
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, (chunk_idx, score) in enumerate(combined_scores[:top_k]):
            if score > 0:  # Only include relevant results
                chunk = chunks[chunk_idx]
                
                # Extract content and metadata
                if isinstance(chunk, dict):
                    content = chunk.get('text', '') or chunk.get('content', '')
                    section = chunk.get('section_title', 'Unknown Section')
                    document_id = chunk.get('document', 'Unknown Document')
                else:
                    content = str(chunk)
                    section = 'Unknown Section'
                    document_id = 'Unknown Document'
                
                results.append({
                    'content': content,
                    'section': section,
                    'document_id': document_id,
                    'similarity_score': float(score),
                    'rank': i + 1,
                    'metadata': {
                        'chunk_index': chunk_idx,
                        'search_method': method,
                        'doc_type': doc_type
                    }
                })
        
        return results
    
    def _tfidf_search(self, query: str, doc_type: str, max_results: int) -> Dict[int, float]:
        """Perform TF-IDF search."""
        if doc_type not in self.tfidf_vectorizers:
            return {}
            
        vectorizer = self.tfidf_vectorizers[doc_type]
        query_vector = vectorizer.transform([query])
        
        # Calculate similarities (already have document vectors from building)
        # For now, rebuild document vectors (in production, store these)
        chunks = self.chunks[doc_type]
        texts = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                text = chunk.get('text', '') or chunk.get('content', '')
            else:
                text = str(chunk)
            texts.append(self._preprocess_text(text))
        
        doc_vectors = vectorizer.transform(texts)
        similarities = (query_vector * doc_vectors.T).toarray()[0]
        
        # Normalize scores
        max_sim = max(similarities) if similarities.max() > 0 else 1
        normalized_scores = similarities / max_sim
        
        return {i: float(score) for i, score in enumerate(normalized_scores)}
    
    def _bm25_search(self, query: str, doc_type: str, max_results: int) -> Dict[int, float]:
        """Perform BM25 search."""
        if doc_type not in self.bm25_models:
            return {}
            
        bm25_model = self.bm25_models[doc_type]
        query_tokens = query.split()
        scores = bm25_model.get_scores(query_tokens)
        
        # Normalize scores
        max_score = max(scores) if scores.max() > 0 else 1
        normalized_scores = scores / max_score
        
        return {i: float(score) for i, score in enumerate(normalized_scores)}
    
    def _vector_search(self, query: str, doc_type: str, max_results: int) -> Dict[int, float]:
        """Perform vector similarity search using FAISS."""
        if doc_type not in self.vector_indices:
            return {}
            
        # Create query vector using same TF-IDF approach
        if doc_type not in self.tfidf_vectorizers:
            return {}
            
        vectorizer = self.tfidf_vectorizers[doc_type]
        query_vector = vectorizer.transform([query]).toarray().astype('float32')
        
        # Search in FAISS index
        index = self.vector_indices[doc_type]
        k = min(max_results, index.ntotal)
        
        distances, indices = index.search(query_vector, k)
        
        # Convert distances to similarity scores
        similarities = {}
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # Valid index
                # Convert distance to similarity (inverse relationship)
                similarity = 1.0 / (1.0 + dist)
                similarities[idx] = similarity
        
        return similarities
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded documents."""
        stats = {
            'total_document_types': len(self.chunks),
            'document_types': {}
        }
        
        total_chunks = 0
        for doc_type, chunks in self.chunks.items():
            chunk_count = len(chunks)
            total_chunks += chunk_count
            
            stats['document_types'][doc_type] = {
                'chunk_count': chunk_count,
                'has_vector_index': doc_type in self.vector_indices,
                'has_tfidf': doc_type in self.tfidf_vectorizers,
                'has_bm25': doc_type in self.bm25_models
            }
        
        stats['total_chunks'] = total_chunks
        return stats