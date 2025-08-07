# src/embeddings/text_embedder.py
"""
Text embedding module for converting text to vectors.
Supports multiple embedding models with caching capabilities.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
import pickle
import logging
from pathlib import Path
import json
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Cache for storing and retrieving embeddings.
    Avoids redundant computation of embeddings.
    """
    
    def __init__(self, cache_dir: Path = Path("./embeddings_cache")):
        """
        Initialize the embedding cache.
        
        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        
        logger.info(f"EmbeddingCache initialized at {cache_dir}")
    
    def get_cache_key(self, text: str, model_name: str) -> str:
        """
        Generate a cache key for text and model combination.
        
        Args:
            text: Input text
            model_name: Name of the embedding model
            
        Returns:
            Cache key string
        """
        combined = f"{model_name}:{text}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """
        Retrieve embedding from cache.
        
        Args:
            text: Input text
            model_name: Name of the embedding model
            
        Returns:
            Cached embedding or None
        """
        cache_key = self.get_cache_key(text, model_name)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                # Store in memory cache for faster access
                self.memory_cache[cache_key] = embedding
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
        
        return None
    
    def set(self, text: str, model_name: str, embedding: np.ndarray):
        """
        Store embedding in cache.
        
        Args:
            text: Input text
            model_name: Name of the embedding model
            embedding: Embedding vector
        """
        cache_key = self.get_cache_key(text, model_name)
        
        # Store in memory cache
        self.memory_cache[cache_key] = embedding
        
        # Store on disk
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    def clear(self):
        """Clear all cached embeddings."""
        self.memory_cache.clear()
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        
        logger.info("Embedding cache cleared")


class TextEmbedder:
    """
    Main text embedding class supporting multiple models.
    Optimized for Persian and multilingual text.
    """
    
    # Recommended models for Persian text
    MODELS = {
        'multilingual-base': 'paraphrase-multilingual-mpnet-base-v2',
        'multilingual-mini': 'paraphrase-multilingual-MiniLM-L12-v2',
        'persian-base': 'HooshvareLab/bert-fa-base-uncased',  # If available
        'e5-base': 'intfloat/multilingual-e5-base',
        'e5-large': 'intfloat/multilingual-e5-large'
    }
    
    def __init__(
        self,
        model_name: str = 'multilingual-base',
        device: str = 'cpu',
        use_cache: bool = True,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize the text embedder.
        
        Args:
            model_name: Name or path of the embedding model
            device: Device to run model on ('cpu' or 'cuda')
            use_cache: Whether to use caching
            cache_dir: Directory for cache storage
        """
        # Resolve model name
        self.model_name = self.MODELS.get(model_name, model_name)
        self.device = device
        self.use_cache = use_cache
        
        # Initialize model
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=device)
        
        # Initialize cache
        self.cache = EmbeddingCache(cache_dir) if use_cache else None
        
        # Get model properties
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.max_seq_length = self.model.max_seq_length
        
        logger.info(f"TextEmbedder initialized with {self.model_name} (dim={self.embedding_dim})")
    
    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings
            
        Returns:
            Embedding vector(s) as numpy array
        """
        # Handle single text input
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        embeddings = []
        
        for text in texts:
            # Check cache
            if self.use_cache and self.cache:
                cached = self.cache.get(text, self.model_name)
                if cached is not None:
                    embeddings.append(cached)
                    continue
            
            # Preprocess text for better embedding
            processed_text = self._preprocess_text(text)
            
            # Generate embedding
            embedding = self.model.encode(
                processed_text,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            
            # Cache the embedding
            if self.use_cache and self.cache:
                self.cache.set(text, self.model_name, embedding)
            
            embeddings.append(embedding)
        
        # Convert to numpy array
        result = np.vstack(embeddings) if len(embeddings) > 1 else embeddings[0]
        
        # Return single embedding if input was single text
        return result[0] if single_input else result
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Efficiently embed a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings
            
        Returns:
            Array of embeddings
        """
        # Separate cached and uncached texts
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []
        
        if self.use_cache and self.cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text, self.model_name)
                if cached is not None:
                    cached_embeddings[i] = cached
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Embed uncached texts
        if uncached_texts:
            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in uncached_texts]
            
            # Generate embeddings
            new_embeddings = self.model.encode(
                processed_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            
            # Cache new embeddings
            if self.use_cache and self.cache:
                for text, embedding in zip(uncached_texts, new_embeddings):
                    self.cache.set(text, self.model_name, embedding)
            
            # Combine with cached embeddings
            for idx, embedding in zip(uncached_indices, new_embeddings):
                cached_embeddings[idx] = embedding
        
        # Reconstruct in original order
        embeddings = [cached_embeddings[i] for i in range(len(texts))]
        
        return np.vstack(embeddings)
    
    def similarity(
        self,
        text1: Union[str, np.ndarray],
        text2: Union[str, np.ndarray],
        metric: str = 'cosine'
    ) -> float:
        """
        Calculate similarity between two texts or embeddings.
        
        Args:
            text1: First text or embedding
            text2: Second text or embedding
            metric: Similarity metric ('cosine', 'euclidean', 'dot')
            
        Returns:
            Similarity score
        """
        # Convert texts to embeddings if necessary
        if isinstance(text1, str):
            embedding1 = self.embed(text1)
        else:
            embedding1 = text1
        
        if isinstance(text2, str):
            embedding2 = self.embed(text2)
        else:
            embedding2 = text2
        
        # Calculate similarity
        if metric == 'cosine':
            return self._cosine_similarity(embedding1, embedding2)
        elif metric == 'euclidean':
            return -np.linalg.norm(embedding1 - embedding2)
        elif metric == 'dot':
            return np.dot(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better embedding quality.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long
        if len(text) > self.max_seq_length * 4:  # Rough character estimate
            text = text[:self.max_seq_length * 4]
        
        # Add instruction prefix for e5 models
        if 'e5' in self.model_name.lower():
            text = f"query: {text}"
        
        return text
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'max_sequence_length': self.max_seq_length,
            'device': self.device,
            'cache_enabled': self.use_cache
        }
    
    def save_model(self, path: Path):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        self.model.save(str(path))
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: Path, **kwargs):
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model
            **kwargs: Additional arguments for initialization
            
        Returns:
            TextEmbedder instance
        """
        embedder = cls(model_name=str(path), **kwargs)
        logger.info(f"Model loaded from {path}")
        return embedder
    
    def clear_cache(self):
        """Clear the embedding cache."""
        if self.cache:
            self.cache.clear()
    
    def __repr__(self):
        """String representation."""
        return f"TextEmbedder(model='{self.model_name}', dim={self.embedding_dim})"