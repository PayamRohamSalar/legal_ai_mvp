# src/database/vector_db.py
"""
Vector database manager for ChromaDB.
Handles vector storage, retrieval, and management for legal documents.
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import json

from config.database_config import ChromaDBConfig

# Configure logging
logger = logging.getLogger(__name__)


class VectorDBManager:
    """
    Manages vector database operations using ChromaDB.
    Handles document embeddings, storage, and similarity search.
    """
    
    def __init__(self, config: ChromaDBConfig):
        """
        Initialize the vector database manager.
        
        Args:
            config: ChromaDB configuration object
        """
        self.config = config
        self.client = None
        self.collection = None
        self.embedding_function = None
        
        # Initialize connection
        self._initialize_client()
        self._initialize_collection()
        
        logger.info("VectorDBManager initialized successfully")
    
    def _initialize_client(self):
        """Initialize ChromaDB client."""
        try:
            # Check if we're in server mode or persistent mode
            if self.config.host and self.config.port:
                # Server mode (for Docker deployment)
                self.client = chromadb.HttpClient(
                    host=self.config.host,
                    port=self.config.port,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                logger.info(f"Connected to ChromaDB server at {self.config.host}:{self.config.port}")
            else:
                # Persistent local mode
                self.client = chromadb.PersistentClient(
                    path=str(self.config.persist_directory),
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                logger.info(f"Using persistent ChromaDB at {self.config.persist_directory}")
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    
    def _initialize_collection(self):
        """Initialize or get the collection."""
        try:
            # Initialize embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config.embedding_model
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            logger.info(f"Collection '{self.config.collection_name}' ready with {self.collection.count()} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> bool:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of unique document IDs
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate inputs
            if not (len(documents) == len(metadatas) == len(ids)):
                raise ValueError("Documents, metadatas, and ids must have the same length")
            
            # Prepare metadatas (ChromaDB requires string values)
            processed_metadatas = []
            for metadata in metadatas:
                processed_meta = self._process_metadata_for_storage(metadata)
                processed_metadatas.append(processed_meta)
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=processed_metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(documents)} documents to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results with documents and metadata
        """
        try:
            # Prepare where clause for filtering
            where = None
            if filter_metadata:
                where = self._build_where_clause(filter_metadata)
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
            
            # Process results
            processed_results = []
            if results and results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i] if results['documents'] else None,
                        'metadata': self._process_metadata_from_storage(
                            results['metadatas'][0][i] if results['metadatas'] else {}
                        ),
                        'distance': results['distances'][0][i] if results['distances'] else None
                    }
                    processed_results.append(result)
            
            logger.info(f"Search returned {len(processed_results)} results for query: {query[:50]}...")
            return processed_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve documents by their IDs.
        
        Args:
            ids: List of document IDs
            
        Returns:
            List of documents with metadata
        """
        try:
            results = self.collection.get(ids=ids)
            
            processed_results = []
            if results and results['ids']:
                for i in range(len(results['ids'])):
                    result = {
                        'id': results['ids'][i],
                        'document': results['documents'][i] if results['documents'] else None,
                        'metadata': self._process_metadata_from_storage(
                            results['metadatas'][i] if results['metadatas'] else {}
                        )
                    }
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to get documents by IDs: {e}")
            return []
    
    def update_document(
        self,
        document_id: str,
        document: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update a document in the database.
        
        Args:
            document_id: ID of the document to update
            document: New document text (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if successful
        """
        try:
            update_args = {'ids': [document_id]}
            
            if document:
                update_args['documents'] = [document]
            
            if metadata:
                processed_metadata = self._process_metadata_for_storage(metadata)
                update_args['metadatas'] = [processed_metadata]
            
            self.collection.update(**update_args)
            
            logger.info(f"Successfully updated document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents from the database.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Successfully deleted {len(ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get a sample to understand the data
            sample = self.collection.peek(1)
            
            stats = {
                'collection_name': self.config.collection_name,
                'document_count': count,
                'embedding_model': self.config.embedding_model,
                'sample_metadata_keys': list(sample['metadatas'][0].keys()) if sample['metadatas'] else []
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.config.collection_name)
            self._initialize_collection()
            
            logger.info(f"Collection '{self.config.collection_name}' cleared")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
    
    def _process_metadata_for_storage(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata for storage in ChromaDB.
        ChromaDB requires certain data types for metadata.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Processed metadata suitable for ChromaDB
        """
        processed = {}
        
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                processed[key] = value
            elif isinstance(value, list):
                # Convert lists to JSON strings
                processed[f"{key}_json"] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, dict):
                # Convert nested dicts to JSON strings
                processed[f"{key}_json"] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, datetime):
                processed[key] = value.isoformat()
            else:
                # Convert other types to string
                processed[key] = str(value)
        
        return processed
    
    def _process_metadata_from_storage(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata retrieved from ChromaDB back to original format.
        
        Args:
            metadata: Metadata from ChromaDB
            
        Returns:
            Processed metadata
        """
        processed = {}
        
        for key, value in metadata.items():
            if key.endswith('_json'):
                # Restore JSON fields
                original_key = key[:-5]
                try:
                    processed[original_key] = json.loads(value)
                except:
                    processed[original_key] = value
            else:
                processed[key] = value
        
        return processed
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a where clause for ChromaDB filtering.
        
        Args:
            filters: Filter conditions
            
        Returns:
            Where clause for ChromaDB query
        """
        where = {}
        
        for key, value in filters.items():
            if isinstance(value, list):
                # Use $in operator for list values
                where[key] = {"$in": value}
            elif isinstance(value, dict):
                # Pass through complex conditions
                where[key] = value
            else:
                # Simple equality
                where[key] = value
        
        return where
    
    def close(self):
        """Close the database connection."""
        # ChromaDB doesn't require explicit closing, but we log it
        logger.info("VectorDBManager closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()