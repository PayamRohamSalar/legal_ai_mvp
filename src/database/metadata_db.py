# src/database/metadata_db.py
"""
PostgreSQL database manager for metadata storage.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict, Any, Optional
import logging
from contextlib import contextmanager

from config.database_config import PostgreSQLConfig
from .models import Base, LegalDocumentModel, DocumentChunkModel, SearchLogModel

# Configure logging
logger = logging.getLogger(__name__)


class MetadataDBManager:
    """
    Manages PostgreSQL database operations for metadata storage.
    """
    
    def __init__(self, config: PostgreSQLConfig):
        """
        Initialize the metadata database manager.
        
        Args:
            config: PostgreSQL configuration object
        """
        self.config = config
        self.engine = None
        self.SessionLocal = None
        
        # Initialize database connection
        self._initialize_database()
        
        logger.info("MetadataDBManager initialized successfully")
    
    def _initialize_database(self):
        """Initialize database connection and create tables."""
        try:
            # Create engine
            self.engine = create_engine(
                self.config.sync_url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=False  # Set to True for SQL debugging
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Create tables if they don't exist
            Base.metadata.create_all(bind=self.engine)
            
            logger.info("Database initialized and tables created")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """
        Get a database session context manager.
        
        Yields:
            Database session
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def save_document(self, document_data: Dict[str, Any]) -> str:
        """
        Save a legal document to the database.
        
        Args:
            document_data: Document data dictionary
            
        Returns:
            Document ID
        """
        with self.get_session() as session:
            try:
                # Create document model
                doc_model = LegalDocumentModel(**document_data)
                
                # Add to session and commit
                session.add(doc_model)
                session.flush()
                
                document_id = doc_model.id
                logger.info(f"Saved document: {document_id}")
                
                return document_id
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to save document: {e}")
                raise
    
    def save_chunks(self, chunks_data: List[Dict[str, Any]]) -> bool:
        """
        Save document chunks to the database.
        
        Args:
            chunks_data: List of chunk data dictionaries
            
        Returns:
            True if successful
        """
        with self.get_session() as session:
            try:
                # Create chunk models
                chunk_models = [
                    DocumentChunkModel(**chunk_data)
                    for chunk_data in chunks_data
                ]
                
                # Bulk insert
                session.bulk_save_objects(chunk_models)
                
                logger.info(f"Saved {len(chunks_data)} chunks")
                return True
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to save chunks: {e}")
                return False
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document data dictionary or None
        """
        with self.get_session() as session:
            try:
                doc = session.query(LegalDocumentModel).filter_by(id=document_id).first()
                
                if doc:
                    return self._model_to_dict(doc)
                
                return None
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to get document {document_id}: {e}")
                return None
    
    def get_documents_by_filter(self, **filters) -> List[Dict[str, Any]]:
        """
        Get documents matching filter criteria.
        
        Args:
            **filters: Filter criteria (e.g., document_type='قانون')
            
        Returns:
            List of document dictionaries
        """
        with self.get_session() as session:
            try:
                query = session.query(LegalDocumentModel)
                
                # Apply filters
                for key, value in filters.items():
                    if hasattr(LegalDocumentModel, key):
                        query = query.filter(getattr(LegalDocumentModel, key) == value)
                
                documents = query.all()
                
                return [self._model_to_dict(doc) for doc in documents]
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to get documents by filter: {e}")
                return []
    
    def get_chunks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of chunk dictionaries
        """
        with self.get_session() as session:
            try:
                chunks = session.query(DocumentChunkModel).filter_by(
                    document_id=document_id
                ).order_by(DocumentChunkModel.sequence_number).all()
                
                return [self._model_to_dict(chunk) for chunk in chunks]
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to get chunks for document {document_id}: {e}")
                return []
    
    def update_document_status(
        self, 
        document_id: str, 
        status: str,
        **additional_fields
    ) -> bool:
        """
        Update document processing status.
        
        Args:
            document_id: Document ID
            status: New status
            **additional_fields: Additional fields to update
            
        Returns:
            True if successful
        """
        with self.get_session() as session:
            try:
                doc = session.query(LegalDocumentModel).filter_by(id=document_id).first()
                
                if doc:
                    doc.processing_status = status
                    doc.updated_at = datetime.utcnow()
                    
                    # Update additional fields
                    for key, value in additional_fields.items():
                        if hasattr(doc, key):
                            setattr(doc, key, value)
                    
                    logger.info(f"Updated document {document_id} status to {status}")
                    return True
                
                return False
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to update document status: {e}")
                return False
    
    def log_search(
        self,
        query_text: str,
        query_type: str,
        num_results: int,
        response_time_ms: float,
        session_id: Optional[str] = None,
        top_result_id: Optional[str] = None
    ) -> bool:
        """
        Log a search query.
        
        Args:
            query_text: The search query
            query_type: Type of search performed
            num_results: Number of results returned
            response_time_ms: Response time in milliseconds
            session_id: Optional session identifier
            top_result_id: ID of the top result
            
        Returns:
            True if successful
        """
        with self.get_session() as session:
            try:
                log_entry = SearchLogModel(
                    query_text=query_text,
                    query_type=query_type,
                    num_results=num_results,
                    response_time_ms=response_time_ms,
                    session_id=session_id,
                    top_result_id=top_result_id
                )
                
                session.add(log_entry)
                
                logger.debug(f"Logged search query: {query_text[:50]}...")
                return True
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to log search: {e}")
                return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self.get_session() as session:
            try:
                stats = {
                    'total_documents': session.query(LegalDocumentModel).count(),
                    'total_chunks': session.query(DocumentChunkModel).count(),
                    'total_searches': session.query(SearchLogModel).count(),
                    'documents_by_type': {},
                    'documents_by_domain': {}
                }
                
                # Documents by type
                type_results = session.query(
                    LegalDocumentModel.document_type,
                    text('COUNT(*) as count')
                ).group_by(LegalDocumentModel.document_type).all()
                
                stats['documents_by_type'] = {
                    doc_type: count for doc_type, count in type_results
                }
                
                # Documents by domain
                domain_results = session.query(
                    LegalDocumentModel.legal_domain,
                    text('COUNT(*) as count')
                ).group_by(LegalDocumentModel.legal_domain).all()
                
                stats['documents_by_domain'] = {
                    domain: count for domain, count in domain_results
                }
                
                return stats
                
            except SQLAlchemyError as e:
                logger.error(f"Failed to get statistics: {e}")
                return {}
    
    def _model_to_dict(self, model) -> Dict[str, Any]:
        """Convert SQLAlchemy model to dictionary."""
        return {
            column.name: getattr(model, column.name)
            for column in model.__table__.columns
        }
    
    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()