# src/database/models.py
"""
SQLAlchemy models for the legal document metadata database.
"""

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, 
    Boolean, JSON, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class LegalDocumentModel(Base):
    """Model for legal documents."""
    __tablename__ = 'legal_documents'
    
    id = Column(String(100), primary_key=True)
    title = Column(Text, nullable=False)
    law_name = Column(Text, nullable=False)
    document_type = Column(String(50))
    
    # Dates
    approval_date = Column(String(50))
    approval_year = Column(Integer)
    effective_date = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Authority
    approval_authority = Column(String(200))
    
    # Structure stats
    num_chapters = Column(Integer, default=0)
    num_articles = Column(Integer, default=0)
    num_notes = Column(Integer, default=0)
    num_clauses = Column(Integer, default=0)
    
    # Content
    legal_domain = Column(String(100))
    keywords = Column(JSON)
    subject_areas = Column(JSON)
    
    # References
    referenced_laws = Column(JSON)
    amending_laws = Column(JSON)
    
    # File info
    source_file = Column(String(255))
    file_hash = Column(String(64))
    
    # Processing info
    processed_at = Column(DateTime)
    processing_status = Column(String(20), default='pending')
    
    # Relationships
    chunks = relationship("DocumentChunkModel", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_law_name', 'law_name'),
        Index('idx_document_type', 'document_type'),
        Index('idx_approval_year', 'approval_year'),
        Index('idx_legal_domain', 'legal_domain'),
    )


class DocumentChunkModel(Base):
    """Model for document chunks."""
    __tablename__ = 'document_chunks'
    
    id = Column(String(100), primary_key=True)
    document_id = Column(String(100), ForeignKey('legal_documents.id'), nullable=False)
    
    # Chunk info
    chunk_type = Column(String(20))  # 'article', 'semantic', 'fixed'
    sequence_number = Column(Integer)
    
    # Content
    content = Column(Text, nullable=False)
    tokens_count = Column(Integer)
    
    # Metadata
    chapter_number = Column(String(20))
    chapter_title = Column(String(200))
    article_number = Column(String(20))
    article_range = Column(String(50))
    
    # Vector info
    vector_id = Column(String(100), unique=True)
    embedding_model = Column(String(100))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("LegalDocumentModel", back_populates="chunks")
    
    # Indexes
    __table_args__ = (
        Index('idx_document_id', 'document_id'),
        Index('idx_chunk_type', 'chunk_type'),
        Index('idx_article_number', 'article_number'),
        Index('idx_vector_id', 'vector_id'),
    )


class SearchLogModel(Base):
    """Model for logging search queries."""
    __tablename__ = 'search_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Query info
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50))  # 'semantic', 'keyword', 'hybrid'
    
    # Results info
    num_results = Column(Integer)
    top_result_id = Column(String(100))
    response_time_ms = Column(Float)
    
    # User info (anonymous)
    session_id = Column(String(100))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_created_at', 'created_at'),
        Index('idx_session_id', 'session_id'),
    )