# src/data_processing/chunking_strategy.py
"""
Document chunking strategies for optimal vector storage and retrieval.
Implements multiple strategies for chunking legal documents.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
import logging

from .text_parser import LegalDocument, LegalArticle, LegalChapter

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a single chunk of a document."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    chunk_type: str  # 'article', 'chapter', 'semantic', 'fixed'
    parent_id: Optional[str] = None
    sequence_number: int = 0
    tokens_count: Optional[int] = None
    
    def generate_hash(self) -> str:
        """Generate a unique hash for this chunk."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()
        return f"{self.chunk_id}_{content_hash[:8]}"


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, document: LegalDocument) -> List[DocumentChunk]:
        """
        Chunk a document according to the strategy.
        
        Args:
            document: Parsed legal document
            
        Returns:
            List of document chunks
        """
        pass
    
    def _create_base_metadata(self, document: LegalDocument) -> Dict[str, Any]:
        """Create base metadata for all chunks from this document."""
        return {
            'law_name': document.law_name,
            'document_title': document.title,
            'approval_date': document.approval_date,
        }


class ArticleBasedChunker(ChunkingStrategy):
    """
    Chunks documents by individual articles (ماده).
    This is the most natural chunking for legal documents.
    """
    
    def __init__(self, include_notes: bool = True, include_clauses: bool = True):
        """
        Initialize the article-based chunker.
        
        Args:
            include_notes: Whether to include notes (تبصره) in the chunk
            include_clauses: Whether to include clauses (بند) in the chunk
        """
        self.include_notes = include_notes
        self.include_clauses = include_clauses
        logger.info("ArticleBasedChunker initialized")
    
    def chunk(self, document: LegalDocument) -> List[DocumentChunk]:
        """
        Create chunks based on articles.
        
        Args:
            document: Parsed legal document
            
        Returns:
            List of chunks, one per article
        """
        chunks = []
        base_metadata = self._create_base_metadata(document)
        sequence = 0
        
        # Process chapters
        for chapter in document.chapters:
            for article in chapter.articles:
                chunk = self._create_article_chunk(
                    article, 
                    chapter.chapter_number,
                    chapter.title,
                    base_metadata,
                    sequence
                )
                chunks.append(chunk)
                sequence += 1
        
        # Process standalone articles
        for article in document.standalone_articles:
            chunk = self._create_article_chunk(
                article,
                None,
                None,
                base_metadata,
                sequence
            )
            chunks.append(chunk)
            sequence += 1
        
        logger.info(f"Created {len(chunks)} article-based chunks")
        return chunks
    
    def _create_article_chunk(
        self,
        article: LegalArticle,
        chapter_num: Optional[str],
        chapter_title: Optional[str],
        base_metadata: Dict[str, Any],
        sequence: int
    ) -> DocumentChunk:
        """Create a chunk from an article."""
        
        # Build chunk content
        content_parts = [f"ماده {article.article_number}: {article.content}"]
        
        # Add clauses if requested
        if self.include_clauses and article.clauses:
            content_parts.append("\nبندها:")
            for i, clause in enumerate(article.clauses, 1):
                content_parts.append(f"{i}. {clause}")
        
        # Add notes if requested
        if self.include_notes and article.notes:
            content_parts.append("\nتبصره‌ها:")
            for i, note in enumerate(article.notes, 1):
                content_parts.append(f"تبصره {i}: {note}")
        
        content = "\n".join(content_parts)
        
        # Build metadata
        metadata = base_metadata.copy()
        metadata.update({
            'article_number': article.article_number,
            'chapter_number': chapter_num,
            'chapter_title': chapter_title,
            'has_notes': len(article.notes) > 0,
            'has_clauses': len(article.clauses) > 0,
            'num_notes': len(article.notes),
            'num_clauses': len(article.clauses)
        })
        
        # Create chunk ID
        chunk_id = f"article_{article.article_number}"
        if chapter_num:
            chunk_id = f"ch{chapter_num}_{chunk_id}"
        
        return DocumentChunk(
            chunk_id=chunk_id,
            content=content,
            metadata=metadata,
            chunk_type='article',
            sequence_number=sequence,
            tokens_count=self._estimate_tokens(content)
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation for Persian text)."""
        # For Persian text, roughly 3-4 characters per token
        return len(text) // 3


class SemanticChunker(ChunkingStrategy):
    """
    Chunks documents based on semantic boundaries.
    Groups related articles or sections together.
    """
    
    def __init__(self, max_chunk_size: int = 1500, overlap_size: int = 200):
        """
        Initialize semantic chunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            overlap_size: Number of overlapping characters between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        logger.info("SemanticChunker initialized")
    
    def chunk(self, document: LegalDocument) -> List[DocumentChunk]:
        """
        Create semantically meaningful chunks.
        
        Args:
            document: Parsed legal document
            
        Returns:
            List of semantic chunks
        """
        chunks = []
        base_metadata = self._create_base_metadata(document)
        
        # Process by chapters for semantic coherence
        for chapter in document.chapters:
            chapter_chunks = self._chunk_chapter(chapter, base_metadata)
            chunks.extend(chapter_chunks)
        
        # Process standalone articles
        if document.standalone_articles:
            standalone_chunks = self._chunk_articles(
                document.standalone_articles,
                base_metadata,
                None,
                None
            )
            chunks.extend(standalone_chunks)
        
        logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def _chunk_chapter(
        self, 
        chapter: LegalChapter, 
        base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Chunk a single chapter semantically."""
        return self._chunk_articles(
            chapter.articles,
            base_metadata,
            chapter.chapter_number,
            chapter.title
        )
    
    def _chunk_articles(
        self,
        articles: List[LegalArticle],
        base_metadata: Dict[str, Any],
        chapter_num: Optional[str],
        chapter_title: Optional[str]
    ) -> List[DocumentChunk]:
        """Group articles into semantic chunks."""
        chunks = []
        current_chunk_content = []
        current_chunk_articles = []
        current_size = 0
        
        for article in articles:
            article_text = self._format_article(article)
            article_size = len(article_text)
            
            # Check if adding this article would exceed max size
            if current_size + article_size > self.max_chunk_size and current_chunk_content:
                # Create chunk from current content
                chunk = self._create_semantic_chunk(
                    current_chunk_content,
                    current_chunk_articles,
                    base_metadata,
                    chapter_num,
                    chapter_title,
                    len(chunks)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                if self.overlap_size > 0 and current_chunk_content:
                    # Keep last part of previous chunk as overlap
                    overlap_text = "\n".join(current_chunk_content[-1:])
                    current_chunk_content = [overlap_text, article_text]
                    current_chunk_articles = [current_chunk_articles[-1], article.article_number]
                    current_size = len(overlap_text) + article_size
                else:
                    current_chunk_content = [article_text]
                    current_chunk_articles = [article.article_number]
                    current_size = article_size
            else:
                current_chunk_content.append(article_text)
                current_chunk_articles.append(article.article_number)
                current_size += article_size
        
        # Create final chunk
        if current_chunk_content:
            chunk = self._create_semantic_chunk(
                current_chunk_content,
                current_chunk_articles,
                base_metadata,
                chapter_num,
                chapter_title,
                len(chunks)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _format_article(self, article: LegalArticle) -> str:
        """Format an article for inclusion in a chunk."""
        parts = [f"ماده {article.article_number}: {article.content}"]
        
        if article.clauses:
            for clause in article.clauses:
                parts.append(f"  - {clause}")
        
        if article.notes:
            for i, note in enumerate(article.notes, 1):
                parts.append(f"  تبصره {i}: {note}")
        
        return "\n".join(parts)
    
    def _create_semantic_chunk(
        self,
        content_parts: List[str],
        article_numbers: List[str],
        base_metadata: Dict[str, Any],
        chapter_num: Optional[str],
        chapter_title: Optional[str],
        sequence: int
    ) -> DocumentChunk:
        """Create a semantic chunk from grouped content."""
        
        content = "\n\n".join(content_parts)
        
        # Build metadata
        metadata = base_metadata.copy()
        metadata.update({
            'chapter_number': chapter_num,
            'chapter_title': chapter_title,
            'articles_included': article_numbers,
            'num_articles': len(set(article_numbers)),
            'article_range': f"{article_numbers[0]}-{article_numbers[-1]}" if len(article_numbers) > 1 else article_numbers[0]
        })
        
        # Create chunk ID
        chunk_id = f"semantic_{sequence}"
        if chapter_num:
            chunk_id = f"ch{chapter_num}_{chunk_id}"
        
        return DocumentChunk(
            chunk_id=chunk_id,
            content=content,
            metadata=metadata,
            chunk_type='semantic',
            sequence_number=sequence,
            tokens_count=self._estimate_tokens(content)
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 3


class FixedSizeChunker(ChunkingStrategy):
    """
    Simple fixed-size chunking strategy.
    Useful for baseline comparisons.
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        """
        Initialize fixed-size chunker.
        
        Args:
            chunk_size: Size of each chunk in characters
            overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        logger.info("FixedSizeChunker initialized")
    
    def chunk(self, document: LegalDocument) -> List[DocumentChunk]:
        """
        Create fixed-size chunks from document.
        
        Args:
            document: Parsed legal document
            
        Returns:
            List of fixed-size chunks
        """
        # Get full text
        full_text = document.raw_text
        chunks = []
        base_metadata = self._create_base_metadata(document)
        
        # Create chunks
        start = 0
        sequence = 0
        
        while start < len(full_text):
            # Calculate end position
            end = min(start + self.chunk_size, len(full_text))
            
            # Extract chunk content
            content = full_text[start:end]
            
            # Create chunk
            metadata = base_metadata.copy()
            metadata['start_pos'] = start
            metadata['end_pos'] = end
            
            chunk = DocumentChunk(
                chunk_id=f"fixed_{sequence}",
                content=content,
                metadata=metadata,
                chunk_type='fixed',
                sequence_number=sequence,
                tokens_count=self._estimate_tokens(content)
            )
            
            chunks.append(chunk)
            
            # Move to next position with overlap
            start = end - self.overlap if end < len(full_text) else end
            sequence += 1
        
        logger.info(f"Created {len(chunks)} fixed-size chunks")
        return chunks
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 3


class ChunkerFactory:
    """Factory class for creating appropriate chunking strategies."""
    
    @staticmethod
    def create_chunker(strategy: str = "article", **kwargs) -> ChunkingStrategy:
        """
        Create a chunker based on the specified strategy.
        
        Args:
            strategy: Name of the chunking strategy
            **kwargs: Additional arguments for the chunker
            
        Returns:
            Appropriate ChunkingStrategy instance
            
        Raises:
            ValueError: If strategy is not recognized
        """
        strategies = {
            'article': ArticleBasedChunker,
            'semantic': SemanticChunker,
            'fixed': FixedSizeChunker
        }
        
        if strategy not in strategies:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        logger.info(f"Creating {strategy} chunker")
        return strategies[strategy](**kwargs)