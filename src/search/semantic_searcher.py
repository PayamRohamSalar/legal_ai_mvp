# src/search/semantic_searcher.py
"""
Semantic search module for finding relevant legal documents.
Combines vector similarity with metadata filtering.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import time

from ..database.vector_db import VectorDBManager
from ..database.metadata_db import MetadataDBManager
from ..embeddings.text_embedder import TextEmbedder

# Configure logging
logger = logging.getLogger(__name__)


class SearchResult:
    """Container for search results."""
    
    def __init__(
        self,
        chunk_id: str,
        content: str,
        score: float,
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
        highlights: Optional[List[str]] = None
    ):
        """
        Initialize a search result.
        
        Args:
            chunk_id: Unique identifier of the chunk
            content: Text content of the result
            score: Relevance score (0-1, higher is better)
            metadata: Associated metadata
            document_id: Parent document ID
            highlights: Highlighted text snippets
        """
        self.chunk_id = chunk_id
        self.content = content
        self.score = score
        self.metadata = metadata
        self.document_id = document_id
        self.highlights = highlights or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'score': self.score,
            'metadata': self.metadata,
            'document_id': self.document_id,
            'highlights': self.highlights
        }
    
    def get_citation(self) -> str:
        """Generate a citation for this result."""
        parts = []
        
        # Add law name
        if 'law_name' in self.metadata:
            parts.append(self.metadata['law_name'])
        
        # Add chapter if available
        if 'chapter_number' in self.metadata and self.metadata['chapter_number']:
            parts.append(f"فصل {self.metadata['chapter_number']}")
        
        # Add article number
        if 'article_number' in self.metadata:
            parts.append(f"ماده {self.metadata['article_number']}")
        
        # Add approval date
        if 'approval_date' in self.metadata and self.metadata['approval_date']:
            parts.append(f"مصوب {self.metadata['approval_date']}")
        
        return " - ".join(parts) if parts else "منبع نامشخص"


class SemanticSearcher:
    """
    Main semantic search engine for legal documents.
    """
    
    def __init__(
        self,
        vector_db: VectorDBManager,
        metadata_db: MetadataDBManager,
        text_embedder: TextEmbedder
    ):
        """
        Initialize the semantic searcher.
        
        Args:
            vector_db: Vector database manager
            metadata_db: Metadata database manager
            text_embedder: Text embedding model
        """
        self.vector_db = vector_db
        self.metadata_db = metadata_db
        self.text_embedder = text_embedder
        
        logger.info("SemanticSearcher initialized")
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
        rerank: bool = True
    ) -> List[SearchResult]:
        """
        Perform semantic search on legal documents.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filters: Optional metadata filters
            min_score: Minimum similarity score threshold
            rerank: Whether to rerank results
            
        Returns:
            List of SearchResult objects
        """
        start_time = time.time()
        
        logger.info(f"Searching for: {query[:100]}...")
        
        # Preprocess query
        processed_query = self._preprocess_query(query)
        
        # Perform vector search
        vector_results = self.vector_db.search(
            query=processed_query,
            n_results=n_results * 2 if rerank else n_results,  # Get more for reranking
            filter_metadata=filters
        )
        
        # Convert to SearchResult objects
        search_results = []
        for result in vector_results:
            # Calculate similarity score (convert distance to similarity)
            score = 1.0 - (result['distance'] / 2.0) if result['distance'] else 1.0
            
            # Skip if below minimum score
            if score < min_score:
                continue
            
            # Create SearchResult
            search_result = SearchResult(
                chunk_id=result['id'],
                content=result['document'],
                score=score,
                metadata=result['metadata'],
                document_id=result['metadata'].get('document_id')
            )
            
            # Add highlights
            search_result.highlights = self._generate_highlights(
                query, 
                result['document']
            )
            
            search_results.append(search_result)
        
        # Rerank if requested
        if rerank and len(search_results) > n_results:
            search_results = self._rerank_results(query, search_results)
        
        # Limit to requested number
        search_results = search_results[:n_results]
        
        # Log search
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        self._log_search(
            query=query,
            num_results=len(search_results),
            response_time_ms=elapsed_time,
            top_result_id=search_results[0].chunk_id if search_results else None
        )
        
        logger.info(f"Search completed in {elapsed_time:.2f}ms with {len(search_results)} results")
        
        return search_results
    
    def search_by_article(
        self,
        law_name: Optional[str] = None,
        article_number: Optional[str] = None,
        chapter_number: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for specific articles by their identifiers.
        
        Args:
            law_name: Name of the law
            article_number: Article number
            chapter_number: Chapter number
            
        Returns:
            List of matching articles
        """
        filters = {}
        
        if law_name:
            filters['law_name'] = law_name
        if article_number:
            filters['article_number'] = article_number
        if chapter_number:
            filters['chapter_number'] = chapter_number
        
        # Use a generic query for structural search
        query = f"{law_name or ''} {f'ماده {article_number}' if article_number else ''}"
        
        return self.search(
            query=query.strip(),
            filters=filters,
            rerank=False  # No need to rerank for exact matches
        )
    
    def multi_query_search(
        self,
        queries: List[str],
        n_results_per_query: int = 3,
        aggregate: str = 'union'
    ) -> List[SearchResult]:
        """
        Search with multiple queries and aggregate results.
        
        Args:
            queries: List of search queries
            n_results_per_query: Results per individual query
            aggregate: Aggregation method ('union', 'intersection', 'weighted')
            
        Returns:
            Aggregated search results
        """
        all_results = {}
        
        for query in queries:
            results = self.search(query, n_results=n_results_per_query)
            
            for result in results:
                if result.chunk_id not in all_results:
                    all_results[result.chunk_id] = result
                elif aggregate == 'weighted':
                    # Average the scores
                    all_results[result.chunk_id].score = (
                        all_results[result.chunk_id].score + result.score
                    ) / 2
        
        # Sort by score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.score,
            reverse=True
        )
        
        return sorted_results
    
    def find_similar(
        self,
        chunk_id: str,
        n_results: int = 5
    ) -> List[SearchResult]:
        """
        Find documents similar to a given chunk.
        
        Args:
            chunk_id: ID of the reference chunk
            n_results: Number of similar documents to find
            
        Returns:
            List of similar documents
        """
        # Get the reference chunk
        ref_chunks = self.vector_db.get_by_ids([chunk_id])
        
        if not ref_chunks:
            logger.warning(f"Chunk {chunk_id} not found")
            return []
        
        ref_chunk = ref_chunks[0]
        
        # Search using the chunk's content
        return self.search(
            query=ref_chunk['document'],
            n_results=n_results + 1  # +1 to exclude self
        )[1:]  # Exclude the first result (self)
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess the search query.
        
        Args:
            query: Raw query text
            
        Returns:
            Preprocessed query
        """
        # Remove excessive whitespace
        query = ' '.join(query.split())
        
        # Add context for better semantic search (optional)
        if len(query) < 20:
            query = f"قانون {query}"
        
        return query
    
    def _rerank_results(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Rerank search results for better relevance.
        
        Args:
            query: Original query
            results: Initial search results
            
        Returns:
            Reranked results
        """
        # Simple reranking based on metadata relevance
        for result in results:
            boost = 0.0
            
            # Boost if query terms appear in article number
            if 'article_number' in result.metadata:
                if result.metadata['article_number'] in query:
                    boost += 0.1
            
            # Boost if it's a complete article (not semantic chunk)
            if result.metadata.get('chunk_type') == 'article':
                boost += 0.05
            
            # Boost recent documents
            if 'approval_year' in result.metadata:
                year = result.metadata['approval_year']
                if isinstance(year, int) and year > 1390:  # Recent laws (after 2011)
                    boost += 0.02
            
            # Apply boost
            result.score = min(1.0, result.score + boost)
        
        # Resort by new scores
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def _generate_highlights(
        self,
        query: str,
        content: str,
        max_highlights: int = 3,
        context_size: int = 50
    ) -> List[str]:
        """
        Generate highlighted snippets from content.
        
        Args:
            query: Search query
            content: Document content
            max_highlights: Maximum number of highlights
            context_size: Characters of context around match
            
        Returns:
            List of highlighted text snippets
        """
        highlights = []
        query_terms = query.lower().split()
        content_lower = content.lower()
        
        for term in query_terms[:3]:  # Use first 3 query terms
            if term in content_lower:
                # Find position of term
                pos = content_lower.find(term)
                if pos != -1:
                    # Extract context
                    start = max(0, pos - context_size)
                    end = min(len(content), pos + len(term) + context_size)
                    
                    snippet = content[start:end]
                    
                    # Add ellipsis if truncated
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(content):
                        snippet = snippet + "..."
                    
                    highlights.append(snippet)
                    
                    if len(highlights) >= max_highlights:
                        break
        
        return highlights
    
    def _log_search(
        self,
        query: str,
        num_results: int,
        response_time_ms: float,
        top_result_id: Optional[str] = None
    ):
        """
        Log search query for analytics.
        
        Args:
            query: Search query
            num_results: Number of results found
            response_time_ms: Response time in milliseconds
            top_result_id: ID of top result
        """
        try:
            self.metadata_db.log_search(
                query_text=query,
                query_type='semantic',
                num_results=num_results,
                response_time_ms=response_time_ms,
                top_result_id=top_result_id,
                session_id=None  # Can be added later for user tracking
            )
        except Exception as e:
            logger.warning(f"Failed to log search: {e}")
    
    def get_search_suggestions(
        self,
        partial_query: str,
        n_suggestions: int = 5
    ) -> List[str]:
        """
        Get search suggestions based on partial query.
        
        Args:
            partial_query: Partial search query
            n_suggestions: Number of suggestions
            
        Returns:
            List of suggested queries
        """
        # This is a placeholder for future implementation
        # Could use search history, popular queries, or autocomplete
        suggestions = [
            f"{partial_query} هیئت علمی",
            f"{partial_query} دانشگاه",
            f"{partial_query} پژوهش",
            f"{partial_query} استخدام",
            f"{partial_query} مناقصه"
        ]
        
        return suggestions[:n_suggestions]