# src/data_processing/pipeline.py
"""
Main pipeline orchestrator for processing legal documents.
Coordinates all components to create a complete knowledge base.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json
from tqdm import tqdm

from .document_processor import DocumentProcessor, ProcessedDocument
from .text_parser import PersianLegalTextParser, LegalDocument
from .chunking_strategy import ChunkerFactory, DocumentChunk
from .metadata_extractor import MetadataExtractor, LegalMetadata
from ..database.vector_db import VectorDBManager
from ..database.metadata_db import MetadataDBManager
from ..embeddings.text_embedder import TextEmbedder
from config.database_config import db_settings
from config.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentProcessingPipeline:
    """
    Main pipeline for processing legal documents end-to-end.
    """
    
    def __init__(
        self,
        vector_db: Optional[VectorDBManager] = None,
        metadata_db: Optional[MetadataDBManager] = None,
        text_embedder: Optional[TextEmbedder] = None,
        chunking_strategy: str = 'article'
    ):
        """
        Initialize the processing pipeline.
        
        Args:
            vector_db: Vector database manager (optional)
            metadata_db: Metadata database manager (optional)
            text_embedder: Text embedding model (optional)
            chunking_strategy: Strategy for chunking documents
        """
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.text_parser = PersianLegalTextParser()
        self.metadata_extractor = MetadataExtractor()
        self.chunking_strategy = chunking_strategy
        
        # Initialize databases if not provided
        if vector_db is None:
            logger.info("Initializing VectorDBManager...")
            chroma_config = db_settings.get_chromadb_config()
            self.vector_db = VectorDBManager(chroma_config)
        else:
            self.vector_db = vector_db
        
        if metadata_db is None:
            logger.info("Initializing MetadataDBManager...")
            postgres_config = db_settings.get_postgres_config()
            self.metadata_db = MetadataDBManager(postgres_config)
        else:
            self.metadata_db = metadata_db
        
        if text_embedder is None:
            logger.info("Initializing TextEmbedder...")
            self.text_embedder = TextEmbedder(
                model_name='multilingual-base',
                use_cache=True
            )
        else:
            self.text_embedder = text_embedder
        
        # Statistics
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
        
        logger.info(f"DocumentProcessingPipeline initialized with {chunking_strategy} chunking")
    
    def process_document(
        self,
        file_path: Path,
        skip_if_exists: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single document through the entire pipeline.
        
        Args:
            file_path: Path to the document file
            skip_if_exists: Skip if document already processed
            
        Returns:
            Processing result dictionary
        """
        logger.info(f"Processing document: {file_path.name}")
        result = {
            'success': False,
            'file': str(file_path),
            'document_id': None,
            'chunks': 0,
            'error': None
        }
        
        try:
            # Step 1: Extract raw content
            logger.info("Step 1: Extracting document content...")
            processed_doc = self.doc_processor.process_file(file_path)
            
            if not self.doc_processor.validate_document(processed_doc):
                raise ValueError("Document validation failed")
            
            # Step 2: Parse legal structure
            logger.info("Step 2: Parsing legal structure...")
            legal_doc = self.text_parser.parse(processed_doc.content)
            
            # Step 3: Extract metadata
            logger.info("Step 3: Extracting metadata...")
            metadata = self.metadata_extractor.extract(legal_doc)
            
            # Check if document already exists
            if skip_if_exists:
                existing = self.metadata_db.get_document(metadata.document_id)
                if existing:
                    logger.info(f"Document {metadata.document_id} already exists, skipping")
                    result['document_id'] = metadata.document_id
                    result['success'] = True
                    result['skipped'] = True
                    return result
            
            # Step 4: Create chunks
            logger.info("Step 4: Creating document chunks...")
            chunker = ChunkerFactory.create_chunker(self.chunking_strategy)
            chunks = chunker.chunk(legal_doc)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 5: Generate embeddings
            logger.info("Step 5: Generating embeddings...")
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.text_embedder.embed_batch(
                chunk_texts,
                show_progress=True
            )
            
            # Step 6: Save to metadata database
            logger.info("Step 6: Saving to metadata database...")
            
            # Prepare document data
            doc_data = {
                'id': metadata.document_id,
                'title': metadata.title,
                'law_name': metadata.law_name,
                'document_type': metadata.document_type,
                'approval_date': metadata.approval_date,
                'approval_year': metadata.approval_year,
                'effective_date': metadata.effective_date,
                'approval_authority': metadata.approval_authority,
                'num_chapters': metadata.num_chapters,
                'num_articles': metadata.num_articles,
                'num_notes': metadata.num_notes,
                'num_clauses': metadata.num_clauses,
                'legal_domain': metadata.legal_domain,
                'keywords': metadata.keywords,
                'subject_areas': metadata.subject_areas,
                'referenced_laws': metadata.referenced_laws,
                'amending_laws': metadata.amending_laws,
                'source_file': file_path.name,
                'processed_at': datetime.now(),
                'processing_status': 'completed'
            }
            
            # Save document
            document_id = self.metadata_db.save_document(doc_data)
            
            # Prepare chunk data
            chunks_data = []
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'id': f"{document_id}_{chunk.chunk_id}",
                    'document_id': document_id,
                    'chunk_type': chunk.chunk_type,
                    'sequence_number': chunk.sequence_number,
                    'content': chunk.content,
                    'tokens_count': chunk.tokens_count,
                    'chapter_number': chunk.metadata.get('chapter_number'),
                    'chapter_title': chunk.metadata.get('chapter_title'),
                    'article_number': chunk.metadata.get('article_number'),
                    'article_range': chunk.metadata.get('article_range'),
                    'vector_id': f"{document_id}_{chunk.chunk_id}",
                    'embedding_model': self.text_embedder.model_name
                }
                chunks_data.append(chunk_data)
            
            # Save chunks
            self.metadata_db.save_chunks(chunks_data)
            
            # Step 7: Save to vector database
            logger.info("Step 7: Saving to vector database...")
            
            # Prepare for vector DB
            vector_ids = [f"{document_id}_{chunk.chunk_id}" for chunk in chunks]
            vector_metadatas = []
            
            for chunk in chunks:
                # Combine chunk metadata with document metadata
                combined_metadata = {
                    'document_id': document_id,
                    'law_name': metadata.law_name,
                    'document_type': metadata.document_type,
                    'approval_date': metadata.approval_date,
                    'legal_domain': metadata.legal_domain,
                    **chunk.metadata
                }
                vector_metadatas.append(combined_metadata)
            
            # Add to vector DB
            self.vector_db.add_documents(
                documents=chunk_texts,
                metadatas=vector_metadatas,
                ids=vector_ids
            )
            
            # Update result
            result['success'] = True
            result['document_id'] = document_id
            result['chunks'] = len(chunks)
            
            # Update statistics
            self.stats['documents_processed'] += 1
            self.stats['chunks_created'] += len(chunks)
            
            logger.info(f"Successfully processed document: {document_id}")
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            result['error'] = str(e)
            self.stats['errors'] += 1
        
        return result
    
    def process_directory(
        self,
        directory_path: Path,
        file_pattern: str = "*.docx",
        skip_existing: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process all documents in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            file_pattern: File pattern to match
            skip_existing: Skip already processed documents
            
        Returns:
            List of processing results
        """
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")
        
        # Find all matching files
        files = list(directory_path.glob(file_pattern))
        logger.info(f"Found {len(files)} files to process in {directory_path}")
        
        # Record start time
        self.stats['start_time'] = datetime.now()
        
        # Process each file
        results = []
        for file_path in tqdm(files, desc="Processing documents"):
            result = self.process_document(file_path, skip_if_exists=skip_existing)
            results.append(result)
        
        # Record end time
        self.stats['end_time'] = datetime.now()
        
        # Log summary
        self._log_processing_summary(results)
        
        return results
    
    def validate_pipeline(self) -> Dict[str, bool]:
        """
        Validate that all pipeline components are working.
        
        Returns:
            Dictionary of component status
        """
        validation = {
            'document_processor': False,
            'text_parser': False,
            'metadata_extractor': False,
            'vector_db': False,
            'metadata_db': False,
            'text_embedder': False
        }
        
        try:
            # Test document processor
            validation['document_processor'] = self.doc_processor is not None
            
            # Test text parser
            test_text = "قانون تست\nماده 1. این یک تست است."
            parsed = self.text_parser.parse(test_text)
            validation['text_parser'] = parsed is not None
            
            # Test metadata extractor
            validation['metadata_extractor'] = self.metadata_extractor is not None
            
            # Test vector DB
            stats = self.vector_db.get_collection_stats()
            validation['vector_db'] = stats is not None
            
            # Test metadata DB
            db_stats = self.metadata_db.get_statistics()
            validation['metadata_db'] = db_stats is not None
            
            # Test text embedder
            test_embedding = self.text_embedder.embed("تست")
            validation['text_embedder'] = test_embedding is not None
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
        
        return validation
    
    def clear_all_data(self, confirm: bool = False):
        """
        Clear all data from databases.
        
        Args:
            confirm: Safety confirmation flag
        """
        if not confirm:
            logger.warning("Clear operation not confirmed. Set confirm=True to proceed.")
            return
        
        logger.warning("Clearing all data from databases...")
        
        # Clear vector database
        self.vector_db.clear_collection()
        
        # Clear metadata database (would need to implement this method)
        # self.metadata_db.clear_all()
        
        # Reset statistics
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
        
        logger.info("All data cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pipeline and database statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'pipeline': self.stats,
            'vector_db': self.vector_db.get_collection_stats(),
            'metadata_db': self.metadata_db.get_statistics(),
            'embedder': self.text_embedder.get_model_info()
        }
        
        # Calculate processing time
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            stats['pipeline']['processing_duration'] = str(duration)
        
        return stats
    
    def _log_processing_summary(self, results: List[Dict[str, Any]]):
        """Log a summary of processing results."""
        successful = sum(1 for r in results if r['success'])
        failed = sum(1 for r in results if not r['success'])
        skipped = sum(1 for r in results if r.get('skipped', False))
        
        logger.info("=" * 50)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total files: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Skipped: {skipped}")
        logger.info(f"Total chunks created: {self.stats['chunks_created']}")
        
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            logger.info(f"Processing time: {duration}")
        
        # Log errors
        if failed > 0:
            logger.warning("Failed documents:")
            for result in results:
                if not result['success'] and not result.get('skipped'):
                    logger.warning(f"  - {result['file']}: {result['error']}")
        
        logger.info("=" * 50)
    
    def export_statistics(self, output_path: Path):
        """
        Export statistics to a JSON file.
        
        Args:
            output_path: Path to save statistics
        """
        stats = self.get_statistics()
        
        # Convert datetime objects to strings
        def convert_dates(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        stats_json = json.dumps(stats, default=convert_dates, indent=2, ensure_ascii=False)
        
        output_path.write_text(stats_json, encoding='utf-8')
        logger.info(f"Statistics exported to {output_path}")