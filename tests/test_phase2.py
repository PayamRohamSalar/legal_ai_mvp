#!/usr/bin/env python3
"""
Phase 2 Integration Test Script
Tests all components of the document processing pipeline.
"""

import sys
from pathlib import Path
import logging
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.document_processor import DocumentProcessor
from src.data_processing.text_parser import PersianLegalTextParser
from src.data_processing.chunking_strategy import ChunkerFactory
from src.data_processing.metadata_extractor import MetadataExtractor
from src.data_processing.pipeline import DocumentProcessingPipeline
from src.database.vector_db import VectorDBManager
from src.database.metadata_db import MetadataDBManager
from src.embeddings.text_embedder import TextEmbedder
from src.search.semantic_searcher import SemanticSearcher
from config.database_config import db_settings
from config.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_document_processor():
    """Test document processor component."""
    print("\n" + "="*50)
    print("Testing Document Processor")
    print("="*50)
    
    processor = DocumentProcessor()
    
    # Test with sample file
    sample_file = Path("data/laws/Part2_Legals.docx")
    
    if sample_file.exists():
        try:
            doc = processor.process_file(sample_file)
            print(f"✓ Successfully processed: {doc.filename}")
            print(f"  - Content length: {len(doc.content)} characters")
            print(f"  - Metadata keys: {list(doc.metadata.keys())}")
            return True
        except Exception as e:
            print(f"✗ Error processing document: {e}")
            return False
    else:
        print(f"⚠ Sample file not found: {sample_file}")
        return False


def test_text_parser():
    """Test Persian legal text parser."""
    print("\n" + "="*50)
    print("Testing Text Parser")
    print("="*50)
    
    parser = PersianLegalTextParser()
    
    # Test with sample legal text
    sample_text = """
    قانون مقررات انتظامی هیئت علمی دانشگاه‌ها
    (مصوب 22/12/1364 مجلس شورای اسلامی)
    
    فصل اول: هیئت‌های رسیدگی انتظامی
    
    ماده 1. هیئت‌های رسیدگی عبارتند از:
    1. هیئت بدوی
    2. هیئت تجدید نظر
    
    تبصره 1. اعضای هیئت باید دارای شرایط خاصی باشند.
    
    ماده 2. هیئت بدوی در هر دانشگاه تشکیل می‌شود.
    """
    
    try:
        legal_doc = parser.parse(sample_text)
        summary = parser.extract_structure_summary(legal_doc)
        
        print(f"✓ Successfully parsed legal document")
        print(f"  - Title: {legal_doc.title}")
        print(f"  - Chapters: {summary['num_chapters']}")
        print(f"  - Articles: {summary['num_articles']}")
        print(f"  - Notes: {summary['num_notes']}")
        return True
    except Exception as e:
        print(f"✗ Error parsing text: {e}")
        return False


def test_chunking_strategies():
    """Test different chunking strategies."""
    print("\n" + "="*50)
    print("Testing Chunking Strategies")
    print("="*50)
    
    # Create sample legal document
    parser = PersianLegalTextParser()
    sample_text = """
    قانون تست
    ماده 1. این یک ماده تست است.
    ماده 2. این ماده دوم است.
    """
    legal_doc = parser.parse(sample_text)
    
    strategies = ['article', 'semantic', 'fixed']
    
    for strategy in strategies:
        try:
            chunker = ChunkerFactory.create_chunker(strategy)
            chunks = chunker.chunk(legal_doc)
            print(f"✓ {strategy.capitalize()} chunker: {len(chunks)} chunks created")
        except Exception as e:
            print(f"✗ Error with {strategy} chunker: {e}")
            return False
    
    return True


def test_metadata_extractor():
    """Test metadata extraction."""
    print("\n" + "="*50)
    print("Testing Metadata Extractor")
    print("="*50)
    
    parser = PersianLegalTextParser()
    extractor = MetadataExtractor()
    
    sample_text = """
    قانون نحوه تأمین هیئت علمی مورد نیاز دانشگاه‌ها
    (مصوب 1/3/1365 مجلس شورای اسلامی)
    """
    
    try:
        legal_doc = parser.parse(sample_text)
        metadata = extractor.extract(legal_doc)
        
        print(f"✓ Successfully extracted metadata")
        print(f"  - Document ID: {metadata.document_id}")
        print(f"  - Document Type: {metadata.document_type}")
        print(f"  - Legal Domain: {metadata.legal_domain}")
        print(f"  - Keywords: {metadata.keywords[:3]}")
        return True
    except Exception as e:
        print(f"✗ Error extracting metadata: {e}")
        return False


def test_text_embedder():
    """Test text embedding."""
    print("\n" + "="*50)
    print("Testing Text Embedder")
    print("="*50)
    
    try:
        embedder = TextEmbedder(model_name='multilingual-base', use_cache=True)
        
        # Test single embedding
        text = "قانون هیئت علمی دانشگاه"
        embedding = embedder.embed(text)
        
        print(f"✓ Successfully created embedding")
        print(f"  - Model: {embedder.model_name}")
        print(f"  - Embedding dimension: {embedding.shape[0]}")
        
        # Test batch embedding
        texts = ["ماده اول", "ماده دوم", "تبصره یک"]
        embeddings = embedder.embed_batch(texts)
        
        print(f"✓ Batch embedding successful")
        print(f"  - Batch size: {len(texts)}")
        print(f"  - Output shape: {embeddings.shape}")
        
        # Test similarity
        similarity = embedder.similarity("دانشگاه", "هیئت علمی")
        print(f"✓ Similarity calculation: {similarity:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Error with text embedder: {e}")
        return False


def test_database_connections():
    """Test database connections."""
    print("\n" + "="*50)
    print("Testing Database Connections")
    print("="*50)
    
    # Test Vector DB
    try:
        chroma_config = db_settings.get_chromadb_config()
        vector_db = VectorDBManager(chroma_config)
        stats = vector_db.get_collection_stats()
        print(f"✓ ChromaDB connected")
        print(f"  - Collection: {stats['collection_name']}")
        print(f"  - Documents: {stats['document_count']}")
    except Exception as e:
        print(f"✗ ChromaDB connection failed: {e}")
        return False
    
    # Test Metadata DB
    try:
        postgres_config = db_settings.get_postgres_config()
        metadata_db = MetadataDBManager(postgres_config)
        db_stats = metadata_db.get_statistics()
        print(f"✓ PostgreSQL connected")
        print(f"  - Total documents: {db_stats['total_documents']}")
        print(f"  - Total chunks: {db_stats['total_chunks']}")
    except Exception as e:
        print(f"✗ PostgreSQL connection failed: {e}")
        return False
    
    return True


def test_full_pipeline():
    """Test the complete processing pipeline."""
    print("\n" + "="*50)
    print("Testing Full Pipeline")
    print("="*50)
    
    try:
        # Initialize pipeline
        pipeline = DocumentProcessingPipeline(chunking_strategy='article')
        
        # Validate components
        validation = pipeline.validate_pipeline()
        all_valid = all(validation.values())
        
        if all_valid:
            print("✓ All pipeline components validated")
            for component, status in validation.items():
                status_icon = "✓" if status else "✗"
                print(f"  {status_icon} {component}")
        else:
            print("✗ Some components failed validation")
            return False
        
        # Process sample document if available
        sample_file = Path("data/laws/Part2_Legals.docx")
        
        if sample_file.exists():
            print("\nProcessing sample document...")
            result = pipeline.process_document(sample_file, skip_if_exists=False)
            
            if result['success']:
                print(f"✓ Document processed successfully")
                print(f"  - Document ID: {result['document_id']}")
                print(f"  - Chunks created: {result['chunks']}")
            else:
                print(f"✗ Document processing failed: {result['error']}")
                return False
        
        # Get statistics
        stats = pipeline.get_statistics()
        print("\nPipeline Statistics:")
        print(f"  - Documents processed: {stats['pipeline']['documents_processed']}")
        print(f"  - Chunks created: {stats['pipeline']['chunks_created']}")
        print(f"  - Errors: {stats['pipeline']['errors']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        return False


def test_semantic_search():
    """Test semantic search functionality."""
    print("\n" + "="*50)
    print("Testing Semantic Search")
    print("="*50)
    
    try:
        # Initialize components
        chroma_config = db_settings.get_chromadb_config()
        postgres_config = db_settings.get_postgres_config()
        
        vector_db = VectorDBManager(chroma_config)
        metadata_db = MetadataDBManager(postgres_config)
        text_embedder = TextEmbedder(model_name='multilingual-base')
        
        # Initialize searcher
        searcher = SemanticSearcher(vector_db, metadata_db, text_embedder)
        
        # Test search (if there's data)
        if vector_db.get_collection_stats()['document_count'] > 0:
            test_query = "هیئت علمی دانشگاه"
            print(f"\nSearching for: '{test_query}'")
            
            results = searcher.search(test_query, n_results=3)
            
            if results:
                print(f"✓ Found {len(results)} results")
                for i, result in enumerate(results, 1):
                    print(f"\n  Result {i}:")
                    print(f"    Score: {result.score:.3f}")
                    print(f"    Citation: {result.get_citation()}")
                    print(f"    Preview: {result.content[:100]}...")
            else:
                print("  No results found")
        else:
            print("⚠ No documents in database to search")
        
        return True
        
    except Exception as e:
        print(f"✗ Search test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print(" PHASE 2 INTEGRATION TESTS ".center(60))
    print("="*60)
    
    tests = [
        ("Document Processor", test_document_processor),
        ("Text Parser", test_text_parser),
        ("Chunking Strategies", test_chunking_strategies),
        ("Metadata Extractor", test_metadata_extractor),
        ("Text Embedder", test_text_embedder),
        ("Database Connections", test_database_connections),
        ("Full Pipeline", test_full_pipeline),
        ("Semantic Search", test_semantic_search),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print(" TEST SUMMARY ".center(60))
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Phase 2 is ready.")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please review the errors.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)