#!/usr/bin/env python3
"""
Quick Start Script for Phase 2
Quickly process documents and test search functionality.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_processing.pipeline import DocumentProcessingPipeline
from src.search.semantic_searcher import SemanticSearcher
from src.database.vector_db import VectorDBManager
from src.database.metadata_db import MetadataDBManager
from src.embeddings.text_embedder import TextEmbedder
from config.database_config import db_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    
    print("\n" + "="*60)
    print(" 🚀 PHASE 2 QUICK START ".center(60))
    print("="*60)
    
    # Step 1: Initialize Pipeline
    print("\n📦 Initializing Document Processing Pipeline...")
    print("-" * 40)
    
    try:
        pipeline = DocumentProcessingPipeline(chunking_strategy='article')
        print("✓ Pipeline initialized successfully")
        
        # Validate components
        validation = pipeline.validate_pipeline()
        all_valid = all(validation.values())
        
        if not all_valid:
            print("⚠ Warning: Some components are not properly initialized")
            for component, status in validation.items():
                if not status:
                    print(f"  ✗ {component}")
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        return False
    
    # Step 2: Process Documents
    print("\n📄 Processing Legal Documents...")
    print("-" * 40)
    
    # Check for sample document
    sample_file = Path("data/laws/Part2_Legals.docx")
    
    if not sample_file.exists():
        print(f"⚠ Sample file not found: {sample_file}")
        print("  Please add legal documents to data/laws/ directory")
        
        # Try to find any document
        doc_dir = Path("data/laws")
        if doc_dir.exists():
            docs = list(doc_dir.glob("*.docx")) + list(doc_dir.glob("*.txt"))
            if docs:
                sample_file = docs[0]
                print(f"  Found alternative: {sample_file.name}")
            else:
                print("  No documents found to process")
                return False
    
    # Process the document
    print(f"\nProcessing: {sample_file.name}")
    result = pipeline.process_document(sample_file, skip_if_exists=False)
    
    if result['success']:
        print(f"✓ Document processed successfully!")
        print(f"  - Document ID: {result['document_id']}")
        print(f"  - Chunks created: {result['chunks']}")
    else:
        print(f"✗ Processing failed: {result['error']}")
        return False
    
    # Step 3: Test Search
    print("\n🔍 Testing Semantic Search...")
    print("-" * 40)
    
    try:
        # Initialize search components
        vector_db = VectorDBManager(db_settings.get_chromadb_config())
        metadata_db = MetadataDBManager(db_settings.get_postgres_config())
        embedder = TextEmbedder(model_name='multilingual-base')
        
        searcher = SemanticSearcher(vector_db, metadata_db, embedder)
        
        # Test queries
        test_queries = [
            "هیئت علمی دانشگاه",
            "شرایط استخدام",
            "مناقصه",
            "آموزش عالی",
            "تخلفات انتظامی"
        ]
        
        print("\nRunning test searches:")
        for query in test_queries[:3]:  # Test first 3 queries
            print(f"\n🔎 Query: '{query}'")
            results = searcher.search(query, n_results=2)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"\n  Result {i}:")
                    print(f"    Score: {result.score:.3f}")
                    print(f"    Source: {result.get_citation()}")
                    print(f"    Preview: {result.content[:100]}...")
            else:
                print("    No results found")
        
        print("\n✓ Search functionality is working!")
        
    except Exception as e:
        print(f"✗ Search test failed: {e}")
        return False
    
    # Step 4: Show Statistics
    print("\n📊 System Statistics")
    print("-" * 40)
    
    stats = pipeline.get_statistics()
    
    print(f"Pipeline Stats:")
    print(f"  - Documents processed: {stats['pipeline']['documents_processed']}")
    print(f"  - Total chunks: {stats['pipeline']['chunks_created']}")
    print(f"  - Errors: {stats['pipeline']['errors']}")
    
    print(f"\nVector Database:")
    print(f"  - Collection: {stats['vector_db']['collection_name']}")
    print(f"  - Documents: {stats['vector_db']['document_count']}")
    
    print(f"\nMetadata Database:")
    print(f"  - Total documents: {stats['metadata_db']['total_documents']}")
    print(f"  - Total chunks: {stats['metadata_db']['total_chunks']}")
    print(f"  - Total searches: {stats['metadata_db']['total_searches']}")
    
    print(f"\nEmbedding Model:")
    print(f"  - Model: {stats['embedder']['model_name']}")
    print(f"  - Dimension: {stats['embedder']['embedding_dimension']}")
    
    # Success message
    print("\n" + "="*60)
    print(" ✅ PHASE 2 IS READY! ".center(60))
    print("="*60)
    print("\nNext steps:")
    print("1. Add more legal documents to data/laws/")
    print("2. Process them with: pipeline.process_directory(Path('data/laws'))")
    print("3. Test searches with different queries")
    print("4. Move to Phase 3: RAG implementation")
    
    return True


def interactive_mode():
    """Run in interactive mode for testing."""
    
    print("\n🎮 Entering Interactive Mode...")
    print("Type 'help' for commands, 'exit' to quit\n")
    
    # Initialize components
    pipeline = DocumentProcessingPipeline()
    vector_db = VectorDBManager(db_settings.get_chromadb_config())
    metadata_db = MetadataDBManager(db_settings.get_postgres_config())
    embedder = TextEmbedder(model_name='multilingual-base')
    searcher = SemanticSearcher(vector_db, metadata_db, embedder)
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == 'exit':
                print("Goodbye!")
                break
            
            elif command == 'help':
                print("\nAvailable commands:")
                print("  search <query>  - Search for legal information")
                print("  process <file>  - Process a document file")
                print("  stats          - Show system statistics")
                print("  clear          - Clear all data (careful!)")
                print("  exit           - Exit interactive mode")
            
            elif command.startswith('search '):
                query = command[7:]
                print(f"\nSearching for: '{query}'")
                results = searcher.search(query, n_results=3)
                
                if results:
                    for i, result in enumerate(results, 1):
                        print(f"\n--- Result {i} ---")
                        print(f"Score: {result.score:.3f}")
                        print(f"Source: {result.get_citation()}")
                        print(f"Content: {result.content[:200]}...")
                else:
                    print("No results found.")
            
            elif command.startswith('process '):
                file_path = Path(command[8:])
                if file_path.exists():
                    print(f"Processing {file_path.name}...")
                    result = pipeline.process_document(file_path)
                    if result['success']:
                        print(f"✓ Success! Created {result['chunks']} chunks")
                    else:
                        print(f"✗ Failed: {result['error']}")
                else:
                    print(f"File not found: {file_path}")
            
            elif command == 'stats':
                stats = pipeline.get_statistics()
                print(f"\nDocuments: {stats['metadata_db']['total_documents']}")
                print(f"Chunks: {stats['metadata_db']['total_chunks']}")
                print(f"Searches: {stats['metadata_db']['total_searches']}")
            
            elif command == 'clear':
                confirm = input("Are you sure? Type 'yes' to confirm: ")
                if confirm.lower() == 'yes':
                    pipeline.clear_all_data(confirm=True)
                    print("All data cleared.")
            
            else:
                print("Unknown command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\n\nUse 'exit' to quit properly.")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 2 Quick Start")
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    else:
        success = main()
        sys.exit(0 if success else 1)