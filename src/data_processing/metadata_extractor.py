# src/data_processing/metadata_extractor.py
"""
Metadata extraction module for legal documents.
Extracts structured metadata from Persian legal texts.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import logging
import json

from .text_parser import LegalDocument, LegalArticle, LegalChapter

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class LegalMetadata:
    """Container for comprehensive legal document metadata."""
    
    # Basic Information
    document_id: str
    title: str
    law_name: str
    document_type: str  # قانون، آیین‌نامه، دستورالعمل، etc.
    
    # Temporal Information
    approval_date: Optional[str]
    approval_date_formatted: Optional[str]
    approval_year: Optional[int]
    effective_date: Optional[str]
    
    # Approval Authority
    approval_authority: Optional[str]  # مجلس شورای اسلامی، هیئت وزیران، etc.
    
    # Structure Information
    num_chapters: int
    num_articles: int
    num_notes: int
    num_clauses: int
    
    # Content Categories
    legal_domain: str  # حوزه حقوقی
    keywords: List[str]
    subject_areas: List[str]
    
    # References
    referenced_laws: List[str]
    amending_laws: List[str]
    
    # Processing Information
    extraction_date: str
    extraction_version: str = "1.0"
    confidence_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert metadata to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class MetadataExtractor:
    """
    Extracts comprehensive metadata from Persian legal documents.
    """
    
    def __init__(self):
        """Initialize the metadata extractor with patterns and mappings."""
        
        # Patterns for extracting specific information
        self.patterns = {
            # Date patterns
            'persian_date': re.compile(
                r'([۰-۹]{1,2})[/\/]([۰-۹]{1,2})[/\/]([۰-۹]{4})',
                re.UNICODE
            ),
            'english_date': re.compile(
                r'([0-9]{1,2})[/\/]([0-9]{1,2})[/\/]([0-9]{4})'
            ),
            
            # Approval authority patterns
            'parliament': re.compile(
                r'مصوب.*?مجلس\s+شورای\s+اسلامی',
                re.UNICODE | re.IGNORECASE
            ),
            'cabinet': re.compile(
                r'مصوب.*?هیئت\s+وزیران',
                re.UNICODE | re.IGNORECASE
            ),
            'council': re.compile(
                r'مصوب.*?شورای\s+عالی',
                re.UNICODE | re.IGNORECASE
            ),
            
            # Document type patterns
            'law': re.compile(r'قانون\s+', re.UNICODE),
            'regulation': re.compile(r'آیین[\s\-]?نامه\s+', re.UNICODE),
            'directive': re.compile(r'دستورالعمل\s+', re.UNICODE),
            'bylaw': re.compile(r'اساسنامه\s+', re.UNICODE),
            
            # Reference patterns
            'law_reference': re.compile(
                r'(?:قانون|ماده|تبصره)\s+(.+?)(?:\s+مصوب|\s+و\s+|\s*[،,]|$)',
                re.UNICODE
            ),
            
            # Amendment patterns
            'amendment': re.compile(
                r'(?:اصلاح|الحاق|حذف)\s+(?:ماده|تبصره|بند)',
                re.UNICODE | re.IGNORECASE
            )
        }
        
        # Domain keywords for classification
        self.domain_keywords = {
            'research_technology': [
                'پژوهش', 'فناوری', 'تحقیقات', 'علمی', 'دانشگاه', 
                'هیئت علمی', 'آموزش عالی', 'فنی', 'نوآوری'
            ],
            'employment': [
                'استخدام', 'کارمند', 'حقوق', 'مزایا', 'بازنشستگی',
                'انتظامی', 'تخلف', 'انضباطی'
            ],
            'education': [
                'آموزش', 'دانشجو', 'تحصیل', 'مدرک', 'دوره', 
                'کارشناسی', 'دکتری', 'تحصیلات'
            ],
            'financial': [
                'مالی', 'بودجه', 'اعتبار', 'هزینه', 'درآمد',
                'محاسبات', 'حسابرسی', 'مناقصه'
            ],
            'administrative': [
                'اداری', 'سازمان', 'وزارت', 'مؤسسه', 'تشکیلات',
                'اختیارات', 'وظایف'
            ]
        }
        
        logger.info("MetadataExtractor initialized")
    
    def extract(self, document: LegalDocument, document_id: str = None) -> LegalMetadata:
        """
        Extract comprehensive metadata from a legal document.
        
        Args:
            document: Parsed legal document
            document_id: Optional document identifier
            
        Returns:
            LegalMetadata object with extracted information
        """
        logger.info(f"Extracting metadata for document: {document.title}")
        
        # Generate document ID if not provided
        if not document_id:
            document_id = self._generate_document_id(document)
        
        # Extract basic information
        document_type = self._identify_document_type(document.title)
        
        # Extract dates
        approval_date, formatted_date, year = self._extract_dates(document)
        
        # Extract approval authority
        approval_authority = self._extract_approval_authority(document.raw_text)
        
        # Calculate structure statistics
        structure_stats = self._calculate_structure_stats(document)
        
        # Extract keywords and domains
        keywords = self._extract_keywords(document)
        legal_domain = self._identify_legal_domain(document.raw_text)
        subject_areas = self._identify_subject_areas(document.raw_text)
        
        # Extract references
        referenced_laws = self._extract_referenced_laws(document.raw_text)
        amending_laws = self._check_amendments(document.raw_text)
        
        # Create metadata object
        metadata = LegalMetadata(
            document_id=document_id,
            title=document.title,
            law_name=document.law_name,
            document_type=document_type,
            approval_date=approval_date,
            approval_date_formatted=formatted_date,
            approval_year=year,
            effective_date=approval_date,  # Default to approval date
            approval_authority=approval_authority,
            num_chapters=structure_stats['num_chapters'],
            num_articles=structure_stats['num_articles'],
            num_notes=structure_stats['num_notes'],
            num_clauses=structure_stats['num_clauses'],
            legal_domain=legal_domain,
            keywords=keywords,
            subject_areas=subject_areas,
            referenced_laws=referenced_laws,
            amending_laws=amending_laws,
            extraction_date=datetime.now().isoformat()
        )
        
        logger.info(f"Metadata extraction completed for {document_id}")
        return metadata
    
    def _generate_document_id(self, document: LegalDocument) -> str:
        """Generate a unique document ID."""
        # Use law name and approval date if available
        base_id = document.law_name.replace(' ', '_')[:30]
        
        if document.approval_date:
            date_part = document.approval_date.replace('/', '')
            return f"{base_id}_{date_part}"
        
        # Fallback to timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{base_id}_{timestamp}"
    
    def _identify_document_type(self, title: str) -> str:
        """Identify the type of legal document."""
        title_lower = title.lower()
        
        if self.patterns['law'].search(title):
            return "قانون"
        elif self.patterns['regulation'].search(title):
            return "آیین‌نامه"
        elif self.patterns['directive'].search(title):
            return "دستورالعمل"
        elif self.patterns['bylaw'].search(title):
            return "اساسنامه"
        else:
            return "سند حقوقی"
    
    def _extract_dates(self, document: LegalDocument) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """
        Extract and format dates from the document.
        
        Returns:
            Tuple of (original_date, formatted_date, year)
        """
        if not document.approval_date:
            return None, None, None
        
        original_date = document.approval_date
        
        # Convert Persian digits to English
        persian_to_english = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')
        english_date = original_date.translate(persian_to_english)
        
        # Parse date
        try:
            parts = english_date.split('/')
            if len(parts) == 3:
                day, month, year = parts
                formatted_date = f"{year}/{month.zfill(2)}/{day.zfill(2)}"
                return original_date, formatted_date, int(year)
        except:
            pass
        
        return original_date, None, None
    
    def _extract_approval_authority(self, text: str) -> Optional[str]:
        """Extract the approval authority from text."""
        if self.patterns['parliament'].search(text):
            return "مجلس شورای اسلامی"
        elif self.patterns['cabinet'].search(text):
            return "هیئت وزیران"
        elif self.patterns['council'].search(text):
            # Try to extract the specific council name
            match = self.patterns['council'].search(text)
            if match:
                return match.group(0)
            return "شورای عالی"
        
        return None
    
    def _calculate_structure_stats(self, document: LegalDocument) -> Dict[str, int]:
        """Calculate structural statistics of the document."""
        stats = {
            'num_chapters': len(document.chapters),
            'num_articles': 0,
            'num_notes': 0,
            'num_clauses': 0
        }
        
        # Count from chapters
        for chapter in document.chapters:
            stats['num_articles'] += len(chapter.articles)
            for article in chapter.articles:
                stats['num_notes'] += len(article.notes)
                stats['num_clauses'] += len(article.clauses)
        
        # Count standalone articles
        stats['num_articles'] += len(document.standalone_articles)
        for article in document.standalone_articles:
            stats['num_notes'] += len(article.notes)
            stats['num_clauses'] += len(article.clauses)
        
        return stats
    
    def _extract_keywords(self, document: LegalDocument) -> List[str]:
        """Extract important keywords from the document."""
        keywords = set()
        
        # Extract from title
        title_words = document.title.split()
        important_title_words = [
            word for word in title_words 
            if len(word) > 3 and word not in ['قانون', 'آیین', 'نامه', 'دستورالعمل']
        ]
        keywords.update(important_title_words[:5])
        
        # Extract frequently mentioned terms
        text = document.raw_text
        
        # Common legal terms to look for
        legal_terms = [
            'هیئت علمی', 'دانشگاه', 'پژوهش', 'فناوری', 'آموزش عالی',
            'تحقیقات', 'مناقصه', 'استخدام', 'مؤسسه', 'وزارت'
        ]
        
        for term in legal_terms:
            if term in text:
                keywords.add(term)
        
        return list(keywords)[:10]  # Limit to 10 keywords
    
    def _identify_legal_domain(self, text: str) -> str:
        """Identify the primary legal domain of the document."""
        domain_scores = {}
        
        # Count keyword occurrences for each domain
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                domain_scores[domain] = score
        
        # Return domain with highest score
        if domain_scores:
            primary_domain = max(domain_scores, key=domain_scores.get)
            
            # Map to Persian names
            domain_names = {
                'research_technology': 'پژوهش و فناوری',
                'employment': 'استخدام و کارگزینی',
                'education': 'آموزش',
                'financial': 'مالی و محاسباتی',
                'administrative': 'اداری و تشکیلاتی'
            }
            
            return domain_names.get(primary_domain, 'عمومی')
        
        return 'عمومی'
    
    def _identify_subject_areas(self, text: str) -> List[str]:
        """Identify multiple subject areas covered in the document."""
        subject_areas = []
        
        # Check for presence of each domain
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in text for keyword in keywords):
                domain_names = {
                    'research_technology': 'پژوهش و فناوری',
                    'employment': 'امور استخدامی',
                    'education': 'آموزش عالی',
                    'financial': 'امور مالی',
                    'administrative': 'امور اداری'
                }
                
                area = domain_names.get(domain)
                if area and area not in subject_areas:
                    subject_areas.append(area)
        
        return subject_areas
    
    def _extract_referenced_laws(self, text: str) -> List[str]:
        """Extract references to other laws in the document."""
        references = set()
        
        # Find law references
        matches = self.patterns['law_reference'].findall(text)
        
        for match in matches:
            # Clean and filter the match
            cleaned = match.strip()
            if len(cleaned) > 5 and len(cleaned) < 100:
                # Avoid duplicates and very short/long matches
                references.add(cleaned)
        
        return list(references)[:10]  # Limit to 10 references
    
    def _check_amendments(self, text: str) -> List[str]:
        """Check if this document amends other laws."""
        amendments = []
        
        if self.patterns['amendment'].search(text):
            # This document contains amendments
            # Try to extract what it amends
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if 'اصلاح' in line or 'الحاق' in line:
                    # Look for law name in surrounding lines
                    context = ' '.join(lines[max(0, i-1):min(len(lines), i+2)])
                    if 'قانون' in context:
                        amendments.append(context[:100])  # First 100 chars
        
        return amendments
    
    def enrich_with_nlp(self, metadata: LegalMetadata, text: str) -> LegalMetadata:
        """
        Enrich metadata using NLP techniques (optional enhancement).
        
        Args:
            metadata: Base metadata object
            text: Document text for NLP analysis
            
        Returns:
            Enriched metadata
        """
        # This method can be extended with more sophisticated NLP
        # For now, it returns the metadata as-is
        return metadata