# src/data_processing/text_parser.py
"""
Persian legal text parser module.
Parses and structures Iranian legal documents with their hierarchical structure.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class LegalArticle:
    """Represents a single article (ماده) in a legal document."""
    article_number: str
    content: str
    clauses: List[str] = field(default_factory=list)  # بندها
    notes: List[str] = field(default_factory=list)    # تبصره‌ها
    raw_text: str = ""


@dataclass
class LegalChapter:
    """Represents a chapter (فصل) in a legal document."""
    chapter_number: str
    title: str
    articles: List[LegalArticle] = field(default_factory=list)


@dataclass
class LegalDocument:
    """Represents a complete legal document with structured content."""
    title: str
    law_name: str
    approval_date: Optional[str] = None
    chapters: List[LegalChapter] = field(default_factory=list)
    standalone_articles: List[LegalArticle] = field(default_factory=list)
    raw_text: str = ""


class PersianLegalTextParser:
    """
    Parser for Persian legal texts with understanding of Iranian legal document structure.
    """
    
    def __init__(self):
        """Initialize the parser with regex patterns for legal structures."""
        
        # Patterns for identifying legal structures
        self.patterns = {
            # Law title pattern
            'law_title': re.compile(
                r'قانون\s+(.+?)(?:\s*[\(（]مصوب|$)', 
                re.UNICODE | re.MULTILINE
            ),
            
            # Approval date pattern
            'approval_date': re.compile(
                r'مصوب\s*([۰-۹0-9]+[/\/][۰-۹0-9]+[/\/][۰-۹0-9]+)',
                re.UNICODE
            ),
            
            # Chapter pattern (فصل)
            'chapter': re.compile(
                r'فصل\s+([اولدومسچهارپنجششفتهشنیکبیستاول‌دوم‌سوم‌چهارم‌پنجم‌ششم‌هفتم‌هشتم‌نهم‌دهم۰-۹0-9]+)[:\s\-\.]*(.+?)(?=\n|$)',
                re.UNICODE | re.MULTILINE
            ),
            
            # Article pattern (ماده)
            'article': re.compile(
                r'ماده\s*([۰-۹0-9]+|[اولدومسچهارپنجششفتهشنیکبیست]+)[\s\-\.]*',
                re.UNICODE
            ),
            
            # Note/clause pattern (تبصره)
            'note': re.compile(
                r'تبصره\s*([۰-۹0-9]+)?[\s\-\.]*',
                re.UNICODE
            ),
            
            # Numbered items pattern (بند)
            'item': re.compile(
                r'^([۰-۹0-9]+|[الف-ی]|[a-z]|[اولدومسچهارپنجششفتهشنیکبیست]+)[\s\-\.)]+',
                re.UNICODE | re.MULTILINE
            )
        }
        
        logger.info("PersianLegalTextParser initialized")
    
    def parse(self, text: str) -> LegalDocument:
        """
        Parse a complete legal document text.
        
        Args:
            text: Raw text of the legal document
            
        Returns:
            Structured LegalDocument object
        """
        logger.info("Starting to parse legal document")
        
        # Extract document metadata
        title = self._extract_title(text)
        law_name = self._extract_law_name(text)
        approval_date = self._extract_approval_date(text)
        
        # Initialize document
        doc = LegalDocument(
            title=title,
            law_name=law_name,
            approval_date=approval_date,
            raw_text=text
        )
        
        # Split into chapters and articles
        chapters = self._extract_chapters(text)
        
        if chapters:
            doc.chapters = chapters
        else:
            # If no chapters, extract articles directly
            doc.standalone_articles = self._extract_articles(text)
        
        logger.info(f"Parsed document with {len(doc.chapters)} chapters and {len(doc.standalone_articles)} standalone articles")
        return doc
    
    def _extract_title(self, text: str) -> str:
        """Extract the document title."""
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            if 'قانون' in line or 'آیین‌نامه' in line or 'دستورالعمل' in line:
                return line.strip()
        return "Unknown Title"
    
    def _extract_law_name(self, text: str) -> str:
        """Extract the law name from text."""
        match = self.patterns['law_title'].search(text)
        if match:
            return match.group(1).strip()
        return self._extract_title(text)
    
    def _extract_approval_date(self, text: str) -> Optional[str]:
        """Extract the approval date."""
        match = self.patterns['approval_date'].search(text)
        if match:
            return match.group(1)
        return None
    
    def _extract_chapters(self, text: str) -> List[LegalChapter]:
        """Extract chapters from the document."""
        chapters = []
        
        # Find all chapter headers
        chapter_matches = list(self.patterns['chapter'].finditer(text))
        
        for i, match in enumerate(chapter_matches):
            chapter_num = match.group(1)
            chapter_title = match.group(2).strip()
            
            # Determine chapter content boundaries
            start_pos = match.end()
            end_pos = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else len(text)
            
            chapter_content = text[start_pos:end_pos]
            
            # Extract articles within this chapter
            articles = self._extract_articles(chapter_content)
            
            chapter = LegalChapter(
                chapter_number=self._normalize_number(chapter_num),
                title=chapter_title,
                articles=articles
            )
            
            chapters.append(chapter)
        
        return chapters
    
    def _extract_articles(self, text: str) -> List[LegalArticle]:
        """Extract articles from text."""
        articles = []
        
        # Find all article headers
        article_matches = list(self.patterns['article'].finditer(text))
        
        for i, match in enumerate(article_matches):
            article_num = match.group(1)
            
            # Determine article content boundaries
            start_pos = match.end()
            
            # Find next article or end of text
            if i + 1 < len(article_matches):
                end_pos = article_matches[i + 1].start()
            else:
                end_pos = len(text)
            
            article_content = text[start_pos:end_pos].strip()
            
            # Extract notes and clauses
            notes = self._extract_notes(article_content)
            clauses = self._extract_clauses(article_content)
            
            # Clean article content (remove notes and clauses markers)
            clean_content = self._clean_article_content(article_content)
            
            article = LegalArticle(
                article_number=self._normalize_number(article_num),
                content=clean_content,
                clauses=clauses,
                notes=notes,
                raw_text=article_content
            )
            
            articles.append(article)
        
        return articles
    
    def _extract_notes(self, text: str) -> List[str]:
        """Extract notes (تبصره) from article text."""
        notes = []
        
        # Find all note markers
        note_matches = list(self.patterns['note'].finditer(text))
        
        for i, match in enumerate(note_matches):
            start_pos = match.end()
            
            # Find next note or end of text
            if i + 1 < len(note_matches):
                end_pos = note_matches[i + 1].start()
            else:
                # Check if there's another article marker
                next_article = self.patterns['article'].search(text[start_pos:])
                end_pos = start_pos + next_article.start() if next_article else len(text)
            
            note_content = text[start_pos:end_pos].strip()
            if note_content:
                notes.append(note_content)
        
        return notes
    
    def _extract_clauses(self, text: str) -> List[str]:
        """Extract numbered clauses/items from text."""
        clauses = []
        
        # Split by lines and check for numbered items
        lines = text.split('\n')
        current_clause = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts with a number or letter marker
            if self.patterns['item'].match(line):
                # Save previous clause if exists
                if current_clause:
                    clauses.append(' '.join(current_clause))
                # Start new clause
                current_clause = [line]
            elif current_clause:
                # Continue current clause
                current_clause.append(line)
        
        # Add last clause
        if current_clause:
            clauses.append(' '.join(current_clause))
        
        return clauses
    
    def _clean_article_content(self, text: str) -> str:
        """
        Clean article content by removing structural markers.
        
        Args:
            text: Raw article text
            
        Returns:
            Cleaned article content
        """
        # Remove note markers
        text = self.patterns['note'].sub('', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _normalize_number(self, number_str: str) -> str:
        """
        Normalize Persian/Arabic numerals and number words.
        
        Args:
            number_str: Number in various formats
            
        Returns:
            Normalized number string
        """
        # Persian to English digit mapping
        persian_to_english = {
            '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
            '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
        }
        
        # Convert Persian digits to English
        for persian, english in persian_to_english.items():
            number_str = number_str.replace(persian, english)
        
        # Map Persian number words to digits (basic mapping)
        word_to_number = {
            'اول': '1', 'یکم': '1', 'اولین': '1',
            'دوم': '2', 'دومین': '2',
            'سوم': '3', 'سومین': '3',
            'چهارم': '4', 'چهارمین': '4',
            'پنجم': '5', 'پنجمین': '5',
            'ششم': '6', 'ششمین': '6',
            'هفتم': '7', 'هفتمین': '7',
            'هشتم': '8', 'هشتمین': '8',
            'نهم': '9', 'نهمین': '9',
            'دهم': '10', 'دهمین': '10'
        }
        
        # Check if it's a word number
        normalized = number_str.strip()
        if normalized in word_to_number:
            return word_to_number[normalized]
        
        return normalized
    
    def extract_structure_summary(self, doc: LegalDocument) -> Dict:
        """
        Extract a summary of the document structure.
        
        Args:
            doc: Parsed LegalDocument
            
        Returns:
            Dictionary with structure summary
        """
        total_articles = len(doc.standalone_articles)
        total_notes = 0
        total_clauses = 0
        
        # Count from chapters
        for chapter in doc.chapters:
            total_articles += len(chapter.articles)
            for article in chapter.articles:
                total_notes += len(article.notes)
                total_clauses += len(article.clauses)
        
        # Count from standalone articles
        for article in doc.standalone_articles:
            total_notes += len(article.notes)
            total_clauses += len(article.clauses)
        
        return {
            'title': doc.title,
            'law_name': doc.law_name,
            'approval_date': doc.approval_date,
            'num_chapters': len(doc.chapters),
            'num_articles': total_articles,
            'num_notes': total_notes,
            'num_clauses': total_clauses
        }