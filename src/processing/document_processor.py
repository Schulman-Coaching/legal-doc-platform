"""
Legal Document Processing Pipeline
==================================
Microservices pipeline for document processing including OCR,
text extraction, entity recognition, and legal-specific parsing.
"""

import asyncio
import hashlib
import io
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import aiofiles
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class ProcessingStage(str, Enum):
    """Document processing stages."""
    FORMAT_CONVERSION = "format_conversion"
    OCR = "ocr"
    TEXT_CLEANING = "text_cleaning"
    METADATA_EXTRACTION = "metadata_extraction"
    ENTITY_EXTRACTION = "entity_extraction"
    CLAUSE_DETECTION = "clause_detection"
    CITATION_EXTRACTION = "citation_extraction"
    LEGAL_PARSING = "legal_parsing"
    CLASSIFICATION = "classification"
    INDEXING = "indexing"


class EntityType(str, Enum):
    """Types of entities extracted from legal documents."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    MONEY = "money"
    CASE_NUMBER = "case_number"
    STATUTE = "statute"
    COURT = "court"
    JUDGE = "judge"
    ATTORNEY = "attorney"
    PARTY = "party"
    DOCKET_NUMBER = "docket_number"


class DocumentType(str, Enum):
    """Legal document types."""
    CONTRACT = "contract"
    BRIEF = "brief"
    MOTION = "motion"
    PLEADING = "pleading"
    DISCOVERY = "discovery"
    CORRESPONDENCE = "correspondence"
    MEMO = "memo"
    OPINION = "opinion"
    ORDER = "order"
    TRANSCRIPT = "transcript"
    EXHIBIT = "exhibit"
    AGREEMENT = "agreement"
    AMENDMENT = "amendment"
    WILL = "will"
    TRUST = "trust"
    DEED = "deed"
    PATENT = "patent"
    TRADEMARK = "trademark"
    OTHER = "other"


class PracticeArea(str, Enum):
    """Legal practice areas."""
    CORPORATE = "corporate"
    LITIGATION = "litigation"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    REAL_ESTATE = "real_estate"
    EMPLOYMENT = "employment"
    TAX = "tax"
    BANKRUPTCY = "bankruptcy"
    CRIMINAL = "criminal"
    FAMILY = "family"
    IMMIGRATION = "immigration"
    ENVIRONMENTAL = "environmental"
    HEALTHCARE = "healthcare"
    REGULATORY = "regulatory"
    ESTATES = "estates_trusts"
    OTHER = "other"


@dataclass
class ExtractedEntity:
    """Represents an extracted entity."""
    entity_type: EntityType
    value: str
    normalized_value: Optional[str] = None
    confidence: float = 0.0
    start_offset: int = 0
    end_offset: int = 0
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedClause:
    """Represents an extracted legal clause."""
    clause_type: str
    text: str
    start_offset: int
    end_offset: int
    confidence: float = 0.0
    risk_level: str = "low"  # low, medium, high
    standard: bool = True  # Is it a standard clause?
    notes: str = ""


@dataclass
class ExtractedCitation:
    """Represents an extracted legal citation."""
    citation_type: str  # case, statute, regulation
    raw_text: str
    normalized_citation: str
    volume: Optional[str] = None
    reporter: Optional[str] = None
    page: Optional[str] = None
    court: Optional[str] = None
    year: Optional[int] = None
    jurisdiction: Optional[str] = None
    is_valid: bool = True
    confidence: float = 0.0


@dataclass
class DocumentSection:
    """Represents a document section/structure element."""
    section_type: str  # title, heading, paragraph, list, table
    level: int  # Hierarchy level (1=top)
    title: Optional[str] = None
    text: str = ""
    start_offset: int = 0
    end_offset: int = 0
    children: list["DocumentSection"] = field(default_factory=list)


class ProcessingResult(BaseModel):
    """Result of document processing."""
    document_id: str
    stage: ProcessingStage
    success: bool
    error_message: Optional[str] = None
    processing_time_ms: int = 0
    output_data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentProcessingState(BaseModel):
    """Complete state of document processing."""
    document_id: str
    original_filename: str
    mime_type: str
    file_size: int
    current_stage: ProcessingStage
    completed_stages: list[ProcessingStage] = Field(default_factory=list)
    failed_stages: list[ProcessingStage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Extracted data
    raw_text: str = ""
    cleaned_text: str = ""
    page_count: int = 0
    word_count: int = 0
    language: str = "en"

    # Metadata
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None

    # Extracted entities
    entities: list[dict[str, Any]] = Field(default_factory=list)

    # Extracted clauses
    clauses: list[dict[str, Any]] = Field(default_factory=list)

    # Extracted citations
    citations: list[dict[str, Any]] = Field(default_factory=list)

    # Document structure
    sections: list[dict[str, Any]] = Field(default_factory=list)

    # Classification results
    document_type: Optional[DocumentType] = None
    practice_areas: list[PracticeArea] = Field(default_factory=list)
    document_type_confidence: float = 0.0

    # Processing metadata
    processing_errors: list[str] = Field(default_factory=list)
    total_processing_time_ms: int = 0


# ============================================================================
# Processing Pipeline Base
# ============================================================================

class ProcessingStageHandler(ABC):
    """Base class for processing stage handlers."""

    @property
    @abstractmethod
    def stage(self) -> ProcessingStage:
        """Return the processing stage this handler implements."""
        pass

    @abstractmethod
    async def process(
        self,
        state: DocumentProcessingState,
        file_path: Path,
    ) -> ProcessingResult:
        """
        Process the document for this stage.

        Args:
            state: Current document processing state
            file_path: Path to the document file

        Returns:
            ProcessingResult with stage output
        """
        pass


# ============================================================================
# Format Conversion Service
# ============================================================================

class FormatConversionHandler(ProcessingStageHandler):
    """Converts documents to standardized formats."""

    @property
    def stage(self) -> ProcessingStage:
        return ProcessingStage.FORMAT_CONVERSION

    async def process(
        self,
        state: DocumentProcessingState,
        file_path: Path,
    ) -> ProcessingResult:
        """Convert document to PDF/A format."""
        import time
        start_time = time.time()

        try:
            # In production, use Apache Tika or LibreOffice
            # This is a placeholder implementation

            output_data = {
                "converted_format": "pdf/a",
                "original_format": state.mime_type,
                "conversion_method": "tika",
            }

            # Simulate format detection and conversion
            if state.mime_type == "application/pdf":
                output_data["conversion_needed"] = False
            else:
                output_data["conversion_needed"] = True

            processing_time = int((time.time() - start_time) * 1000)

            return ProcessingResult(
                document_id=state.document_id,
                stage=self.stage,
                success=True,
                processing_time_ms=processing_time,
                output_data=output_data,
            )

        except Exception as e:
            logger.error(f"Format conversion failed: {e}")
            return ProcessingResult(
                document_id=state.document_id,
                stage=self.stage,
                success=False,
                error_message=str(e),
            )


# ============================================================================
# OCR Service
# ============================================================================

class OCRHandler(ProcessingStageHandler):
    """Extracts text from scanned documents using OCR."""

    def __init__(self, tesseract_path: Optional[str] = None):
        self.tesseract_path = tesseract_path

    @property
    def stage(self) -> ProcessingStage:
        return ProcessingStage.OCR

    async def process(
        self,
        state: DocumentProcessingState,
        file_path: Path,
    ) -> ProcessingResult:
        """Perform OCR on document."""
        import time
        start_time = time.time()

        try:
            # Check if OCR is needed
            if state.raw_text and len(state.raw_text) > 100:
                return ProcessingResult(
                    document_id=state.document_id,
                    stage=self.stage,
                    success=True,
                    processing_time_ms=0,
                    output_data={"ocr_needed": False},
                )

            # In production, use Tesseract or Google Vision API
            # This is a placeholder implementation
            extracted_text = await self._perform_ocr(file_path)

            output_data = {
                "ocr_needed": True,
                "ocr_engine": "tesseract",
                "confidence_score": 0.95,
                "extracted_text": extracted_text,
                "languages_detected": ["en"],
            }

            processing_time = int((time.time() - start_time) * 1000)

            return ProcessingResult(
                document_id=state.document_id,
                stage=self.stage,
                success=True,
                processing_time_ms=processing_time,
                output_data=output_data,
            )

        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ProcessingResult(
                document_id=state.document_id,
                stage=self.stage,
                success=False,
                error_message=str(e),
            )

    async def _perform_ocr(self, file_path: Path) -> str:
        """
        Perform OCR on the document.
        In production, integrate with Tesseract or cloud OCR.
        """
        # Placeholder - in production use pytesseract
        # import pytesseract
        # from PIL import Image
        # image = Image.open(file_path)
        # return pytesseract.image_to_string(image)
        return ""


# ============================================================================
# Text Cleaning Service
# ============================================================================

class TextCleaningHandler(ProcessingStageHandler):
    """Cleans and normalizes extracted text."""

    # Legal-specific corrections
    LEGAL_CORRECTIONS = {
        r'\bplaintif\b': 'plaintiff',
        r'\bdefendanl\b': 'defendant',
        r'\bhereinafler\b': 'hereinafter',
        r'\bwherefore\b': 'wherefore',
        r'\bjudgemenl\b': 'judgment',
    }

    @property
    def stage(self) -> ProcessingStage:
        return ProcessingStage.TEXT_CLEANING

    async def process(
        self,
        state: DocumentProcessingState,
        file_path: Path,
    ) -> ProcessingResult:
        """Clean and normalize document text."""
        import time
        start_time = time.time()

        try:
            raw_text = state.raw_text
            if not raw_text:
                # Try to extract text from file
                raw_text = await self._extract_text(file_path, state.mime_type)

            # Apply cleaning steps
            cleaned_text = await self._clean_text(raw_text)

            output_data = {
                "original_length": len(raw_text),
                "cleaned_length": len(cleaned_text),
                "word_count": len(cleaned_text.split()),
                "corrections_applied": [],
            }

            processing_time = int((time.time() - start_time) * 1000)

            return ProcessingResult(
                document_id=state.document_id,
                stage=self.stage,
                success=True,
                processing_time_ms=processing_time,
                output_data={
                    **output_data,
                    "cleaned_text": cleaned_text,
                },
            )

        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")
            return ProcessingResult(
                document_id=state.document_id,
                stage=self.stage,
                success=False,
                error_message=str(e),
            )

    async def _extract_text(self, file_path: Path, mime_type: str) -> str:
        """Extract text from document based on type."""
        # In production, use Apache Tika
        # This is a placeholder
        if mime_type == "text/plain":
            async with aiofiles.open(file_path, 'r') as f:
                return await f.read()
        return ""

    async def _clean_text(self, text: str) -> str:
        """Apply text cleaning rules."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Normalize line breaks
        text = re.sub(r'(\r\n|\r|\n)+', '\n', text)

        # Fix common OCR errors
        for pattern, replacement in self.LEGAL_CORRECTIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Standardize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        # Remove header/footer artifacts
        text = re.sub(r'Page \d+ of \d+', '', text)

        # Normalize dates
        # Convert MM/DD/YYYY to standardized format
        text = re.sub(
            r'(\d{1,2})/(\d{1,2})/(\d{4})',
            r'\3-\1-\2',
            text
        )

        return text.strip()


# ============================================================================
# Metadata Extraction Service
# ============================================================================

class MetadataExtractionHandler(ProcessingStageHandler):
    """Extracts document metadata."""

    @property
    def stage(self) -> ProcessingStage:
        return ProcessingStage.METADATA_EXTRACTION

    async def process(
        self,
        state: DocumentProcessingState,
        file_path: Path,
    ) -> ProcessingResult:
        """Extract metadata from document."""
        import time
        start_time = time.time()

        try:
            # In production, use Apache Tika for comprehensive metadata
            metadata = await self._extract_metadata(file_path)

            processing_time = int((time.time() - start_time) * 1000)

            return ProcessingResult(
                document_id=state.document_id,
                stage=self.stage,
                success=True,
                processing_time_ms=processing_time,
                output_data=metadata,
            )

        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return ProcessingResult(
                document_id=state.document_id,
                stage=self.stage,
                success=False,
                error_message=str(e),
            )

    async def _extract_metadata(self, file_path: Path) -> dict[str, Any]:
        """Extract metadata from file."""
        # Placeholder - in production use Apache Tika
        stat = file_path.stat()

        return {
            "file_size": stat.st_size,
            "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "title": None,
            "author": None,
            "subject": None,
            "keywords": [],
            "page_count": 1,
            "has_digital_signature": False,
        }


# ============================================================================
# Entity Extraction Service
# ============================================================================

class EntityExtractionHandler(ProcessingStageHandler):
    """Extracts named entities from legal documents."""

    # Legal-specific entity patterns
    CASE_NUMBER_PATTERN = re.compile(
        r'\b(\d{1,2}[-:\s]?(?:cv|cr|civ|crim|mc|md|bk|ap|mj|po|sw)[-:\s]?\d{2,8}(?:[-:\s][A-Z]{1,3})?)\b',
        re.IGNORECASE
    )

    DOCKET_PATTERN = re.compile(
        r'\b(?:No\.|Docket\s+(?:No\.?)?|Case\s+(?:No\.?)?)\s*[:.]?\s*([A-Z0-9\-:]+)\b',
        re.IGNORECASE
    )

    MONEY_PATTERN = re.compile(
        r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand))?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD)\b',
        re.IGNORECASE
    )

    DATE_PATTERN = re.compile(
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|'
        r'\b\d{1,2}/\d{1,2}/\d{4}\b|'
        r'\b\d{4}-\d{2}-\d{2}\b',
        re.IGNORECASE
    )

    COURT_PATTERN = re.compile(
        r'\b(?:United\s+States\s+)?(?:Supreme|District|Circuit|Appeals?|Bankruptcy|Tax|Claims?)\s+Court\s+(?:of\s+)?(?:the\s+)?(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b|'
        r'\b(?:Superior|Municipal|Family|Probate|Surrogate(?:\'?s)?)\s+Court\b',
        re.IGNORECASE
    )

    @property
    def stage(self) -> ProcessingStage:
        return ProcessingStage.ENTITY_EXTRACTION

    async def process(
        self,
        state: DocumentProcessingState,
        file_path: Path,
    ) -> ProcessingResult:
        """Extract entities from document text."""
        import time
        start_time = time.time()

        try:
            text = state.cleaned_text or state.raw_text
            if not text:
                return ProcessingResult(
                    document_id=state.document_id,
                    stage=self.stage,
                    success=True,
                    processing_time_ms=0,
                    output_data={"entities": [], "message": "No text to process"},
                )

            entities = await self._extract_entities(text)

            processing_time = int((time.time() - start_time) * 1000)

            return ProcessingResult(
                document_id=state.document_id,
                stage=self.stage,
                success=True,
                processing_time_ms=processing_time,
                output_data={
                    "entities": [self._entity_to_dict(e) for e in entities],
                    "entity_count": len(entities),
                    "entity_types": list(set(e.entity_type.value for e in entities)),
                },
            )

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return ProcessingResult(
                document_id=state.document_id,
                stage=self.stage,
                success=False,
                error_message=str(e),
            )

    async def _extract_entities(self, text: str) -> list[ExtractedEntity]:
        """Extract all entities from text."""
        entities = []

        # Extract case numbers
        for match in self.CASE_NUMBER_PATTERN.finditer(text):
            entities.append(ExtractedEntity(
                entity_type=EntityType.CASE_NUMBER,
                value=match.group(1),
                normalized_value=match.group(1).upper().replace(' ', '-'),
                confidence=0.9,
                start_offset=match.start(),
                end_offset=match.end(),
                context=text[max(0, match.start()-50):min(len(text), match.end()+50)],
            ))

        # Extract docket numbers
        for match in self.DOCKET_PATTERN.finditer(text):
            entities.append(ExtractedEntity(
                entity_type=EntityType.DOCKET_NUMBER,
                value=match.group(1),
                confidence=0.85,
                start_offset=match.start(),
                end_offset=match.end(),
            ))

        # Extract monetary amounts
        for match in self.MONEY_PATTERN.finditer(text):
            entities.append(ExtractedEntity(
                entity_type=EntityType.MONEY,
                value=match.group(),
                normalized_value=self._normalize_money(match.group()),
                confidence=0.95,
                start_offset=match.start(),
                end_offset=match.end(),
            ))

        # Extract dates
        for match in self.DATE_PATTERN.finditer(text):
            entities.append(ExtractedEntity(
                entity_type=EntityType.DATE,
                value=match.group(),
                normalized_value=self._normalize_date(match.group()),
                confidence=0.9,
                start_offset=match.start(),
                end_offset=match.end(),
            ))

        # Extract courts
        for match in self.COURT_PATTERN.finditer(text):
            entities.append(ExtractedEntity(
                entity_type=EntityType.COURT,
                value=match.group(),
                confidence=0.85,
                start_offset=match.start(),
                end_offset=match.end(),
            ))

        # In production, also use spaCy/Stanford NER for:
        # - Person names
        # - Organizations
        # - Locations

        return entities

    def _normalize_money(self, value: str) -> str:
        """Normalize monetary amount to decimal."""
        # Remove $ and commas
        cleaned = re.sub(r'[$,]', '', value.lower())
        # Handle million/billion
        if 'billion' in cleaned:
            cleaned = re.sub(r'\s*billion.*', '', cleaned)
            try:
                return str(float(cleaned) * 1_000_000_000)
            except ValueError:
                return value
        elif 'million' in cleaned:
            cleaned = re.sub(r'\s*million.*', '', cleaned)
            try:
                return str(float(cleaned) * 1_000_000)
            except ValueError:
                return value
        return cleaned.strip()

    def _normalize_date(self, value: str) -> str:
        """Normalize date to ISO format."""
        # This is simplified - use dateutil.parser in production
        return value

    def _entity_to_dict(self, entity: ExtractedEntity) -> dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "type": entity.entity_type.value,
            "value": entity.value,
            "normalized_value": entity.normalized_value,
            "confidence": entity.confidence,
            "start_offset": entity.start_offset,
            "end_offset": entity.end_offset,
            "context": entity.context,
            "metadata": entity.metadata,
        }


# ============================================================================
# Clause Detection Service
# ============================================================================

class ClauseDetectionHandler(ProcessingStageHandler):
    """Detects and extracts legal clauses from contracts."""

    # Common clause patterns
    CLAUSE_PATTERNS = {
        "indemnification": [
            r'indemni(?:fy|fication|fied)',
            r'hold\s+harmless',
            r'defend\s+and\s+indemnify',
        ],
        "limitation_of_liability": [
            r'limit(?:ation)?\s+(?:of\s+)?liability',
            r'consequential\s+damages',
            r'in\s+no\s+event\s+shall.*liable',
        ],
        "confidentiality": [
            r'confidential(?:ity)?',
            r'non[\s-]?disclosure',
            r'proprietary\s+information',
        ],
        "termination": [
            r'terminat(?:e|ion)',
            r'cancel(?:lation)?',
            r'upon\s+(?:\d+|thirty|sixty|ninety)\s+days?\s+(?:prior\s+)?(?:written\s+)?notice',
        ],
        "governing_law": [
            r'govern(?:ing|ed\s+by)\s+(?:the\s+)?law(?:s)?',
            r'choice\s+of\s+law',
            r'jurisdiction',
        ],
        "arbitration": [
            r'arbitrat(?:ion|e|or)',
            r'dispute\s+resolution',
            r'binding\s+arbitration',
        ],
        "force_majeure": [
            r'force\s+majeure',
            r'act(?:s)?\s+of\s+god',
            r'beyond\s+(?:the\s+)?(?:reasonable\s+)?control',
        ],
        "assignment": [
            r'assign(?:ment|able)?',
            r'transfer(?:ability)?',
            r'(?:may|shall)\s+not\s+assign',
        ],
        "severability": [
            r'severab(?:ility|le)',
            r'invalid\s+or\s+unenforceable',
            r'remaining\s+provisions',
        ],
        "entire_agreement": [
            r'entire\s+agreement',
            r'complete\s+agreement',
            r'supersedes?\s+(?:all\s+)?prior',
            r'integration\s+clause',
        ],
        "non_compete": [
            r'non[\s-]?compet(?:e|ition)',
            r'restrictive\s+covenant',
            r'compete\s+(?:directly\s+or\s+indirectly)',
        ],
        "intellectual_property": [
            r'intellectual\s+property',
            r'patent(?:s)?',
            r'trademark(?:s)?',
            r'copyright(?:s)?',
            r'trade\s+secret(?:s)?',
        ],
        "warranty": [
            r'warrant(?:y|ies)',
            r'as[\s-]is',
            r'merchantability',
            r'fitness\s+for\s+(?:a\s+)?particular\s+purpose',
        ],
        "payment_terms": [
            r'payment\s+term(?:s)?',
            r'net\s+\d+',
            r'invoice(?:s)?',
            r'interest\s+rate',
        ],
    }

    # High-risk clause indicators
    HIGH_RISK_INDICATORS = [
        r'unlimited\s+liability',
        r'sole\s+discretion',
        r'waive(?:s|r)?\s+(?:all\s+)?(?:rights?|claims?)',
        r'perpetual',
        r'irrevocable',
        r'automatic\s+renewal',
    ]

    @property
    def stage(self) -> ProcessingStage:
        return ProcessingStage.CLAUSE_DETECTION

    async def process(
        self,
        state: DocumentProcessingState,
        file_path: Path,
    ) -> ProcessingResult:
        """Detect clauses in document."""
        import time
        start_time = time.time()

        try:
            text = state.cleaned_text or state.raw_text
            if not text:
                return ProcessingResult(
                    document_id=state.document_id,
                    stage=self.stage,
                    success=True,
                    processing_time_ms=0,
                    output_data={"clauses": [], "message": "No text to process"},
                )

            clauses = await self._detect_clauses(text)

            processing_time = int((time.time() - start_time) * 1000)

            return ProcessingResult(
                document_id=state.document_id,
                stage=self.stage,
                success=True,
                processing_time_ms=processing_time,
                output_data={
                    "clauses": [self._clause_to_dict(c) for c in clauses],
                    "clause_count": len(clauses),
                    "clause_types": list(set(c.clause_type for c in clauses)),
                    "high_risk_count": sum(1 for c in clauses if c.risk_level == "high"),
                },
            )

        except Exception as e:
            logger.error(f"Clause detection failed: {e}")
            return ProcessingResult(
                document_id=state.document_id,
                stage=self.stage,
                success=False,
                error_message=str(e),
            )

    async def _detect_clauses(self, text: str) -> list[ExtractedClause]:
        """Detect all clauses in text."""
        clauses = []

        # Split text into paragraphs/sections for analysis
        paragraphs = re.split(r'\n\s*\n', text)

        for para_idx, paragraph in enumerate(paragraphs):
            for clause_type, patterns in self.CLAUSE_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, paragraph, re.IGNORECASE):
                        # Found a clause
                        risk_level = self._assess_risk(paragraph)

                        clause = ExtractedClause(
                            clause_type=clause_type,
                            text=paragraph[:500],  # Truncate for storage
                            start_offset=text.find(paragraph),
                            end_offset=text.find(paragraph) + len(paragraph),
                            confidence=0.8,
                            risk_level=risk_level,
                            standard=self._is_standard_clause(paragraph, clause_type),
                        )
                        clauses.append(clause)
                        break  # Only add clause once per type per paragraph

        return clauses

    def _assess_risk(self, text: str) -> str:
        """Assess risk level of a clause."""
        risk_count = 0
        for indicator in self.HIGH_RISK_INDICATORS:
            if re.search(indicator, text, re.IGNORECASE):
                risk_count += 1

        if risk_count >= 2:
            return "high"
        elif risk_count == 1:
            return "medium"
        return "low"

    def _is_standard_clause(self, text: str, clause_type: str) -> bool:
        """Check if clause appears to be standard/boilerplate."""
        # In production, compare against clause library
        # This is a simplified heuristic
        text_lower = text.lower()

        # Non-standard indicators
        non_standard = [
            'notwithstanding anything to the contrary',
            'except as otherwise',
            'modified',
            'amended',
        ]

        for indicator in non_standard:
            if indicator in text_lower:
                return False

        return True

    def _clause_to_dict(self, clause: ExtractedClause) -> dict[str, Any]:
        """Convert clause to dictionary."""
        return {
            "type": clause.clause_type,
            "text": clause.text,
            "start_offset": clause.start_offset,
            "end_offset": clause.end_offset,
            "confidence": clause.confidence,
            "risk_level": clause.risk_level,
            "standard": clause.standard,
            "notes": clause.notes,
        }


# ============================================================================
# Citation Extraction Service
# ============================================================================

class CitationExtractionHandler(ProcessingStageHandler):
    """Extracts legal citations from documents."""

    # Citation patterns
    CASE_CITATION_PATTERN = re.compile(
        r'(\d+)\s+([A-Z][a-z]*\.?(?:\s+[A-Z][a-z]*\.?)*)\s+(\d+)(?:\s*,\s*(\d+))?\s*\(([^)]+)\)',
        re.IGNORECASE
    )

    USC_PATTERN = re.compile(
        r'(\d+)\s+U\.?S\.?C\.?\s+(?:§\s*)?(\d+(?:[a-z])?(?:\(\w+\))*)',
        re.IGNORECASE
    )

    CFR_PATTERN = re.compile(
        r'(\d+)\s+C\.?F\.?R\.?\s+(?:§\s*)?(\d+(?:\.\d+)?)',
        re.IGNORECASE
    )

    @property
    def stage(self) -> ProcessingStage:
        return ProcessingStage.CITATION_EXTRACTION

    async def process(
        self,
        state: DocumentProcessingState,
        file_path: Path,
    ) -> ProcessingResult:
        """Extract citations from document."""
        import time
        start_time = time.time()

        try:
            text = state.cleaned_text or state.raw_text
            if not text:
                return ProcessingResult(
                    document_id=state.document_id,
                    stage=self.stage,
                    success=True,
                    processing_time_ms=0,
                    output_data={"citations": [], "message": "No text to process"},
                )

            citations = await self._extract_citations(text)

            processing_time = int((time.time() - start_time) * 1000)

            return ProcessingResult(
                document_id=state.document_id,
                stage=self.stage,
                success=True,
                processing_time_ms=processing_time,
                output_data={
                    "citations": [self._citation_to_dict(c) for c in citations],
                    "citation_count": len(citations),
                    "citation_types": list(set(c.citation_type for c in citations)),
                },
            )

        except Exception as e:
            logger.error(f"Citation extraction failed: {e}")
            return ProcessingResult(
                document_id=state.document_id,
                stage=self.stage,
                success=False,
                error_message=str(e),
            )

    async def _extract_citations(self, text: str) -> list[ExtractedCitation]:
        """Extract all citations from text."""
        citations = []

        # Extract case citations
        for match in self.CASE_CITATION_PATTERN.finditer(text):
            volume, reporter, page, pinpoint, court_year = match.groups()
            year = self._extract_year(court_year)
            court = self._extract_court(court_year)

            citations.append(ExtractedCitation(
                citation_type="case",
                raw_text=match.group(),
                normalized_citation=f"{volume} {reporter} {page}",
                volume=volume,
                reporter=reporter,
                page=page,
                court=court,
                year=year,
                confidence=0.85,
            ))

        # Extract USC citations
        for match in self.USC_PATTERN.finditer(text):
            title, section = match.groups()
            citations.append(ExtractedCitation(
                citation_type="statute",
                raw_text=match.group(),
                normalized_citation=f"{title} U.S.C. § {section}",
                confidence=0.9,
            ))

        # Extract CFR citations
        for match in self.CFR_PATTERN.finditer(text):
            title, section = match.groups()
            citations.append(ExtractedCitation(
                citation_type="regulation",
                raw_text=match.group(),
                normalized_citation=f"{title} C.F.R. § {section}",
                confidence=0.9,
            ))

        return citations

    def _extract_year(self, court_year: str) -> Optional[int]:
        """Extract year from court/year parenthetical."""
        year_match = re.search(r'\d{4}', court_year)
        return int(year_match.group()) if year_match else None

    def _extract_court(self, court_year: str) -> Optional[str]:
        """Extract court from court/year parenthetical."""
        court = re.sub(r'\d{4}', '', court_year).strip()
        return court if court else None

    def _citation_to_dict(self, citation: ExtractedCitation) -> dict[str, Any]:
        """Convert citation to dictionary."""
        return {
            "type": citation.citation_type,
            "raw_text": citation.raw_text,
            "normalized": citation.normalized_citation,
            "volume": citation.volume,
            "reporter": citation.reporter,
            "page": citation.page,
            "court": citation.court,
            "year": citation.year,
            "jurisdiction": citation.jurisdiction,
            "is_valid": citation.is_valid,
            "confidence": citation.confidence,
        }


# ============================================================================
# Legal Parser Service
# ============================================================================

class LegalParserHandler(ProcessingStageHandler):
    """Parses document structure and legal elements."""

    @property
    def stage(self) -> ProcessingStage:
        return ProcessingStage.LEGAL_PARSING

    async def process(
        self,
        state: DocumentProcessingState,
        file_path: Path,
    ) -> ProcessingResult:
        """Parse document structure."""
        import time
        start_time = time.time()

        try:
            text = state.cleaned_text or state.raw_text
            if not text:
                return ProcessingResult(
                    document_id=state.document_id,
                    stage=self.stage,
                    success=True,
                    processing_time_ms=0,
                    output_data={"sections": [], "message": "No text to process"},
                )

            sections = await self._parse_structure(text)
            defined_terms = await self._extract_defined_terms(text)

            processing_time = int((time.time() - start_time) * 1000)

            return ProcessingResult(
                document_id=state.document_id,
                stage=self.stage,
                success=True,
                processing_time_ms=processing_time,
                output_data={
                    "sections": [self._section_to_dict(s) for s in sections],
                    "section_count": len(sections),
                    "defined_terms": defined_terms,
                    "defined_term_count": len(defined_terms),
                },
            )

        except Exception as e:
            logger.error(f"Legal parsing failed: {e}")
            return ProcessingResult(
                document_id=state.document_id,
                stage=self.stage,
                success=False,
                error_message=str(e),
            )

    async def _parse_structure(self, text: str) -> list[DocumentSection]:
        """Parse document into hierarchical sections."""
        sections = []

        # Pattern for section headers
        section_pattern = re.compile(
            r'^(?:'
            r'(?:ARTICLE|SECTION|Part)\s+(?:[IVXLCDM]+|\d+)|'  # ARTICLE I, SECTION 1
            r'(?:\d+(?:\.\d+)*\.?\s+[A-Z])|'  # 1. Title, 1.1 Subtitle
            r'(?:[A-Z][A-Z\s]+:)'  # ALL CAPS HEADER:
            r')',
            re.MULTILINE
        )

        lines = text.split('\n')
        current_section = None

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            if section_pattern.match(line):
                # New section header
                if current_section:
                    sections.append(current_section)

                level = self._determine_level(line)
                current_section = DocumentSection(
                    section_type="heading",
                    level=level,
                    title=line,
                    start_offset=text.find(line),
                )
            elif current_section:
                # Add to current section
                current_section.text += line + "\n"
            else:
                # Preamble or unstructured text
                sections.append(DocumentSection(
                    section_type="paragraph",
                    level=0,
                    text=line,
                    start_offset=text.find(line),
                ))

        if current_section:
            sections.append(current_section)

        return sections

    def _determine_level(self, header: str) -> int:
        """Determine hierarchy level of a section header."""
        if re.match(r'^ARTICLE', header, re.IGNORECASE):
            return 1
        elif re.match(r'^SECTION', header, re.IGNORECASE):
            return 2
        elif re.match(r'^\d+\.\d+\.\d+', header):
            return 4
        elif re.match(r'^\d+\.\d+', header):
            return 3
        elif re.match(r'^\d+\.', header):
            return 2
        return 2

    async def _extract_defined_terms(self, text: str) -> list[dict[str, str]]:
        """Extract defined terms from document."""
        defined_terms = []

        # Pattern for defined terms
        patterns = [
            r'"([^"]+)"\s+(?:means|shall mean|refers to|is defined as)',
            r'"([^"]+)"\s+\((?:the\s+)?"[^"]+"\)',
            r'(?:hereinafter|as used herein),?\s+"([^"]+)"',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                term = match.group(1)
                # Get context around definition
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 200)
                context = text[start:end]

                defined_terms.append({
                    "term": term,
                    "context": context,
                })

        return defined_terms

    def _section_to_dict(self, section: DocumentSection) -> dict[str, Any]:
        """Convert section to dictionary."""
        return {
            "type": section.section_type,
            "level": section.level,
            "title": section.title,
            "text": section.text[:500] if section.text else "",
            "start_offset": section.start_offset,
            "end_offset": section.end_offset,
        }


# ============================================================================
# Document Classification Service
# ============================================================================

class DocumentClassificationHandler(ProcessingStageHandler):
    """Classifies documents by type and practice area."""

    # Keywords for document type classification
    DOCTYPE_KEYWORDS = {
        DocumentType.CONTRACT: [
            'agreement', 'contract', 'parties', 'whereas', 'hereby',
            'consideration', 'covenant', 'execute',
        ],
        DocumentType.BRIEF: [
            'brief', 'argument', 'appellant', 'appellee', 'court of appeals',
            'points and authorities', 'statement of facts',
        ],
        DocumentType.MOTION: [
            'motion', 'movant', 'relief', 'court is requested',
            'hereby moves', 'grant this motion',
        ],
        DocumentType.PLEADING: [
            'complaint', 'answer', 'counterclaim', 'cross-claim',
            'prayer for relief', 'cause of action',
        ],
        DocumentType.DISCOVERY: [
            'interrogator', 'request for production', 'deposition',
            'subpoena', 'request for admission',
        ],
        DocumentType.OPINION: [
            'opinion', 'this court holds', 'affirmed', 'reversed',
            'delivered the opinion', 'judgment of the court',
        ],
        DocumentType.ORDER: [
            'order', 'it is hereby ordered', 'so ordered',
            'the court orders', 'pursuant to this order',
        ],
    }

    # Keywords for practice area classification
    PRACTICE_KEYWORDS = {
        PracticeArea.CORPORATE: [
            'corporation', 'shareholders', 'board of directors', 'merger',
            'acquisition', 'bylaws', 'articles of incorporation',
        ],
        PracticeArea.LITIGATION: [
            'plaintiff', 'defendant', 'damages', 'lawsuit', 'trial',
            'verdict', 'judgment', 'discovery',
        ],
        PracticeArea.INTELLECTUAL_PROPERTY: [
            'patent', 'trademark', 'copyright', 'trade secret',
            'infringement', 'licensing', 'intellectual property',
        ],
        PracticeArea.REAL_ESTATE: [
            'deed', 'mortgage', 'lease', 'tenant', 'landlord',
            'property', 'easement', 'title', 'conveyance',
        ],
        PracticeArea.EMPLOYMENT: [
            'employment', 'employee', 'employer', 'termination',
            'compensation', 'benefits', 'discrimination', 'harassment',
        ],
        PracticeArea.BANKRUPTCY: [
            'bankruptcy', 'debtor', 'creditor', 'chapter 7', 'chapter 11',
            'reorganization', 'discharge', 'trustee',
        ],
    }

    @property
    def stage(self) -> ProcessingStage:
        return ProcessingStage.CLASSIFICATION

    async def process(
        self,
        state: DocumentProcessingState,
        file_path: Path,
    ) -> ProcessingResult:
        """Classify document type and practice area."""
        import time
        start_time = time.time()

        try:
            text = state.cleaned_text or state.raw_text
            if not text:
                return ProcessingResult(
                    document_id=state.document_id,
                    stage=self.stage,
                    success=True,
                    processing_time_ms=0,
                    output_data={
                        "document_type": DocumentType.OTHER.value,
                        "practice_areas": [],
                        "message": "No text to process"
                    },
                )

            # Classify document type
            doc_type, doc_confidence = await self._classify_document_type(text)

            # Classify practice areas
            practice_areas = await self._classify_practice_areas(text)

            processing_time = int((time.time() - start_time) * 1000)

            return ProcessingResult(
                document_id=state.document_id,
                stage=self.stage,
                success=True,
                processing_time_ms=processing_time,
                output_data={
                    "document_type": doc_type.value,
                    "document_type_confidence": doc_confidence,
                    "practice_areas": [pa.value for pa in practice_areas],
                },
            )

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return ProcessingResult(
                document_id=state.document_id,
                stage=self.stage,
                success=False,
                error_message=str(e),
            )

    async def _classify_document_type(
        self,
        text: str,
    ) -> tuple[DocumentType, float]:
        """Classify document type using keyword matching."""
        text_lower = text.lower()
        scores = {}

        for doc_type, keywords in self.DOCTYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[doc_type] = score / len(keywords)

        if not scores:
            return DocumentType.OTHER, 0.0

        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]

        # In production, use a fine-tuned BERT model for better accuracy
        return best_type, min(confidence * 2, 1.0)  # Scale confidence

    async def _classify_practice_areas(
        self,
        text: str,
    ) -> list[PracticeArea]:
        """Classify practice areas (can be multiple)."""
        text_lower = text.lower()
        results = []

        for practice_area, keywords in self.PRACTICE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score >= 2:  # Require at least 2 keyword matches
                results.append(practice_area)

        return results or [PracticeArea.OTHER]


# ============================================================================
# Processing Pipeline Orchestrator
# ============================================================================

class ProcessingPipeline:
    """Orchestrates the document processing pipeline."""

    def __init__(self):
        self.handlers: dict[ProcessingStage, ProcessingStageHandler] = {
            ProcessingStage.FORMAT_CONVERSION: FormatConversionHandler(),
            ProcessingStage.OCR: OCRHandler(),
            ProcessingStage.TEXT_CLEANING: TextCleaningHandler(),
            ProcessingStage.METADATA_EXTRACTION: MetadataExtractionHandler(),
            ProcessingStage.ENTITY_EXTRACTION: EntityExtractionHandler(),
            ProcessingStage.CLAUSE_DETECTION: ClauseDetectionHandler(),
            ProcessingStage.CITATION_EXTRACTION: CitationExtractionHandler(),
            ProcessingStage.LEGAL_PARSING: LegalParserHandler(),
            ProcessingStage.CLASSIFICATION: DocumentClassificationHandler(),
        }

        # Define pipeline order
        self.pipeline_order = [
            ProcessingStage.FORMAT_CONVERSION,
            ProcessingStage.OCR,
            ProcessingStage.TEXT_CLEANING,
            ProcessingStage.METADATA_EXTRACTION,
            ProcessingStage.ENTITY_EXTRACTION,
            ProcessingStage.CLAUSE_DETECTION,
            ProcessingStage.CITATION_EXTRACTION,
            ProcessingStage.LEGAL_PARSING,
            ProcessingStage.CLASSIFICATION,
        ]

    async def process_document(
        self,
        document_id: str,
        file_path: Path,
        filename: str,
        mime_type: str,
        file_size: int,
    ) -> DocumentProcessingState:
        """
        Process a document through the entire pipeline.

        Args:
            document_id: Unique document identifier
            file_path: Path to the document file
            filename: Original filename
            mime_type: MIME type of the document
            file_size: File size in bytes

        Returns:
            Complete processing state
        """
        import time
        start_time = time.time()

        state = DocumentProcessingState(
            document_id=document_id,
            original_filename=filename,
            mime_type=mime_type,
            file_size=file_size,
            current_stage=self.pipeline_order[0],
        )

        for stage in self.pipeline_order:
            state.current_stage = stage
            handler = self.handlers.get(stage)

            if not handler:
                logger.warning(f"No handler for stage {stage}")
                continue

            logger.info(f"Processing {document_id} - Stage: {stage.value}")

            try:
                result = await handler.process(state, file_path)

                if result.success:
                    state.completed_stages.append(stage)
                    state = self._apply_result(state, result)
                else:
                    state.failed_stages.append(stage)
                    state.processing_errors.append(
                        f"{stage.value}: {result.error_message}"
                    )
                    # Continue with next stage on failure

                state.total_processing_time_ms += result.processing_time_ms

            except Exception as e:
                logger.error(f"Stage {stage} failed with exception: {e}")
                state.failed_stages.append(stage)
                state.processing_errors.append(f"{stage.value}: {str(e)}")

        state.updated_at = datetime.utcnow()
        total_time = int((time.time() - start_time) * 1000)
        state.total_processing_time_ms = total_time

        logger.info(
            f"Document {document_id} processing complete. "
            f"Completed: {len(state.completed_stages)}, "
            f"Failed: {len(state.failed_stages)}, "
            f"Time: {total_time}ms"
        )

        return state

    def _apply_result(
        self,
        state: DocumentProcessingState,
        result: ProcessingResult,
    ) -> DocumentProcessingState:
        """Apply processing result to state."""
        data = result.output_data

        if result.stage == ProcessingStage.OCR:
            if data.get("extracted_text"):
                state.raw_text = data["extracted_text"]

        elif result.stage == ProcessingStage.TEXT_CLEANING:
            if data.get("cleaned_text"):
                state.cleaned_text = data["cleaned_text"]
                state.word_count = data.get("word_count", 0)

        elif result.stage == ProcessingStage.METADATA_EXTRACTION:
            state.title = data.get("title")
            state.author = data.get("author")
            state.page_count = data.get("page_count", 1)

        elif result.stage == ProcessingStage.ENTITY_EXTRACTION:
            state.entities = data.get("entities", [])

        elif result.stage == ProcessingStage.CLAUSE_DETECTION:
            state.clauses = data.get("clauses", [])

        elif result.stage == ProcessingStage.CITATION_EXTRACTION:
            state.citations = data.get("citations", [])

        elif result.stage == ProcessingStage.LEGAL_PARSING:
            state.sections = data.get("sections", [])

        elif result.stage == ProcessingStage.CLASSIFICATION:
            doc_type = data.get("document_type")
            if doc_type:
                state.document_type = DocumentType(doc_type)
            state.document_type_confidence = data.get("document_type_confidence", 0)
            practice_areas = data.get("practice_areas", [])
            state.practice_areas = [PracticeArea(pa) for pa in practice_areas]

        return state


# ============================================================================
# Kafka Consumer for Processing
# ============================================================================

async def process_documents_from_kafka():
    """
    Consume documents from Kafka and process them.
    This is the main entry point for the processing service.
    """
    from aiokafka import AIOKafkaConsumer
    import json
    import os

    consumer = AIOKafkaConsumer(
        'legal-documents-ingested',
        bootstrap_servers=os.getenv('KAFKA_SERVERS', 'localhost:9092'),
        group_id='document-processor',
        auto_offset_reset='earliest',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    )

    pipeline = ProcessingPipeline()
    storage_path = Path(os.getenv('STORAGE_PATH', '/data/documents'))

    await consumer.start()
    logger.info("Started document processing consumer")

    try:
        async for msg in consumer:
            document_data = msg.value
            document_id = document_data['document_id']

            logger.info(f"Received document {document_id} for processing")

            # Get file path
            file_path = storage_path / f"{document_id}.enc"

            if not file_path.exists():
                logger.error(f"File not found for document {document_id}")
                continue

            # Process document
            state = await pipeline.process_document(
                document_id=document_id,
                file_path=file_path,
                filename=document_data['original_filename'],
                mime_type=document_data['mime_type'],
                file_size=document_data['file_size'],
            )

            # Publish results
            # In production, publish to results topic and store in database
            logger.info(f"Document {document_id} processed: {state.document_type}")

    finally:
        await consumer.stop()


if __name__ == "__main__":
    asyncio.run(process_documents_from_kafka())
