"""
Data Models for AI/ML Analytics
================================
Pydantic and dataclass models for the AI/ML layer.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AnalysisType(str, Enum):
    """Types of AI analysis available."""
    SUMMARIZATION = "summarization"
    QA = "question_answering"
    SIMILARITY = "similarity"
    CLUSTERING = "clustering"
    RISK_ANALYSIS = "risk_analysis"
    CONTRACT_REVIEW = "contract_review"
    SENTIMENT = "sentiment"
    KEY_POINTS = "key_points"
    COMPARISON = "comparison"
    ANOMALY_DETECTION = "anomaly_detection"
    PREDICTION = "prediction"


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SentimentType(str, Enum):
    """Sentiment classification."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    ADVERSARIAL = "adversarial"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    LOCAL = "local"


class ChunkingStrategy(str, Enum):
    """Document chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    RECURSIVE = "recursive"


class LLMConfig(BaseModel):
    """Configuration for LLM provider."""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.1
    top_p: float = 0.95
    timeout: float = 60.0
    max_retries: int = 3


@dataclass
class DocumentChunk:
    """A chunk of a document for processing."""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None


@dataclass
class DocumentEmbedding:
    """Vector embedding for a document."""
    document_id: str
    embedding: list[float]
    model_name: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None


@dataclass
class SimilarDocument:
    """Represents a similar document match."""
    document_id: str
    similarity_score: float
    title: Optional[str] = None
    snippet: Optional[str] = None
    chunk_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskFactor:
    """Individual risk factor in an assessment."""
    factor: str
    severity: str
    weight: float
    explanation: Optional[str] = None
    location: Optional[str] = None


@dataclass
class RiskAssessment:
    """Risk assessment result."""
    overall_risk: RiskLevel
    risk_score: float
    risk_factors: list[dict[str, Any]]
    recommendations: list[str]
    confidence: float


@dataclass
class ContractObligation:
    """Contract obligation extracted from document."""
    party: str
    obligation: str
    deadline: Optional[str] = None
    condition: Optional[str] = None
    consequence: Optional[str] = None
    section: Optional[str] = None


@dataclass
class ContractDeadline:
    """Important date or deadline in a contract."""
    deadline_type: str
    description: str
    date: Optional[str] = None
    is_recurring: bool = False
    notice_period: Optional[str] = None


@dataclass
class NegotiationPoint:
    """Point to negotiate in a contract."""
    clause: str
    issue: str
    recommendation: str
    priority: str
    current_language: Optional[str] = None
    suggested_language: Optional[str] = None


@dataclass
class ContractReviewResult:
    """Contract analysis result."""
    document_id: str
    summary: str
    key_terms: list[dict[str, Any]]
    obligations: list[dict[str, Any]]
    deadlines: list[dict[str, Any]]
    risk_assessment: RiskAssessment
    missing_clauses: list[str]
    non_standard_provisions: list[dict[str, Any]]
    negotiation_points: list[dict[str, Any]]
    processing_time_ms: int = 0


class AnalysisRequest(BaseModel):
    """Request for AI analysis."""
    document_id: str
    analysis_type: AnalysisType
    options: dict[str, Any] = Field(default_factory=dict)
    context: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Response from AI analysis."""
    document_id: str
    analysis_type: AnalysisType
    success: bool
    result: dict[str, Any]
    processing_time_ms: int
    model_used: str
    confidence: float
    error_message: Optional[str] = None


class QuestionAnswerRequest(BaseModel):
    """Request for question answering."""
    question: str
    document_ids: Optional[list[str]] = None
    context: Optional[str] = None
    max_sources: int = 5


class QuestionAnswerResponse(BaseModel):
    """Response from question answering."""
    question: str
    answer: str
    sources: list[dict[str, Any]]
    confidence: float
    processing_time_ms: int


class LLMResponse(BaseModel):
    """Standardized response from LLM."""
    text: str
    model: str
    provider: str
    usage: dict[str, int]
    finish_reason: Optional[str] = None
    cached: bool = False
