"""Tests for AI/ML data models."""

import pytest
from datetime import datetime

from src.ai_ml.models import (
    AnalysisType,
    RiskLevel,
    SentimentType,
    LLMProvider,
    ChunkingStrategy,
    LLMConfig,
    DocumentChunk,
    DocumentEmbedding,
    SimilarDocument,
    RiskAssessment,
    ContractReviewResult,
    AnalysisRequest,
    AnalysisResponse,
    QuestionAnswerRequest,
    QuestionAnswerResponse,
    LLMResponse,
)


class TestEnums:
    """Test enum classes."""

    def test_analysis_type_values(self):
        """Test AnalysisType enum values."""
        assert AnalysisType.SUMMARIZATION == "summarization"
        assert AnalysisType.QA == "question_answering"
        assert AnalysisType.CONTRACT_REVIEW == "contract_review"
        assert AnalysisType.RISK_ANALYSIS == "risk_analysis"
        assert len(AnalysisType) == 11

    def test_risk_level_ordering(self):
        """Test RiskLevel enum values."""
        assert RiskLevel.LOW == "low"
        assert RiskLevel.MEDIUM == "medium"
        assert RiskLevel.HIGH == "high"
        assert RiskLevel.CRITICAL == "critical"

    def test_llm_provider_values(self):
        """Test LLMProvider enum values."""
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.ANTHROPIC == "anthropic"
        assert LLMProvider.AZURE_OPENAI == "azure_openai"
        assert LLMProvider.LOCAL == "local"

    def test_chunking_strategy_values(self):
        """Test ChunkingStrategy enum values."""
        assert ChunkingStrategy.FIXED_SIZE == "fixed_size"
        assert ChunkingStrategy.SEMANTIC == "semantic"
        assert ChunkingStrategy.RECURSIVE == "recursive"


class TestLLMConfig:
    """Test LLMConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
        )
        assert config.provider == LLMProvider.OPENAI
        assert config.model_name == "gpt-4"
        assert config.max_tokens == 4096
        assert config.temperature == 0.1
        assert config.top_p == 0.95
        assert config.timeout == 60.0
        assert config.max_retries == 3
        assert config.api_key is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model_name="claude-3-opus",
            api_key="test-key",
            max_tokens=8000,
            temperature=0.5,
        )
        assert config.api_key == "test-key"
        assert config.max_tokens == 8000
        assert config.temperature == 0.5


class TestDocumentChunk:
    """Test DocumentChunk dataclass."""

    def test_creation(self):
        """Test chunk creation."""
        chunk = DocumentChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Test content",
            chunk_index=0,
            start_char=0,
            end_char=12,
        )
        assert chunk.chunk_id == "chunk-1"
        assert chunk.document_id == "doc-1"
        assert chunk.content == "Test content"
        assert chunk.chunk_index == 0
        assert chunk.metadata == {}
        assert chunk.embedding is None

    def test_with_metadata(self):
        """Test chunk with metadata."""
        chunk = DocumentChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Content",
            chunk_index=0,
            start_char=0,
            end_char=7,
            metadata={"section": "intro"},
            embedding=[0.1, 0.2, 0.3],
        )
        assert chunk.metadata == {"section": "intro"}
        assert chunk.embedding == [0.1, 0.2, 0.3]


class TestDocumentEmbedding:
    """Test DocumentEmbedding dataclass."""

    def test_creation(self):
        """Test embedding creation."""
        emb = DocumentEmbedding(
            document_id="doc-1",
            embedding=[0.1, 0.2, 0.3],
            model_name="test-model",
        )
        assert emb.document_id == "doc-1"
        assert emb.embedding == [0.1, 0.2, 0.3]
        assert emb.model_name == "test-model"
        assert isinstance(emb.created_at, datetime)
        assert emb.metadata == {}

    def test_with_chunk_id(self):
        """Test embedding with chunk ID."""
        emb = DocumentEmbedding(
            document_id="doc-1",
            embedding=[0.1],
            model_name="model",
            chunk_id="chunk-1",
        )
        assert emb.chunk_id == "chunk-1"


class TestSimilarDocument:
    """Test SimilarDocument dataclass."""

    def test_creation(self):
        """Test similar document creation."""
        similar = SimilarDocument(
            document_id="doc-1",
            similarity_score=0.95,
        )
        assert similar.document_id == "doc-1"
        assert similar.similarity_score == 0.95
        assert similar.title is None
        assert similar.snippet is None

    def test_with_all_fields(self):
        """Test with all fields populated."""
        similar = SimilarDocument(
            document_id="doc-1",
            similarity_score=0.85,
            title="Test Document",
            snippet="This is a snippet...",
            chunk_id="chunk-1",
            metadata={"type": "contract"},
        )
        assert similar.title == "Test Document"
        assert similar.snippet == "This is a snippet..."
        assert similar.chunk_id == "chunk-1"


class TestRiskAssessment:
    """Test RiskAssessment dataclass."""

    def test_creation(self):
        """Test risk assessment creation."""
        assessment = RiskAssessment(
            overall_risk=RiskLevel.HIGH,
            risk_score=0.75,
            risk_factors=[{"factor": "unlimited liability", "severity": "high"}],
            recommendations=["Negotiate liability cap"],
            confidence=0.9,
        )
        assert assessment.overall_risk == RiskLevel.HIGH
        assert assessment.risk_score == 0.75
        assert len(assessment.risk_factors) == 1
        assert len(assessment.recommendations) == 1
        assert assessment.confidence == 0.9


class TestContractReviewResult:
    """Test ContractReviewResult dataclass."""

    def test_creation(self):
        """Test contract review result creation."""
        risk = RiskAssessment(
            overall_risk=RiskLevel.MEDIUM,
            risk_score=0.5,
            risk_factors=[],
            recommendations=[],
            confidence=0.8,
        )

        result = ContractReviewResult(
            document_id="doc-1",
            summary="Contract summary",
            key_terms=[{"term": "Effective Date"}],
            obligations=[{"party": "A", "obligation": "Pay"}],
            deadlines=[{"type": "payment", "date": "2024-01-01"}],
            risk_assessment=risk,
            missing_clauses=["force majeure"],
            non_standard_provisions=[],
            negotiation_points=[],
        )

        assert result.document_id == "doc-1"
        assert result.summary == "Contract summary"
        assert len(result.key_terms) == 1
        assert len(result.missing_clauses) == 1
        assert result.processing_time_ms == 0


class TestAnalysisRequest:
    """Test AnalysisRequest model."""

    def test_basic_request(self):
        """Test basic analysis request."""
        request = AnalysisRequest(
            document_id="doc-1",
            analysis_type=AnalysisType.SUMMARIZATION,
        )
        assert request.document_id == "doc-1"
        assert request.analysis_type == AnalysisType.SUMMARIZATION
        assert request.options == {}
        assert request.context is None

    def test_with_options(self):
        """Test request with options."""
        request = AnalysisRequest(
            document_id="doc-1",
            analysis_type=AnalysisType.CONTRACT_REVIEW,
            options={"perspective": "buyer"},
            context="Additional context",
        )
        assert request.options == {"perspective": "buyer"}
        assert request.context == "Additional context"


class TestAnalysisResponse:
    """Test AnalysisResponse model."""

    def test_successful_response(self):
        """Test successful analysis response."""
        response = AnalysisResponse(
            document_id="doc-1",
            analysis_type=AnalysisType.SUMMARIZATION,
            success=True,
            result={"summary": "Test summary"},
            processing_time_ms=1500,
            model_used="gpt-4",
            confidence=0.9,
        )
        assert response.success is True
        assert response.result == {"summary": "Test summary"}
        assert response.error_message is None

    def test_failed_response(self):
        """Test failed analysis response."""
        response = AnalysisResponse(
            document_id="doc-1",
            analysis_type=AnalysisType.SUMMARIZATION,
            success=False,
            result={},
            processing_time_ms=100,
            model_used="",
            confidence=0.0,
            error_message="API error",
        )
        assert response.success is False
        assert response.error_message == "API error"


class TestQuestionAnswerModels:
    """Test Q&A models."""

    def test_qa_request(self):
        """Test Q&A request model."""
        request = QuestionAnswerRequest(
            question="What is the contract term?",
        )
        assert request.question == "What is the contract term?"
        assert request.document_ids is None
        assert request.max_sources == 5

    def test_qa_response(self):
        """Test Q&A response model."""
        response = QuestionAnswerResponse(
            question="What is the contract term?",
            answer="The contract term is 2 years.",
            sources=[{"document_id": "doc-1", "section": "2.1"}],
            confidence=0.85,
            processing_time_ms=2000,
        )
        assert response.answer == "The contract term is 2 years."
        assert len(response.sources) == 1


class TestLLMResponse:
    """Test LLMResponse model."""

    def test_creation(self):
        """Test LLM response creation."""
        response = LLMResponse(
            text="Generated text",
            model="gpt-4",
            provider="openai",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        assert response.text == "Generated text"
        assert response.model == "gpt-4"
        assert response.cached is False

    def test_cached_response(self):
        """Test cached response."""
        response = LLMResponse(
            text="Cached text",
            model="gpt-4",
            provider="openai",
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            cached=True,
        )
        assert response.cached is True
