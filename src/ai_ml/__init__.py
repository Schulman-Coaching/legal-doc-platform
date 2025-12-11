"""
Legal Document AI/ML & Analytics Package
=========================================
Advanced AI/ML capabilities for legal document analysis.
"""

from .models import (
    AnalysisType,
    RiskLevel,
    SentimentType,
    DocumentEmbedding,
    SimilarDocument,
    RiskAssessment,
    ContractReviewResult,
    AnalysisRequest,
    AnalysisResponse,
    LLMProvider,
    LLMConfig,
    ChunkingStrategy,
    DocumentChunk,
)
from .llm_gateway import LLMGateway
from .embedding_service import EmbeddingService
from .vector_store import VectorStore, FAISSVectorStore
from .document_chunker import DocumentChunker
from .summarization_service import SummarizationService
from .similarity_service import SimilarityService
from .contract_analysis_service import ContractAnalysisService
from .risk_detection_service import RiskDetectionService
from .knowledge_graph_service import KnowledgeGraphService
from .rag_service import RAGService
from .cache_service import CacheService
from .ai_analytics_service import AIAnalyticsService

__all__ = [
    # Models
    "AnalysisType",
    "RiskLevel",
    "SentimentType",
    "DocumentEmbedding",
    "SimilarDocument",
    "RiskAssessment",
    "ContractReviewResult",
    "AnalysisRequest",
    "AnalysisResponse",
    "LLMProvider",
    "LLMConfig",
    "ChunkingStrategy",
    "DocumentChunk",
    # Services
    "LLMGateway",
    "EmbeddingService",
    "VectorStore",
    "FAISSVectorStore",
    "DocumentChunker",
    "SummarizationService",
    "SimilarityService",
    "ContractAnalysisService",
    "RiskDetectionService",
    "KnowledgeGraphService",
    "RAGService",
    "CacheService",
    "AIAnalyticsService",
]
