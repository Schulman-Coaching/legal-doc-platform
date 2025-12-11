"""
AI Analytics Service
=====================
Main service coordinating all AI/ML capabilities for legal document analysis.
"""

import logging
import os
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware

from .cache_service import CacheService
from .contract_analysis_service import ContractAnalysisService
from .document_chunker import DocumentChunker
from .embedding_service import EmbeddingService, SentenceTransformerEmbedding, OpenAIEmbedding
from .knowledge_graph_service import KnowledgeGraphService
from .llm_gateway import LLMGateway
from .models import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisType,
    LLMConfig,
    LLMProvider,
    ChunkingStrategy,
    QuestionAnswerRequest,
    QuestionAnswerResponse,
)
from .rag_service import RAGService
from .risk_detection_service import RiskDetectionService
from .similarity_service import SimilarityService
from .summarization_service import SummarizationService
from .vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class AIAnalyticsService:
    """
    Main service coordinating all AI/ML capabilities.
    Provides a unified interface for document analysis.
    """

    def __init__(
        self,
        llm_configs: Optional[list[LLMConfig]] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_openai_embeddings: bool = False,
        neo4j_uri: Optional[str] = None,
        vector_store_path: Optional[str] = None,
        redis_url: Optional[str] = None,
    ):
        """
        Initialize AI Analytics Service.

        Args:
            llm_configs: List of LLM provider configurations
            embedding_model: Embedding model name
            use_openai_embeddings: Whether to use OpenAI embeddings
            neo4j_uri: Neo4j connection URI
            vector_store_path: Path to persist vector store
            redis_url: Redis URL for caching
        """
        # Initialize cache
        self.cache = CacheService(redis_url=redis_url)

        # Initialize LLM Gateway
        if llm_configs:
            self.llm_gateway = LLMGateway(llm_configs, cache_service=self.cache)
        else:
            # Default configuration from environment
            configs = self._get_default_llm_configs()
            self.llm_gateway = LLMGateway(configs, cache_service=self.cache)

        # Initialize embedding service
        if use_openai_embeddings:
            embedding_model_instance = OpenAIEmbedding()
        else:
            embedding_model_instance = SentenceTransformerEmbedding(embedding_model)

        self.embedding_service = EmbeddingService(
            model=embedding_model_instance,
            cache_service=self.cache,
        )

        # Initialize vector store
        self.vector_store = FAISSVectorStore(
            dimension=self.embedding_service.embedding_dimension,
            persist_path=vector_store_path,
        )

        # Initialize document chunker
        self.chunker = DocumentChunker(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=1000,
            chunk_overlap=200,
        )

        # Initialize services
        self.summarization = SummarizationService(
            llm_gateway=self.llm_gateway,
            chunker=self.chunker,
        )

        self.similarity = SimilarityService(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
            chunker=self.chunker,
        )

        self.contract_analysis = ContractAnalysisService(
            llm_gateway=self.llm_gateway,
        )

        self.risk_detection = RiskDetectionService(
            llm_gateway=self.llm_gateway,
        )

        self.knowledge_graph = KnowledgeGraphService(
            uri=neo4j_uri,
        )

        self.rag = RAGService(
            llm_gateway=self.llm_gateway,
            similarity_service=self.similarity,
        )

        logger.info("AI Analytics Service initialized")

    def _get_default_llm_configs(self) -> list[LLMConfig]:
        """Get default LLM configurations from environment."""
        configs = []

        # OpenAI
        if os.getenv("OPENAI_API_KEY"):
            configs.append(LLMConfig(
                provider=LLMProvider.OPENAI,
                model_name=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
            ))

        # Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            configs.append(LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name=os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229"),
            ))

        # Azure OpenAI
        if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            configs.append(LLMConfig(
                provider=LLMProvider.AZURE_OPENAI,
                model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            ))

        # Local model
        if os.getenv("LOCAL_LLM_ENDPOINT"):
            configs.append(LLMConfig(
                provider=LLMProvider.LOCAL,
                model_name=os.getenv("LOCAL_LLM_MODEL", "llama-2-70b"),
                endpoint=os.getenv("LOCAL_LLM_ENDPOINT"),
            ))

        if not configs:
            # Default to OpenAI if no config (will fail gracefully if no key)
            configs.append(LLMConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4-turbo-preview",
            ))

        return configs

    async def analyze(
        self,
        request: AnalysisRequest,
        document_text: str,
    ) -> AnalysisResponse:
        """
        Perform AI analysis on a document.

        Args:
            request: Analysis request with type and options
            document_text: Full document text

        Returns:
            Analysis response with results
        """
        start_time = time.time()

        try:
            if request.analysis_type == AnalysisType.SUMMARIZATION:
                result = await self.summarization.summarize(
                    document_text,
                    summary_type=request.options.get("summary_type", "standard"),
                    document_id=request.document_id,
                )

            elif request.analysis_type == AnalysisType.CONTRACT_REVIEW:
                review = await self.contract_analysis.analyze_contract(
                    request.document_id,
                    document_text,
                    party_perspective=request.options.get("perspective", "neutral"),
                )
                result = {
                    "summary": review.summary,
                    "key_terms": review.key_terms,
                    "obligations": review.obligations,
                    "deadlines": review.deadlines,
                    "risk_score": review.risk_assessment.risk_score,
                    "risk_level": review.risk_assessment.overall_risk.value,
                    "risk_factors": review.risk_assessment.risk_factors,
                    "missing_clauses": review.missing_clauses,
                    "non_standard_provisions": review.non_standard_provisions,
                    "negotiation_points": review.negotiation_points,
                    "recommendations": review.risk_assessment.recommendations,
                }

            elif request.analysis_type == AnalysisType.RISK_ANALYSIS:
                assessment = await self.risk_detection.assess_document_risk(
                    document_text,
                    document_type=request.options.get("document_type", "contract"),
                    perspective=request.options.get("perspective", "neutral"),
                )
                result = {
                    "overall_risk": assessment.overall_risk.value,
                    "risk_score": assessment.risk_score,
                    "risk_factors": assessment.risk_factors,
                    "recommendations": assessment.recommendations,
                    "confidence": assessment.confidence,
                }

            elif request.analysis_type == AnalysisType.SIMILARITY:
                similar = await self.similarity.find_similar(
                    query_text=document_text,
                    top_k=request.options.get("top_k", 10),
                    min_score=request.options.get("min_score", 0.5),
                )
                result = {
                    "similar_documents": [
                        {
                            "document_id": s.document_id,
                            "similarity_score": s.similarity_score,
                            "metadata": s.metadata,
                        }
                        for s in similar
                    ]
                }

            elif request.analysis_type == AnalysisType.QA:
                answer = await self.rag.answer_question(
                    question=request.options.get("question", request.context or ""),
                    document_ids=[request.document_id] if request.document_id else None,
                )
                result = {
                    "question": answer.question,
                    "answer": answer.answer,
                    "sources": answer.sources,
                    "confidence": answer.confidence,
                }

            elif request.analysis_type == AnalysisType.KEY_POINTS:
                facts = await self.summarization.extract_key_facts(document_text)
                result = {"key_facts": facts}

            elif request.analysis_type == AnalysisType.CLUSTERING:
                clusters = await self.similarity.cluster_documents(
                    n_clusters=request.options.get("n_clusters", 5),
                )
                result = {"clusters": clusters}

            elif request.analysis_type == AnalysisType.ANOMALY_DETECTION:
                anomalies = await self.risk_detection.detect_anomalies(
                    document_text,
                    document_type=request.options.get("document_type", "contract"),
                )
                result = {"anomalies": anomalies}

            else:
                result = {"message": f"Analysis type {request.analysis_type} not implemented"}

            processing_time = int((time.time() - start_time) * 1000)

            return AnalysisResponse(
                document_id=request.document_id,
                analysis_type=request.analysis_type,
                success=True,
                result=result,
                processing_time_ms=processing_time,
                model_used=self.llm_gateway.primary_provider.value if self.llm_gateway.primary_provider else "unknown",
                confidence=result.get("confidence", 0.85),
            )

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return AnalysisResponse(
                document_id=request.document_id,
                analysis_type=request.analysis_type,
                success=False,
                result={},
                processing_time_ms=int((time.time() - start_time) * 1000),
                model_used="",
                confidence=0.0,
                error_message=str(e),
            )

    async def index_document(
        self,
        document_id: str,
        document_text: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Index a document for similarity search and RAG.

        Args:
            document_id: Document identifier
            document_text: Full document text
            metadata: Optional metadata

        Returns:
            Indexing result
        """
        return await self.similarity.add_document(
            document_id=document_id,
            text=document_text,
            metadata=metadata,
        )

    async def answer_question(
        self,
        question: str,
        document_ids: Optional[list[str]] = None,
    ) -> QuestionAnswerResponse:
        """
        Answer a question using indexed documents.

        Args:
            question: Question to answer
            document_ids: Optional specific documents to search

        Returns:
            Answer with sources
        """
        return await self.rag.answer_question(
            question=question,
            document_ids=document_ids,
        )

    async def close(self) -> None:
        """Close all connections and save state."""
        await self.llm_gateway.close()
        await self.vector_store.save()
        await self.knowledge_graph.close()

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        return {
            "llm_gateway": self.llm_gateway.get_stats(),
            "embedding_model": self.embedding_service.model.model_name,
            "embedding_dimension": self.embedding_service.embedding_dimension,
        }


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Legal Document AI/ML Service",
    description="Advanced AI analytics for legal document processing",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
ai_service: Optional[AIAnalyticsService] = None


@app.on_event("startup")
async def startup_event():
    """Initialize AI services on startup."""
    global ai_service
    ai_service = AIAnalyticsService()
    logger.info("AI Analytics Service started")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    if ai_service:
        await ai_service.close()


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ai-analytics",
        "version": "2.0.0",
    }


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_document(
    request: AnalysisRequest,
    document_text: str = Body(..., embed=True),
):
    """Perform AI analysis on a document."""
    if not ai_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return await ai_service.analyze(request, document_text)


@app.post("/api/v1/summarize")
async def summarize_document(
    document_id: str = Body(...),
    document_text: str = Body(...),
    summary_type: str = Body("standard"),
):
    """Generate document summary."""
    if not ai_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    result = await ai_service.summarization.summarize(
        document_text,
        summary_type,
        document_id,
    )
    return result


@app.post("/api/v1/contract-review")
async def review_contract(
    document_id: str = Body(...),
    contract_text: str = Body(...),
    perspective: str = Body("neutral"),
):
    """Perform comprehensive contract review."""
    if not ai_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    result = await ai_service.contract_analysis.analyze_contract(
        document_id,
        contract_text,
        party_perspective=perspective,
    )

    return {
        "document_id": result.document_id,
        "summary": result.summary,
        "key_terms": result.key_terms,
        "obligations": result.obligations,
        "deadlines": result.deadlines,
        "risk_assessment": {
            "level": result.risk_assessment.overall_risk.value,
            "score": result.risk_assessment.risk_score,
            "factors": result.risk_assessment.risk_factors,
            "recommendations": result.risk_assessment.recommendations,
        },
        "missing_clauses": result.missing_clauses,
        "non_standard_provisions": result.non_standard_provisions,
        "negotiation_points": result.negotiation_points,
        "processing_time_ms": result.processing_time_ms,
    }


@app.post("/api/v1/risk-assessment")
async def assess_risk(
    document_text: str = Body(...),
    document_type: str = Body("contract"),
    perspective: str = Body("neutral"),
):
    """Assess document risk."""
    if not ai_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    result = await ai_service.risk_detection.assess_document_risk(
        document_text,
        document_type,
        perspective,
    )

    return {
        "overall_risk": result.overall_risk.value,
        "risk_score": result.risk_score,
        "risk_factors": result.risk_factors,
        "recommendations": result.recommendations,
        "confidence": result.confidence,
    }


@app.post("/api/v1/index")
async def index_document(
    document_id: str = Body(...),
    document_text: str = Body(...),
    metadata: Optional[dict] = Body(None),
):
    """Index document for similarity search."""
    if not ai_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    result = await ai_service.index_document(document_id, document_text, metadata)
    return result


@app.post("/api/v1/similarity/search")
async def search_similar(
    query_text: str = Body(...),
    top_k: int = Body(10),
    min_score: float = Body(0.5),
):
    """Find similar documents."""
    if not ai_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    results = await ai_service.similarity.find_similar(
        query_text,
        top_k,
        min_score,
    )

    return {
        "results": [
            {
                "document_id": r.document_id,
                "score": r.similarity_score,
                "metadata": r.metadata,
            }
            for r in results
        ]
    }


@app.post("/api/v1/qa")
async def question_answer(
    question: str = Body(...),
    document_ids: Optional[list[str]] = Body(None),
):
    """Answer a question using indexed documents."""
    if not ai_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    result = await ai_service.answer_question(question, document_ids)

    return {
        "question": result.question,
        "answer": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
        "processing_time_ms": result.processing_time_ms,
    }


@app.get("/api/v1/stats")
async def get_stats():
    """Get service statistics."""
    if not ai_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return ai_service.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
