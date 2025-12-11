"""
RAG (Retrieval-Augmented Generation) Service
=============================================
Question answering over legal documents using retrieval and LLM.
"""

import logging
import time
from typing import Any, Optional

from .document_chunker import DocumentChunker
from .embedding_service import EmbeddingService
from .llm_gateway import LLMGateway
from .models import ChunkingStrategy, QuestionAnswerRequest, QuestionAnswerResponse
from .similarity_service import SimilarityService
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGService:
    """
    Retrieval-Augmented Generation service for question answering.
    Combines document retrieval with LLM generation for accurate answers.
    """

    SYSTEM_PROMPT = """You are a legal research assistant with expertise in analyzing
legal documents. Answer questions accurately based ONLY on the provided context.
If the context doesn't contain enough information to answer, say so clearly.
Always cite the specific document and section when providing information."""

    QA_PROMPT = """Answer the following question based ONLY on the provided context.
Be specific and cite sources. If the context doesn't contain the answer, say
"I cannot find this information in the provided documents."

Question: {question}

Context from documents:
{context}

Answer:"""

    QA_WITH_REASONING_PROMPT = """Answer the following question based on the provided context.
Think step by step:
1. First, identify which parts of the context are relevant
2. Extract the specific information that answers the question
3. Formulate a clear, accurate answer with citations

Question: {question}

Context from documents:
{context}

Step-by-step analysis and answer:"""

    MULTI_DOC_SYNTHESIS_PROMPT = """Synthesize information from multiple documents to answer this question.
Identify any contradictions or variations between documents.

Question: {question}

Document excerpts:
{context}

Synthesized answer (note any conflicts between sources):"""

    def __init__(
        self,
        llm_gateway: LLMGateway,
        similarity_service: SimilarityService,
        max_context_length: int = 8000,
        default_top_k: int = 5,
    ):
        """
        Initialize RAG service.

        Args:
            llm_gateway: LLM gateway for text generation
            similarity_service: Service for document similarity search
            max_context_length: Maximum context characters to include
            default_top_k: Default number of chunks to retrieve
        """
        self.llm = llm_gateway
        self.similarity = similarity_service
        self.max_context_length = max_context_length
        self.default_top_k = default_top_k

    async def answer_question(
        self,
        question: str,
        document_ids: Optional[list[str]] = None,
        top_k: Optional[int] = None,
        min_relevance: float = 0.3,
        include_reasoning: bool = False,
    ) -> QuestionAnswerResponse:
        """
        Answer a question using document retrieval.

        Args:
            question: The question to answer
            document_ids: Optional list of document IDs to search within
            top_k: Number of chunks to retrieve
            min_relevance: Minimum relevance score for chunks
            include_reasoning: Whether to include step-by-step reasoning

        Returns:
            Question answer response with sources
        """
        start_time = time.time()
        top_k = top_k or self.default_top_k

        # Build filter if specific documents requested
        filter_metadata = None
        if document_ids:
            filter_metadata = {"document_id": document_ids}

        # Retrieve relevant chunks
        similar_docs = await self.similarity.find_similar(
            query_text=question,
            top_k=top_k * 2,  # Get extra to filter
            min_score=min_relevance,
            filter_metadata=filter_metadata,
            deduplicate_documents=False,  # Keep all relevant chunks
        )

        if not similar_docs:
            return QuestionAnswerResponse(
                question=question,
                answer="I could not find any relevant information in the documents to answer this question.",
                sources=[],
                confidence=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000),
            )

        # Build context from retrieved chunks
        context_parts = []
        sources = []
        total_length = 0

        for doc in similar_docs:
            # Get chunk content from metadata or fetch it
            chunk_content = doc.metadata.get("content", "")
            if not chunk_content:
                # Try to get from vector store
                embedding = await self.similarity.vector_store.get(doc.document_id)
                if embedding:
                    chunk_content = embedding.metadata.get("content", f"[Content for {doc.document_id}]")

            # Check if we have room
            chunk_text = f"\n---\nDocument: {doc.metadata.get('document_id', doc.document_id)}\n"
            if doc.metadata.get("section"):
                chunk_text += f"Section: {doc.metadata.get('section')}\n"
            chunk_text += f"Content: {chunk_content}\n"

            if total_length + len(chunk_text) > self.max_context_length:
                break

            context_parts.append(chunk_text)
            total_length += len(chunk_text)

            sources.append({
                "document_id": doc.metadata.get("document_id", doc.document_id),
                "chunk_id": doc.metadata.get("chunk_id", doc.document_id),
                "relevance_score": doc.similarity_score,
                "section": doc.metadata.get("section"),
            })

            if len(sources) >= top_k:
                break

        context = "\n".join(context_parts)

        # Generate answer
        if include_reasoning:
            prompt = self.QA_WITH_REASONING_PROMPT.format(
                question=question,
                context=context,
            )
        else:
            prompt = self.QA_PROMPT.format(
                question=question,
                context=context,
            )

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
        )

        # Estimate confidence based on retrieval scores
        avg_relevance = sum(s["relevance_score"] for s in sources) / len(sources) if sources else 0
        confidence = min(avg_relevance + 0.2, 1.0)  # Slight boost for having retrieved context

        processing_time = int((time.time() - start_time) * 1000)

        return QuestionAnswerResponse(
            question=question,
            answer=response.text,
            sources=sources,
            confidence=confidence,
            processing_time_ms=processing_time,
        )

    async def answer_with_synthesis(
        self,
        question: str,
        document_ids: Optional[list[str]] = None,
        top_k: int = 10,
    ) -> QuestionAnswerResponse:
        """
        Answer a question by synthesizing information from multiple documents.
        Better for complex questions requiring cross-document analysis.
        """
        start_time = time.time()

        filter_metadata = None
        if document_ids:
            filter_metadata = {"document_id": document_ids}

        # Get more chunks for synthesis
        similar_docs = await self.similarity.find_similar(
            query_text=question,
            top_k=top_k,
            min_score=0.2,
            filter_metadata=filter_metadata,
            deduplicate_documents=False,
        )

        if not similar_docs:
            return QuestionAnswerResponse(
                question=question,
                answer="No relevant documents found.",
                sources=[],
                confidence=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000),
            )

        # Group by document for better synthesis
        doc_groups: dict[str, list] = {}
        for doc in similar_docs:
            doc_id = doc.metadata.get("document_id", doc.document_id.split(":")[0])
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(doc)

        # Build context with document grouping
        context_parts = []
        sources = []

        for doc_id, chunks in doc_groups.items():
            context_parts.append(f"\n=== Document: {doc_id} ===")
            for chunk in chunks[:3]:  # Max 3 chunks per doc
                chunk_content = chunk.metadata.get("content", "")
                context_parts.append(f"- {chunk_content[:500]}")
                sources.append({
                    "document_id": doc_id,
                    "chunk_id": chunk.metadata.get("chunk_id"),
                    "relevance_score": chunk.similarity_score,
                })

        context = "\n".join(context_parts)[:self.max_context_length]

        prompt = self.MULTI_DOC_SYNTHESIS_PROMPT.format(
            question=question,
            context=context,
        )

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
        )

        processing_time = int((time.time() - start_time) * 1000)

        return QuestionAnswerResponse(
            question=question,
            answer=response.text,
            sources=sources,
            confidence=0.8 if sources else 0.0,
            processing_time_ms=processing_time,
        )

    async def get_relevant_context(
        self,
        question: str,
        document_ids: Optional[list[str]] = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Get relevant context for a question without generating an answer.
        Useful for manual review or custom prompting.
        """
        filter_metadata = None
        if document_ids:
            filter_metadata = {"document_id": document_ids}

        similar_docs = await self.similarity.find_similar(
            query_text=question,
            top_k=top_k,
            min_score=0.2,
            filter_metadata=filter_metadata,
            deduplicate_documents=False,
        )

        return [
            {
                "document_id": doc.metadata.get("document_id", doc.document_id),
                "chunk_id": doc.metadata.get("chunk_id"),
                "content": doc.metadata.get("content", ""),
                "relevance_score": doc.similarity_score,
                "metadata": doc.metadata,
            }
            for doc in similar_docs
        ]

    async def conversational_qa(
        self,
        question: str,
        conversation_history: list[dict[str, str]],
        document_ids: Optional[list[str]] = None,
    ) -> QuestionAnswerResponse:
        """
        Answer a question in a conversational context.
        Uses conversation history for context resolution.
        """
        start_time = time.time()

        # Rewrite question with conversation context if needed
        if conversation_history:
            rewrite_prompt = f"""Given this conversation history, rewrite the follow-up question
to be a standalone question that can be understood without context.

Conversation:
{self._format_history(conversation_history)}

Follow-up question: {question}

Standalone question:"""

            rewrite_response = await self.llm.generate(rewrite_prompt)
            standalone_question = rewrite_response.text.strip()
        else:
            standalone_question = question

        # Now answer the standalone question
        response = await self.answer_question(
            question=standalone_question,
            document_ids=document_ids,
        )

        # Update processing time to include rewrite
        response.processing_time_ms = int((time.time() - start_time) * 1000)

        return response

    def _format_history(self, history: list[dict[str, str]]) -> str:
        """Format conversation history for prompt."""
        formatted = []
        for turn in history[-5:]:  # Last 5 turns
            role = turn.get("role", "user")
            content = turn.get("content", "")
            formatted.append(f"{role.capitalize()}: {content}")
        return "\n".join(formatted)

    async def answer_with_citations(
        self,
        question: str,
        document_ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Answer a question with inline citations.
        Returns answer with citation markers and a citation list.
        """
        response = await self.answer_question(
            question=question,
            document_ids=document_ids,
            include_reasoning=False,
        )

        # Post-process to add citation markers
        prompt = f"""Add citation markers [1], [2], etc. to this answer where appropriate,
based on the source documents.

Answer: {response.answer}

Sources:
{self._format_sources(response.sources)}

Answer with citations:"""

        cited_response = await self.llm.generate(prompt)

        return {
            "question": question,
            "answer": cited_response.text,
            "citations": [
                {
                    "marker": f"[{i + 1}]",
                    "document_id": src["document_id"],
                    "section": src.get("section"),
                }
                for i, src in enumerate(response.sources)
            ],
            "confidence": response.confidence,
            "processing_time_ms": response.processing_time_ms,
        }

    def _format_sources(self, sources: list[dict[str, Any]]) -> str:
        """Format sources for citation prompt."""
        formatted = []
        for i, src in enumerate(sources):
            formatted.append(f"[{i + 1}] Document: {src['document_id']}, Section: {src.get('section', 'N/A')}")
        return "\n".join(formatted)
