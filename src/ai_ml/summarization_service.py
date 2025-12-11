"""
Document Summarization Service
===============================
Generate summaries of legal documents using LLM.
"""

import logging
import time
from typing import Any, Optional

from .document_chunker import DocumentChunker
from .llm_gateway import LLMGateway
from .models import ChunkingStrategy

logger = logging.getLogger(__name__)


class SummarizationService:
    """Service for document summarization using LLM."""

    LEGAL_SUMMARY_SYSTEM_PROMPT = """You are an expert legal document analyst with extensive experience
in contract law, litigation, corporate legal matters, and regulatory compliance.
Provide accurate, professional, and objective analysis.
Focus on legally significant details and potential issues."""

    LEGAL_SUMMARY_PROMPT = """Summarize the following legal document. Focus on:
1. The main purpose and subject matter
2. Key parties involved and their roles
3. Important terms, conditions, or obligations
4. Critical dates, deadlines, and timeframes
5. Notable risks, liabilities, or concerns
6. Any unusual or non-standard provisions

Document:
{document_text}

Provide a concise summary (3-5 paragraphs) followed by a bullet-point list of key facts."""

    EXECUTIVE_SUMMARY_PROMPT = """Create an executive summary of this legal document for a busy attorney.
Be concise but comprehensive. Highlight anything unusual or requiring immediate attention.
Structure your response as:

EXECUTIVE SUMMARY
[2-3 sentence overview]

KEY POINTS
• [bullet points of critical information]

ACTION ITEMS / CONCERNS
• [items requiring attention]

Document:
{document_text}"""

    BULLET_POINTS_PROMPT = """Extract the key points from this legal document as a structured list.
Organize by category:

PARTIES & RELATIONSHIPS
TERM & DURATION
FINANCIAL TERMS
OBLIGATIONS & RESPONSIBILITIES
RIGHTS & REMEDIES
TERMINATION CONDITIONS
RISKS & LIABILITIES

Document:
{document_text}"""

    MAP_REDUCE_MAP_PROMPT = """Summarize this section of a legal document, focusing on key legal points:

Section {chunk_index} of {total_chunks}:
{chunk_text}

Provide a concise summary of this section's key legal content."""

    MAP_REDUCE_REDUCE_PROMPT = """Combine these section summaries into a coherent overall summary.
Maintain all important legal details while eliminating redundancy:

Section Summaries:
{summaries}

Create a unified, well-organized summary covering:
1. Document purpose and parties
2. Key terms and obligations
3. Important dates and deadlines
4. Risks and notable provisions"""

    def __init__(
        self,
        llm_gateway: LLMGateway,
        chunker: Optional[DocumentChunker] = None,
        max_input_tokens: int = 8000,
    ):
        """
        Initialize summarization service.

        Args:
            llm_gateway: LLM gateway for text generation
            chunker: Document chunker for long documents
            max_input_tokens: Maximum tokens for single LLM call
        """
        self.llm = llm_gateway
        self.chunker = chunker or DocumentChunker(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=4000,
            chunk_overlap=200,
        )
        self.max_input_tokens = max_input_tokens
        # Approximate chars per token
        self.chars_per_token = 4
        self.max_input_chars = max_input_tokens * self.chars_per_token

    async def summarize(
        self,
        document_text: str,
        summary_type: str = "standard",
        document_id: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Generate document summary.

        Args:
            document_text: Full document text
            summary_type: "standard", "executive", "bullet_points", or "comprehensive"
            document_id: Optional document identifier
            max_length: Maximum summary length (approximate)

        Returns:
            Summary and metadata
        """
        start_time = time.time()

        # Check if document needs chunking
        if len(document_text) > self.max_input_chars:
            result = await self._summarize_long_document(
                document_text,
                summary_type,
                document_id,
            )
        else:
            result = await self._summarize_short_document(
                document_text,
                summary_type,
            )

        processing_time = int((time.time() - start_time) * 1000)

        return {
            **result,
            "processing_time_ms": processing_time,
            "original_length": len(document_text),
            "summary_type": summary_type,
            "document_id": document_id,
        }

    async def _summarize_short_document(
        self,
        document_text: str,
        summary_type: str,
    ) -> dict[str, Any]:
        """Summarize document that fits in single context."""
        # Select prompt template
        if summary_type == "executive":
            prompt = self.EXECUTIVE_SUMMARY_PROMPT.format(
                document_text=document_text
            )
        elif summary_type == "bullet_points":
            prompt = self.BULLET_POINTS_PROMPT.format(
                document_text=document_text
            )
        else:
            prompt = self.LEGAL_SUMMARY_PROMPT.format(
                document_text=document_text
            )

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=self.LEGAL_SUMMARY_SYSTEM_PROMPT,
        )

        return {
            "summary": response.text,
            "model_used": response.model,
            "summary_length": len(response.text),
            "method": "direct",
        }

    async def _summarize_long_document(
        self,
        document_text: str,
        summary_type: str,
        document_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Summarize long document using map-reduce approach."""
        # Chunk the document
        chunks = self.chunker.chunk_document(
            document_id=document_id or "temp",
            text=document_text,
        )

        logger.info(f"Summarizing long document in {len(chunks)} chunks")

        # Map: Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            prompt = self.MAP_REDUCE_MAP_PROMPT.format(
                chunk_index=i + 1,
                total_chunks=len(chunks),
                chunk_text=chunk.content,
            )

            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=self.LEGAL_SUMMARY_SYSTEM_PROMPT,
            )
            chunk_summaries.append(response.text)

        # Reduce: Combine summaries
        combined_summaries = "\n\n---\n\n".join(
            f"Section {i + 1}:\n{summary}"
            for i, summary in enumerate(chunk_summaries)
        )

        # If combined summaries are still too long, recursively reduce
        if len(combined_summaries) > self.max_input_chars:
            return await self._summarize_long_document(
                combined_summaries,
                summary_type,
                document_id,
            )

        reduce_prompt = self.MAP_REDUCE_REDUCE_PROMPT.format(
            summaries=combined_summaries
        )

        final_response = await self.llm.generate(
            prompt=reduce_prompt,
            system_prompt=self.LEGAL_SUMMARY_SYSTEM_PROMPT,
        )

        # Apply summary type formatting if needed
        if summary_type == "executive":
            formatting_prompt = f"""Reformat this summary as an executive summary:

{final_response.text}

Format as:
EXECUTIVE SUMMARY
[2-3 sentence overview]

KEY POINTS
• [bullet points]

ACTION ITEMS / CONCERNS
• [items requiring attention]"""

            formatted = await self.llm.generate(
                prompt=formatting_prompt,
                system_prompt=self.LEGAL_SUMMARY_SYSTEM_PROMPT,
            )
            final_text = formatted.text
        else:
            final_text = final_response.text

        return {
            "summary": final_text,
            "model_used": final_response.model,
            "summary_length": len(final_text),
            "method": "map_reduce",
            "chunks_processed": len(chunks),
        }

    async def generate_title(self, document_text: str) -> str:
        """Generate a descriptive title for the document."""
        prompt = f"""Generate a concise, descriptive title for this legal document.
The title should identify the document type and main parties/subject.
Return only the title, no explanation.

Document excerpt:
{document_text[:3000]}"""

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=self.LEGAL_SUMMARY_SYSTEM_PROMPT,
            max_tokens=100,
        )

        return response.text.strip().strip('"\'')

    async def extract_key_facts(self, document_text: str) -> list[dict[str, Any]]:
        """Extract key facts from document as structured data."""
        prompt = f"""Extract key facts from this legal document.
Return as JSON array with objects containing:
- "fact": the key fact
- "category": one of [parties, dates, financial, obligations, rights, conditions, definitions]
- "importance": one of [critical, high, medium, low]

Document:
{document_text[:8000]}

Return only valid JSON array:"""

        try:
            result = await self.llm.generate_json(
                prompt=prompt,
                system_prompt=self.LEGAL_SUMMARY_SYSTEM_PROMPT,
            )

            if isinstance(result, dict) and "items" in result:
                return result["items"]
            elif isinstance(result, list):
                return result
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to extract key facts: {e}")
            return []

    async def compare_documents(
        self,
        document1_text: str,
        document2_text: str,
        document1_name: str = "Document 1",
        document2_name: str = "Document 2",
    ) -> dict[str, Any]:
        """Compare two documents and identify differences."""
        # Summarize both documents first if they're long
        if len(document1_text) > self.max_input_chars:
            summary1 = await self.summarize(document1_text, "bullet_points")
            doc1_content = summary1["summary"]
        else:
            doc1_content = document1_text

        if len(document2_text) > self.max_input_chars:
            summary2 = await self.summarize(document2_text, "bullet_points")
            doc2_content = summary2["summary"]
        else:
            doc2_content = document2_text

        prompt = f"""Compare these two legal documents and identify:
1. Key similarities
2. Key differences
3. Missing provisions in each
4. Conflicting terms
5. Recommendation on which is more favorable (and to whom)

{document1_name}:
{doc1_content[:4000]}

{document2_name}:
{doc2_content[:4000]}

Provide structured analysis:"""

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=self.LEGAL_SUMMARY_SYSTEM_PROMPT,
        )

        return {
            "comparison": response.text,
            "document1_name": document1_name,
            "document2_name": document2_name,
        }
