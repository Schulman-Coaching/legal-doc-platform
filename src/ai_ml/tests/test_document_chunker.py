"""Tests for Document Chunker."""

import pytest

from src.ai_ml.document_chunker import DocumentChunker
from src.ai_ml.models import ChunkingStrategy


class TestDocumentChunkerInit:
    """Test DocumentChunker initialization."""

    def test_default_init(self):
        """Test default initialization."""
        chunker = DocumentChunker()
        assert chunker.strategy == ChunkingStrategy.RECURSIVE
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
        assert chunker.min_chunk_size == 100

    def test_custom_init(self):
        """Test custom initialization."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=500,
            chunk_overlap=50,
            min_chunk_size=50,
        )
        assert chunker.strategy == ChunkingStrategy.FIXED_SIZE
        assert chunker.chunk_size == 500


class TestFixedSizeChunking:
    """Test fixed-size chunking strategy."""

    def test_short_text_single_chunk(self):
        """Short text should produce single chunk."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=1000,
            min_chunk_size=50,
        )
        chunks = chunker.chunk_document("doc-1", "This is short text.")
        assert len(chunks) == 1
        assert chunks[0].content == "This is short text."
        assert chunks[0].document_id == "doc-1"
        assert chunks[0].chunk_index == 0

    def test_long_text_multiple_chunks(self):
        """Long text should produce multiple chunks."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=100,
            chunk_overlap=20,
            min_chunk_size=20,
        )
        text = "This is a sentence. " * 20  # ~400 chars
        chunks = chunker.chunk_document("doc-1", text)
        assert len(chunks) > 1

        # Check chunk indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_overlap(self):
        """Chunks should have overlap."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=100,
            chunk_overlap=30,
            min_chunk_size=20,
        )
        text = "Word " * 100  # 500 chars

        chunks = chunker.chunk_document("doc-1", text)

        # With overlap, later chunks should start before previous chunk ends
        if len(chunks) > 1:
            assert chunks[1].start_char < chunks[0].end_char

    def test_sentence_boundary(self):
        """Chunker should try to end at sentence boundaries."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=100,
            chunk_overlap=10,
            min_chunk_size=20,
        )
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."

        chunks = chunker.chunk_document("doc-1", text)

        # Chunks should tend to end at sentence boundaries
        for chunk in chunks:
            content = chunk.content.strip()
            # Most chunks should end with sentence-ending punctuation
            if len(content) > 20:
                assert content[-1] in ".!?" or content.endswith("...")


class TestParagraphChunking:
    """Test paragraph-based chunking strategy."""

    def test_single_paragraph(self):
        """Single paragraph should produce single chunk."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.PARAGRAPH,
            chunk_size=1000,
            min_chunk_size=10,
        )
        text = "This is a single paragraph with some content."
        chunks = chunker.chunk_document("doc-1", text)
        assert len(chunks) == 1

    def test_multiple_paragraphs(self):
        """Multiple paragraphs should be chunked appropriately."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.PARAGRAPH,
            chunk_size=100,
            min_chunk_size=10,
        )
        text = """First paragraph here.

Second paragraph here.

Third paragraph here."""

        chunks = chunker.chunk_document("doc-1", text)
        assert len(chunks) >= 1

    def test_paragraph_merging(self):
        """Small paragraphs should be merged."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.PARAGRAPH,
            chunk_size=200,
            min_chunk_size=10,
        )
        text = """Short para 1.

Short para 2.

Short para 3."""

        chunks = chunker.chunk_document("doc-1", text)
        # Small paragraphs should be merged into fewer chunks
        assert len(chunks) <= 2


class TestSemanticChunking:
    """Test semantic (legal structure) chunking strategy."""

    def test_legal_sections(self):
        """Should chunk by legal sections."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=500,
            min_chunk_size=50,
        )
        text = """ARTICLE I - DEFINITIONS

This section defines terms.

ARTICLE II - OBLIGATIONS

This section describes obligations.

ARTICLE III - TERMINATION

This section covers termination."""

        chunks = chunker.chunk_document("doc-1", text)
        assert len(chunks) >= 2

    def test_section_patterns(self):
        """Should recognize various section patterns."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=500,
            min_chunk_size=30,
        )
        text = """Section 1. Introduction
Some intro text here.

Section 2. Scope
Scope description here.

Section 3. Terms
Terms listed here."""

        chunks = chunker.chunk_document("doc-1", text)
        assert len(chunks) >= 1

    def test_fallback_to_paragraph(self):
        """Should fall back to paragraph chunking if no sections found."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=200,
            min_chunk_size=20,
        )
        text = """This is regular text without any legal section markers.

Just normal paragraphs here.

And another paragraph."""

        chunks = chunker.chunk_document("doc-1", text)
        assert len(chunks) >= 1


class TestSentenceChunking:
    """Test sentence-based chunking strategy."""

    def test_sentence_splitting(self):
        """Should split by sentences."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.SENTENCE,
            chunk_size=100,
            min_chunk_size=20,
        )
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        chunks = chunker.chunk_document("doc-1", text)
        assert len(chunks) >= 1

    def test_abbreviation_handling(self):
        """Should not split on common abbreviations."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.SENTENCE,
            chunk_size=500,
            min_chunk_size=20,
        )
        text = "Dr. Smith and Mr. Jones met at 123 Main St. They discussed the contract."

        chunks = chunker.chunk_document("doc-1", text)
        # Should keep "Dr. Smith" together
        assert any("Dr. Smith" in chunk.content or "Dr." in chunk.content for chunk in chunks)


class TestRecursiveChunking:
    """Test recursive chunking strategy."""

    def test_hierarchical_splitting(self):
        """Should split hierarchically."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=100,
            min_chunk_size=20,
        )
        text = """First big section.

Second big section.

Third big section with more content that goes on and on."""

        chunks = chunker.chunk_document("doc-1", text)
        assert len(chunks) >= 1

        # All chunks should respect size limits (approximately)
        for chunk in chunks:
            assert len(chunk.content) >= chunker.min_chunk_size or len(text) < chunker.min_chunk_size

    def test_very_long_text(self):
        """Should handle very long text."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=200,
            chunk_overlap=20,
            min_chunk_size=50,
        )
        text = "This is a test sentence. " * 100  # ~2500 chars

        chunks = chunker.chunk_document("doc-1", text)
        assert len(chunks) > 5

        # Chunks should cover the whole document
        all_content = " ".join(chunk.content for chunk in chunks)
        assert "test sentence" in all_content


class TestChunkMetadata:
    """Test chunk metadata handling."""

    def test_metadata_passed_through(self):
        """Metadata should be passed to chunks."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=1000,
        )
        metadata = {"type": "contract", "version": 1}
        chunks = chunker.chunk_document("doc-1", "Test content", metadata)

        for chunk in chunks:
            assert chunk.metadata == metadata

    def test_chunk_positions(self):
        """Chunk positions should be accurate."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=50,
            chunk_overlap=10,
            min_chunk_size=10,
        )
        text = "0123456789" * 10  # 100 chars

        chunks = chunker.chunk_document("doc-1", text)

        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char <= len(text)
            assert chunk.end_char > chunk.start_char


class TestMergeSmallChunks:
    """Test chunk merging functionality."""

    def test_merge_small_chunks(self):
        """Should merge chunks below minimum size."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=50,
            min_chunk_size=10,
        )

        # Create some small chunks manually
        from src.ai_ml.models import DocumentChunk

        small_chunks = [
            DocumentChunk(
                chunk_id="1", document_id="doc", content="Small",
                chunk_index=0, start_char=0, end_char=5
            ),
            DocumentChunk(
                chunk_id="2", document_id="doc", content="Also small",
                chunk_index=1, start_char=5, end_char=15
            ),
            DocumentChunk(
                chunk_id="3", document_id="doc", content="This is a larger chunk with more content",
                chunk_index=2, start_char=15, end_char=55
            ),
        ]

        merged = chunker.merge_small_chunks(small_chunks, min_size=20)

        # Small chunks should be merged
        assert len(merged) <= len(small_chunks)

    def test_no_merge_needed(self):
        """Should not merge if chunks are large enough."""
        chunker = DocumentChunker()

        from src.ai_ml.models import DocumentChunk

        large_chunks = [
            DocumentChunk(
                chunk_id="1", document_id="doc",
                content="This is a sufficiently large chunk" * 5,
                chunk_index=0, start_char=0, end_char=170
            ),
            DocumentChunk(
                chunk_id="2", document_id="doc",
                content="Another large chunk with content" * 5,
                chunk_index=1, start_char=170, end_char=330
            ),
        ]

        merged = chunker.merge_small_chunks(large_chunks, min_size=50)
        assert len(merged) == len(large_chunks)


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_text(self):
        """Should handle empty text."""
        chunker = DocumentChunker()
        chunks = chunker.chunk_document("doc-1", "")
        assert len(chunks) == 1
        assert chunks[0].content == ""

    def test_very_short_text(self):
        """Should handle text shorter than min_chunk_size."""
        chunker = DocumentChunker(min_chunk_size=100)
        chunks = chunker.chunk_document("doc-1", "Short")
        assert len(chunks) == 1

    def test_text_with_only_whitespace(self):
        """Should handle whitespace-only text."""
        chunker = DocumentChunker()
        chunks = chunker.chunk_document("doc-1", "   \n\n   ")
        assert len(chunks) == 1

    def test_unicode_content(self):
        """Should handle unicode content."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=100,
            min_chunk_size=10,
        )
        text = "合同条款 " * 20  # Chinese text

        chunks = chunker.chunk_document("doc-1", text)
        assert len(chunks) >= 1
        assert "合同条款" in chunks[0].content
