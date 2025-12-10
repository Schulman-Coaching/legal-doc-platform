"""
Tests for Deduplication Service
===============================
"""

import pytest
import hashlib

from ..services.deduplication_service import (
    DeduplicationService,
    DuplicateType,
    DuplicateAction,
    DocumentFingerprint,
)


class TestDeduplicationService:
    """Test suite for DeduplicationService."""

    @pytest.fixture
    def dedup_service(self):
        """Create deduplication service instance."""
        return DeduplicationService(
            exact_match_threshold=1.0,
            simhash_threshold=3,
            minhash_threshold=0.85,
        )

    @pytest.fixture
    def sample_content(self):
        """Create sample document content."""
        return b"This is a sample legal document with some content for testing deduplication."

    @pytest.fixture
    def similar_content(self):
        """Create similar document content."""
        return b"This is a sample legal document with slightly different content for testing deduplication."

    @pytest.fixture
    def different_content(self):
        """Create completely different document content."""
        return b"Completely unrelated content that should not match anything else in the system."

    @pytest.mark.asyncio
    async def test_exact_duplicate_detection(self, dedup_service, sample_content):
        """Test detection of exact duplicates via hash matching."""
        # Register first document
        await dedup_service.register_document(
            document_id="doc-001",
            content=sample_content,
            filename="document1.pdf",
        )

        # Check same content
        result = await dedup_service.check_duplicate(
            content=sample_content,
            filename="document2.pdf",
        )

        assert result.is_duplicate is True
        assert result.duplicate_type == DuplicateType.EXACT
        assert result.original_document_id == "doc-001"
        assert result.similarity_score == 1.0
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_no_duplicate_different_content(self, dedup_service, sample_content, different_content):
        """Test that different content is not flagged as duplicate."""
        # Register first document
        await dedup_service.register_document(
            document_id="doc-001",
            content=sample_content,
            filename="document1.pdf",
        )

        # Check different content
        result = await dedup_service.check_duplicate(
            content=different_content,
            filename="document2.pdf",
        )

        assert result.is_duplicate is False

    @pytest.mark.asyncio
    async def test_fingerprint_generation(self, dedup_service, sample_content):
        """Test document fingerprint generation."""
        fingerprint = await dedup_service._generate_fingerprint(
            content=sample_content,
            filename="test.txt",
            metadata={"author": "test"},
        )

        assert fingerprint.sha256 == hashlib.sha256(sample_content).hexdigest()
        assert fingerprint.md5 == hashlib.md5(sample_content).hexdigest()
        assert fingerprint.file_size == len(sample_content)

    @pytest.mark.asyncio
    async def test_register_multiple_documents(self, dedup_service):
        """Test registering multiple documents."""
        docs = [
            ("doc-001", b"First document content"),
            ("doc-002", b"Second document content"),
            ("doc-003", b"Third document content"),
        ]

        for doc_id, content in docs:
            fingerprint = await dedup_service.register_document(
                document_id=doc_id,
                content=content,
                filename=f"{doc_id}.txt",
            )
            assert fingerprint.document_id == doc_id

        # Verify index stats
        stats = dedup_service.get_stats()
        assert stats["exact_hash_count"] == 3

    @pytest.mark.asyncio
    async def test_simhash_computation(self, dedup_service):
        """Test SimHash computation for text similarity."""
        text = "This is a test document with some legal text for testing simhash computation"

        simhash = dedup_service._compute_simhash(text)

        assert isinstance(simhash, int)
        assert simhash >= 0

    @pytest.mark.asyncio
    async def test_simhash_similar_texts(self, dedup_service):
        """Test that similar texts produce similar SimHashes."""
        text1 = "This is a sample legal document about contract law"
        text2 = "This is a sample legal document about contract terms"

        hash1 = dedup_service._compute_simhash(text1)
        hash2 = dedup_service._compute_simhash(text2)

        distance = dedup_service._hamming_distance(hash1, hash2)

        # Similar texts should have small Hamming distance
        assert distance < 20  # Reasonable threshold for similarity

    @pytest.mark.asyncio
    async def test_simhash_different_texts(self, dedup_service):
        """Test that different texts produce different SimHashes."""
        text1 = "This is about contract law and legal obligations"
        text2 = "Football statistics and player performance analysis"

        hash1 = dedup_service._compute_simhash(text1)
        hash2 = dedup_service._compute_simhash(text2)

        distance = dedup_service._hamming_distance(hash1, hash2)

        # Different texts should have large Hamming distance
        assert distance > 10

    @pytest.mark.asyncio
    async def test_minhash_computation(self, dedup_service):
        """Test MinHash signature computation."""
        text = "This is a test document with enough words for creating shingles and computing minhash"

        minhash = dedup_service._compute_minhash(text)

        assert isinstance(minhash, list)
        assert len(minhash) == dedup_service.num_perm

    @pytest.mark.asyncio
    async def test_hamming_distance(self, dedup_service):
        """Test Hamming distance calculation."""
        # Same values
        assert dedup_service._hamming_distance(0, 0) == 0

        # One bit different
        assert dedup_service._hamming_distance(1, 0) == 1
        assert dedup_service._hamming_distance(2, 0) == 1

        # Multiple bits
        assert dedup_service._hamming_distance(7, 0) == 3  # 111 vs 000
        assert dedup_service._hamming_distance(15, 0) == 4  # 1111 vs 0000

    @pytest.mark.asyncio
    async def test_clear_index(self, dedup_service, sample_content):
        """Test clearing the deduplication index."""
        # Register some documents
        await dedup_service.register_document("doc-001", sample_content, "test.pdf")

        stats_before = dedup_service.get_stats()
        assert stats_before["exact_hash_count"] > 0

        # Clear index
        await dedup_service.clear_index()

        stats_after = dedup_service.get_stats()
        assert stats_after["exact_hash_count"] == 0

    @pytest.mark.asyncio
    async def test_duplicate_result_action(self, dedup_service, sample_content):
        """Test that correct action is assigned to duplicates."""
        await dedup_service.register_document("doc-001", sample_content, "test.pdf")

        result = await dedup_service.check_duplicate(sample_content, "test2.pdf")

        assert result.is_duplicate is True
        assert result.action == DuplicateAction.REJECT  # Exact match

    @pytest.mark.asyncio
    async def test_metadata_in_check(self, dedup_service, sample_content):
        """Test duplicate check with metadata."""
        result = await dedup_service.check_duplicate(
            content=sample_content,
            filename="test.pdf",
            metadata={"author": "John Doe", "date": "2024-01-15"},
        )

        # With no registered documents, should not be duplicate
        assert result.is_duplicate is False


class TestPhashComputation:
    """Test perceptual hash computation for images."""

    @pytest.fixture
    def dedup_service(self):
        return DeduplicationService()

    def test_is_image_detection(self, dedup_service):
        """Test image file detection."""
        image_files = ["photo.jpg", "scan.png", "document.tiff", "image.gif"]
        non_image_files = ["document.pdf", "contract.docx", "data.csv"]

        for filename in image_files:
            assert dedup_service._is_image(filename) is True

        for filename in non_image_files:
            assert dedup_service._is_image(filename) is False

    @pytest.mark.asyncio
    async def test_phash_returns_none_for_invalid_image(self, dedup_service):
        """Test that pHash returns None for invalid image data."""
        invalid_data = b"This is not an image"

        phash = dedup_service._compute_phash(invalid_data)

        assert phash is None
