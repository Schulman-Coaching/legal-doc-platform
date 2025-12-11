"""Tests for Vector Store implementations."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from src.ai_ml.vector_store import FAISSVectorStore, VectorStore
from src.ai_ml.models import DocumentEmbedding, SimilarDocument


class TestFAISSVectorStoreInit:
    """Test FAISSVectorStore initialization."""

    def test_default_init(self):
        """Test default initialization."""
        store = FAISSVectorStore(dimension=384)
        assert store.dimension == 384
        assert store.index_type == "flat"
        assert store.persist_path is None

    def test_custom_init(self):
        """Test custom initialization."""
        store = FAISSVectorStore(
            dimension=512,
            index_type="hnsw",
            persist_path="/tmp/test",
        )
        assert store.dimension == 512
        assert store.index_type == "hnsw"
        assert store.persist_path == "/tmp/test"


class TestFAISSVectorStoreOperations:
    """Test FAISSVectorStore basic operations."""

    @pytest.mark.asyncio
    async def test_add_single(self):
        """Test adding single document."""
        store = FAISSVectorStore(dimension=3)

        await store.add(
            document_id="doc-1",
            embedding=[1.0, 0.0, 0.0],
            metadata={"type": "contract"},
        )

        count = await store.count()
        assert count == 1

    @pytest.mark.asyncio
    async def test_add_multiple(self):
        """Test adding multiple documents."""
        store = FAISSVectorStore(dimension=3)

        await store.add("doc-1", [1.0, 0.0, 0.0])
        await store.add("doc-2", [0.0, 1.0, 0.0])
        await store.add("doc-3", [0.0, 0.0, 1.0])

        count = await store.count()
        assert count == 3

    @pytest.mark.asyncio
    async def test_add_batch(self):
        """Test batch adding documents."""
        store = FAISSVectorStore(dimension=3)

        await store.add_batch(
            document_ids=["doc-1", "doc-2"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            metadatas=[{"idx": 1}, {"idx": 2}],
        )

        count = await store.count()
        assert count == 2

    @pytest.mark.asyncio
    async def test_get_existing(self):
        """Test getting existing document."""
        store = FAISSVectorStore(dimension=3)

        await store.add("doc-1", [1.0, 0.0, 0.0], {"type": "test"})

        result = await store.get("doc-1")

        assert result is not None
        assert result.document_id == "doc-1"
        assert result.metadata == {"type": "test"}

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        """Test getting non-existent document."""
        store = FAISSVectorStore(dimension=3)

        result = await store.get("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_existing(self):
        """Test deleting existing document."""
        store = FAISSVectorStore(dimension=3)

        await store.add("doc-1", [1.0, 0.0, 0.0])
        deleted = await store.delete("doc-1")

        assert deleted is True
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        """Test deleting non-existent document."""
        store = FAISSVectorStore(dimension=3)

        deleted = await store.delete("nonexistent")

        assert deleted is False

    @pytest.mark.asyncio
    async def test_update_existing(self):
        """Test updating existing document."""
        store = FAISSVectorStore(dimension=3)

        await store.add("doc-1", [1.0, 0.0, 0.0], {"version": 1})
        await store.add("doc-1", [0.0, 1.0, 0.0], {"version": 2})

        result = await store.get("doc-1")

        assert result.metadata["version"] == 2
        # Embedding should be updated (normalized)
        assert await store.count() == 1


class TestFAISSVectorStoreSearch:
    """Test FAISSVectorStore search operations."""

    @pytest.mark.asyncio
    async def test_search_empty_store(self):
        """Test searching empty store."""
        store = FAISSVectorStore(dimension=3)

        results = await store.search([1.0, 0.0, 0.0])

        assert results == []

    @pytest.mark.asyncio
    async def test_search_basic(self):
        """Test basic search."""
        store = FAISSVectorStore(dimension=3)

        await store.add("doc-1", [1.0, 0.0, 0.0])
        await store.add("doc-2", [0.0, 1.0, 0.0])
        await store.add("doc-3", [0.0, 0.0, 1.0])

        results = await store.search([1.0, 0.0, 0.0], top_k=2)

        assert len(results) <= 2
        # First result should be most similar
        assert results[0].document_id == "doc-1"
        assert results[0].similarity_score > 0.9

    @pytest.mark.asyncio
    async def test_search_with_min_score(self):
        """Test search with minimum score filter."""
        store = FAISSVectorStore(dimension=3)

        await store.add("doc-1", [1.0, 0.0, 0.0])
        await store.add("doc-2", [0.0, 1.0, 0.0])

        # Query orthogonal to both - should get low scores
        results = await store.search([0.0, 0.0, 1.0], min_score=0.9)

        # No results should meet the minimum score
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_returns_similar_documents(self):
        """Test that search returns SimilarDocument objects."""
        store = FAISSVectorStore(dimension=3)

        await store.add("doc-1", [1.0, 0.0, 0.0], {"category": "legal"})

        results = await store.search([1.0, 0.0, 0.0])

        assert len(results) > 0
        assert isinstance(results[0], SimilarDocument)
        assert results[0].metadata == {"category": "legal"}

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self):
        """Test search with metadata filtering."""
        store = FAISSVectorStore(dimension=3)

        await store.add("doc-1", [1.0, 0.0, 0.0], {"type": "contract"})
        await store.add("doc-2", [0.9, 0.1, 0.0], {"type": "memo"})
        await store.add("doc-3", [0.8, 0.2, 0.0], {"type": "contract"})

        results = await store.search(
            [1.0, 0.0, 0.0],
            filter_metadata={"type": "contract"},
        )

        # Should only return contract documents
        for result in results:
            assert result.metadata.get("type") == "contract"

    @pytest.mark.asyncio
    async def test_search_top_k_limit(self):
        """Test that top_k limits results."""
        store = FAISSVectorStore(dimension=3)

        for i in range(10):
            emb = [1.0 - i * 0.1, i * 0.1, 0.0]
            await store.add(f"doc-{i}", emb)

        results = await store.search([1.0, 0.0, 0.0], top_k=3)

        assert len(results) == 3


class TestFAISSVectorStoreNormalization:
    """Test embedding normalization for cosine similarity."""

    @pytest.mark.asyncio
    async def test_embeddings_are_normalized(self):
        """Test that embeddings are normalized when added."""
        store = FAISSVectorStore(dimension=3)

        # Add unnormalized embedding
        await store.add("doc-1", [2.0, 0.0, 0.0])

        # Retrieved embedding should be normalized
        result = await store.get("doc-1")
        emb = np.array(result.embedding)
        norm = np.linalg.norm(emb)

        assert abs(norm - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_similar_directions_high_score(self):
        """Test that similar direction vectors get high scores."""
        store = FAISSVectorStore(dimension=3)

        await store.add("doc-1", [1.0, 1.0, 0.0])

        # Same direction, different magnitude
        results = await store.search([2.0, 2.0, 0.0])

        assert results[0].similarity_score > 0.99


class TestFAISSVectorStoreMetadataFilter:
    """Test metadata filtering functionality."""

    @pytest.mark.asyncio
    async def test_filter_exact_match(self):
        """Test exact match filtering."""
        store = FAISSVectorStore(dimension=3)

        await store.add("doc-1", [1.0, 0.0, 0.0], {"status": "active"})
        await store.add("doc-2", [0.9, 0.1, 0.0], {"status": "inactive"})

        results = await store.search(
            [1.0, 0.0, 0.0],
            filter_metadata={"status": "active"},
        )

        assert all(r.metadata.get("status") == "active" for r in results)

    @pytest.mark.asyncio
    async def test_filter_list_membership(self):
        """Test list membership filtering."""
        store = FAISSVectorStore(dimension=3)

        await store.add("doc-1", [1.0, 0.0, 0.0], {"type": "contract"})
        await store.add("doc-2", [0.9, 0.1, 0.0], {"type": "memo"})
        await store.add("doc-3", [0.8, 0.2, 0.0], {"type": "letter"})

        results = await store.search(
            [1.0, 0.0, 0.0],
            filter_metadata={"type": ["contract", "memo"]},
        )

        for result in results:
            assert result.metadata.get("type") in ["contract", "memo"]

    @pytest.mark.asyncio
    async def test_filter_missing_key(self):
        """Test filtering when key is missing."""
        store = FAISSVectorStore(dimension=3)

        await store.add("doc-1", [1.0, 0.0, 0.0], {"type": "contract"})
        await store.add("doc-2", [0.9, 0.1, 0.0], {})  # No type

        results = await store.search(
            [1.0, 0.0, 0.0],
            filter_metadata={"type": "contract"},
        )

        # Should only return doc with matching type
        assert len(results) == 1
        assert results[0].document_id == "doc-1"


class TestFAISSVectorStoreNumpySearch:
    """Test numpy-based fallback search."""

    @pytest.mark.asyncio
    async def test_numpy_search_matches_faiss(self):
        """Test that numpy search produces similar results."""
        store = FAISSVectorStore(dimension=3)

        await store.add("doc-1", [1.0, 0.0, 0.0])
        await store.add("doc-2", [0.0, 1.0, 0.0])

        # Force numpy search by calling internal method
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        query = query / np.linalg.norm(query)

        results = await store._numpy_search(query, top_k=2, min_score=0.0, filter_metadata=None)

        assert len(results) == 2
        assert results[0].document_id == "doc-1"


class TestEdgeCases:
    """Test edge cases."""

    @pytest.mark.asyncio
    async def test_empty_embedding(self):
        """Test handling of zero vector."""
        store = FAISSVectorStore(dimension=3)

        # Zero vector - should handle gracefully
        await store.add("doc-1", [0.0, 0.0, 0.0])

        count = await store.count()
        assert count == 1

    @pytest.mark.asyncio
    async def test_single_element_search(self):
        """Test search with single element in store."""
        store = FAISSVectorStore(dimension=3)

        await store.add("doc-1", [1.0, 0.0, 0.0])

        results = await store.search([1.0, 0.0, 0.0], top_k=10)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_large_top_k(self):
        """Test top_k larger than store size."""
        store = FAISSVectorStore(dimension=3)

        await store.add("doc-1", [1.0, 0.0, 0.0])
        await store.add("doc-2", [0.0, 1.0, 0.0])

        results = await store.search([1.0, 0.0, 0.0], top_k=100)

        # Should return all documents
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test concurrent add operations."""
        import asyncio

        store = FAISSVectorStore(dimension=3)

        async def add_doc(doc_id):
            await store.add(doc_id, [1.0, 0.0, 0.0])

        # Add documents concurrently
        await asyncio.gather(*[add_doc(f"doc-{i}") for i in range(10)])

        count = await store.count()
        assert count == 10
