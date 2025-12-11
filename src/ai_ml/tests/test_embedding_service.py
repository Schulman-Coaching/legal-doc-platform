"""Tests for Embedding Service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.ai_ml.embedding_service import (
    BaseEmbeddingModel,
    SentenceTransformerEmbedding,
    OpenAIEmbedding,
    EmbeddingService,
)
from src.ai_ml.models import DocumentEmbedding


class TestSentenceTransformerEmbedding:
    """Test SentenceTransformerEmbedding class."""

    def test_init(self):
        """Test initialization."""
        model = SentenceTransformerEmbedding()
        assert model._model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert model._device == "cpu"
        assert model._model is None

    def test_model_name_property(self):
        """Test model_name property."""
        model = SentenceTransformerEmbedding("custom-model")
        assert model.model_name == "custom-model"

    def test_embedding_dimension_fallback(self):
        """Test embedding dimension with fallback."""
        model = SentenceTransformerEmbedding()
        # Without loading the actual model, should use fallback
        dim = model.embedding_dimension
        assert dim == 384  # Fallback dimension

    @pytest.mark.asyncio
    async def test_embed_fallback(self):
        """Test embedding generation with fallback."""
        model = SentenceTransformerEmbedding()
        # Force fallback mode
        model._model = "fallback"
        model._dimension = 384

        embedding = await model.embed("Test text")

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_batch_fallback(self):
        """Test batch embedding with fallback."""
        model = SentenceTransformerEmbedding()
        model._model = "fallback"
        model._dimension = 384

        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = await model.embed_batch(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 384

    @pytest.mark.asyncio
    async def test_different_texts_different_embeddings(self):
        """Test that different texts produce different embeddings."""
        model = SentenceTransformerEmbedding()
        model._model = "fallback"
        model._dimension = 384

        emb1 = await model.embed("Contract terms")
        emb2 = await model.embed("Different content")

        # Different texts should produce different embeddings
        assert emb1 != emb2


class TestOpenAIEmbedding:
    """Test OpenAIEmbedding class."""

    def test_init(self):
        """Test initialization."""
        model = OpenAIEmbedding()
        assert model._model_name == "text-embedding-3-small"
        assert model._dimensions is None

    def test_model_dimensions(self):
        """Test embedding dimensions for different models."""
        model_small = OpenAIEmbedding("text-embedding-3-small")
        assert model_small.embedding_dimension == 1536

        model_large = OpenAIEmbedding("text-embedding-3-large")
        assert model_large.embedding_dimension == 3072

        model_ada = OpenAIEmbedding("text-embedding-ada-002")
        assert model_ada.embedding_dimension == 1536

    def test_custom_dimensions(self):
        """Test custom dimension override."""
        model = OpenAIEmbedding("text-embedding-3-small", dimensions=512)
        assert model.embedding_dimension == 512

    @pytest.mark.asyncio
    async def test_embed_without_api_key(self):
        """Test that embedding fails gracefully without API key."""
        model = OpenAIEmbedding()
        model._api_key = None

        with pytest.raises(ValueError, match="API key not configured"):
            await model.embed("Test text")

    @pytest.mark.asyncio
    async def test_embed_with_mock_api(self):
        """Test embedding with mocked API."""
        model = OpenAIEmbedding(api_key="test-key")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1] * 1536}
            ]
        }

        with patch.object(model, '_get_client') as mock_client:
            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_response
            mock_client.return_value = mock_http_client

            embedding = await model.embed("Test")

            assert len(embedding) == 1536
            mock_http_client.post.assert_called_once()


class TestEmbeddingService:
    """Test EmbeddingService class."""

    def test_init_default(self):
        """Test default initialization."""
        service = EmbeddingService()
        assert service.model is not None
        assert service.cache is None
        assert service.batch_size == 32

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        custom_model = SentenceTransformerEmbedding("custom-model")
        service = EmbeddingService(model=custom_model)
        assert service.model == custom_model

    def test_embedding_dimension(self):
        """Test embedding dimension property."""
        service = EmbeddingService()
        dim = service.embedding_dimension
        assert isinstance(dim, int)
        assert dim > 0

    @pytest.mark.asyncio
    async def test_get_embedding(self):
        """Test getting single embedding."""
        # Create service with fallback model
        model = SentenceTransformerEmbedding()
        model._model = "fallback"
        model._dimension = 384
        service = EmbeddingService(model=model)

        result = await service.get_embedding("Test text", document_id="doc-1")

        assert isinstance(result, DocumentEmbedding)
        assert result.document_id == "doc-1"
        assert len(result.embedding) == 384

    @pytest.mark.asyncio
    async def test_get_embedding_generates_id(self):
        """Test that document_id is generated if not provided."""
        model = SentenceTransformerEmbedding()
        model._model = "fallback"
        model._dimension = 384
        service = EmbeddingService(model=model)

        result = await service.get_embedding("Test text")

        assert result.document_id is not None
        assert len(result.document_id) > 0

    @pytest.mark.asyncio
    async def test_get_embeddings_batch(self):
        """Test batch embedding generation."""
        model = SentenceTransformerEmbedding()
        model._model = "fallback"
        model._dimension = 384
        service = EmbeddingService(model=model, batch_size=2)

        texts = ["Text 1", "Text 2", "Text 3"]
        doc_ids = ["doc-1", "doc-2", "doc-3"]

        results = await service.get_embeddings_batch(texts, doc_ids)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.document_id == doc_ids[i]
            assert len(result.embedding) == 384

    @pytest.mark.asyncio
    async def test_batch_generates_ids(self):
        """Test that batch generates IDs if not provided."""
        model = SentenceTransformerEmbedding()
        model._model = "fallback"
        model._dimension = 384
        service = EmbeddingService(model=model)

        texts = ["Text 1", "Text 2"]
        results = await service.get_embeddings_batch(texts)

        assert len(results) == 2
        assert results[0].document_id == "doc_0"
        assert results[1].document_id == "doc_1"

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        service = EmbeddingService()

        # Identical vectors should have similarity 1.0
        vec = [1.0, 0.0, 0.0]
        assert abs(service.cosine_similarity(vec, vec) - 1.0) < 0.001

        # Orthogonal vectors should have similarity 0.0
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        assert abs(service.cosine_similarity(vec1, vec2)) < 0.001

        # Opposite vectors should have similarity -1.0
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]
        assert abs(service.cosine_similarity(vec1, vec2) - (-1.0)) < 0.001

    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector."""
        service = EmbeddingService()

        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]

        # Zero vector should return 0.0
        assert service.cosine_similarity(vec1, vec2) == 0.0


class TestEmbeddingServiceWithCache:
    """Test EmbeddingService with caching."""

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test that cached embeddings are returned."""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = {"embedding": [0.5] * 384}

        model = SentenceTransformerEmbedding()
        model._model = "fallback"
        model._dimension = 384
        service = EmbeddingService(model=model, cache_service=mock_cache)

        result = await service.get_embedding("Test text", use_cache=True)

        mock_cache.get.assert_called_once()
        assert result.metadata.get("cached") is True
        assert result.embedding == [0.5] * 384

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss and storage."""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None

        model = SentenceTransformerEmbedding()
        model._model = "fallback"
        model._dimension = 384
        service = EmbeddingService(model=model, cache_service=mock_cache)

        result = await service.get_embedding("Test text", use_cache=True)

        mock_cache.get.assert_called_once()
        mock_cache.set.assert_called_once()
        assert result.metadata.get("cached") is not True

    @pytest.mark.asyncio
    async def test_bypass_cache(self):
        """Test bypassing cache."""
        mock_cache = AsyncMock()

        model = SentenceTransformerEmbedding()
        model._model = "fallback"
        model._dimension = 384
        service = EmbeddingService(model=model, cache_service=mock_cache)

        await service.get_embedding("Test text", use_cache=False)

        mock_cache.get.assert_not_called()
        mock_cache.set.assert_not_called()
