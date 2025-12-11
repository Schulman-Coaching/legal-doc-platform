"""
Embedding Service
==================
Generate embeddings using sentence-transformers or API-based models.
"""

import asyncio
import hashlib
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx
import numpy as np

from .models import DocumentEmbedding

logger = logging.getLogger(__name__)


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced by this model."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        pass


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """Embedding model using sentence-transformers library."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name, device=self._device)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Loaded embedding model: {self._model_name} (dim={self._dimension})")
            except ImportError:
                logger.warning("sentence-transformers not installed, using fallback")
                self._model = "fallback"
                self._dimension = 384

    @property
    def embedding_dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension or 384

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        self._load_model()

        if self._model == "fallback":
            return self._fallback_embed(text)

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode(text, convert_to_numpy=True)
        )
        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        self._load_model()

        if self._model == "fallback":
            return [self._fallback_embed(text) for text in texts]

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        )
        return embeddings.tolist()

    def _fallback_embed(self, text: str) -> list[float]:
        """Fallback embedding using hash when model not available."""
        text_hash = hashlib.sha256(text.encode()).digest()
        embedding = [float(b) / 255.0 - 0.5 for b in text_hash]
        # Pad or truncate to dimension
        while len(embedding) < 384:
            embedding.extend(embedding[:384 - len(embedding)])
        return embedding[:384]


class OpenAIEmbedding(BaseEmbeddingModel):
    """Embedding model using OpenAI API."""

    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        self._model_name = model_name
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._dimensions = dimensions
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def embedding_dimension(self) -> int:
        if self._dimensions:
            return self._dimensions
        return self.MODEL_DIMENSIONS.get(self._model_name, 1536)

    @property
    def model_name(self) -> str:
        return self._model_name

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Generate embedding using OpenAI API."""
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts using OpenAI API."""
        if not self._api_key:
            raise ValueError("OpenAI API key not configured")

        client = await self._get_client()

        request_body = {
            "model": self._model_name,
            "input": texts,
        }

        if self._dimensions and self._model_name.startswith("text-embedding-3"):
            request_body["dimensions"] = self._dimensions

        response = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=request_body,
        )

        response.raise_for_status()
        data = response.json()

        # Sort by index to ensure order matches input
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


class CohereEmbedding(BaseEmbeddingModel):
    """Embedding model using Cohere API."""

    def __init__(
        self,
        model_name: str = "embed-english-v3.0",
        api_key: Optional[str] = None,
        input_type: str = "search_document",
    ):
        self._model_name = model_name
        self._api_key = api_key or os.getenv("COHERE_API_KEY")
        self._input_type = input_type
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def embedding_dimension(self) -> int:
        return 1024

    @property
    def model_name(self) -> str:
        return self._model_name

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def embed(self, text: str) -> list[float]:
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not self._api_key:
            raise ValueError("Cohere API key not configured")

        client = await self._get_client()

        response = await client.post(
            "https://api.cohere.ai/v1/embed",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model_name,
                "texts": texts,
                "input_type": self._input_type,
            },
        )

        response.raise_for_status()
        data = response.json()
        return data["embeddings"]

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


class EmbeddingService:
    """
    High-level service for generating and managing embeddings.
    Supports multiple backends with caching and batching.
    """

    def __init__(
        self,
        model: Optional[BaseEmbeddingModel] = None,
        cache_service: Optional[Any] = None,
        batch_size: int = 32,
    ):
        self.model = model or SentenceTransformerEmbedding()
        self.cache = cache_service
        self.batch_size = batch_size

    @property
    def embedding_dimension(self) -> int:
        return self.model.embedding_dimension

    async def get_embedding(
        self,
        text: str,
        document_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> DocumentEmbedding:
        """Get embedding for a single text, with caching."""
        cache_key = f"emb:{hashlib.sha256(text.encode()).hexdigest()[:32]}"

        if use_cache and self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                return DocumentEmbedding(
                    document_id=document_id or cache_key,
                    embedding=cached["embedding"],
                    model_name=self.model.model_name,
                    metadata={"cached": True},
                )

        embedding = await self.model.embed(text)

        if use_cache and self.cache:
            await self.cache.set(cache_key, {"embedding": embedding}, ttl=86400)

        return DocumentEmbedding(
            document_id=document_id or cache_key,
            embedding=embedding,
            model_name=self.model.model_name,
        )

    async def get_embeddings_batch(
        self,
        texts: list[str],
        document_ids: Optional[list[str]] = None,
    ) -> list[DocumentEmbedding]:
        """Get embeddings for multiple texts."""
        if document_ids is None:
            document_ids = [f"doc_{i}" for i in range(len(texts))]

        results = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_ids = document_ids[i:i + self.batch_size]

            embeddings = await self.model.embed_batch(batch_texts)

            for doc_id, embedding in zip(batch_ids, embeddings):
                results.append(DocumentEmbedding(
                    document_id=doc_id,
                    embedding=embedding,
                    model_name=self.model.model_name,
                ))

        return results

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        a_np = np.array(a)
        b_np = np.array(b)

        dot_product = np.dot(a_np, b_np)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))
