"""
Vector Store
=============
Vector storage and similarity search implementations.
"""

import asyncio
import json
import logging
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .models import DocumentEmbedding, SimilarDocument

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def add(
        self,
        document_id: str,
        embedding: list[float],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add a document embedding to the store."""
        pass

    @abstractmethod
    async def add_batch(
        self,
        document_ids: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Add multiple document embeddings to the store."""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        min_score: float = 0.0,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[SimilarDocument]:
        """Search for similar documents."""
        pass

    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        """Delete a document from the store."""
        pass

    @abstractmethod
    async def get(self, document_id: str) -> Optional[DocumentEmbedding]:
        """Get a document embedding by ID."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Get the number of documents in the store."""
        pass


class FAISSVectorStore(VectorStore):
    """Vector store using FAISS for similarity search."""

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        persist_path: Optional[str] = None,
    ):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Embedding dimension
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            persist_path: Path to persist index to disk
        """
        self.dimension = dimension
        self.index_type = index_type
        self.persist_path = persist_path

        self._index = None
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._next_idx = 0
        self._lock = asyncio.Lock()

        self._faiss_available = self._check_faiss()

        if persist_path and os.path.exists(persist_path):
            self._load()

    def _check_faiss(self) -> bool:
        """Check if FAISS is available."""
        try:
            import faiss
            return True
        except ImportError:
            logger.warning("FAISS not installed, using numpy-based search")
            return False

    def _create_index(self):
        """Create FAISS index."""
        if not self._faiss_available:
            return None

        import faiss

        if self.index_type == "flat":
            index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, 100, faiss.METRIC_INNER_PRODUCT)
        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexFlatIP(self.dimension)

        return index

    async def add(
        self,
        document_id: str,
        embedding: list[float],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add a document embedding to the store."""
        async with self._lock:
            # Normalize embedding for cosine similarity
            emb_np = np.array(embedding, dtype=np.float32)
            emb_np = emb_np / np.linalg.norm(emb_np)
            embedding = emb_np.tolist()

            if document_id in self._id_to_idx:
                # Update existing
                idx = self._id_to_idx[document_id]
                self._embeddings[document_id] = embedding
                if self._faiss_available and self._index is not None:
                    # FAISS doesn't support in-place update, rebuild needed
                    pass
            else:
                idx = self._next_idx
                self._next_idx += 1
                self._id_to_idx[document_id] = idx
                self._idx_to_id[idx] = document_id
                self._embeddings[document_id] = embedding

                if self._faiss_available:
                    if self._index is None:
                        self._index = self._create_index()
                    self._index.add(emb_np.reshape(1, -1))

            if metadata:
                self._metadata[document_id] = metadata

    async def add_batch(
        self,
        document_ids: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Add multiple document embeddings to the store."""
        if metadatas is None:
            metadatas = [None] * len(document_ids)

        for doc_id, emb, meta in zip(document_ids, embeddings, metadatas):
            await self.add(doc_id, emb, meta)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        min_score: float = 0.0,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[SimilarDocument]:
        """Search for similar documents using cosine similarity."""
        async with self._lock:
            if not self._embeddings:
                return []

            # Normalize query
            query_np = np.array(query_embedding, dtype=np.float32)
            query_np = query_np / np.linalg.norm(query_np)

            if self._faiss_available and self._index is not None and self._index.ntotal > 0:
                # Use FAISS search
                scores, indices = self._index.search(query_np.reshape(1, -1), min(top_k * 2, self._index.ntotal))
                scores = scores[0]
                indices = indices[0]

                results = []
                for score, idx in zip(scores, indices):
                    if idx < 0 or score < min_score:
                        continue

                    doc_id = self._idx_to_id.get(idx)
                    if doc_id is None:
                        continue

                    metadata = self._metadata.get(doc_id, {})

                    # Apply metadata filter
                    if filter_metadata:
                        if not self._matches_filter(metadata, filter_metadata):
                            continue

                    results.append(SimilarDocument(
                        document_id=doc_id,
                        similarity_score=float(score),
                        metadata=metadata,
                    ))

                    if len(results) >= top_k:
                        break

                return results
            else:
                # Numpy-based search
                return await self._numpy_search(query_np, top_k, min_score, filter_metadata)

    async def _numpy_search(
        self,
        query_np: np.ndarray,
        top_k: int,
        min_score: float,
        filter_metadata: Optional[dict[str, Any]],
    ) -> list[SimilarDocument]:
        """Fallback search using numpy."""
        scores = []

        for doc_id, embedding in self._embeddings.items():
            emb_np = np.array(embedding, dtype=np.float32)
            score = float(np.dot(query_np, emb_np))

            if score < min_score:
                continue

            metadata = self._metadata.get(doc_id, {})

            if filter_metadata and not self._matches_filter(metadata, filter_metadata):
                continue

            scores.append((doc_id, score, metadata))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return [
            SimilarDocument(
                document_id=doc_id,
                similarity_score=score,
                metadata=metadata,
            )
            for doc_id, score, metadata in scores[:top_k]
        ]

    def _matches_filter(self, metadata: dict[str, Any], filter_metadata: dict[str, Any]) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter_metadata.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True

    async def delete(self, document_id: str) -> bool:
        """Delete a document from the store."""
        async with self._lock:
            if document_id not in self._id_to_idx:
                return False

            idx = self._id_to_idx[document_id]
            del self._id_to_idx[document_id]
            del self._idx_to_id[idx]
            del self._embeddings[document_id]
            self._metadata.pop(document_id, None)

            # Note: FAISS doesn't support efficient deletion
            # For production, consider rebuilding index periodically
            return True

    async def get(self, document_id: str) -> Optional[DocumentEmbedding]:
        """Get a document embedding by ID."""
        if document_id not in self._embeddings:
            return None

        return DocumentEmbedding(
            document_id=document_id,
            embedding=self._embeddings[document_id],
            model_name="unknown",
            metadata=self._metadata.get(document_id, {}),
        )

    async def count(self) -> int:
        """Get the number of documents in the store."""
        return len(self._embeddings)

    def _load(self) -> None:
        """Load index from disk."""
        if not self.persist_path:
            return

        metadata_path = Path(self.persist_path) / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
                self._id_to_idx = data["id_to_idx"]
                self._idx_to_id = data["idx_to_id"]
                self._metadata = data["metadata"]
                self._embeddings = data["embeddings"]
                self._next_idx = data["next_idx"]

        if self._faiss_available:
            index_path = Path(self.persist_path) / "index.faiss"
            if index_path.exists():
                import faiss
                self._index = faiss.read_index(str(index_path))

        logger.info(f"Loaded vector store with {len(self._embeddings)} documents")

    async def save(self) -> None:
        """Save index to disk."""
        if not self.persist_path:
            return

        os.makedirs(self.persist_path, exist_ok=True)

        metadata_path = Path(self.persist_path) / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump({
                "id_to_idx": self._id_to_idx,
                "idx_to_id": self._idx_to_id,
                "metadata": self._metadata,
                "embeddings": self._embeddings,
                "next_idx": self._next_idx,
            }, f)

        if self._faiss_available and self._index is not None:
            import faiss
            index_path = Path(self.persist_path) / "index.faiss"
            faiss.write_index(self._index, str(index_path))

        logger.info(f"Saved vector store with {len(self._embeddings)} documents")


class ChromaVectorStore(VectorStore):
    """Vector store using ChromaDB."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None

    def _init_client(self):
        """Initialize ChromaDB client."""
        if self._client is not None:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            if self.persist_directory:
                self._client = chromadb.Client(Settings(
                    persist_directory=self.persist_directory,
                    anonymized_telemetry=False,
                ))
            else:
                self._client = chromadb.Client()

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(f"Initialized ChromaDB collection: {self.collection_name}")

        except ImportError:
            logger.error("ChromaDB not installed")
            raise

    async def add(
        self,
        document_id: str,
        embedding: list[float],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self._init_client()

        # Clean metadata for ChromaDB (only str, int, float, bool allowed)
        clean_metadata = {}
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_metadata[k] = v
                else:
                    clean_metadata[k] = str(v)

        self._collection.upsert(
            ids=[document_id],
            embeddings=[embedding],
            metadatas=[clean_metadata] if clean_metadata else None,
        )

    async def add_batch(
        self,
        document_ids: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self._init_client()

        clean_metadatas = None
        if metadatas:
            clean_metadatas = []
            for meta in metadatas:
                clean = {}
                if meta:
                    for k, v in meta.items():
                        if isinstance(v, (str, int, float, bool)):
                            clean[k] = v
                        else:
                            clean[k] = str(v)
                clean_metadatas.append(clean)

        self._collection.upsert(
            ids=document_ids,
            embeddings=embeddings,
            metadatas=clean_metadatas,
        )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        min_score: float = 0.0,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[SimilarDocument]:
        self._init_client()

        where_filter = None
        if filter_metadata:
            where_filter = filter_metadata

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
        )

        documents = []
        for i, doc_id in enumerate(results["ids"][0]):
            # ChromaDB returns distances, convert to similarity
            distance = results["distances"][0][i] if results.get("distances") else 0
            score = 1 - distance  # For cosine distance

            if score < min_score:
                continue

            metadata = results["metadatas"][0][i] if results.get("metadatas") else {}

            documents.append(SimilarDocument(
                document_id=doc_id,
                similarity_score=score,
                metadata=metadata,
            ))

        return documents

    async def delete(self, document_id: str) -> bool:
        self._init_client()
        try:
            self._collection.delete(ids=[document_id])
            return True
        except Exception:
            return False

    async def get(self, document_id: str) -> Optional[DocumentEmbedding]:
        self._init_client()

        results = self._collection.get(
            ids=[document_id],
            include=["embeddings", "metadatas"],
        )

        if not results["ids"]:
            return None

        return DocumentEmbedding(
            document_id=document_id,
            embedding=results["embeddings"][0],
            model_name="unknown",
            metadata=results["metadatas"][0] if results.get("metadatas") else {},
        )

    async def count(self) -> int:
        self._init_client()
        return self._collection.count()
