"""
Document Similarity Service
============================
Find similar documents using embeddings and vector search.
"""

import logging
from typing import Any, Optional

from .document_chunker import DocumentChunker
from .embedding_service import EmbeddingService
from .models import ChunkingStrategy, DocumentEmbedding, SimilarDocument
from .vector_store import FAISSVectorStore, VectorStore

logger = logging.getLogger(__name__)


class SimilarityService:
    """Service for document similarity search and clustering."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: Optional[VectorStore] = None,
        chunker: Optional[DocumentChunker] = None,
        use_chunking: bool = True,
    ):
        """
        Initialize similarity service.

        Args:
            embedding_service: Service for generating embeddings
            vector_store: Vector store for similarity search
            chunker: Document chunker for long documents
            use_chunking: Whether to chunk documents before embedding
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store or FAISSVectorStore(
            dimension=embedding_service.embedding_dimension
        )
        self.chunker = chunker or DocumentChunker(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=500,
            chunk_overlap=50,
        )
        self.use_chunking = use_chunking

    async def add_document(
        self,
        document_id: str,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Add document to similarity index.

        Args:
            document_id: Document identifier
            text: Document text
            metadata: Optional metadata

        Returns:
            Indexing result with chunk count
        """
        metadata = metadata or {}
        metadata["document_id"] = document_id

        if self.use_chunking and len(text) > 1000:
            # Chunk document and embed each chunk
            chunks = self.chunker.chunk_document(document_id, text, metadata)

            for chunk in chunks:
                chunk_metadata = {
                    **metadata,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }

                embedding = await self.embedding_service.get_embedding(
                    chunk.content,
                    document_id=f"{document_id}:{chunk.chunk_id}",
                )

                await self.vector_store.add(
                    document_id=f"{document_id}:{chunk.chunk_id}",
                    embedding=embedding.embedding,
                    metadata=chunk_metadata,
                )

            logger.info(f"Indexed document {document_id} with {len(chunks)} chunks")

            return {
                "document_id": document_id,
                "chunks_indexed": len(chunks),
                "embedding_dimension": self.embedding_service.embedding_dimension,
            }
        else:
            # Embed whole document
            embedding = await self.embedding_service.get_embedding(text, document_id)

            await self.vector_store.add(
                document_id=document_id,
                embedding=embedding.embedding,
                metadata=metadata,
            )

            logger.info(f"Indexed document {document_id}")

            return {
                "document_id": document_id,
                "chunks_indexed": 1,
                "embedding_dimension": self.embedding_service.embedding_dimension,
            }

    async def find_similar(
        self,
        query_text: str,
        top_k: int = 10,
        min_score: float = 0.5,
        filter_metadata: Optional[dict[str, Any]] = None,
        deduplicate_documents: bool = True,
    ) -> list[SimilarDocument]:
        """
        Find documents similar to query text.

        Args:
            query_text: Query text to find similar documents for
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            filter_metadata: Optional metadata filter
            deduplicate_documents: If True, return only best chunk per document

        Returns:
            List of similar documents
        """
        # Generate query embedding
        query_embedding = await self.embedding_service.get_embedding(query_text)

        # Search vector store
        results = await self.vector_store.search(
            query_embedding=query_embedding.embedding,
            top_k=top_k * 3 if deduplicate_documents else top_k,
            min_score=min_score,
            filter_metadata=filter_metadata,
        )

        if deduplicate_documents:
            # Keep only best match per document
            seen_docs = set()
            deduplicated = []

            for result in results:
                # Extract base document_id (without chunk suffix)
                doc_id = result.metadata.get("document_id", result.document_id)
                if ":" in result.document_id:
                    doc_id = result.document_id.split(":")[0]

                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    result.document_id = doc_id
                    deduplicated.append(result)

                    if len(deduplicated) >= top_k:
                        break

            return deduplicated

        return results[:top_k]

    async def find_similar_to_document(
        self,
        document_id: str,
        top_k: int = 10,
        min_score: float = 0.5,
        exclude_same_document: bool = True,
    ) -> list[SimilarDocument]:
        """
        Find documents similar to an existing indexed document.

        Args:
            document_id: ID of document to find similar documents for
            top_k: Number of results
            min_score: Minimum similarity score
            exclude_same_document: Whether to exclude the source document

        Returns:
            List of similar documents
        """
        # Get document embedding
        doc_embedding = await self.vector_store.get(document_id)

        if doc_embedding is None:
            # Try with chunk suffix
            chunk_id = f"{document_id}:0"
            doc_embedding = await self.vector_store.get(chunk_id)

        if doc_embedding is None:
            raise ValueError(f"Document {document_id} not found in index")

        # Search
        results = await self.vector_store.search(
            query_embedding=doc_embedding.embedding,
            top_k=top_k + 10,  # Extra to account for filtering
            min_score=min_score,
        )

        # Filter and deduplicate
        seen_docs = set()
        filtered = []

        for result in results:
            # Get base document ID
            base_id = result.metadata.get("document_id", result.document_id)
            if ":" in result.document_id:
                base_id = result.document_id.split(":")[0]

            # Skip same document if requested
            if exclude_same_document and base_id == document_id:
                continue

            if base_id not in seen_docs:
                seen_docs.add(base_id)
                result.document_id = base_id
                filtered.append(result)

                if len(filtered) >= top_k:
                    break

        return filtered

    async def cluster_documents(
        self,
        document_ids: Optional[list[str]] = None,
        n_clusters: int = 5,
        method: str = "kmeans",
    ) -> dict[str, list[str]]:
        """
        Cluster documents by similarity.

        Args:
            document_ids: Optional list of document IDs to cluster
            n_clusters: Number of clusters
            method: Clustering method ("kmeans" or "hierarchical")

        Returns:
            Dictionary mapping cluster labels to document IDs
        """
        import numpy as np

        # Collect embeddings
        embeddings = []
        doc_ids = []

        if document_ids:
            for doc_id in document_ids:
                emb = await self.vector_store.get(doc_id)
                if emb:
                    embeddings.append(emb.embedding)
                    doc_ids.append(doc_id)
        else:
            # Get all documents (this could be slow for large indexes)
            count = await self.vector_store.count()
            if count > 10000:
                logger.warning(f"Clustering {count} documents may be slow")

            # For FAISS store, access internal data
            if hasattr(self.vector_store, '_embeddings'):
                for doc_id, emb in self.vector_store._embeddings.items():
                    embeddings.append(emb)
                    doc_ids.append(doc_id)

        if len(embeddings) < n_clusters:
            n_clusters = max(1, len(embeddings))

        embeddings_np = np.array(embeddings)

        try:
            if method == "kmeans":
                from sklearn.cluster import KMeans

                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings_np)

            elif method == "hierarchical":
                from sklearn.cluster import AgglomerativeClustering

                clustering = AgglomerativeClustering(n_clusters=n_clusters)
                labels = clustering.fit_predict(embeddings_np)

            else:
                raise ValueError(f"Unknown clustering method: {method}")

        except ImportError:
            logger.warning("scikit-learn not installed, using simple clustering")
            # Simple fallback clustering
            labels = [i % n_clusters for i in range(len(doc_ids))]

        # Organize results
        clusters: dict[str, list[str]] = {f"cluster_{i}": [] for i in range(n_clusters)}

        for doc_id, label in zip(doc_ids, labels):
            # Get base document ID
            base_id = doc_id.split(":")[0] if ":" in doc_id else doc_id
            cluster_key = f"cluster_{label}"
            if base_id not in clusters[cluster_key]:
                clusters[cluster_key].append(base_id)

        return clusters

    async def get_document_similarity(
        self,
        document_id_1: str,
        document_id_2: str,
    ) -> float:
        """
        Calculate similarity between two documents.

        Args:
            document_id_1: First document ID
            document_id_2: Second document ID

        Returns:
            Similarity score (0-1)
        """
        emb1 = await self.vector_store.get(document_id_1)
        emb2 = await self.vector_store.get(document_id_2)

        if emb1 is None or emb2 is None:
            raise ValueError("One or both documents not found in index")

        return self.embedding_service.cosine_similarity(
            emb1.embedding,
            emb2.embedding,
        )

    async def remove_document(self, document_id: str) -> bool:
        """
        Remove document from similarity index.

        Args:
            document_id: Document ID to remove

        Returns:
            True if removed, False if not found
        """
        # Try removing base document
        removed = await self.vector_store.delete(document_id)

        # Also remove any chunks
        chunk_index = 0
        while True:
            chunk_id = f"{document_id}:{chunk_index}"
            chunk_removed = await self.vector_store.delete(chunk_id)
            if not chunk_removed:
                break
            removed = True
            chunk_index += 1

        return removed

    async def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        count = await self.vector_store.count()
        return {
            "documents_indexed": count,
            "embedding_dimension": self.embedding_service.embedding_dimension,
            "embedding_model": self.embedding_service.model.model_name,
            "use_chunking": self.use_chunking,
        }
