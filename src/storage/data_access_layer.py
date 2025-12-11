"""
Data Access Layer
=================
Unified interface for all storage backends.

Provides coordinated access to PostgreSQL, Elasticsearch, MinIO,
Redis, Neo4j, and ClickHouse for document storage and retrieval.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime
from typing import Any, Optional

from .config import StorageConfig
from .models import DocumentRecord, SearchDocument, DocumentStorageClass, GraphNode
from .repositories.postgres import PostgresRepository
from .repositories.elasticsearch import ElasticsearchRepository
from .repositories.minio import MinIORepository
from .repositories.redis import RedisRepository
from .repositories.neo4j import Neo4jRepository
from .repositories.clickhouse import ClickHouseRepository

logger = logging.getLogger(__name__)


class DataAccessLayer:
    """
    Unified data access layer for the legal document platform.

    Orchestrates all storage backends to provide:
    - Coordinated document storage across all backends
    - Caching with automatic invalidation
    - Full-text and semantic search
    - Knowledge graph operations
    - Analytics tracking

    Usage:
        config = StorageConfig.from_env()
        dal = DataAccessLayer(config)
        await dal.connect()

        # Store document
        await dal.store_document(document_id, file_data, metadata, content, entities)

        # Search documents
        results = await dal.search("contract termination")

        # Get document with caching
        doc = await dal.get_document(document_id)

        await dal.disconnect()
    """

    def __init__(self, config: StorageConfig):
        """Initialize with configuration."""
        self.config = config

        # Initialize repositories
        self.postgres = PostgresRepository(config.postgres)
        self.elasticsearch = ElasticsearchRepository(config.elasticsearch)
        self.minio = MinIORepository(config.minio)
        self.redis = RedisRepository(config.redis)
        self.neo4j = Neo4jRepository(config.neo4j)
        self.clickhouse = ClickHouseRepository(config.clickhouse)

        self._connected = False

    async def connect(self) -> None:
        """Connect to all storage backends."""
        logger.info("Connecting to all storage backends...")

        # Connect in parallel where possible
        await asyncio.gather(
            self.postgres.connect(),
            self.elasticsearch.connect(),
            self.minio.connect(),
            self.redis.connect(),
            self.neo4j.connect(),
            self.clickhouse.connect(),
        )

        # Initialize schemas
        await asyncio.gather(
            self.postgres.initialize_schema(),
            self.elasticsearch.create_indices(),
            self.minio.create_bucket(),
            self.neo4j.initialize_schema(),
            self.clickhouse.initialize_schema(),
        )

        self._connected = True
        logger.info("All storage backends connected and initialized")

    async def disconnect(self) -> None:
        """Disconnect from all storage backends."""
        logger.info("Disconnecting from all storage backends...")

        await asyncio.gather(
            self.postgres.disconnect(),
            self.elasticsearch.disconnect(),
            self.minio.disconnect(),
            self.redis.disconnect(),
            self.neo4j.disconnect(),
            self.clickhouse.disconnect(),
            return_exceptions=True,
        )

        self._connected = False
        logger.info("All storage backends disconnected")

    def _check_connected(self) -> None:
        """Verify connected state."""
        if not self._connected:
            raise RuntimeError("DataAccessLayer not connected. Call connect() first.")

    # ==========================================================================
    # Document Storage Operations
    # ==========================================================================

    async def store_document(
        self,
        document_id: str,
        file_data: bytes,
        metadata: DocumentRecord,
        text_content: str,
        entities: list[dict[str, Any]],
        content_vector: Optional[list[float]] = None,
    ) -> bool:
        """
        Store a document across all backends.

        Coordinates storage across:
        1. MinIO - File binary storage
        2. PostgreSQL - Metadata storage
        3. Elasticsearch - Full-text search indexing
        4. Neo4j - Knowledge graph nodes/relationships
        5. ClickHouse - Analytics event
        6. Redis - Cache invalidation

        Args:
            document_id: Unique document identifier
            file_data: Raw file bytes
            metadata: Document metadata record
            text_content: Extracted text content for search
            entities: Extracted entities for knowledge graph
            content_vector: Optional embedding vector for semantic search

        Returns:
            True if successful, raises exception on failure
        """
        self._check_connected()
        start_time = datetime.utcnow()

        try:
            # 1. Store file in MinIO
            storage_path = await self.minio.upload_file(
                document_id=document_id,
                data=file_data,
                content_type=metadata.mime_type,
                metadata={
                    "classification": metadata.classification,
                    "user_id": metadata.user_id,
                },
                storage_class=DocumentStorageClass(metadata.storage_class)
                    if isinstance(metadata.storage_class, str)
                    else metadata.storage_class,
            )
            metadata.storage_path = storage_path
            logger.debug(f"Stored file in MinIO: {storage_path}")

            # 2. Store metadata in PostgreSQL
            await self.postgres.create(metadata)
            logger.debug(f"Stored metadata in PostgreSQL: {document_id}")

            # 3. Index for search in Elasticsearch
            search_doc = SearchDocument(
                id=document_id,
                title=metadata.original_filename,
                content=text_content,
                content_vector=content_vector,
                document_type=metadata.document_type,
                practice_areas=metadata.practice_areas,
                client_id=metadata.client_id,
                matter_id=metadata.matter_id,
                classification=metadata.classification,
                tags=metadata.tags,
                persons=[e["value"] for e in entities if e.get("type") == "person"],
                organizations=[e["value"] for e in entities if e.get("type") == "organization"],
                locations=[e["value"] for e in entities if e.get("type") == "location"],
                case_numbers=[e["value"] for e in entities if e.get("type") == "case_number"],
                citations=[e["value"] for e in entities if e.get("type") == "citation"],
                monetary_amounts=[e["value"] for e in entities if e.get("type") == "monetary"],
            )
            await self.elasticsearch.index_document(search_doc)
            logger.debug(f"Indexed in Elasticsearch: {document_id}")

            # 4. Create knowledge graph nodes
            await self._create_knowledge_graph(document_id, metadata, entities)
            logger.debug(f"Created knowledge graph nodes: {document_id}")

            # 5. Record analytics event
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.clickhouse.record_document_event(
                document_id=document_id,
                event_type="document_stored",
                user_id=metadata.user_id,
                client_id=metadata.client_id,
                matter_id=metadata.matter_id,
                document_type=metadata.document_type,
                practice_area=metadata.practice_areas[0] if metadata.practice_areas else None,
                processing_time_ms=processing_time,
                file_size=metadata.file_size,
            )

            # 6. Invalidate any related caches
            await self.redis.invalidate_document(document_id)
            await self.redis.invalidate_search_cache()

            logger.info(f"Document {document_id} stored successfully in {processing_time}ms")
            return True

        except Exception as e:
            logger.error(f"Failed to store document {document_id}: {e}")
            # Attempt cleanup on failure
            await self._cleanup_partial_storage(document_id, metadata.storage_path)
            raise

    async def _create_knowledge_graph(
        self,
        document_id: str,
        metadata: DocumentRecord,
        entities: list[dict[str, Any]],
    ) -> None:
        """Create knowledge graph nodes and relationships."""
        # Create document node
        await self.neo4j.create_document_node(
            document_id=document_id,
            properties={
                "title": metadata.original_filename,
                "document_type": metadata.document_type,
                "classification": metadata.classification,
                "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
            },
        )

        # Link to matter/client
        if metadata.matter_id and metadata.client_id:
            await self.neo4j.create_matter_node(
                matter_id=metadata.matter_id,
                client_id=metadata.client_id,
            )
            await self.neo4j.link_document_matter(document_id, metadata.matter_id)

        # Create entity nodes and relationships
        for entity in entities:
            entity_id = hashlib.md5(
                f"{entity['type']}:{entity['value']}".encode()
            ).hexdigest()

            await self.neo4j.create_entity_node(
                entity_id=entity_id,
                entity_type=entity["type"],
                value=entity["value"],
                properties=entity.get("properties", {}),
            )

            await self.neo4j.link_document_entity(
                document_id=document_id,
                entity_id=entity_id,
                relationship_type="MENTIONS",
                properties={
                    "confidence": entity.get("confidence", 1.0),
                    "count": entity.get("count", 1),
                },
            )

    async def _cleanup_partial_storage(
        self,
        document_id: str,
        storage_path: Optional[str],
    ) -> None:
        """Clean up partial storage on failure."""
        try:
            if storage_path:
                await self.minio.delete_file(storage_path)
            await self.postgres.delete(document_id, "system")
            await self.elasticsearch.delete_document(document_id)
            await self.neo4j.delete_document_node(document_id)
        except Exception as e:
            logger.warning(f"Cleanup failed for {document_id}: {e}")

    # ==========================================================================
    # Document Retrieval Operations
    # ==========================================================================

    async def get_document(
        self,
        document_id: str,
        include_content: bool = False,
        use_cache: bool = True,
    ) -> Optional[dict[str, Any]]:
        """
        Retrieve a document with optional caching.

        Args:
            document_id: Document identifier
            include_content: Whether to include file content
            use_cache: Whether to use Redis cache

        Returns:
            Document data or None if not found
        """
        self._check_connected()

        # Check cache first (unless including content)
        if use_cache and not include_content:
            cached = await self.redis.get_cached_document(document_id)
            if cached:
                logger.debug(f"Cache hit for document {document_id}")
                return cached

        # Get from PostgreSQL
        metadata = await self.postgres.get(document_id)
        if not metadata:
            return None

        result = metadata.model_dump()

        # Get file content if requested
        if include_content and metadata.storage_path:
            file_data = await self.minio.download_file(metadata.storage_path)
            result["content"] = file_data

        # Cache metadata (without content)
        if use_cache and not include_content:
            await self.redis.cache_document(document_id, result)

        # Track access
        await self.clickhouse.record_document_event(
            document_id=document_id,
            event_type="document_accessed",
            user_id=metadata.user_id,
            client_id=metadata.client_id,
        )

        return result

    async def get_document_with_context(
        self,
        document_id: str,
    ) -> Optional[dict[str, Any]]:
        """
        Get document with full context from knowledge graph.

        Returns document with related entities, documents, matter, and client.
        """
        self._check_connected()

        # Get basic document
        doc = await self.get_document(document_id)
        if not doc:
            return None

        # Get knowledge graph context
        context = await self.neo4j.get_document_context(document_id)
        if context:
            doc["context"] = context

        return doc

    async def delete_document(
        self,
        document_id: str,
        deleted_by: str,
    ) -> bool:
        """
        Delete a document from all backends.

        Performs soft delete in PostgreSQL and removes from other backends.
        """
        self._check_connected()

        # Get document first
        metadata = await self.postgres.get(document_id)
        if not metadata:
            return False

        # Check legal hold
        if metadata.legal_hold:
            raise ValueError("Cannot delete document under legal hold")

        try:
            # Delete from all backends
            await asyncio.gather(
                self.postgres.delete(document_id, deleted_by),
                self.elasticsearch.delete_document(document_id),
                self.minio.delete_file(metadata.storage_path) if metadata.storage_path else asyncio.sleep(0),
                self.neo4j.delete_document_node(document_id),
                self.redis.invalidate_document(document_id),
            )

            # Record deletion event
            await self.clickhouse.record_document_event(
                document_id=document_id,
                event_type="document_deleted",
                user_id=deleted_by,
                client_id=metadata.client_id,
            )

            # Invalidate search cache
            await self.redis.invalidate_search_cache()

            logger.info(f"Document {document_id} deleted by {deleted_by}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            raise

    # ==========================================================================
    # Search Operations
    # ==========================================================================

    async def search(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        from_: int = 0,
        size: int = 20,
        use_cache: bool = True,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Search documents with caching and analytics.

        Args:
            query: Search query string
            filters: Optional filters (document_type, client_id, etc.)
            from_: Pagination offset
            size: Results per page
            use_cache: Whether to use cached results
            user_id: User ID for analytics

        Returns:
            Search results with total count and hits
        """
        self._check_connected()
        start_time = datetime.utcnow()

        # Check cache
        cache_key = ElasticsearchRepository.generate_cache_key(query, filters, from_, size)
        if use_cache:
            cached = await self.redis.get_cached_search(cache_key)
            if cached:
                logger.debug(f"Search cache hit for: {query}")
                return cached

        # Execute search
        results = await self.elasticsearch.search(
            query=query,
            filters=filters,
            from_=from_,
            size=size,
        )

        # Cache results
        if use_cache and results.get("total", 0) > 0:
            await self.redis.cache_search_results(cache_key, results)

        # Track search analytics
        response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        if user_id:
            await self.clickhouse.record_search_event(
                user_id=user_id,
                query=query,
                results_count=results.get("total", 0),
                response_time_ms=response_time,
                filters=filters,
                search_type="text",
            )

        return results

    async def semantic_search(
        self,
        query_vector: list[float],
        filters: Optional[dict[str, Any]] = None,
        size: int = 20,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Semantic search using vector similarity.

        Args:
            query_vector: Query embedding vector
            filters: Optional filters
            size: Number of results
            user_id: User ID for analytics

        Returns:
            Search results ranked by similarity
        """
        self._check_connected()
        start_time = datetime.utcnow()

        results = await self.elasticsearch.semantic_search(
            query_vector=query_vector,
            filters=filters,
            size=size,
        )

        # Track analytics
        response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        if user_id:
            await self.clickhouse.record_search_event(
                user_id=user_id,
                query="[vector_search]",
                results_count=results.get("total", 0),
                response_time_ms=response_time,
                filters=filters,
                search_type="semantic",
            )

        return results

    async def hybrid_search(
        self,
        query: str,
        query_vector: list[float],
        filters: Optional[dict[str, Any]] = None,
        size: int = 20,
        text_weight: float = 0.5,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Hybrid search combining text and semantic search.

        Args:
            query: Text query
            query_vector: Query embedding vector
            filters: Optional filters
            size: Number of results
            text_weight: Weight for text search (0-1)
            user_id: User ID for analytics

        Returns:
            Combined search results
        """
        self._check_connected()
        start_time = datetime.utcnow()

        results = await self.elasticsearch.hybrid_search(
            query=query,
            query_vector=query_vector,
            filters=filters,
            size=size,
            text_weight=text_weight,
        )

        # Track analytics
        response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        if user_id:
            await self.clickhouse.record_search_event(
                user_id=user_id,
                query=query,
                results_count=results.get("total", 0),
                response_time_ms=response_time,
                filters=filters,
                search_type="hybrid",
            )

        return results

    async def find_similar_documents(
        self,
        document_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find documents similar to a given document."""
        self._check_connected()

        # Try Elasticsearch "more like this" first
        es_similar = await self.elasticsearch.more_like_this(document_id, size=limit)

        # Also get graph-based similarity
        graph_similar = await self.neo4j.find_similar_documents(
            document_id, min_shared_entities=2, limit=limit
        )

        # Combine and deduplicate
        seen_ids = set()
        combined = []

        for doc in es_similar:
            if doc["id"] not in seen_ids:
                doc["similarity_source"] = "content"
                combined.append(doc)
                seen_ids.add(doc["id"])

        for item in graph_similar:
            doc = item["document"]
            if doc.get("id") and doc["id"] not in seen_ids:
                doc["similarity_source"] = "entities"
                doc["shared_entities"] = item["shared_entities"]
                combined.append(doc)
                seen_ids.add(doc["id"])

        return combined[:limit]

    # ==========================================================================
    # Knowledge Graph Operations
    # ==========================================================================

    async def get_document_relationships(
        self,
        document_id: str,
        relationship_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get documents related to a given document."""
        self._check_connected()
        return await self.neo4j.get_related_documents(
            document_id, relationship_type
        )

    async def create_document_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Create a relationship between two documents."""
        self._check_connected()
        return await self.neo4j.create_document_relationship(
            source_id, target_id, relationship_type, properties
        )

    async def find_documents_by_entity(
        self,
        entity_value: str,
        entity_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Find documents mentioning a specific entity."""
        self._check_connected()
        return await self.neo4j.find_documents_by_entity(
            entity_value, entity_type, limit
        )

    # ==========================================================================
    # Analytics Operations
    # ==========================================================================

    async def get_document_analytics(
        self,
        client_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get document analytics."""
        self._check_connected()
        return await self.clickhouse.get_document_stats(
            client_id, start_date, end_date
        )

    async def get_search_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get search analytics."""
        self._check_connected()
        return await self.clickhouse.get_search_analytics(start_date, end_date)

    async def get_api_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get API usage analytics."""
        self._check_connected()
        return await self.clickhouse.get_api_metrics(start_date, end_date)

    # ==========================================================================
    # Health & Status
    # ==========================================================================

    async def health_check(self) -> dict[str, Any]:
        """Check health of all backends."""
        checks = await asyncio.gather(
            self.postgres.health_check(),
            self.elasticsearch.health_check(),
            self.minio.health_check(),
            self.redis.health_check(),
            self.neo4j.health_check(),
            self.clickhouse.health_check(),
            return_exceptions=True,
        )

        backend_names = [
            "postgresql", "elasticsearch", "minio",
            "redis", "neo4j", "clickhouse"
        ]

        results = {}
        all_healthy = True

        for name, check in zip(backend_names, checks):
            if isinstance(check, Exception):
                results[name] = {"status": "error", "error": str(check)}
                all_healthy = False
            else:
                results[name] = check
                if check.get("status") != "healthy":
                    all_healthy = False

        return {
            "status": "healthy" if all_healthy else "degraded",
            "backends": results,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics from all backends."""
        self._check_connected()

        pg_stats = await self.postgres.get_statistics()
        es_stats = await self.elasticsearch.get_statistics()
        minio_stats = await self.minio.get_bucket_size()
        redis_stats = await self.redis.get_stats()
        neo4j_stats = await self.neo4j.get_statistics()

        return {
            "postgresql": pg_stats,
            "elasticsearch": es_stats,
            "minio": minio_stats,
            "redis": redis_stats,
            "neo4j": neo4j_stats,
            "timestamp": datetime.utcnow().isoformat(),
        }
