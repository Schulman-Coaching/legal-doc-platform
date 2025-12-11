"""
Tests for the unified Data Access Layer.
"""

from __future__ import annotations

import pytest
from datetime import datetime
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from ..data_access_layer import DataAccessLayer
from ..config import StorageConfig
from ..models import DocumentRecord, SearchDocument, DocumentStorageClass


class TestDataAccessLayerUnit:
    """Unit tests for DataAccessLayer using mocks."""

    @pytest.fixture
    def mock_dal(
        self,
        storage_config,
        mock_postgres,
        mock_elasticsearch,
        mock_minio,
        mock_redis,
        mock_neo4j,
        mock_clickhouse,
    ):
        """Create DAL with mocked repositories."""
        dal = DataAccessLayer(storage_config)
        dal.postgres = mock_postgres
        dal.elasticsearch = mock_elasticsearch
        dal.minio = mock_minio
        dal.redis = mock_redis
        dal.neo4j = mock_neo4j
        dal.clickhouse = mock_clickhouse
        dal._connected = True
        return dal

    @pytest.mark.asyncio
    async def test_store_document(
        self,
        mock_dal,
        sample_document_record,
        sample_file_content,
        sample_entities,
    ):
        """Test storing a document across all backends."""
        result = await mock_dal.store_document(
            document_id=sample_document_record.id,
            file_data=sample_file_content,
            metadata=sample_document_record,
            text_content="This is the document content.",
            entities=sample_entities,
        )

        assert result is True

        # Verify storage in each backend
        assert sample_document_record.id in mock_dal.postgres.documents
        assert sample_document_record.id in mock_dal.elasticsearch.documents
        assert f"hot/{sample_document_record.id}" in mock_dal.minio.files

    @pytest.mark.asyncio
    async def test_get_document_cached(self, mock_dal, sample_document_record):
        """Test getting a cached document."""
        # Pre-cache the document
        await mock_dal.redis.cache_document(
            sample_document_record.id,
            sample_document_record.model_dump(),
        )

        result = await mock_dal.get_document(sample_document_record.id)

        assert result is not None
        assert result["id"] == sample_document_record.id

    @pytest.mark.asyncio
    async def test_get_document_from_database(self, mock_dal, sample_document_record):
        """Test getting document from database (cache miss)."""
        # Store in postgres
        await mock_dal.postgres.create(sample_document_record)

        result = await mock_dal.get_document(sample_document_record.id, use_cache=False)

        assert result is not None
        assert result["id"] == sample_document_record.id

    @pytest.mark.asyncio
    async def test_get_nonexistent_document(self, mock_dal):
        """Test getting a non-existent document."""
        result = await mock_dal.get_document("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_document(
        self,
        mock_dal,
        sample_document_record,
        sample_file_content,
    ):
        """Test deleting a document from all backends."""
        # First store the document
        await mock_dal.postgres.create(sample_document_record)
        await mock_dal.elasticsearch.index_document(
            SearchDocument(id=sample_document_record.id, content="test")
        )
        await mock_dal.minio.upload_file(
            sample_document_record.id,
            sample_file_content,
        )
        sample_document_record.storage_path = f"hot/{sample_document_record.id}"

        result = await mock_dal.delete_document(
            sample_document_record.id,
            deleted_by="user-001",
        )

        assert result is True

        # Verify deletion from all backends
        assert await mock_dal.postgres.get(sample_document_record.id) is None
        assert sample_document_record.id not in mock_dal.elasticsearch.documents

    @pytest.mark.asyncio
    async def test_delete_document_with_legal_hold(self, mock_dal, sample_document_record):
        """Test that document with legal hold cannot be deleted."""
        sample_document_record.legal_hold = True
        await mock_dal.postgres.create(sample_document_record)

        with pytest.raises(ValueError, match="legal hold"):
            await mock_dal.delete_document(sample_document_record.id, "user-001")

    @pytest.mark.asyncio
    async def test_search(self, mock_dal, sample_search_document):
        """Test searching documents."""
        await mock_dal.elasticsearch.index_document(sample_search_document)

        results = await mock_dal.search("Non-Disclosure", user_id="user-001")

        assert results["total"] >= 1

    @pytest.mark.asyncio
    async def test_search_cached(self, mock_dal, sample_search_document):
        """Test that search results are cached."""
        await mock_dal.elasticsearch.index_document(sample_search_document)

        # First search - should cache
        results1 = await mock_dal.search("Non-Disclosure", use_cache=True)

        # Verify caching
        assert len([k for k in mock_dal.redis.cache if k.startswith("search:")]) == 1

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, mock_dal):
        """Test health check when all backends are healthy."""
        result = await mock_dal.health_check()

        assert result["status"] == "healthy"
        assert "backends" in result
        assert all(
            b["status"] == "healthy"
            for b in result["backends"].values()
        )

    @pytest.mark.asyncio
    async def test_not_connected_error(self, storage_config):
        """Test that operations fail when not connected."""
        dal = DataAccessLayer(storage_config)

        with pytest.raises(RuntimeError, match="not connected"):
            await dal.get_document("doc-123")


class TestDataAccessLayerKnowledgeGraph:
    """Tests for knowledge graph operations."""

    @pytest.fixture
    def mock_dal(
        self,
        storage_config,
        mock_postgres,
        mock_elasticsearch,
        mock_minio,
        mock_redis,
        mock_neo4j,
        mock_clickhouse,
    ):
        """Create DAL with mocked repositories."""
        dal = DataAccessLayer(storage_config)
        dal.postgres = mock_postgres
        dal.elasticsearch = mock_elasticsearch
        dal.minio = mock_minio
        dal.redis = mock_redis
        dal.neo4j = mock_neo4j
        dal.clickhouse = mock_clickhouse
        dal._connected = True
        return dal

    @pytest.mark.asyncio
    async def test_create_knowledge_graph(
        self,
        mock_dal,
        sample_document_record,
        sample_entities,
    ):
        """Test knowledge graph creation during document storage."""
        await mock_dal._create_knowledge_graph(
            sample_document_record.id,
            sample_document_record,
            sample_entities,
        )

        # Verify document node created
        assert sample_document_record.id in mock_dal.neo4j.nodes

        # Verify entity nodes created
        assert len(mock_dal.neo4j.nodes) > 1

        # Verify relationships created
        assert len(mock_dal.neo4j.relationships) > 0

    @pytest.mark.asyncio
    async def test_find_documents_by_entity(self, mock_dal):
        """Test finding documents by entity."""
        result = await mock_dal.find_documents_by_entity(
            entity_value="Acme Corporation",
            entity_type="organization",
        )

        assert isinstance(result, list)


class TestDataAccessLayerSearch:
    """Tests for search operations."""

    @pytest.fixture
    def mock_dal(
        self,
        storage_config,
        mock_postgres,
        mock_elasticsearch,
        mock_minio,
        mock_redis,
        mock_neo4j,
        mock_clickhouse,
    ):
        """Create DAL with mocked repositories."""
        dal = DataAccessLayer(storage_config)
        dal.postgres = mock_postgres
        dal.elasticsearch = mock_elasticsearch
        dal.minio = mock_minio
        dal.redis = mock_redis
        dal.neo4j = mock_neo4j
        dal.clickhouse = mock_clickhouse
        dal._connected = True
        return dal

    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_dal, sample_search_document):
        """Test search with filters."""
        await mock_dal.elasticsearch.index_document(sample_search_document)

        results = await mock_dal.search(
            query="Agreement",
            filters={"document_type": "Contract"},
        )

        assert "total" in results
        assert "hits" in results

    @pytest.mark.asyncio
    async def test_find_similar_documents(self, mock_dal, sample_search_document):
        """Test finding similar documents."""
        await mock_dal.elasticsearch.index_document(sample_search_document)

        results = await mock_dal.find_similar_documents(sample_search_document.id)

        assert isinstance(results, list)


class TestDataAccessLayerAnalytics:
    """Tests for analytics operations."""

    @pytest.fixture
    def mock_dal(
        self,
        storage_config,
        mock_postgres,
        mock_elasticsearch,
        mock_minio,
        mock_redis,
        mock_neo4j,
        mock_clickhouse,
    ):
        """Create DAL with mocked repositories."""
        dal = DataAccessLayer(storage_config)
        dal.postgres = mock_postgres
        dal.elasticsearch = mock_elasticsearch
        dal.minio = mock_minio
        dal.redis = mock_redis
        dal.neo4j = mock_neo4j
        dal.clickhouse = mock_clickhouse
        dal._connected = True
        return dal

    @pytest.mark.asyncio
    async def test_record_analytics_on_store(
        self,
        mock_dal,
        sample_document_record,
        sample_file_content,
    ):
        """Test that analytics are recorded when storing documents."""
        await mock_dal.store_document(
            document_id=sample_document_record.id,
            file_data=sample_file_content,
            metadata=sample_document_record,
            text_content="Content",
            entities=[],
        )

        # Verify analytics event recorded
        assert len(mock_dal.clickhouse.events) >= 1
        assert mock_dal.clickhouse.events[-1]["type"] == "document"

    @pytest.mark.asyncio
    async def test_record_analytics_on_search(self, mock_dal, sample_search_document):
        """Test that analytics are recorded when searching."""
        await mock_dal.elasticsearch.index_document(sample_search_document)

        await mock_dal.search("test query", user_id="user-001")

        # Verify search analytics recorded
        search_events = [e for e in mock_dal.clickhouse.events if e["type"] == "search"]
        assert len(search_events) >= 1


class TestDataAccessLayerCaching:
    """Tests for caching behavior."""

    @pytest.fixture
    def mock_dal(
        self,
        storage_config,
        mock_postgres,
        mock_elasticsearch,
        mock_minio,
        mock_redis,
        mock_neo4j,
        mock_clickhouse,
    ):
        """Create DAL with mocked repositories."""
        dal = DataAccessLayer(storage_config)
        dal.postgres = mock_postgres
        dal.elasticsearch = mock_elasticsearch
        dal.minio = mock_minio
        dal.redis = mock_redis
        dal.neo4j = mock_neo4j
        dal.clickhouse = mock_clickhouse
        dal._connected = True
        return dal

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_store(
        self,
        mock_dal,
        sample_document_record,
        sample_file_content,
    ):
        """Test cache invalidation when storing document."""
        # Pre-cache some search results
        await mock_dal.redis.cache_search_results("query1", {"total": 5})

        # Store document
        await mock_dal.store_document(
            document_id=sample_document_record.id,
            file_data=sample_file_content,
            metadata=sample_document_record,
            text_content="Content",
            entities=[],
        )

        # Verify search cache invalidated
        cached = await mock_dal.redis.get_cached_search("query1")
        assert cached is None

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_delete(
        self,
        mock_dal,
        sample_document_record,
    ):
        """Test cache invalidation when deleting document."""
        # Store document
        await mock_dal.postgres.create(sample_document_record)
        sample_document_record.storage_path = f"hot/{sample_document_record.id}"

        # Cache the document
        await mock_dal.redis.cache_document(
            sample_document_record.id,
            sample_document_record.model_dump(),
        )

        # Delete document
        await mock_dal.delete_document(sample_document_record.id, "user-001")

        # Verify document cache invalidated
        cached = await mock_dal.redis.get_cached_document(sample_document_record.id)
        assert cached is None


class TestDataAccessLayerStatistics:
    """Tests for statistics operations."""

    @pytest.fixture
    def mock_dal(
        self,
        storage_config,
        mock_postgres,
        mock_elasticsearch,
        mock_minio,
        mock_redis,
        mock_neo4j,
        mock_clickhouse,
    ):
        """Create DAL with mocked repositories."""
        dal = DataAccessLayer(storage_config)
        dal.postgres = mock_postgres
        dal.elasticsearch = mock_elasticsearch
        dal.minio = mock_minio
        dal.redis = mock_redis
        dal.neo4j = mock_neo4j
        dal.clickhouse = mock_clickhouse
        dal._connected = True
        return dal

    @pytest.mark.asyncio
    async def test_get_statistics(self, mock_dal, sample_document_record, sample_file_content):
        """Test getting statistics from all backends."""
        # Add some data
        await mock_dal.postgres.create(sample_document_record)
        await mock_dal.elasticsearch.index_document(
            SearchDocument(id=sample_document_record.id, content="test")
        )
        await mock_dal.minio.upload_file(
            sample_document_record.id,
            sample_file_content,
        )

        stats = await mock_dal.get_statistics()

        assert "postgresql" in stats
        assert "elasticsearch" in stats
        assert "minio" in stats
        assert "redis" in stats
        assert "neo4j" in stats
        assert "timestamp" in stats
