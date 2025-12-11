"""
Tests for storage repositories.

These tests use mocks for unit testing. Integration tests require
running services and are marked with @pytest.mark.integration.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from ..models import DocumentRecord, SearchDocument, DocumentStorageClass
from ..config import RedisConfig


class TestPostgresRepository:
    """Tests for PostgreSQL repository."""

    @pytest.mark.asyncio
    async def test_create_document(self, mock_postgres, sample_document_record):
        """Test creating a document."""
        result = await mock_postgres.create(sample_document_record)

        assert result.id == sample_document_record.id
        assert result.original_filename == sample_document_record.original_filename

    @pytest.mark.asyncio
    async def test_get_document(self, mock_postgres, sample_document_record):
        """Test retrieving a document."""
        await mock_postgres.create(sample_document_record)
        result = await mock_postgres.get(sample_document_record.id)

        assert result is not None
        assert result.id == sample_document_record.id

    @pytest.mark.asyncio
    async def test_get_nonexistent_document(self, mock_postgres):
        """Test retrieving non-existent document."""
        result = await mock_postgres.get("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_document(self, mock_postgres, sample_document_record):
        """Test deleting a document."""
        await mock_postgres.create(sample_document_record)
        result = await mock_postgres.delete(sample_document_record.id, "user-001")

        assert result is True
        assert await mock_postgres.get(sample_document_record.id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_document(self, mock_postgres):
        """Test deleting non-existent document."""
        result = await mock_postgres.delete("nonexistent-id", "user-001")
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check(self, mock_postgres):
        """Test health check."""
        result = await mock_postgres.health_check()
        assert result["status"] == "healthy"


class TestElasticsearchRepository:
    """Tests for Elasticsearch repository."""

    @pytest.mark.asyncio
    async def test_index_document(self, mock_elasticsearch, sample_search_document):
        """Test indexing a document."""
        result = await mock_elasticsearch.index_document(sample_search_document)
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_document(self, mock_elasticsearch, sample_search_document):
        """Test deleting a document."""
        await mock_elasticsearch.index_document(sample_search_document)
        result = await mock_elasticsearch.delete_document(sample_search_document.id)
        assert result is True

    @pytest.mark.asyncio
    async def test_search_documents(self, mock_elasticsearch, sample_search_document):
        """Test searching documents."""
        await mock_elasticsearch.index_document(sample_search_document)
        results = await mock_elasticsearch.search("Non-Disclosure")

        assert results["total"] >= 1

    @pytest.mark.asyncio
    async def test_search_no_results(self, mock_elasticsearch, sample_search_document):
        """Test search with no results."""
        await mock_elasticsearch.index_document(sample_search_document)
        results = await mock_elasticsearch.search("xyznonexistent")

        assert results["total"] == 0

    @pytest.mark.asyncio
    async def test_health_check(self, mock_elasticsearch):
        """Test health check."""
        result = await mock_elasticsearch.health_check()
        assert result["status"] == "healthy"


class TestMinIORepository:
    """Tests for MinIO repository."""

    @pytest.mark.asyncio
    async def test_upload_file(self, mock_minio, sample_file_content):
        """Test uploading a file."""
        doc_id = str(uuid4())
        path = await mock_minio.upload_file(
            document_id=doc_id,
            data=sample_file_content,
            content_type="application/pdf",
        )

        assert path == f"hot/{doc_id}"

    @pytest.mark.asyncio
    async def test_download_file(self, mock_minio, sample_file_content):
        """Test downloading a file."""
        doc_id = str(uuid4())
        path = await mock_minio.upload_file(
            document_id=doc_id,
            data=sample_file_content,
            content_type="application/pdf",
        )

        data = await mock_minio.download_file(path)
        assert data == sample_file_content

    @pytest.mark.asyncio
    async def test_delete_file(self, mock_minio, sample_file_content):
        """Test deleting a file."""
        doc_id = str(uuid4())
        path = await mock_minio.upload_file(
            document_id=doc_id,
            data=sample_file_content,
            content_type="application/pdf",
        )

        result = await mock_minio.delete_file(path)
        assert result is True

        # File should no longer exist
        data = await mock_minio.download_file(path)
        assert data == b""

    @pytest.mark.asyncio
    async def test_health_check(self, mock_minio):
        """Test health check."""
        result = await mock_minio.health_check()
        assert result["status"] == "healthy"


class TestRedisRepository:
    """Tests for Redis repository."""

    @pytest.mark.asyncio
    async def test_cache_document(self, mock_redis):
        """Test caching a document."""
        doc_id = "doc-123"
        doc_data = {"id": doc_id, "title": "Test"}

        result = await mock_redis.cache_document(doc_id, doc_data)
        assert result is True

        cached = await mock_redis.get_cached_document(doc_id)
        assert cached == doc_data

    @pytest.mark.asyncio
    async def test_invalidate_document(self, mock_redis):
        """Test invalidating document cache."""
        doc_id = "doc-123"
        doc_data = {"id": doc_id, "title": "Test"}

        await mock_redis.cache_document(doc_id, doc_data)
        await mock_redis.invalidate_document(doc_id)

        cached = await mock_redis.get_cached_document(doc_id)
        assert cached is None

    @pytest.mark.asyncio
    async def test_cache_search_results(self, mock_redis):
        """Test caching search results."""
        query_hash = "abc123"
        results = {"total": 10, "hits": []}

        await mock_redis.cache_search_results(query_hash, results)
        cached = await mock_redis.get_cached_search(query_hash)

        assert cached == results

    @pytest.mark.asyncio
    async def test_invalidate_search_cache(self, mock_redis):
        """Test invalidating search cache."""
        await mock_redis.cache_search_results("query1", {"total": 5})
        await mock_redis.cache_search_results("query2", {"total": 10})

        count = await mock_redis.invalidate_search_cache()
        assert count == 2

        assert await mock_redis.get_cached_search("query1") is None
        assert await mock_redis.get_cached_search("query2") is None

    @pytest.mark.asyncio
    async def test_health_check(self, mock_redis):
        """Test health check."""
        result = await mock_redis.health_check()
        assert result["status"] == "healthy"


class TestNeo4jRepository:
    """Tests for Neo4j repository."""

    @pytest.mark.asyncio
    async def test_create_document_node(self, mock_neo4j):
        """Test creating a document node."""
        doc_id = "doc-123"
        properties = {"title": "Contract", "type": "NDA"}

        node = await mock_neo4j.create_document_node(doc_id, properties)

        assert node["id"] == doc_id
        assert node["title"] == "Contract"

    @pytest.mark.asyncio
    async def test_delete_document_node(self, mock_neo4j):
        """Test deleting a document node."""
        doc_id = "doc-123"
        await mock_neo4j.create_document_node(doc_id, {})

        result = await mock_neo4j.delete_document_node(doc_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_create_entity_node(self, mock_neo4j):
        """Test creating an entity node."""
        entity_id = "entity-123"
        node = await mock_neo4j.create_entity_node(
            entity_id=entity_id,
            entity_type="person",
            value="John Doe",
        )

        assert node["id"] == entity_id
        assert node["type"] == "person"
        assert node["value"] == "John Doe"

    @pytest.mark.asyncio
    async def test_link_document_entity(self, mock_neo4j):
        """Test linking document to entity."""
        doc_id = "doc-123"
        entity_id = "entity-123"

        await mock_neo4j.create_document_node(doc_id, {})
        await mock_neo4j.create_entity_node(entity_id, "person", "John Doe")

        result = await mock_neo4j.link_document_entity(doc_id, entity_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check(self, mock_neo4j):
        """Test health check."""
        result = await mock_neo4j.health_check()
        assert result["status"] == "healthy"


class TestClickHouseRepository:
    """Tests for ClickHouse repository."""

    @pytest.mark.asyncio
    async def test_record_document_event(self, mock_clickhouse):
        """Test recording a document event."""
        event_id = await mock_clickhouse.record_document_event(
            document_id="doc-123",
            event_type="document_created",
            user_id="user-001",
        )

        assert event_id is not None
        assert len(mock_clickhouse.events) == 1

    @pytest.mark.asyncio
    async def test_record_search_event(self, mock_clickhouse):
        """Test recording a search event."""
        event_id = await mock_clickhouse.record_search_event(
            user_id="user-001",
            query="contract termination",
            results_count=15,
            response_time_ms=125,
        )

        assert event_id is not None
        assert mock_clickhouse.events[-1]["type"] == "search"

    @pytest.mark.asyncio
    async def test_health_check(self, mock_clickhouse):
        """Test health check."""
        result = await mock_clickhouse.health_check()
        assert result["status"] == "healthy"


class TestRedisRepositoryAdvanced:
    """Advanced tests for Redis repository features."""

    @pytest.mark.asyncio
    async def test_multiple_documents_cache(self, mock_redis):
        """Test caching multiple documents."""
        for i in range(5):
            doc_id = f"doc-{i}"
            await mock_redis.cache_document(doc_id, {"id": doc_id, "index": i})

        for i in range(5):
            cached = await mock_redis.get_cached_document(f"doc-{i}")
            assert cached is not None
            assert cached["index"] == i

    @pytest.mark.asyncio
    async def test_cache_update(self, mock_redis):
        """Test updating cached document."""
        doc_id = "doc-123"

        await mock_redis.cache_document(doc_id, {"version": 1})
        await mock_redis.cache_document(doc_id, {"version": 2})

        cached = await mock_redis.get_cached_document(doc_id)
        assert cached["version"] == 2


class TestRepositoryStatistics:
    """Tests for repository statistics."""

    @pytest.mark.asyncio
    async def test_postgres_statistics(self, mock_postgres, sample_document_record):
        """Test PostgreSQL statistics."""
        await mock_postgres.create(sample_document_record)
        stats = await mock_postgres.get_statistics()

        assert stats["total_documents"] == 1

    @pytest.mark.asyncio
    async def test_elasticsearch_statistics(self, mock_elasticsearch, sample_search_document):
        """Test Elasticsearch statistics."""
        await mock_elasticsearch.index_document(sample_search_document)
        stats = await mock_elasticsearch.get_statistics()

        assert stats["total_documents"] == 1

    @pytest.mark.asyncio
    async def test_minio_statistics(self, mock_minio, sample_file_content):
        """Test MinIO statistics."""
        await mock_minio.upload_file("doc-1", sample_file_content)
        await mock_minio.upload_file("doc-2", sample_file_content)

        stats = await mock_minio.get_bucket_size()
        assert stats["total_size_bytes"] == len(sample_file_content) * 2

    @pytest.mark.asyncio
    async def test_neo4j_statistics(self, mock_neo4j):
        """Test Neo4j statistics."""
        await mock_neo4j.create_document_node("doc-1", {})
        await mock_neo4j.create_document_node("doc-2", {})

        stats = await mock_neo4j.get_statistics()
        assert stats["total_nodes"] == 2
