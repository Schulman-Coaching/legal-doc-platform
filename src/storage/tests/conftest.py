"""
Test fixtures for storage layer tests.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import AsyncGenerator, Generator
from uuid import uuid4

import pytest
import pytest_asyncio

from ..config import (
    StorageConfig,
    PostgresConfig,
    ElasticsearchConfig,
    MinIOConfig,
    RedisConfig,
    Neo4jConfig,
    ClickHouseConfig,
)
from ..models import DocumentRecord, SearchDocument, DocumentStorageClass
from ..repositories.postgres import PostgresRepository
from ..repositories.elasticsearch import ElasticsearchRepository
from ..repositories.minio import MinIORepository
from ..repositories.redis import RedisRepository
from ..repositories.neo4j import Neo4jRepository
from ..repositories.clickhouse import ClickHouseRepository
from ..data_access_layer import DataAccessLayer


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Configuration fixtures

@pytest.fixture
def postgres_config() -> PostgresConfig:
    """PostgreSQL test configuration."""
    return PostgresConfig(
        host=os.getenv("TEST_POSTGRES_HOST", "localhost"),
        port=int(os.getenv("TEST_POSTGRES_PORT", "5432")),
        database=os.getenv("TEST_POSTGRES_DB", "legal_docs_test"),
        user=os.getenv("TEST_POSTGRES_USER", "postgres"),
        password=os.getenv("TEST_POSTGRES_PASSWORD", "postgres"),
    )


@pytest.fixture
def elasticsearch_config() -> ElasticsearchConfig:
    """Elasticsearch test configuration."""
    return ElasticsearchConfig(
        hosts=[os.getenv("TEST_ELASTICSEARCH_HOST", "http://localhost:9200")],
        index_prefix="legal_docs_test",
    )


@pytest.fixture
def minio_config() -> MinIOConfig:
    """MinIO test configuration."""
    return MinIOConfig(
        endpoint=os.getenv("TEST_MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("TEST_MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("TEST_MINIO_SECRET_KEY", "minioadmin"),
        bucket_name=f"test-bucket-{uuid4().hex[:8]}",
        secure=False,
    )


@pytest.fixture
def redis_config() -> RedisConfig:
    """Redis test configuration."""
    return RedisConfig(
        host=os.getenv("TEST_REDIS_HOST", "localhost"),
        port=int(os.getenv("TEST_REDIS_PORT", "6379")),
        db=15,  # Use a separate DB for testing
    )


@pytest.fixture
def neo4j_config() -> Neo4jConfig:
    """Neo4j test configuration."""
    return Neo4jConfig(
        uri=os.getenv("TEST_NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("TEST_NEO4J_USER", "neo4j"),
        password=os.getenv("TEST_NEO4J_PASSWORD", "password"),
        database="neo4j",
    )


@pytest.fixture
def clickhouse_config() -> ClickHouseConfig:
    """ClickHouse test configuration."""
    return ClickHouseConfig(
        host=os.getenv("TEST_CLICKHOUSE_HOST", "localhost"),
        port=int(os.getenv("TEST_CLICKHOUSE_PORT", "9000")),
        database=os.getenv("TEST_CLICKHOUSE_DB", "legal_analytics_test"),
    )


@pytest.fixture
def storage_config(
    postgres_config: PostgresConfig,
    elasticsearch_config: ElasticsearchConfig,
    minio_config: MinIOConfig,
    redis_config: RedisConfig,
    neo4j_config: Neo4jConfig,
    clickhouse_config: ClickHouseConfig,
) -> StorageConfig:
    """Combined storage configuration."""
    return StorageConfig(
        postgres=postgres_config,
        elasticsearch=elasticsearch_config,
        minio=minio_config,
        redis=redis_config,
        neo4j=neo4j_config,
        clickhouse=clickhouse_config,
    )


# Sample data fixtures

@pytest.fixture
def sample_document_record() -> DocumentRecord:
    """Sample document record for testing."""
    return DocumentRecord(
        id=str(uuid4()),
        original_filename="test_contract.pdf",
        mime_type="application/pdf",
        file_size=1024 * 100,  # 100KB
        checksum_sha256="abc123def456",
        storage_path="hot/test-doc-id",
        storage_class=DocumentStorageClass.HOT,
        client_id="client-001",
        matter_id="matter-001",
        user_id="user-001",
        classification="confidential",
        tags=["contract", "nda"],
        processing_status="pending",
        document_type="Contract",
        practice_areas=["Corporate", "M&A"],
    )


@pytest.fixture
def sample_search_document() -> SearchDocument:
    """Sample search document for testing."""
    return SearchDocument(
        id=str(uuid4()),
        title="Non-Disclosure Agreement",
        content="""
        This Non-Disclosure Agreement ("Agreement") is entered into between
        Acme Corporation ("Disclosing Party") and Beta Inc. ("Receiving Party").
        The parties agree to maintain confidentiality of all proprietary information
        shared during the course of their business relationship.
        """,
        document_type="Contract",
        practice_areas=["Corporate"],
        client_id="client-001",
        matter_id="matter-001",
        classification="confidential",
        tags=["nda", "confidentiality"],
        persons=["John Smith", "Jane Doe"],
        organizations=["Acme Corporation", "Beta Inc."],
    )


@pytest.fixture
def sample_entities() -> list[dict]:
    """Sample entities for knowledge graph testing."""
    return [
        {"type": "person", "value": "John Smith", "confidence": 0.95},
        {"type": "person", "value": "Jane Doe", "confidence": 0.92},
        {"type": "organization", "value": "Acme Corporation", "confidence": 0.98},
        {"type": "organization", "value": "Beta Inc.", "confidence": 0.97},
        {"type": "date", "value": "2024-01-15", "confidence": 0.99},
        {"type": "monetary", "value": "$1,000,000", "confidence": 0.85},
    ]


@pytest.fixture
def sample_file_content() -> bytes:
    """Sample file content for testing."""
    return b"This is sample PDF content for testing purposes." * 100


# Mock repository fixtures for unit tests

class MockPostgresRepository:
    """Mock PostgreSQL repository for unit testing."""

    def __init__(self):
        self.documents = {}

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def initialize_schema(self):
        pass

    async def create(self, document: DocumentRecord) -> DocumentRecord:
        self.documents[document.id] = document
        return document

    async def get(self, document_id: str) -> DocumentRecord | None:
        return self.documents.get(document_id)

    async def delete(self, document_id: str, deleted_by: str) -> bool:
        if document_id in self.documents:
            del self.documents[document_id]
            return True
        return False

    async def health_check(self):
        return {"status": "healthy"}

    async def get_statistics(self):
        return {"total_documents": len(self.documents)}


class MockElasticsearchRepository:
    """Mock Elasticsearch repository for unit testing."""

    def __init__(self):
        self.documents = {}

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def create_indices(self):
        pass

    async def index_document(self, doc: SearchDocument) -> bool:
        self.documents[doc.id] = doc
        return True

    async def delete_document(self, document_id: str) -> bool:
        if document_id in self.documents:
            del self.documents[document_id]
            return True
        return False

    async def search(self, query: str, **kwargs):
        results = [
            doc.model_dump() for doc in self.documents.values()
            if query.lower() in doc.content.lower()
        ]
        return {"total": len(results), "hits": results}

    async def more_like_this(self, document_id: str, size: int = 10):
        """Find similar documents."""
        return []

    async def health_check(self):
        return {"status": "healthy"}

    async def get_statistics(self):
        return {"total_documents": len(self.documents)}


class MockMinIORepository:
    """Mock MinIO repository for unit testing."""

    def __init__(self):
        self.files = {}

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def create_bucket(self):
        pass

    async def upload_file(self, document_id: str, data: bytes, **kwargs) -> str:
        path = f"hot/{document_id}"
        self.files[path] = data
        return path

    async def download_file(self, object_name: str) -> bytes:
        return self.files.get(object_name, b"")

    async def delete_file(self, object_name: str) -> bool:
        if object_name in self.files:
            del self.files[object_name]
            return True
        return False

    async def health_check(self):
        return {"status": "healthy"}

    async def get_bucket_size(self):
        return {"total_size_bytes": sum(len(f) for f in self.files.values())}


class MockRedisRepository:
    """Mock Redis repository for unit testing."""

    def __init__(self):
        self.cache = {}

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def get_cached_document(self, document_id: str):
        return self.cache.get(f"doc:{document_id}")

    async def cache_document(self, document_id: str, data: dict, ttl: int = None):
        self.cache[f"doc:{document_id}"] = data
        return True

    async def invalidate_document(self, document_id: str):
        key = f"doc:{document_id}"
        if key in self.cache:
            del self.cache[key]
        return True

    async def get_cached_search(self, query_hash: str):
        return self.cache.get(f"search:{query_hash}")

    async def cache_search_results(self, query_hash: str, results: dict, ttl: int = None):
        self.cache[f"search:{query_hash}"] = results
        return True

    async def invalidate_search_cache(self):
        keys_to_delete = [k for k in self.cache if k.startswith("search:")]
        for k in keys_to_delete:
            del self.cache[k]
        return len(keys_to_delete)

    async def health_check(self):
        return {"status": "healthy"}

    async def get_stats(self):
        return {"keys": len(self.cache)}


class MockNeo4jRepository:
    """Mock Neo4j repository for unit testing."""

    def __init__(self):
        self.nodes = {}
        self.relationships = []

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def initialize_schema(self):
        pass

    async def create_document_node(self, document_id: str, properties: dict):
        self.nodes[document_id] = {"id": document_id, **properties}
        return self.nodes[document_id]

    async def delete_document_node(self, document_id: str) -> bool:
        if document_id in self.nodes:
            del self.nodes[document_id]
            return True
        return False

    async def create_entity_node(self, entity_id: str, entity_type: str, value: str, **kwargs):
        self.nodes[entity_id] = {"id": entity_id, "type": entity_type, "value": value}
        return self.nodes[entity_id]

    async def link_document_entity(self, document_id: str, entity_id: str, **kwargs):
        self.relationships.append((document_id, entity_id))
        return True

    async def create_matter_node(self, matter_id: str, client_id: str, **kwargs):
        self.nodes[matter_id] = {"id": matter_id, "client_id": client_id}
        return self.nodes[matter_id]

    async def link_document_matter(self, document_id: str, matter_id: str):
        self.relationships.append((document_id, matter_id))
        return True

    async def get_document_context(self, document_id: str):
        return {"document": self.nodes.get(document_id)}

    async def find_similar_documents(self, document_id: str, **kwargs):
        return []

    async def find_documents_by_entity(self, entity_value: str, entity_type: str = None, limit: int = 50):
        """Find documents by entity."""
        return []

    async def health_check(self):
        return {"status": "healthy"}

    async def get_statistics(self):
        return {"total_nodes": len(self.nodes)}


class MockClickHouseRepository:
    """Mock ClickHouse repository for unit testing."""

    def __init__(self):
        self.events = []

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def initialize_schema(self):
        pass

    async def record_document_event(self, **kwargs):
        self.events.append({"type": "document", **kwargs})
        return str(uuid4())

    async def record_search_event(self, **kwargs):
        self.events.append({"type": "search", **kwargs})
        return str(uuid4())

    async def health_check(self):
        return {"status": "healthy"}


@pytest.fixture
def mock_postgres():
    """Mock PostgreSQL repository."""
    return MockPostgresRepository()


@pytest.fixture
def mock_elasticsearch():
    """Mock Elasticsearch repository."""
    return MockElasticsearchRepository()


@pytest.fixture
def mock_minio():
    """Mock MinIO repository."""
    return MockMinIORepository()


@pytest.fixture
def mock_redis():
    """Mock Redis repository."""
    return MockRedisRepository()


@pytest.fixture
def mock_neo4j():
    """Mock Neo4j repository."""
    return MockNeo4jRepository()


@pytest.fixture
def mock_clickhouse():
    """Mock ClickHouse repository."""
    return MockClickHouseRepository()
