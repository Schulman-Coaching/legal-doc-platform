"""
Legal Document Storage Layer
============================
Polyglot persistence layer supporting multiple storage backends.

Provides unified access to:
- PostgreSQL: Document metadata and relational data
- Elasticsearch: Full-text and semantic search
- MinIO/S3: Object/file storage
- Redis: Caching and session management
- Neo4j: Knowledge graph and relationships
- ClickHouse: Analytics and time-series data
"""

from .config import StorageConfig
from .repositories.postgres import PostgresRepository, PostgresConfig
from .repositories.elasticsearch import ElasticsearchRepository, ElasticsearchConfig
from .repositories.minio import MinIORepository, MinIOConfig
from .repositories.redis import RedisRepository, RedisConfig
from .repositories.neo4j import Neo4jRepository, Neo4jConfig
from .repositories.clickhouse import ClickHouseRepository, ClickHouseConfig
from .data_access_layer import DataAccessLayer
from .models import (
    DocumentRecord,
    SearchDocument,
    StorageBackend,
    DocumentStorageClass,
    RetentionPolicy,
    StorageMetrics,
)

__all__ = [
    # Config
    "StorageConfig",
    # Repositories
    "PostgresRepository",
    "PostgresConfig",
    "ElasticsearchRepository",
    "ElasticsearchConfig",
    "MinIORepository",
    "MinIOConfig",
    "RedisRepository",
    "RedisConfig",
    "Neo4jRepository",
    "Neo4jConfig",
    "ClickHouseRepository",
    "ClickHouseConfig",
    # Unified Layer
    "DataAccessLayer",
    # Models
    "DocumentRecord",
    "SearchDocument",
    "StorageBackend",
    "DocumentStorageClass",
    "RetentionPolicy",
    "StorageMetrics",
]
