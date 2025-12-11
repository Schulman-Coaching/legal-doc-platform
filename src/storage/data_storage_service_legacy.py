"""
Legal Document Data Storage Service
====================================
Polyglot persistence layer supporting multiple storage backends
for documents, metadata, search, caching, and analytics.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Optional, TypeVar, Generic
from uuid import uuid4

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
# Data Models
# ============================================================================

class StorageBackend(str, Enum):
    """Available storage backends."""
    POSTGRESQL = "postgresql"
    ELASTICSEARCH = "elasticsearch"
    MINIO = "minio"
    REDIS = "redis"
    NEO4J = "neo4j"
    CLICKHOUSE = "clickhouse"


class DocumentStorageClass(str, Enum):
    """Storage classes for documents."""
    HOT = "hot"  # Frequently accessed, fast storage
    WARM = "warm"  # Moderately accessed
    COLD = "cold"  # Archive storage
    GLACIER = "glacier"  # Long-term archive


class RetentionPolicy(str, Enum):
    """Data retention policies."""
    STANDARD = "standard"  # 7 years
    LITIGATION_HOLD = "litigation_hold"  # Indefinite
    SHORT_TERM = "short_term"  # 1 year
    REGULATORY = "regulatory"  # Based on regulation


@dataclass
class StorageMetrics:
    """Metrics for storage operations."""
    operation: str
    backend: StorageBackend
    duration_ms: int
    success: bool
    bytes_processed: int = 0
    error_message: Optional[str] = None


class DocumentRecord(BaseModel):
    """Document metadata record for PostgreSQL."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    original_filename: str
    mime_type: str
    file_size: int
    checksum_sha256: str
    storage_path: str
    storage_class: DocumentStorageClass = DocumentStorageClass.HOT
    encryption_key_id: Optional[str] = None

    # Organizational metadata
    client_id: Optional[str] = None
    matter_id: Optional[str] = None
    user_id: str
    classification: str = "internal"
    tags: list[str] = Field(default_factory=list)

    # Processing state
    processing_status: str = "pending"
    document_type: Optional[str] = None
    practice_areas: list[str] = Field(default_factory=list)

    # Extracted data references
    entity_count: int = 0
    clause_count: int = 0
    citation_count: int = 0
    page_count: int = 1
    word_count: int = 0

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None

    # Retention
    retention_policy: RetentionPolicy = RetentionPolicy.STANDARD
    retention_until: Optional[datetime] = None
    legal_hold: bool = False

    # Custom metadata
    custom_metadata: dict[str, Any] = Field(default_factory=dict)


class SearchDocument(BaseModel):
    """Document model for Elasticsearch indexing."""
    id: str
    title: Optional[str] = None
    content: str
    content_vector: Optional[list[float]] = None  # For semantic search

    # Metadata
    document_type: Optional[str] = None
    practice_areas: list[str] = Field(default_factory=list)
    client_id: Optional[str] = None
    matter_id: Optional[str] = None
    classification: str = "internal"
    tags: list[str] = Field(default_factory=list)

    # Extracted entities
    persons: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    case_numbers: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)

    # Dates
    document_date: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Search metadata
    language: str = "en"
    suggest: Optional[dict] = None  # For autocomplete


# ============================================================================
# Repository Pattern Base
# ============================================================================

class BaseRepository(ABC, Generic[T]):
    """Base repository interface."""

    @abstractmethod
    async def get(self, id: str) -> Optional[T]:
        """Get entity by ID."""
        pass

    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create new entity."""
        pass

    @abstractmethod
    async def update(self, id: str, entity: T) -> T:
        """Update existing entity."""
        pass

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete entity by ID."""
        pass

    @abstractmethod
    async def list(
        self,
        filters: Optional[dict] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list[T]:
        """List entities with optional filters."""
        pass


# ============================================================================
# PostgreSQL Repository (Metadata Store)
# ============================================================================

class PostgreSQLConfig(BaseModel):
    """PostgreSQL configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "legal_docs"
    user: str = "postgres"
    password: str = ""
    pool_size: int = 20
    ssl_mode: str = "require"


class DocumentRepository(BaseRepository[DocumentRecord]):
    """PostgreSQL repository for document metadata."""

    def __init__(self, config: PostgreSQLConfig):
        self.config = config
        self._pool = None  # Connection pool

    async def connect(self) -> None:
        """Establish database connection pool."""
        # In production, use asyncpg
        # import asyncpg
        # self._pool = await asyncpg.create_pool(
        #     host=self.config.host,
        #     port=self.config.port,
        #     database=self.config.database,
        #     user=self.config.user,
        #     password=self.config.password,
        #     min_size=5,
        #     max_size=self.config.pool_size,
        # )
        logger.info(f"Connected to PostgreSQL at {self.config.host}:{self.config.port}")

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
        logger.info("Disconnected from PostgreSQL")

    async def get(self, id: str) -> Optional[DocumentRecord]:
        """Get document by ID."""
        # In production:
        # async with self._pool.acquire() as conn:
        #     row = await conn.fetchrow(
        #         "SELECT * FROM documents WHERE id = $1",
        #         id
        #     )
        #     return DocumentRecord(**dict(row)) if row else None

        # Placeholder
        logger.debug(f"Getting document {id}")
        return None

    async def create(self, entity: DocumentRecord) -> DocumentRecord:
        """Create new document record."""
        # In production:
        # async with self._pool.acquire() as conn:
        #     await conn.execute("""
        #         INSERT INTO documents (id, original_filename, mime_type, ...)
        #         VALUES ($1, $2, $3, ...)
        #     """, entity.id, entity.original_filename, ...)

        entity.created_at = datetime.utcnow()
        entity.updated_at = datetime.utcnow()
        logger.info(f"Created document record {entity.id}")
        return entity

    async def update(self, id: str, entity: DocumentRecord) -> DocumentRecord:
        """Update document record."""
        entity.updated_at = datetime.utcnow()
        logger.info(f"Updated document record {id}")
        return entity

    async def delete(self, id: str) -> bool:
        """Delete document record (soft delete)."""
        logger.info(f"Soft deleted document record {id}")
        return True

    async def list(
        self,
        filters: Optional[dict] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list[DocumentRecord]:
        """List documents with filters."""
        # Build query based on filters
        # In production, construct parameterized SQL query

        logger.debug(f"Listing documents with filters: {filters}")
        return []

    async def find_by_client(
        self,
        client_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> list[DocumentRecord]:
        """Find documents by client ID."""
        return await self.list({"client_id": client_id}, offset, limit)

    async def find_by_matter(
        self,
        matter_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> list[DocumentRecord]:
        """Find documents by matter ID."""
        return await self.list({"matter_id": matter_id}, offset, limit)

    async def set_legal_hold(self, id: str, hold: bool) -> bool:
        """Set or remove legal hold on document."""
        logger.info(f"Set legal hold for {id}: {hold}")
        return True

    async def update_processing_status(
        self,
        id: str,
        status: str,
        document_type: Optional[str] = None,
        practice_areas: Optional[list[str]] = None,
    ) -> bool:
        """Update document processing status."""
        logger.info(f"Updated processing status for {id}: {status}")
        return True

    async def archive_document(self, id: str) -> bool:
        """Archive document (move to cold storage)."""
        logger.info(f"Archived document {id}")
        return True


# ============================================================================
# Elasticsearch Repository (Search Engine)
# ============================================================================

class ElasticsearchConfig(BaseModel):
    """Elasticsearch configuration."""
    hosts: list[str] = ["http://localhost:9200"]
    index_name: str = "legal_documents"
    username: Optional[str] = None
    password: Optional[str] = None
    ca_cert_path: Optional[str] = None
    number_of_shards: int = 5
    number_of_replicas: int = 1


class SearchRepository:
    """Elasticsearch repository for document search."""

    # Index mapping for legal documents
    INDEX_MAPPING = {
        "settings": {
            "number_of_shards": 5,
            "number_of_replicas": 1,
            "analysis": {
                "analyzer": {
                    "legal_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "legal_synonyms",
                            "english_stemmer"
                        ]
                    },
                    "citation_analyzer": {
                        "type": "custom",
                        "tokenizer": "keyword",
                        "filter": ["lowercase", "trim"]
                    }
                },
                "filter": {
                    "legal_synonyms": {
                        "type": "synonym",
                        "synonyms": [
                            "plaintiff,claimant,petitioner",
                            "defendant,respondent",
                            "court,tribunal,judiciary",
                            "contract,agreement,covenant",
                        ]
                    },
                    "english_stemmer": {
                        "type": "stemmer",
                        "language": "english"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "legal_analyzer",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "content": {
                    "type": "text",
                    "analyzer": "legal_analyzer",
                    "term_vector": "with_positions_offsets"
                },
                "content_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                },
                "document_type": {"type": "keyword"},
                "practice_areas": {"type": "keyword"},
                "client_id": {"type": "keyword"},
                "matter_id": {"type": "keyword"},
                "classification": {"type": "keyword"},
                "tags": {"type": "keyword"},
                "persons": {"type": "keyword"},
                "organizations": {"type": "keyword"},
                "locations": {"type": "keyword"},
                "dates": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                "case_numbers": {
                    "type": "text",
                    "analyzer": "citation_analyzer"
                },
                "citations": {
                    "type": "text",
                    "analyzer": "citation_analyzer"
                },
                "document_date": {"type": "date"},
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"},
                "language": {"type": "keyword"},
                "suggest": {
                    "type": "completion",
                    "analyzer": "simple",
                    "preserve_separators": True,
                    "preserve_position_increments": True,
                    "max_input_length": 50
                }
            }
        }
    }

    def __init__(self, config: ElasticsearchConfig):
        self.config = config
        self._client = None

    async def connect(self) -> None:
        """Connect to Elasticsearch cluster."""
        # In production, use elasticsearch-py[async]
        # from elasticsearch import AsyncElasticsearch
        # self._client = AsyncElasticsearch(
        #     hosts=self.config.hosts,
        #     basic_auth=(self.config.username, self.config.password),
        #     ca_certs=self.config.ca_cert_path,
        # )

        logger.info(f"Connected to Elasticsearch at {self.config.hosts}")

    async def disconnect(self) -> None:
        """Close Elasticsearch connection."""
        if self._client:
            await self._client.close()
        logger.info("Disconnected from Elasticsearch")

    async def create_index(self) -> None:
        """Create the search index with mappings."""
        # In production:
        # if not await self._client.indices.exists(index=self.config.index_name):
        #     await self._client.indices.create(
        #         index=self.config.index_name,
        #         body=self.INDEX_MAPPING
        #     )

        logger.info(f"Created index {self.config.index_name}")

    async def index_document(self, doc: SearchDocument) -> bool:
        """Index a document for search."""
        # In production:
        # await self._client.index(
        #     index=self.config.index_name,
        #     id=doc.id,
        #     document=doc.dict(),
        #     refresh=True
        # )

        logger.info(f"Indexed document {doc.id}")
        return True

    async def bulk_index(self, docs: list[SearchDocument]) -> dict[str, int]:
        """Bulk index multiple documents."""
        # In production, use bulk API
        success = 0
        failed = 0

        for doc in docs:
            try:
                await self.index_document(doc)
                success += 1
            except Exception as e:
                logger.error(f"Failed to index {doc.id}: {e}")
                failed += 1

        return {"success": success, "failed": failed}

    async def search(
        self,
        query: str,
        filters: Optional[dict] = None,
        from_: int = 0,
        size: int = 20,
        highlight: bool = True,
    ) -> dict[str, Any]:
        """Full-text search with filters."""
        # Build Elasticsearch query
        es_query = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "title^3",
                                "content",
                                "persons^2",
                                "organizations^2",
                                "case_numbers^2",
                                "citations"
                            ],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    }
                ],
                "filter": []
            }
        }

        # Add filters
        if filters:
            for field, value in filters.items():
                if isinstance(value, list):
                    es_query["bool"]["filter"].append(
                        {"terms": {field: value}}
                    )
                else:
                    es_query["bool"]["filter"].append(
                        {"term": {field: value}}
                    )

        # In production:
        # response = await self._client.search(
        #     index=self.config.index_name,
        #     query=es_query,
        #     from_=from_,
        #     size=size,
        #     highlight={
        #         "fields": {
        #             "content": {"fragment_size": 200, "number_of_fragments": 3}
        #         }
        #     } if highlight else None
        # )

        # Placeholder response
        return {
            "total": 0,
            "hits": [],
            "took_ms": 10,
        }

    async def semantic_search(
        self,
        query_vector: list[float],
        filters: Optional[dict] = None,
        size: int = 20,
    ) -> dict[str, Any]:
        """Semantic search using vector similarity."""
        # In production:
        # response = await self._client.search(
        #     index=self.config.index_name,
        #     knn={
        #         "field": "content_vector",
        #         "query_vector": query_vector,
        #         "k": size,
        #         "num_candidates": size * 10
        #     },
        #     filter=filters
        # )

        return {
            "total": 0,
            "hits": [],
            "took_ms": 15,
        }

    async def autocomplete(
        self,
        prefix: str,
        size: int = 10,
    ) -> list[str]:
        """Get autocomplete suggestions."""
        # In production:
        # response = await self._client.search(
        #     index=self.config.index_name,
        #     suggest={
        #         "doc-suggest": {
        #             "prefix": prefix,
        #             "completion": {
        #                 "field": "suggest",
        #                 "size": size
        #             }
        #         }
        #     }
        # )

        return []

    async def delete_document(self, id: str) -> bool:
        """Delete document from search index."""
        # In production:
        # await self._client.delete(index=self.config.index_name, id=id)

        logger.info(f"Deleted document {id} from search index")
        return True

    async def aggregate(
        self,
        agg_field: str,
        filters: Optional[dict] = None,
        size: int = 10,
    ) -> list[dict[str, Any]]:
        """Get aggregations (facets) for a field."""
        # In production, build aggregation query

        return []


# ============================================================================
# MinIO Repository (Object Storage)
# ============================================================================

class MinIOConfig(BaseModel):
    """MinIO configuration."""
    endpoint: str = "localhost:9000"
    access_key: str = ""
    secret_key: str = ""
    bucket_name: str = "legal-documents"
    secure: bool = True
    region: str = "us-east-1"


class ObjectStorageRepository:
    """MinIO repository for document file storage."""

    def __init__(self, config: MinIOConfig):
        self.config = config
        self._client = None

    async def connect(self) -> None:
        """Initialize MinIO client."""
        # In production, use minio-py
        # from minio import Minio
        # self._client = Minio(
        #     self.config.endpoint,
        #     access_key=self.config.access_key,
        #     secret_key=self.config.secret_key,
        #     secure=self.config.secure,
        # )

        logger.info(f"Connected to MinIO at {self.config.endpoint}")

    async def create_bucket(self) -> None:
        """Create storage bucket if it doesn't exist."""
        # In production:
        # if not self._client.bucket_exists(self.config.bucket_name):
        #     self._client.make_bucket(self.config.bucket_name, location=self.config.region)

        logger.info(f"Created bucket {self.config.bucket_name}")

    async def upload_file(
        self,
        document_id: str,
        data: bytes,
        content_type: str,
        metadata: Optional[dict] = None,
        storage_class: DocumentStorageClass = DocumentStorageClass.HOT,
    ) -> str:
        """Upload file to object storage."""
        # Determine storage path based on class
        path_prefix = {
            DocumentStorageClass.HOT: "hot",
            DocumentStorageClass.WARM: "warm",
            DocumentStorageClass.COLD: "cold",
            DocumentStorageClass.GLACIER: "glacier",
        }

        object_name = f"{path_prefix[storage_class]}/{document_id}"

        # In production:
        # from io import BytesIO
        # self._client.put_object(
        #     self.config.bucket_name,
        #     object_name,
        #     BytesIO(data),
        #     length=len(data),
        #     content_type=content_type,
        #     metadata=metadata or {}
        # )

        logger.info(f"Uploaded {object_name} ({len(data)} bytes)")
        return object_name

    async def download_file(self, object_name: str) -> bytes:
        """Download file from object storage."""
        # In production:
        # response = self._client.get_object(self.config.bucket_name, object_name)
        # data = response.read()
        # response.close()
        # response.release_conn()

        logger.info(f"Downloaded {object_name}")
        return b""

    async def get_presigned_url(
        self,
        object_name: str,
        expires: timedelta = timedelta(hours=1),
    ) -> str:
        """Generate presigned URL for temporary access."""
        # In production:
        # url = self._client.presigned_get_object(
        #     self.config.bucket_name,
        #     object_name,
        #     expires=expires
        # )

        return f"https://{self.config.endpoint}/{self.config.bucket_name}/{object_name}?signed=true"

    async def delete_file(self, object_name: str) -> bool:
        """Delete file from object storage."""
        # In production:
        # self._client.remove_object(self.config.bucket_name, object_name)

        logger.info(f"Deleted {object_name}")
        return True

    async def move_storage_class(
        self,
        object_name: str,
        new_class: DocumentStorageClass,
    ) -> str:
        """Move file to different storage class."""
        # Download and re-upload to new location
        data = await self.download_file(object_name)

        # Extract document ID from path
        document_id = object_name.split("/")[-1]

        # Upload to new class
        new_object_name = await self.upload_file(
            document_id,
            data,
            "application/octet-stream",
            storage_class=new_class,
        )

        # Delete old file
        await self.delete_file(object_name)

        return new_object_name

    async def list_objects(
        self,
        prefix: Optional[str] = None,
        recursive: bool = True,
    ) -> list[dict[str, Any]]:
        """List objects in bucket."""
        # In production:
        # objects = self._client.list_objects(
        #     self.config.bucket_name,
        #     prefix=prefix,
        #     recursive=recursive
        # )

        return []


# ============================================================================
# Redis Repository (Cache & Session)
# ============================================================================

class RedisConfig(BaseModel):
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    ssl: bool = False
    cluster_mode: bool = False
    cluster_nodes: list[str] = Field(default_factory=list)


class CacheRepository:
    """Redis repository for caching and sessions."""

    # Cache TTL defaults (seconds)
    DEFAULT_TTL = 3600  # 1 hour
    SESSION_TTL = 86400  # 24 hours
    SEARCH_CACHE_TTL = 300  # 5 minutes

    def __init__(self, config: RedisConfig):
        self.config = config
        self._client = None

    async def connect(self) -> None:
        """Connect to Redis."""
        # In production, use redis-py[async]
        # import redis.asyncio as redis
        # if self.config.cluster_mode:
        #     from redis.asyncio.cluster import RedisCluster
        #     self._client = RedisCluster(...)
        # else:
        #     self._client = redis.Redis(
        #         host=self.config.host,
        #         port=self.config.port,
        #         password=self.config.password,
        #         db=self.config.db,
        #         ssl=self.config.ssl
        #     )

        logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
        logger.info("Disconnected from Redis")

    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        # In production:
        # return await self._client.get(key)

        return None

    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in cache."""
        # In production:
        # await self._client.set(key, value, ex=ttl or self.DEFAULT_TTL)

        logger.debug(f"Cache set: {key}")
        return True

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        # In production:
        # await self._client.delete(key)

        logger.debug(f"Cache delete: {key}")
        return True

    async def get_json(self, key: str) -> Optional[dict]:
        """Get JSON value from cache."""
        value = await self.get(key)
        if value:
            return json.loads(value)
        return None

    async def set_json(
        self,
        key: str,
        value: dict,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set JSON value in cache."""
        return await self.set(key, json.dumps(value), ttl)

    # Session management
    async def get_session(self, session_id: str) -> Optional[dict]:
        """Get user session."""
        return await self.get_json(f"session:{session_id}")

    async def set_session(
        self,
        session_id: str,
        data: dict,
    ) -> bool:
        """Set user session."""
        return await self.set_json(
            f"session:{session_id}",
            data,
            self.SESSION_TTL
        )

    async def delete_session(self, session_id: str) -> bool:
        """Delete user session."""
        return await self.delete(f"session:{session_id}")

    # Search result caching
    async def cache_search_results(
        self,
        query_hash: str,
        results: dict,
    ) -> bool:
        """Cache search results."""
        return await self.set_json(
            f"search:{query_hash}",
            results,
            self.SEARCH_CACHE_TTL
        )

    async def get_cached_search(self, query_hash: str) -> Optional[dict]:
        """Get cached search results."""
        return await self.get_json(f"search:{query_hash}")

    # Rate limiting
    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        """Check and update rate limit."""
        # In production, use Redis INCR with EXPIRE
        # count = await self._client.incr(f"ratelimit:{key}")
        # if count == 1:
        #     await self._client.expire(f"ratelimit:{key}", window_seconds)
        # return count <= limit, limit - count

        return True, limit

    # Pub/Sub for real-time updates
    async def publish(self, channel: str, message: dict) -> int:
        """Publish message to channel."""
        # In production:
        # return await self._client.publish(channel, json.dumps(message))

        logger.debug(f"Published to {channel}")
        return 1

    async def subscribe(self, channel: str) -> AsyncIterator[dict]:
        """Subscribe to channel."""
        # In production, use pubsub
        # pubsub = self._client.pubsub()
        # await pubsub.subscribe(channel)
        # async for message in pubsub.listen():
        #     if message['type'] == 'message':
        #         yield json.loads(message['data'])

        yield {}


# ============================================================================
# ClickHouse Repository (Analytics)
# ============================================================================

class ClickHouseConfig(BaseModel):
    """ClickHouse configuration."""
    host: str = "localhost"
    port: int = 9000
    database: str = "legal_analytics"
    user: str = "default"
    password: str = ""


class AnalyticsRepository:
    """ClickHouse repository for analytics and reporting."""

    # Table schemas
    CREATE_TABLES = """
    -- Document processing events
    CREATE TABLE IF NOT EXISTS document_events (
        event_id UUID,
        document_id String,
        event_type String,
        event_time DateTime64(3),
        user_id String,
        client_id Nullable(String),
        matter_id Nullable(String),
        document_type Nullable(String),
        practice_area Nullable(String),
        processing_time_ms UInt32,
        file_size UInt64,
        metadata Map(String, String)
    ) ENGINE = MergeTree()
    PARTITION BY toYYYYMM(event_time)
    ORDER BY (event_time, document_id);

    -- Search analytics
    CREATE TABLE IF NOT EXISTS search_events (
        search_id UUID,
        user_id String,
        query String,
        filters Map(String, String),
        results_count UInt32,
        clicked_document_id Nullable(String),
        search_time DateTime64(3),
        response_time_ms UInt32
    ) ENGINE = MergeTree()
    PARTITION BY toYYYYMM(search_time)
    ORDER BY (search_time, user_id);

    -- API usage metrics
    CREATE TABLE IF NOT EXISTS api_metrics (
        request_id UUID,
        endpoint String,
        method String,
        user_id String,
        client_id Nullable(String),
        status_code UInt16,
        response_time_ms UInt32,
        request_size UInt32,
        response_size UInt32,
        timestamp DateTime64(3)
    ) ENGINE = MergeTree()
    PARTITION BY toYYYYMM(timestamp)
    ORDER BY (timestamp, endpoint);
    """

    def __init__(self, config: ClickHouseConfig):
        self.config = config
        self._client = None

    async def connect(self) -> None:
        """Connect to ClickHouse."""
        # In production, use clickhouse-driver[asyncio]
        # from clickhouse_driver import Client
        # self._client = Client(
        #     host=self.config.host,
        #     port=self.config.port,
        #     database=self.config.database,
        #     user=self.config.user,
        #     password=self.config.password
        # )

        logger.info(f"Connected to ClickHouse at {self.config.host}:{self.config.port}")

    async def initialize_tables(self) -> None:
        """Create analytics tables."""
        # In production:
        # for statement in self.CREATE_TABLES.split(';'):
        #     if statement.strip():
        #         self._client.execute(statement)

        logger.info("Initialized ClickHouse tables")

    async def record_document_event(
        self,
        document_id: str,
        event_type: str,
        user_id: str,
        client_id: Optional[str] = None,
        matter_id: Optional[str] = None,
        document_type: Optional[str] = None,
        processing_time_ms: int = 0,
        file_size: int = 0,
        metadata: Optional[dict] = None,
    ) -> None:
        """Record a document processing event."""
        # In production:
        # self._client.execute(
        #     "INSERT INTO document_events VALUES",
        #     [(uuid4(), document_id, event_type, datetime.utcnow(), ...)]
        # )

        logger.debug(f"Recorded event {event_type} for {document_id}")

    async def record_search_event(
        self,
        user_id: str,
        query: str,
        results_count: int,
        response_time_ms: int,
        filters: Optional[dict] = None,
        clicked_document_id: Optional[str] = None,
    ) -> None:
        """Record a search event."""
        logger.debug(f"Recorded search event for user {user_id}")

    async def record_api_metric(
        self,
        endpoint: str,
        method: str,
        user_id: str,
        status_code: int,
        response_time_ms: int,
        request_size: int = 0,
        response_size: int = 0,
        client_id: Optional[str] = None,
    ) -> None:
        """Record an API metric."""
        logger.debug(f"Recorded API metric: {method} {endpoint}")

    async def get_document_stats(
        self,
        client_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get document processing statistics."""
        # In production, execute aggregation query

        return {
            "total_documents": 0,
            "documents_by_type": {},
            "documents_by_practice_area": {},
            "avg_processing_time_ms": 0,
            "total_storage_bytes": 0,
        }

    async def get_search_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get search analytics."""
        return {
            "total_searches": 0,
            "avg_response_time_ms": 0,
            "top_queries": [],
            "zero_result_queries": [],
        }

    async def get_api_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get API usage metrics."""
        return {
            "total_requests": 0,
            "requests_by_endpoint": {},
            "avg_response_time_ms": 0,
            "error_rate": 0.0,
        }


# ============================================================================
# Unified Data Access Layer
# ============================================================================

class DataAccessLayer:
    """Unified interface for all storage backends."""

    def __init__(
        self,
        postgres_config: PostgreSQLConfig,
        elasticsearch_config: ElasticsearchConfig,
        minio_config: MinIOConfig,
        redis_config: RedisConfig,
        clickhouse_config: ClickHouseConfig,
    ):
        self.documents = DocumentRepository(postgres_config)
        self.search = SearchRepository(elasticsearch_config)
        self.storage = ObjectStorageRepository(minio_config)
        self.cache = CacheRepository(redis_config)
        self.analytics = AnalyticsRepository(clickhouse_config)

    async def connect_all(self) -> None:
        """Connect to all storage backends."""
        await asyncio.gather(
            self.documents.connect(),
            self.search.connect(),
            self.storage.connect(),
            self.cache.connect(),
            self.analytics.connect(),
        )

        # Initialize schemas
        await self.search.create_index()
        await self.storage.create_bucket()
        await self.analytics.initialize_tables()

        logger.info("All storage backends connected")

    async def disconnect_all(self) -> None:
        """Disconnect from all storage backends."""
        await asyncio.gather(
            self.documents.disconnect(),
            self.search.disconnect(),
            self.cache.disconnect(),
        )

        logger.info("All storage backends disconnected")

    async def store_document(
        self,
        document_id: str,
        file_data: bytes,
        metadata: DocumentRecord,
        text_content: str,
        entities: list[dict],
    ) -> bool:
        """
        Store a document across all relevant backends.

        Args:
            document_id: Unique document identifier
            file_data: Raw file bytes
            metadata: Document metadata record
            text_content: Extracted text content
            entities: Extracted entities

        Returns:
            Success status
        """
        try:
            # 1. Store file in MinIO
            storage_path = await self.storage.upload_file(
                document_id,
                file_data,
                metadata.mime_type,
                {"classification": metadata.classification},
            )
            metadata.storage_path = storage_path

            # 2. Store metadata in PostgreSQL
            await self.documents.create(metadata)

            # 3. Index for search in Elasticsearch
            search_doc = SearchDocument(
                id=document_id,
                title=metadata.original_filename,
                content=text_content,
                document_type=metadata.document_type,
                practice_areas=metadata.practice_areas,
                client_id=metadata.client_id,
                matter_id=metadata.matter_id,
                classification=metadata.classification,
                tags=metadata.tags,
                persons=[e["value"] for e in entities if e["type"] == "person"],
                organizations=[e["value"] for e in entities if e["type"] == "organization"],
                case_numbers=[e["value"] for e in entities if e["type"] == "case_number"],
                citations=[e["value"] for e in entities if e["type"] == "citation"],
            )
            await self.search.index_document(search_doc)

            # 4. Record analytics event
            await self.analytics.record_document_event(
                document_id=document_id,
                event_type="document_stored",
                user_id=metadata.user_id,
                client_id=metadata.client_id,
                matter_id=metadata.matter_id,
                document_type=metadata.document_type,
                file_size=metadata.file_size,
            )

            # 5. Invalidate any related caches
            await self.cache.delete(f"doc:{document_id}")

            logger.info(f"Document {document_id} stored successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to store document {document_id}: {e}")
            return False

    async def retrieve_document(
        self,
        document_id: str,
        include_content: bool = False,
    ) -> Optional[dict[str, Any]]:
        """
        Retrieve a document and its metadata.

        Args:
            document_id: Document identifier
            include_content: Whether to include file content

        Returns:
            Document data or None
        """
        # Check cache first
        cached = await self.cache.get_json(f"doc:{document_id}")
        if cached and not include_content:
            return cached

        # Get metadata from PostgreSQL
        metadata = await self.documents.get(document_id)
        if not metadata:
            return None

        result = metadata.dict()

        # Get file content if requested
        if include_content and metadata.storage_path:
            file_data = await self.storage.download_file(metadata.storage_path)
            result["content"] = file_data

        # Cache metadata (without content)
        if not include_content:
            await self.cache.set_json(f"doc:{document_id}", result, ttl=3600)

        return result

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from all backends."""
        try:
            # Get metadata first
            metadata = await self.documents.get(document_id)
            if not metadata:
                return False

            # Check for legal hold
            if metadata.legal_hold:
                raise ValueError("Cannot delete document under legal hold")

            # Delete from all backends
            await asyncio.gather(
                self.documents.delete(document_id),
                self.search.delete_document(document_id),
                self.storage.delete_file(metadata.storage_path),
                self.cache.delete(f"doc:{document_id}"),
            )

            # Record deletion event
            await self.analytics.record_document_event(
                document_id=document_id,
                event_type="document_deleted",
                user_id=metadata.user_id,
            )

            logger.info(f"Document {document_id} deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False


# ============================================================================
# FastAPI Application
# ============================================================================

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

app = FastAPI(
    title="Legal Document Storage Service",
    description="Polyglot persistence layer for legal document platform",
    version="1.0.0",
)

dal: Optional[DataAccessLayer] = None


@app.on_event("startup")
async def startup_event():
    """Initialize storage connections."""
    global dal

    dal = DataAccessLayer(
        postgres_config=PostgreSQLConfig(),
        elasticsearch_config=ElasticsearchConfig(),
        minio_config=MinIOConfig(),
        redis_config=RedisConfig(),
        clickhouse_config=ClickHouseConfig(),
    )

    await dal.connect_all()
    logger.info("Storage service initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Close storage connections."""
    global dal
    if dal:
        await dal.disconnect_all()


@app.get("/api/v1/documents/{document_id}")
async def get_document(
    document_id: str,
    include_content: bool = Query(False),
):
    """Get document metadata and optionally content."""
    if not dal:
        raise HTTPException(status_code=503, detail="Service not initialized")

    result = await dal.retrieve_document(document_id, include_content)
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")

    return result


@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document."""
    if not dal:
        raise HTTPException(status_code=503, detail="Service not initialized")

    success = await dal.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found or deletion failed")

    return {"message": "Document deleted successfully"}


@app.get("/api/v1/search")
async def search_documents(
    q: str = Query(..., min_length=1),
    document_type: Optional[str] = None,
    client_id: Optional[str] = None,
    matter_id: Optional[str] = None,
    from_: int = Query(0, alias="from"),
    size: int = Query(20, le=100),
):
    """Search documents."""
    if not dal:
        raise HTTPException(status_code=503, detail="Service not initialized")

    filters = {}
    if document_type:
        filters["document_type"] = document_type
    if client_id:
        filters["client_id"] = client_id
    if matter_id:
        filters["matter_id"] = matter_id

    results = await dal.search.search(q, filters, from_, size)
    return results


@app.get("/api/v1/documents/{document_id}/download")
async def download_document(document_id: str):
    """Get presigned URL for document download."""
    if not dal:
        raise HTTPException(status_code=503, detail="Service not initialized")

    metadata = await dal.documents.get(document_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Document not found")

    url = await dal.storage.get_presigned_url(metadata.storage_path)
    return {"download_url": url, "expires_in_seconds": 3600}


@app.get("/api/v1/analytics/documents")
async def get_document_analytics(
    client_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
):
    """Get document analytics."""
    if not dal:
        raise HTTPException(status_code=503, detail="Service not initialized")

    stats = await dal.analytics.get_document_stats(client_id, start_date, end_date)
    return stats


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "storage"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
