"""
Storage Configuration
=====================
Centralized configuration for all storage backends.
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel, Field


class PostgresConfig(BaseModel):
    """PostgreSQL configuration."""
    host: str = Field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    port: int = Field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    database: str = Field(default_factory=lambda: os.getenv("POSTGRES_DB", "legal_docs"))
    user: str = Field(default_factory=lambda: os.getenv("POSTGRES_USER", "postgres"))
    password: str = Field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))
    pool_min_size: int = 5
    pool_max_size: int = 20
    ssl_mode: str = "prefer"
    statement_cache_size: int = 100
    command_timeout: float = 60.0

    @property
    def dsn(self) -> str:
        """Get connection DSN string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class ElasticsearchConfig(BaseModel):
    """Elasticsearch configuration."""
    hosts: list[str] = Field(
        default_factory=lambda: os.getenv("ELASTICSEARCH_HOSTS", "http://localhost:9200").split(",")
    )
    index_prefix: str = "legal_docs"
    username: Optional[str] = Field(default_factory=lambda: os.getenv("ELASTICSEARCH_USER"))
    password: Optional[str] = Field(default_factory=lambda: os.getenv("ELASTICSEARCH_PASSWORD"))
    ca_cert_path: Optional[str] = Field(default_factory=lambda: os.getenv("ELASTICSEARCH_CA_CERT"))
    verify_certs: bool = True
    number_of_shards: int = 5
    number_of_replicas: int = 1
    refresh_interval: str = "1s"
    max_result_window: int = 10000
    request_timeout: int = 30


class MinIOConfig(BaseModel):
    """MinIO/S3 configuration."""
    endpoint: str = Field(default_factory=lambda: os.getenv("MINIO_ENDPOINT", "localhost:9000"))
    access_key: str = Field(default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", ""))
    secret_key: str = Field(default_factory=lambda: os.getenv("MINIO_SECRET_KEY", ""))
    bucket_name: str = Field(default_factory=lambda: os.getenv("MINIO_BUCKET", "legal-documents"))
    secure: bool = Field(default_factory=lambda: os.getenv("MINIO_SECURE", "false").lower() == "true")
    region: str = "us-east-1"
    part_size: int = 10 * 1024 * 1024  # 10MB multipart threshold


class RedisConfig(BaseModel):
    """Redis configuration."""
    host: str = Field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = Field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    password: Optional[str] = Field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))
    db: int = 0
    ssl: bool = False
    cluster_mode: bool = False
    cluster_nodes: list[str] = Field(default_factory=list)
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    max_connections: int = 50
    decode_responses: bool = True

    # TTL defaults (seconds)
    default_ttl: int = 3600  # 1 hour
    session_ttl: int = 86400  # 24 hours
    search_cache_ttl: int = 300  # 5 minutes


class Neo4jConfig(BaseModel):
    """Neo4j configuration."""
    uri: str = Field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    user: str = Field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    password: str = Field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    database: str = Field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_acquisition_timeout: float = 60.0
    encrypted: bool = False


class ClickHouseConfig(BaseModel):
    """ClickHouse configuration."""
    host: str = Field(default_factory=lambda: os.getenv("CLICKHOUSE_HOST", "localhost"))
    port: int = Field(default_factory=lambda: int(os.getenv("CLICKHOUSE_PORT", "9000")))
    http_port: int = Field(default_factory=lambda: int(os.getenv("CLICKHOUSE_HTTP_PORT", "8123")))
    database: str = Field(default_factory=lambda: os.getenv("CLICKHOUSE_DB", "legal_analytics"))
    user: str = Field(default_factory=lambda: os.getenv("CLICKHOUSE_USER", "default"))
    password: str = Field(default_factory=lambda: os.getenv("CLICKHOUSE_PASSWORD", ""))
    secure: bool = False
    verify: bool = True
    connect_timeout: int = 10
    send_receive_timeout: int = 300


class StorageConfig(BaseModel):
    """Combined storage configuration."""
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    elasticsearch: ElasticsearchConfig = Field(default_factory=ElasticsearchConfig)
    minio: MinIOConfig = Field(default_factory=MinIOConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    clickhouse: ClickHouseConfig = Field(default_factory=ClickHouseConfig)

    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Create configuration from environment variables."""
        return cls()
