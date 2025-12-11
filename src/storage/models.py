"""
Storage Layer Models
====================
Data models for the storage layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


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
    duration_ms: float
    success: bool
    bytes_processed: int = 0
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


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

    # Extracted data counts
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

    class Config:
        use_enum_values = True


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
    monetary_amounts: list[str] = Field(default_factory=list)

    # Dates
    document_date: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Search metadata
    language: str = "en"
    suggest: Optional[dict] = None  # For autocomplete


class GraphNode(BaseModel):
    """Node model for Neo4j knowledge graph."""
    id: str
    label: str  # Node type: Document, Entity, Clause, etc.
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphRelationship(BaseModel):
    """Relationship model for Neo4j knowledge graph."""
    source_id: str
    target_id: str
    relationship_type: str  # REFERENCES, CONTAINS, MENTIONS, etc.
    properties: dict[str, Any] = Field(default_factory=dict)


class AnalyticsEvent(BaseModel):
    """Event model for ClickHouse analytics."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    document_id: Optional[str] = None
    user_id: str
    client_id: Optional[str] = None
    matter_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    properties: dict[str, Any] = Field(default_factory=dict)
