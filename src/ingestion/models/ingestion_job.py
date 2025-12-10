"""
Ingestion Job Models
====================
Models for tracking ingestion jobs, batches, and processing status.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from .document import Base


class IngestionSource(str, Enum):
    """Source of document ingestion."""
    API_UPLOAD = "api_upload"
    EMAIL = "email"
    SFTP = "sftp"
    SCANNER = "scanner"
    CLOUD_S3 = "cloud_s3"
    CLOUD_AZURE = "cloud_azure"
    CLOUD_GCS = "cloud_gcs"
    BATCH_IMPORT = "batch_import"
    MANUAL = "manual"


class IngestionStatus(str, Enum):
    """Status of ingestion job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"  # Some items failed
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ItemStatus(str, Enum):
    """Status of individual ingestion item."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    DUPLICATE = "duplicate"
    QUARANTINED = "quarantined"


class IngestionJob(Base):
    """
    Ingestion job tracking.

    Represents a batch of documents being ingested together.
    """
    __tablename__ = "ingestion_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Job identification
    name = Column(String(200))
    description = Column(Text)
    batch_id = Column(String(100), index=True)  # External batch reference

    # Source
    source = Column(SQLEnum(IngestionSource), nullable=False)
    source_identifier = Column(String(500))  # e.g., email address, SFTP path, bucket name
    connector_id = Column(UUID(as_uuid=True), ForeignKey("connector_configs.id"))

    # Status
    status = Column(SQLEnum(IngestionStatus), default=IngestionStatus.PENDING, index=True)
    priority = Column(Integer, default=0)  # Higher = more priority

    # Ownership
    created_by = Column(String(100), nullable=False, index=True)
    client_id = Column(String(100), index=True)
    matter_id = Column(String(100), index=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Progress tracking
    total_items = Column(Integer, default=0)
    completed_items = Column(Integer, default=0)
    failed_items = Column(Integer, default=0)
    skipped_items = Column(Integer, default=0)
    total_bytes = Column(Integer, default=0)
    processed_bytes = Column(Integer, default=0)

    # Processing configuration
    max_concurrent = Column(Integer, default=5)
    max_retries = Column(Integer, default=3)
    stop_on_error = Column(Boolean, default=False)

    # Error tracking
    last_error = Column(Text)
    error_summary = Column(JSONB, default=dict)

    # Metadata
    options = Column(JSONB, default=dict)  # Job-specific options

    # Relationships
    documents = relationship("Document", back_populates="ingestion_job")
    items = relationship("IngestionJobItem", back_populates="job", cascade="all, delete-orphan")
    connector = relationship("ConnectorConfig", back_populates="jobs")

    __table_args__ = (
        Index("ix_jobs_status_created", "status", "created_at"),
        Index("ix_jobs_source_status", "source", "status"),
        Index("ix_jobs_user_status", "created_by", "status"),
    )

    def update_progress(self) -> None:
        """Update progress counters from items."""
        if self.items:
            self.total_items = len(self.items)
            self.completed_items = sum(1 for i in self.items if i.status == ItemStatus.COMPLETED)
            self.failed_items = sum(1 for i in self.items if i.status == ItemStatus.FAILED)
            self.skipped_items = sum(1 for i in self.items if i.status == ItemStatus.SKIPPED)
            self.processed_bytes = sum(i.file_size for i in self.items if i.status == ItemStatus.COMPLETED)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "source": self.source.value if self.source else None,
            "status": self.status.value if self.status else None,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "progress": {
                "total": self.total_items,
                "completed": self.completed_items,
                "failed": self.failed_items,
                "skipped": self.skipped_items,
                "bytes_total": self.total_bytes,
                "bytes_processed": self.processed_bytes,
            },
        }


class IngestionJobItem(Base):
    """
    Individual item within an ingestion job.

    Tracks the status of each file in a batch.
    """
    __tablename__ = "ingestion_job_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("ingestion_jobs.id", ondelete="CASCADE"), nullable=False)

    # File information
    original_filename = Column(String(500), nullable=False)
    file_size = Column(Integer, default=0)
    mime_type = Column(String(100))
    checksum_sha256 = Column(String(64))

    # Source information
    source_path = Column(String(1000))  # Original path/URL
    source_metadata = Column(JSONB, default=dict)

    # Status
    status = Column(SQLEnum(ItemStatus), default=ItemStatus.PENDING, index=True)
    retries = Column(Integer, default=0)
    error_message = Column(Text)

    # Result
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    duplicate_of_id = Column(UUID(as_uuid=True))  # If detected as duplicate

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Processing details
    processing_time_ms = Column(Integer)
    validation_results = Column(JSONB, default=dict)
    scan_results = Column(JSONB, default=dict)

    # Relationships
    job = relationship("IngestionJob", back_populates="items")

    __table_args__ = (
        Index("ix_items_job_status", "job_id", "status"),
        Index("ix_items_checksum", "checksum_sha256"),
    )


class IngestionQueue(Base):
    """
    Priority queue for ingestion jobs.

    Supports priority-based scheduling and rate limiting.
    """
    __tablename__ = "ingestion_queue"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("ingestion_jobs.id", ondelete="CASCADE"), nullable=False)

    # Priority (negative for max-heap behavior)
    priority = Column(Integer, default=0, index=True)

    # Scheduling
    scheduled_at = Column(DateTime, default=datetime.utcnow)
    not_before = Column(DateTime)  # Don't process before this time

    # Rate limiting
    rate_limit_key = Column(String(100))  # e.g., client_id for per-client limits

    # Status
    is_processing = Column(Boolean, default=False)
    locked_by = Column(String(100))  # Worker ID
    locked_at = Column(DateTime)

    __table_args__ = (
        Index("ix_queue_priority_scheduled", "priority", "scheduled_at"),
        Index("ix_queue_not_before", "not_before"),
    )
