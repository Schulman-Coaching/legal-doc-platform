"""
Document Models
===============
SQLAlchemy models for document storage and tracking.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
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
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class DocumentStatus(str, Enum):
    """Document lifecycle status."""
    RECEIVED = "received"
    VALIDATING = "validating"
    VALIDATED = "validated"
    QUARANTINED = "quarantined"
    PROCESSING = "processing"
    PROCESSED = "processed"
    INDEXED = "indexed"
    ARCHIVED = "archived"
    DELETED = "deleted"
    FAILED = "failed"


class SecurityClassification(str, Enum):
    """Document security classification."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    ATTORNEY_CLIENT_PRIVILEGED = "attorney_client_privileged"


class DocumentType(str, Enum):
    """Legal document types."""
    CONTRACT = "contract"
    BRIEF = "brief"
    MOTION = "motion"
    PLEADING = "pleading"
    DISCOVERY = "discovery"
    CORRESPONDENCE = "correspondence"
    MEMO = "memo"
    OPINION = "opinion"
    ORDER = "order"
    EVIDENCE = "evidence"
    EXHIBIT = "exhibit"
    TRANSCRIPT = "transcript"
    UNKNOWN = "unknown"


# Association table for document tags
document_tags = Table(
    "document_tags",
    Base.metadata,
    Column("document_id", UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE")),
    Column("tag_id", UUID(as_uuid=True), ForeignKey("tags.id", ondelete="CASCADE")),
)


class Tag(Base):
    """Document tags for categorization."""
    __tablename__ = "tags"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), nullable=False, unique=True)
    category = Column(String(50))  # e.g., "practice_area", "document_type", "status"
    created_at = Column(DateTime, default=datetime.utcnow)

    documents = relationship("Document", secondary=document_tags, back_populates="tags")


class Document(Base):
    """
    Core document model.

    Stores document metadata and references to actual file storage.
    Files are stored in object storage (MinIO/S3), not in the database.
    """
    __tablename__ = "documents"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # File information
    original_filename = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    storage_path = Column(String(1000), nullable=False)  # Path in object storage
    storage_bucket = Column(String(100), default="documents")

    # Checksums
    checksum_sha256 = Column(String(64), nullable=False, index=True)
    checksum_md5 = Column(String(32))

    # Status
    status = Column(SQLEnum(DocumentStatus), default=DocumentStatus.RECEIVED, index=True)
    classification = Column(SQLEnum(SecurityClassification), default=SecurityClassification.INTERNAL)
    document_type = Column(SQLEnum(DocumentType), default=DocumentType.UNKNOWN)

    # Legal context
    client_id = Column(String(100), index=True)
    matter_id = Column(String(100), index=True)
    case_number = Column(String(100), index=True)
    court = Column(String(200))

    # Ownership and access
    created_by = Column(String(100), nullable=False, index=True)
    owned_by = Column(String(100), index=True)
    department = Column(String(100))

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    received_at = Column(DateTime)  # When originally received
    document_date = Column(DateTime)  # Date on the document itself

    # Ingestion metadata
    ingestion_source = Column(String(50))  # email, api, sftp, scanner, cloud
    ingestion_source_id = Column(String(500))  # Original source identifier
    ingestion_job_id = Column(UUID(as_uuid=True), ForeignKey("ingestion_jobs.id"))

    # Encryption
    encryption_key_id = Column(String(100))
    is_encrypted = Column(Boolean, default=True)

    # Retention and holds
    retention_policy = Column(String(50), default="standard")
    retention_until = Column(DateTime)
    legal_hold = Column(Boolean, default=False, index=True)
    legal_hold_reason = Column(Text)

    # Processing status
    ocr_status = Column(String(20))  # pending, completed, failed, not_needed
    processing_error = Column(Text)

    # Extended metadata (JSON)
    custom_metadata = Column(JSONB, default=dict)

    # Soft delete
    is_deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime)
    deleted_by = Column(String(100))

    # Relationships
    tags = relationship("Tag", secondary=document_tags, back_populates="documents")
    versions = relationship("DocumentVersion", back_populates="document", cascade="all, delete-orphan")
    metadata_records = relationship("DocumentMetadata", back_populates="document", cascade="all, delete-orphan")
    audit_logs = relationship("DocumentAuditLog", back_populates="document", cascade="all, delete-orphan")
    ingestion_job = relationship("IngestionJob", back_populates="documents")

    # Indexes
    __table_args__ = (
        Index("ix_documents_client_matter", "client_id", "matter_id"),
        Index("ix_documents_status_created", "status", "created_at"),
        Index("ix_documents_classification_status", "classification", "status"),
        Index("ix_documents_search", "original_filename", "case_number", "client_id"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "original_filename": self.original_filename,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "status": self.status.value if self.status else None,
            "classification": self.classification.value if self.classification else None,
            "document_type": self.document_type.value if self.document_type else None,
            "client_id": self.client_id,
            "matter_id": self.matter_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "checksum_sha256": self.checksum_sha256,
            "tags": [t.name for t in self.tags],
            "custom_metadata": self.custom_metadata,
        }


class DocumentVersion(Base):
    """
    Document version history.

    Tracks all versions of a document for audit and rollback.
    """
    __tablename__ = "document_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)

    version_number = Column(Integer, nullable=False)
    storage_path = Column(String(1000), nullable=False)
    file_size = Column(Integer, nullable=False)
    checksum_sha256 = Column(String(64), nullable=False)

    # Version metadata
    created_by = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    change_description = Column(Text)

    # Status
    is_current = Column(Boolean, default=True)

    # Relationships
    document = relationship("Document", back_populates="versions")

    __table_args__ = (
        UniqueConstraint("document_id", "version_number", name="uq_document_version"),
        Index("ix_versions_document_current", "document_id", "is_current"),
    )


class DocumentMetadata(Base):
    """
    Extended document metadata.

    Stores extracted metadata, entities, and analysis results.
    """
    __tablename__ = "document_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)

    # Metadata category
    category = Column(String(50), nullable=False)  # extracted, analysis, custom
    key = Column(String(100), nullable=False)
    value = Column(JSONB, nullable=False)

    # Source
    source = Column(String(50))  # ocr, nlp, manual, import
    confidence = Column(Integer)  # 0-100

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="metadata_records")

    __table_args__ = (
        Index("ix_metadata_document_category", "document_id", "category"),
        Index("ix_metadata_key", "key"),
    )


class DocumentAuditLog(Base):
    """
    Document audit trail.

    Records all actions taken on documents for compliance.
    """
    __tablename__ = "document_audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL"))

    # Action details
    action = Column(String(50), nullable=False)  # create, read, update, delete, download, share
    action_detail = Column(Text)

    # Actor
    user_id = Column(String(100), nullable=False, index=True)
    user_email = Column(String(200))
    user_ip = Column(String(45))
    user_agent = Column(Text)

    # Context
    client_id = Column(String(100))
    matter_id = Column(String(100))
    session_id = Column(String(100))

    # Changes (for updates)
    old_values = Column(JSONB)
    new_values = Column(JSONB)

    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    document = relationship("Document", back_populates="audit_logs")

    __table_args__ = (
        Index("ix_audit_document_timestamp", "document_id", "timestamp"),
        Index("ix_audit_user_timestamp", "user_id", "timestamp"),
        Index("ix_audit_action_timestamp", "action", "timestamp"),
    )
