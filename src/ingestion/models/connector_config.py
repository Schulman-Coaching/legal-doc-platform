"""
Connector Configuration Models
==============================
Database models for storing connector configurations securely.
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
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from .document import Base


class ConnectorType(str, Enum):
    """Types of ingestion connectors."""
    EMAIL_IMAP = "email_imap"
    EMAIL_POP3 = "email_pop3"
    SFTP = "sftp"
    CLOUD_S3 = "cloud_s3"
    CLOUD_AZURE = "cloud_azure"
    CLOUD_GCS = "cloud_gcs"
    SCANNER = "scanner"
    WEBHOOK = "webhook"


class ConnectorStatus(str, Enum):
    """Status of connector."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    TESTING = "testing"


class ConnectorConfig(Base):
    """
    Base connector configuration.

    Stores common settings for all connector types.
    Sensitive credentials should be stored in Vault and referenced by ID.
    """
    __tablename__ = "connector_configs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Identification
    name = Column(String(200), nullable=False)
    description = Column(Text)
    connector_type = Column(SQLEnum(ConnectorType), nullable=False)

    # Status
    status = Column(SQLEnum(ConnectorStatus), default=ConnectorStatus.INACTIVE)
    is_enabled = Column(Boolean, default=True)

    # Ownership
    created_by = Column(String(100), nullable=False)
    owned_by = Column(String(100))
    department = Column(String(100))

    # Target settings
    default_client_id = Column(String(100))
    default_matter_id = Column(String(100))
    default_classification = Column(String(50), default="internal")

    # Polling/scheduling
    poll_interval_seconds = Column(Integer, default=300)
    is_polling_enabled = Column(Boolean, default=True)
    last_poll_at = Column(DateTime)
    next_poll_at = Column(DateTime)

    # Error tracking
    consecutive_errors = Column(Integer, default=0)
    last_error = Column(Text)
    last_error_at = Column(DateTime)
    max_consecutive_errors = Column(Integer, default=5)

    # Statistics
    total_documents_ingested = Column(Integer, default=0)
    total_bytes_ingested = Column(Integer, default=0)
    last_successful_ingestion = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Vault reference for credentials
    vault_secret_path = Column(String(500))

    # Type-specific configuration (stored as JSON)
    config = Column(JSONB, default=dict)

    # Relationships
    jobs = relationship("IngestionJob", back_populates="connector")

    # Polymorphic configuration
    __mapper_args__ = {
        "polymorphic_on": connector_type,
        "polymorphic_identity": "base",
    }

    __table_args__ = (
        Index("ix_connectors_type_status", "connector_type", "status"),
        Index("ix_connectors_owner", "owned_by"),
        Index("ix_connectors_next_poll", "next_poll_at"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary (excluding sensitive data)."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "connector_type": self.connector_type.value if self.connector_type else None,
            "status": self.status.value if self.status else None,
            "is_enabled": self.is_enabled,
            "poll_interval_seconds": self.poll_interval_seconds,
            "last_poll_at": self.last_poll_at.isoformat() if self.last_poll_at else None,
            "total_documents_ingested": self.total_documents_ingested,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class EmailConnectorConfig(ConnectorConfig):
    """Email connector specific configuration."""
    __tablename__ = "email_connector_configs"

    id = Column(UUID(as_uuid=True), ForeignKey("connector_configs.id", ondelete="CASCADE"), primary_key=True)

    # Server settings (stored in Vault, referenced here)
    host = Column(String(200))
    port = Column(Integer)
    use_ssl = Column(Boolean, default=True)

    # Mailbox settings
    mailbox = Column(String(100), default="INBOX")
    processed_folder = Column(String(100), default="Processed")
    error_folder = Column(String(100), default="Errors")

    # Filtering
    subject_filters = Column(JSONB, default=list)  # List of subject patterns
    sender_whitelist = Column(JSONB, default=list)
    sender_blacklist = Column(JSONB, default=list)

    # Processing options
    extract_attachments = Column(Boolean, default=True)
    include_email_body = Column(Boolean, default=True)
    mark_as_read = Column(Boolean, default=True)
    move_after_processing = Column(Boolean, default=True)

    # Legal metadata extraction
    detect_matter_from_subject = Column(Boolean, default=True)
    matter_pattern = Column(String(200), default=r"(?:Matter|Case)[:\s#]+(\w+-?\d+)")
    client_pattern = Column(String(200), default=r"(?:Client|Account)[:\s#]+(\w+)")

    __mapper_args__ = {
        "polymorphic_identity": ConnectorType.EMAIL_IMAP,
    }


class SFTPConnectorConfig(ConnectorConfig):
    """SFTP connector specific configuration."""
    __tablename__ = "sftp_connector_configs"

    id = Column(UUID(as_uuid=True), ForeignKey("connector_configs.id", ondelete="CASCADE"), primary_key=True)

    # Server settings
    host = Column(String(200))
    port = Column(Integer, default=22)

    # Authentication type (password, key_file, key_agent)
    auth_method = Column(String(20), default="password")

    # Directory settings
    remote_path = Column(String(1000), default="/")
    processed_path = Column(String(1000), default="/processed")
    error_path = Column(String(1000), default="/errors")

    # Scanning options
    recursive = Column(Boolean, default=True)
    max_depth = Column(Integer, default=10)

    # File filtering
    file_patterns = Column(JSONB, default=lambda: ["*.pdf", "*.doc*"])
    exclude_patterns = Column(JSONB, default=lambda: [".*", "*.tmp"])
    min_file_age_seconds = Column(Integer, default=60)
    max_file_size = Column(Integer, default=1073741824)  # 1GB

    # Post-processing
    delete_after_processing = Column(Boolean, default=False)
    move_after_processing = Column(Boolean, default=True)

    __mapper_args__ = {
        "polymorphic_identity": ConnectorType.SFTP,
    }


class CloudConnectorConfig(ConnectorConfig):
    """Cloud storage connector configuration (S3, Azure, GCS)."""
    __tablename__ = "cloud_connector_configs"

    id = Column(UUID(as_uuid=True), ForeignKey("connector_configs.id", ondelete="CASCADE"), primary_key=True)

    # Bucket/container
    bucket_name = Column(String(200), nullable=False)
    prefix = Column(String(500), default="")

    # Region (for S3/Azure)
    region = Column(String(50))

    # File filtering
    file_patterns = Column(JSONB, default=lambda: ["*.pdf", "*.doc*"])
    exclude_patterns = Column(JSONB, default=lambda: [".*", "*.tmp"])
    max_file_size = Column(Integer, default=1073741824)  # 1GB

    # Post-processing
    delete_after_processing = Column(Boolean, default=False)
    move_after_processing = Column(Boolean, default=True)
    processed_prefix = Column(String(500), default="processed/")
    error_prefix = Column(String(500), default="errors/")

    # Event-driven settings
    use_events = Column(Boolean, default=False)
    event_queue_url = Column(String(500))  # SQS/Service Bus/Pub-Sub

    __mapper_args__ = {
        "polymorphic_identity": ConnectorType.CLOUD_S3,
    }


class ScannerConnectorConfig(ConnectorConfig):
    """Scanner/watched folder connector configuration."""
    __tablename__ = "scanner_connector_configs"

    id = Column(UUID(as_uuid=True), ForeignKey("connector_configs.id", ondelete="CASCADE"), primary_key=True)

    # Watch path
    watch_path = Column(String(1000), nullable=False)
    processed_folder = Column(String(200), default="processed")
    error_folder = Column(String(200), default="errors")

    # File filtering
    file_patterns = Column(JSONB, default=lambda: ["*.pdf", "*.tif*", "*.png", "*.jpg"])
    min_file_age_seconds = Column(Integer, default=5)

    # Processing options
    process_existing = Column(Boolean, default=True)
    delete_after_processing = Column(Boolean, default=False)
    move_after_processing = Column(Boolean, default=True)

    # Image processing
    auto_deskew = Column(Boolean, default=True)
    remove_blank_pages = Column(Boolean, default=True)

    # Scanner identification
    scanner_id = Column(String(100))
    scanner_name = Column(String(200))

    __mapper_args__ = {
        "polymorphic_identity": ConnectorType.SCANNER,
    }
