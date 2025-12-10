"""
Ingestion Models Package
========================
Database models and schemas for the ingestion layer.
"""

from .document import (
    Document,
    DocumentVersion,
    DocumentMetadata,
    DocumentAuditLog,
)
from .ingestion_job import (
    IngestionJob,
    IngestionSource,
    IngestionStatus,
)
from .connector_config import (
    ConnectorConfig,
    EmailConnectorConfig,
    SFTPConnectorConfig,
    CloudConnectorConfig,
)

__all__ = [
    "Document",
    "DocumentVersion",
    "DocumentMetadata",
    "DocumentAuditLog",
    "IngestionJob",
    "IngestionSource",
    "IngestionStatus",
    "ConnectorConfig",
    "EmailConnectorConfig",
    "SFTPConnectorConfig",
    "CloudConnectorConfig",
]
