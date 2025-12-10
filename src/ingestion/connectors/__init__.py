"""
Ingestion Connectors Package
============================
Contains connectors for various document ingestion sources:
- Email (IMAP/POP3)
- SFTP
- Cloud Storage (S3, Azure Blob, Google Cloud Storage)
- Scanner integration
"""

from .email_connector import EmailConnector, EmailConfig
from .sftp_connector import SFTPConnector, SFTPConfig
from .cloud_storage_connector import (
    CloudStorageConnector,
    S3Connector,
    AzureBlobConnector,
    GCSConnector,
    CloudStorageConfig,
)
from .scanner_connector import ScannerConnector, ScannerConfig

__all__ = [
    "EmailConnector",
    "EmailConfig",
    "SFTPConnector",
    "SFTPConfig",
    "CloudStorageConnector",
    "S3Connector",
    "AzureBlobConnector",
    "GCSConnector",
    "CloudStorageConfig",
    "ScannerConnector",
    "ScannerConfig",
]
