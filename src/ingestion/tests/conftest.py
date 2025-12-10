"""
Pytest Configuration and Fixtures
==================================
Shared fixtures for ingestion layer tests.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf_content():
    """Sample PDF file content."""
    return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"


@pytest.fixture
def sample_docx_content():
    """Sample DOCX file content (ZIP signature)."""
    return b"PK\x03\x04\x14\x00\x06\x00" + b"\x00" * 100


@pytest.fixture
def sample_text_content():
    """Sample text file content."""
    return b"This is a sample legal document.\nIt contains multiple lines.\nFor testing purposes."


@pytest.fixture
def sample_image_content():
    """Sample PNG image content."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer for testing."""
    producer = MagicMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send_and_wait = AsyncMock()
    return producer


@pytest.fixture
def mock_database_session():
    """Mock database session for repository tests."""
    session = MagicMock()
    session.execute = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.add = MagicMock()
    session.delete = AsyncMock()
    return session


@pytest.fixture
def legal_document_samples():
    """Collection of sample legal document metadata."""
    return [
        {
            "filename": "contract_2024.pdf",
            "mime_type": "application/pdf",
            "client_id": "CLIENT-001",
            "matter_id": "MATTER-2024-001",
            "classification": "confidential",
            "tags": ["contract", "real-estate"],
        },
        {
            "filename": "motion_to_dismiss.docx",
            "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "client_id": "CLIENT-002",
            "matter_id": "MATTER-2024-002",
            "classification": "attorney_client_privileged",
            "tags": ["litigation", "motion"],
        },
        {
            "filename": "deposition_transcript.txt",
            "mime_type": "text/plain",
            "client_id": "CLIENT-001",
            "matter_id": "MATTER-2024-001",
            "classification": "confidential",
            "tags": ["discovery", "deposition"],
        },
    ]


@pytest.fixture
def email_connector_config():
    """Sample email connector configuration."""
    return {
        "host": "mail.example.com",
        "port": 993,
        "protocol": "imap_ssl",
        "username": "legal@example.com",
        "mailbox": "INBOX",
        "processed_folder": "Processed",
        "error_folder": "Errors",
        "subject_filters": ["Legal:", "Case:"],
        "detect_matter_from_subject": True,
    }


@pytest.fixture
def sftp_connector_config():
    """Sample SFTP connector configuration."""
    return {
        "host": "sftp.example.com",
        "port": 22,
        "username": "legal_user",
        "auth_method": "key_file",
        "remote_path": "/legal_docs",
        "processed_path": "/processed",
        "file_patterns": ["*.pdf", "*.docx"],
        "recursive": True,
    }


@pytest.fixture
def s3_connector_config():
    """Sample S3 connector configuration."""
    return {
        "provider": "aws_s3",
        "bucket_name": "legal-documents",
        "prefix": "incoming/",
        "region_name": "us-east-1",
        "file_patterns": ["*.pdf", "*.doc*"],
        "processed_prefix": "processed/",
    }
