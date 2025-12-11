"""
Tests for Repository Implementations
=====================================
Tests for DocumentRepository, ConnectorRepository, and IngestionJobRepository.
Uses mock database sessions for unit testing.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4, UUID

from ..repositories.document_repository import DocumentRepository
from ..repositories.connector_repository import ConnectorRepository
from ..repositories.job_repository import IngestionJobRepository


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_session():
    """Create a mock async database session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.delete = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def mock_document():
    """Create a mock document object."""
    doc = MagicMock()
    doc.id = uuid4()
    doc.original_filename = "contract.pdf"
    doc.file_size = 1024
    doc.mime_type = "application/pdf"
    doc.storage_path = "/documents/contract.pdf"
    doc.checksum_sha256 = "abc123"
    doc.created_by = "user-123"
    doc.status = MagicMock()
    doc.status.value = "validated"
    doc.is_deleted = False
    doc.legal_hold = False
    doc.tags = []
    doc.versions = []
    doc.metadata_records = []
    return doc


@pytest.fixture
def mock_connector():
    """Create a mock connector config object."""
    connector = MagicMock()
    connector.id = uuid4()
    connector.name = "Email Connector"
    connector.connector_type = MagicMock()
    connector.connector_type.value = "email_imap"
    connector.status = MagicMock()
    connector.status.value = "active"
    connector.is_enabled = True
    connector.poll_interval_seconds = 300
    connector.consecutive_errors = 0
    connector.max_consecutive_errors = 5
    connector.total_documents_ingested = 100
    connector.total_bytes_ingested = 1024 * 1024
    return connector


@pytest.fixture
def mock_job():
    """Create a mock ingestion job object."""
    job = MagicMock()
    job.id = uuid4()
    job.name = "Test Job"
    job.source = MagicMock()
    job.source.value = "api_upload"
    job.status = MagicMock()
    job.status.value = "pending"
    job.created_by = "user-123"
    job.total_items = 10
    job.completed_items = 0
    job.failed_items = 0
    job.skipped_items = 0
    job.total_bytes = 10240
    job.processed_bytes = 0
    job.items = []
    return job


@pytest.fixture
def mock_job_item():
    """Create a mock ingestion job item."""
    item = MagicMock()
    item.id = uuid4()
    item.job_id = uuid4()
    item.original_filename = "document.pdf"
    item.file_size = 1024
    item.status = MagicMock()
    item.status.value = "pending"
    item.retries = 0
    return item


# =============================================================================
# DocumentRepository Tests
# =============================================================================

class TestDocumentRepository:
    """Test suite for DocumentRepository."""

    @pytest.fixture
    def repo(self, mock_session):
        """Create repository with mock session."""
        return DocumentRepository(mock_session)

    # Create operations

    @pytest.mark.asyncio
    async def test_create_document(self, repo, mock_session):
        """Test creating a new document."""
        with patch.object(repo, 'session', mock_session):
            doc = await repo.create(
                original_filename="test.pdf",
                file_size=1024,
                mime_type="application/pdf",
                storage_path="/documents/test.pdf",
                checksum_sha256="abc123",
                created_by="user-123",
            )

            mock_session.add.assert_called_once()
            mock_session.flush.assert_called_once()

    # Read operations

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, repo, mock_session, mock_document):
        """Test getting document by ID when found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_document
        mock_session.execute.return_value = mock_result

        doc = await repo.get_by_id(mock_document.id)

        assert doc == mock_document
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, repo, mock_session):
        """Test getting document by ID when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        doc = await repo.get_by_id(uuid4())

        assert doc is None

    @pytest.mark.asyncio
    async def test_get_by_checksum(self, repo, mock_session, mock_document):
        """Test getting document by checksum."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_document
        mock_session.execute.return_value = mock_result

        doc = await repo.get_by_checksum("abc123")

        assert doc == mock_document

    @pytest.mark.asyncio
    async def test_search_with_filters(self, repo, mock_session, mock_document):
        """Test searching documents with filters."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_document]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        docs = await repo.search(
            client_id="CLIENT-001",
            matter_id="MATTER-001",
            filename_contains="contract",
        )

        assert len(docs) == 1
        assert docs[0] == mock_document

    @pytest.mark.asyncio
    async def test_search_with_date_filters(self, repo, mock_session):
        """Test searching documents with date filters."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        docs = await repo.search(
            created_after=datetime.utcnow() - timedelta(days=7),
            created_before=datetime.utcnow(),
            limit=50,
            offset=10,
        )

        assert docs == []
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_documents(self, repo, mock_session):
        """Test counting documents."""
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 42
        mock_session.execute.return_value = mock_result

        count = await repo.count(client_id="CLIENT-001")

        assert count == 42

    # Update operations

    @pytest.mark.asyncio
    async def test_update_document(self, repo, mock_session, mock_document):
        """Test updating document fields."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_document
        mock_session.execute.return_value = mock_result

        doc = await repo.update(
            mock_document.id,
            original_filename="renamed.pdf",
            client_id="NEW-CLIENT",
        )

        assert doc.original_filename == "renamed.pdf"
        mock_session.flush.assert_called()

    @pytest.mark.asyncio
    async def test_update_document_not_found(self, repo, mock_session):
        """Test updating non-existent document."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        doc = await repo.update(uuid4(), original_filename="test.pdf")

        assert doc is None

    @pytest.mark.asyncio
    async def test_update_status(self, repo, mock_session):
        """Test updating document status."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        success = await repo.update_status(
            uuid4(),
            MagicMock(),  # DocumentStatus
            error_message="Processing error",
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_update_status_not_found(self, repo, mock_session):
        """Test updating status of non-existent document."""
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session.execute.return_value = mock_result

        success = await repo.update_status(uuid4(), MagicMock())

        assert success is False

    # Delete operations

    @pytest.mark.asyncio
    async def test_soft_delete(self, repo, mock_session, mock_document):
        """Test soft deleting a document."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_document
        mock_session.execute.return_value = mock_result

        success = await repo.soft_delete(mock_document.id, "user-123")

        assert success is True
        assert mock_document.is_deleted is True
        mock_session.flush.assert_called()

    @pytest.mark.asyncio
    async def test_soft_delete_not_found(self, repo, mock_session):
        """Test soft deleting non-existent document."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        success = await repo.soft_delete(uuid4(), "user-123")

        assert success is False

    @pytest.mark.asyncio
    async def test_soft_delete_legal_hold(self, repo, mock_session, mock_document):
        """Test that documents under legal hold cannot be deleted."""
        mock_document.legal_hold = True
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_document
        mock_session.execute.return_value = mock_result

        with pytest.raises(ValueError, match="legal hold"):
            await repo.soft_delete(mock_document.id, "user-123")

    # Legal hold operations

    @pytest.mark.asyncio
    async def test_set_legal_hold(self, repo, mock_session):
        """Test setting legal hold on document."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        success = await repo.set_legal_hold(
            uuid4(),
            hold=True,
            reason="Pending litigation",
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_release_legal_hold(self, repo, mock_session):
        """Test releasing legal hold."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        success = await repo.set_legal_hold(uuid4(), hold=False)

        assert success is True

    # Version operations

    @pytest.mark.asyncio
    async def test_create_version(self, repo, mock_session):
        """Test creating a new document version."""
        # Mock max version query
        max_version_result = MagicMock()
        max_version_result.scalar_one.return_value = 1

        # Mock update current versions
        update_result = MagicMock()

        mock_session.execute.side_effect = [max_version_result, update_result]

        version = await repo.create_version(
            document_id=uuid4(),
            storage_path="/documents/v2/doc.pdf",
            file_size=2048,
            checksum_sha256="def456",
            created_by="user-123",
            change_description="Updated content",
        )

        mock_session.add.assert_called()
        mock_session.flush.assert_called()

    @pytest.mark.asyncio
    async def test_get_versions(self, repo, mock_session):
        """Test getting all versions of a document."""
        mock_version = MagicMock()
        mock_version.version_number = 2

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_version]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        versions = await repo.get_versions(uuid4())

        assert len(versions) == 1
        assert versions[0].version_number == 2

    # Metadata operations

    @pytest.mark.asyncio
    async def test_add_metadata(self, repo, mock_session):
        """Test adding metadata to document."""
        metadata = await repo.add_metadata(
            document_id=uuid4(),
            category="extracted",
            key="author",
            value="John Doe",
            source="ocr",
            confidence=95,
        )

        mock_session.add.assert_called()
        mock_session.flush.assert_called()

    @pytest.mark.asyncio
    async def test_get_metadata(self, repo, mock_session):
        """Test getting document metadata."""
        mock_meta = MagicMock()
        mock_meta.key = "author"
        mock_meta.value = "John Doe"

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_meta]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        metadata = await repo.get_metadata(uuid4(), category="extracted")

        assert len(metadata) == 1
        assert metadata[0].key == "author"

    # Tag operations

    @pytest.mark.asyncio
    async def test_add_tag_new(self, repo, mock_session, mock_document):
        """Test adding a new tag to document."""
        # First execute: tag lookup returns None
        tag_result = MagicMock()
        tag_result.scalar_one_or_none.return_value = None

        # Second execute: document lookup
        doc_result = MagicMock()
        doc_result.scalar_one_or_none.return_value = mock_document

        mock_session.execute.side_effect = [tag_result, doc_result]

        success = await repo.add_tag(mock_document.id, "urgent")

        assert success is True
        mock_session.add.assert_called()

    @pytest.mark.asyncio
    async def test_remove_tag(self, repo, mock_session, mock_document):
        """Test removing a tag from document."""
        mock_tag = MagicMock()
        mock_tag.name = "urgent"
        mock_document.tags = [mock_tag]

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_document
        mock_session.execute.return_value = mock_result

        success = await repo.remove_tag(mock_document.id, "urgent")

        assert success is True
        assert mock_tag not in mock_document.tags

    # Audit operations

    @pytest.mark.asyncio
    async def test_log_action(self, repo, mock_session):
        """Test logging an action on document."""
        audit = await repo.log_action(
            document_id=uuid4(),
            action="update",
            user_id="user-123",
            action_detail="Changed classification",
            old_values={"classification": "internal"},
            new_values={"classification": "confidential"},
            user_ip="192.168.1.1",
        )

        mock_session.add.assert_called()
        mock_session.flush.assert_called()

    @pytest.mark.asyncio
    async def test_get_audit_logs(self, repo, mock_session):
        """Test getting audit logs."""
        mock_log = MagicMock()
        mock_log.action = "update"

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_log]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        logs = await repo.get_audit_logs(
            document_id=uuid4(),
            action="update",
            since=datetime.utcnow() - timedelta(days=7),
        )

        assert len(logs) == 1

    # Statistics

    @pytest.mark.asyncio
    async def test_get_statistics(self, repo, mock_session):
        """Test getting document statistics."""
        # Total count
        total_result = MagicMock()
        total_result.scalar_one.return_value = 100

        # By status
        status_result = MagicMock()
        status_result.all.return_value = [(MagicMock(value="validated"), 80), (MagicMock(value="processing"), 20)]

        # Total size
        size_result = MagicMock()
        size_result.scalar_one.return_value = 1024 * 1024 * 100

        mock_session.execute.side_effect = [total_result, status_result, size_result]

        stats = await repo.get_statistics(client_id="CLIENT-001")

        assert stats["total_documents"] == 100
        assert stats["total_size_bytes"] == 1024 * 1024 * 100


# =============================================================================
# ConnectorRepository Tests
# =============================================================================

class TestConnectorRepository:
    """Test suite for ConnectorRepository."""

    @pytest.fixture
    def repo(self, mock_session):
        """Create repository with mock session."""
        return ConnectorRepository(mock_session)

    # Create operations

    @pytest.mark.asyncio
    async def test_create_connector(self, repo, mock_session):
        """Test creating a new connector."""
        with patch.object(repo, '_get_model_class') as mock_get_class:
            mock_get_class.return_value = MagicMock

            connector = await repo.create(
                name="Email Connector",
                connector_type=MagicMock(),
                created_by="admin",
            )

            mock_session.add.assert_called_once()
            mock_session.flush.assert_called_once()

    def test_get_model_class(self, repo):
        """Test getting model class for connector type."""
        from ..models.connector_config import ConnectorType, EmailConnectorConfig

        model_class = repo._get_model_class(ConnectorType.EMAIL_IMAP)
        assert model_class == EmailConnectorConfig

    # Read operations

    @pytest.mark.asyncio
    async def test_get_by_id(self, repo, mock_session, mock_connector):
        """Test getting connector by ID."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_connector
        mock_session.execute.return_value = mock_result

        connector = await repo.get_by_id(mock_connector.id)

        assert connector == mock_connector

    @pytest.mark.asyncio
    async def test_list_connectors(self, repo, mock_session, mock_connector):
        """Test listing connectors with filters."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_connector]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        connectors = await repo.list_connectors(is_enabled=True)

        assert len(connectors) == 1

    @pytest.mark.asyncio
    async def test_get_connectors_due_for_poll(self, repo, mock_session, mock_connector):
        """Test getting connectors due for polling."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_connector]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        connectors = await repo.get_connectors_due_for_poll(limit=10)

        assert len(connectors) == 1

    # Update operations

    @pytest.mark.asyncio
    async def test_update_connector(self, repo, mock_session, mock_connector):
        """Test updating connector configuration."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_connector
        mock_session.execute.return_value = mock_result

        connector = await repo.update(
            mock_connector.id,
            name="Updated Connector",
            poll_interval_seconds=600,
        )

        assert connector.name == "Updated Connector"

    @pytest.mark.asyncio
    async def test_update_status(self, repo, mock_session):
        """Test updating connector status."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        success = await repo.update_status(
            uuid4(),
            MagicMock(),  # ConnectorStatus
            error="Connection failed",
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_record_poll_success(self, repo, mock_session, mock_connector):
        """Test recording successful poll."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_connector
        mock_session.execute.return_value = mock_result

        success = await repo.record_poll(
            mock_connector.id,
            success=True,
            documents_ingested=5,
            bytes_ingested=10240,
        )

        assert success is True
        assert mock_connector.total_documents_ingested == 105
        mock_session.flush.assert_called()

    @pytest.mark.asyncio
    async def test_record_poll_failure(self, repo, mock_session, mock_connector):
        """Test recording failed poll."""
        mock_connector.consecutive_errors = 0
        mock_connector.max_consecutive_errors = 5

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_connector
        mock_session.execute.return_value = mock_result

        success = await repo.record_poll(
            mock_connector.id,
            success=False,
            error="Connection timeout",
        )

        assert success is True
        assert mock_connector.consecutive_errors == 1

    @pytest.mark.asyncio
    async def test_record_poll_failure_exceeds_max(self, repo, mock_session, mock_connector):
        """Test that connector goes to error status after max failures."""
        from ..models.connector_config import ConnectorStatus

        mock_connector.consecutive_errors = 4
        mock_connector.max_consecutive_errors = 5

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_connector
        mock_session.execute.return_value = mock_result

        await repo.record_poll(mock_connector.id, success=False, error="Timeout")

        assert mock_connector.status == ConnectorStatus.ERROR

    # Enable/Disable operations

    @pytest.mark.asyncio
    async def test_enable_connector(self, repo, mock_session):
        """Test enabling a connector."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        success = await repo.enable(uuid4())

        assert success is True

    @pytest.mark.asyncio
    async def test_disable_connector(self, repo, mock_session):
        """Test disabling a connector."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        success = await repo.disable(uuid4())

        assert success is True

    # Delete operations

    @pytest.mark.asyncio
    async def test_delete_connector(self, repo, mock_session, mock_connector):
        """Test deleting a connector."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_connector
        mock_session.execute.return_value = mock_result

        success = await repo.delete(mock_connector.id)

        assert success is True
        mock_session.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_connector_not_found(self, repo, mock_session):
        """Test deleting non-existent connector."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        success = await repo.delete(uuid4())

        assert success is False

    # Statistics

    @pytest.mark.asyncio
    async def test_get_statistics(self, repo, mock_session):
        """Test getting connector statistics."""
        # By type
        type_result = MagicMock()
        type_result.all.return_value = [(MagicMock(value="email_imap"), 3)]

        # By status
        status_result = MagicMock()
        status_result.all.return_value = [(MagicMock(value="active"), 2)]

        # Totals
        totals_result = MagicMock()
        totals_result.one.return_value = (5, 1000, 1024000)

        mock_session.execute.side_effect = [type_result, status_result, totals_result]

        stats = await repo.get_statistics()

        assert stats["total_connectors"] == 5
        assert stats["total_documents_ingested"] == 1000


# =============================================================================
# IngestionJobRepository Tests
# =============================================================================

class TestIngestionJobRepository:
    """Test suite for IngestionJobRepository."""

    @pytest.fixture
    def repo(self, mock_session):
        """Create repository with mock session."""
        return IngestionJobRepository(mock_session)

    # Job operations

    @pytest.mark.asyncio
    async def test_create_job(self, repo, mock_session):
        """Test creating a new ingestion job."""
        from ..models.ingestion_job import IngestionSource

        job = await repo.create_job(
            source=IngestionSource.API_UPLOAD,
            created_by="user-123",
            name="Test Upload",
            client_id="CLIENT-001",
            priority=5,
        )

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_job(self, repo, mock_session, mock_job):
        """Test getting job by ID."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_session.execute.return_value = mock_result

        job = await repo.get_job(mock_job.id)

        assert job == mock_job

    @pytest.mark.asyncio
    async def test_get_job_with_items(self, repo, mock_session, mock_job):
        """Test getting job with items loaded."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_session.execute.return_value = mock_result

        job = await repo.get_job(mock_job.id, include_items=True)

        assert job == mock_job

    @pytest.mark.asyncio
    async def test_list_jobs(self, repo, mock_session, mock_job):
        """Test listing jobs with filters."""
        from ..models.ingestion_job import IngestionStatus

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_job]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        jobs = await repo.list_jobs(
            status=IngestionStatus.PENDING,
            created_by="user-123",
            limit=50,
        )

        assert len(jobs) == 1

    @pytest.mark.asyncio
    async def test_update_job_status_running(self, repo, mock_session):
        """Test updating job status to running."""
        from ..models.ingestion_job import IngestionStatus

        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        success = await repo.update_job_status(
            uuid4(),
            IngestionStatus.RUNNING,
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_update_job_status_completed(self, repo, mock_session):
        """Test updating job status to completed."""
        from ..models.ingestion_job import IngestionStatus

        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        success = await repo.update_job_status(
            uuid4(),
            IngestionStatus.COMPLETED,
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_update_job_progress(self, repo, mock_session):
        """Test updating job progress counters."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        success = await repo.update_job_progress(
            uuid4(),
            completed=5,
            failed=1,
            skipped=2,
            bytes_processed=5120,
        )

        assert success is True

    # Item operations

    @pytest.mark.asyncio
    async def test_add_item(self, repo, mock_session, mock_job):
        """Test adding item to job."""
        mock_update_result = MagicMock()
        mock_session.execute.return_value = mock_update_result

        item = await repo.add_item(
            job_id=mock_job.id,
            original_filename="document.pdf",
            file_size=1024,
            mime_type="application/pdf",
            source_path="/uploads/document.pdf",
            checksum_sha256="abc123",
        )

        mock_session.add.assert_called()
        mock_session.flush.assert_called()

    @pytest.mark.asyncio
    async def test_add_items_batch(self, repo, mock_session, mock_job):
        """Test adding multiple items efficiently."""
        mock_update_result = MagicMock()
        mock_session.execute.return_value = mock_update_result

        items_data = [
            {"filename": "doc1.pdf", "size": 1024, "mime_type": "application/pdf"},
            {"filename": "doc2.pdf", "size": 2048, "mime_type": "application/pdf"},
            {"filename": "doc3.docx", "size": 512, "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
        ]

        items = await repo.add_items_batch(mock_job.id, items_data)

        assert len(items) == 3
        mock_session.flush.assert_called()

    @pytest.mark.asyncio
    async def test_get_item(self, repo, mock_session, mock_job_item):
        """Test getting item by ID."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job_item
        mock_session.execute.return_value = mock_result

        item = await repo.get_item(mock_job_item.id)

        assert item == mock_job_item

    @pytest.mark.asyncio
    async def test_get_pending_items(self, repo, mock_session, mock_job_item):
        """Test getting pending items for job."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_job_item]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        items = await repo.get_pending_items(uuid4(), limit=50)

        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_update_item_status_processing(self, repo, mock_session):
        """Test updating item status to processing."""
        from ..models.ingestion_job import ItemStatus

        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        success = await repo.update_item_status(
            uuid4(),
            ItemStatus.PROCESSING,
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_update_item_status_completed(self, repo, mock_session):
        """Test updating item status to completed."""
        from ..models.ingestion_job import ItemStatus

        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        success = await repo.update_item_status(
            uuid4(),
            ItemStatus.COMPLETED,
            document_id=uuid4(),
            processing_time_ms=150,
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_update_item_status_failed(self, repo, mock_session):
        """Test updating item status to failed."""
        from ..models.ingestion_job import ItemStatus

        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        success = await repo.update_item_status(
            uuid4(),
            ItemStatus.FAILED,
            error_message="Validation failed",
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_increment_retry(self, repo, mock_session):
        """Test incrementing retry count."""
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 2
        mock_session.execute.return_value = mock_result

        retries = await repo.increment_retry(uuid4())

        assert retries == 2

    @pytest.mark.asyncio
    async def test_find_by_checksum(self, repo, mock_session, mock_job_item):
        """Test finding item by checksum within job."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job_item
        mock_session.execute.return_value = mock_result

        item = await repo.find_by_checksum(uuid4(), "abc123")

        assert item == mock_job_item

    # Queue operations

    @pytest.mark.asyncio
    async def test_enqueue_job(self, repo, mock_session):
        """Test adding job to queue."""
        queue_item = await repo.enqueue_job(
            job_id=uuid4(),
            priority=5,
            not_before=datetime.utcnow() + timedelta(minutes=10),
            rate_limit_key="CLIENT-001",
        )

        mock_session.add.assert_called()
        mock_session.flush.assert_called()

    @pytest.mark.asyncio
    async def test_dequeue_job(self, repo, mock_session):
        """Test getting next job from queue."""
        mock_queue_item = MagicMock()
        mock_queue_item.is_processing = False

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_queue_item
        mock_session.execute.return_value = mock_result

        item = await repo.dequeue_job(worker_id="worker-1")

        assert item.is_processing is True
        mock_session.flush.assert_called()

    @pytest.mark.asyncio
    async def test_dequeue_job_empty_queue(self, repo, mock_session):
        """Test dequeueing from empty queue."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        item = await repo.dequeue_job(worker_id="worker-1")

        assert item is None

    @pytest.mark.asyncio
    async def test_release_queue_item_complete(self, repo, mock_session):
        """Test releasing and removing queue item."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        success = await repo.release_queue_item(uuid4(), reschedule=False)

        assert success is True

    @pytest.mark.asyncio
    async def test_release_queue_item_reschedule(self, repo, mock_session):
        """Test releasing and rescheduling queue item."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        success = await repo.release_queue_item(uuid4(), reschedule=True)

        assert success is True

    # Statistics

    @pytest.mark.asyncio
    async def test_get_job_statistics(self, repo, mock_session):
        """Test getting job statistics."""
        # By status
        status_result = MagicMock()
        status_result.all.return_value = [
            (MagicMock(value="completed"), 80),
            (MagicMock(value="failed"), 5),
        ]

        # By source
        source_result = MagicMock()
        source_result.all.return_value = [
            (MagicMock(value="api_upload"), 50),
            (MagicMock(value="email"), 35),
        ]

        # Items totals
        items_result = MagicMock()
        items_result.one.return_value = (1000, 50, 1024 * 1024 * 100)

        # Queue size
        queue_result = MagicMock()
        queue_result.scalar_one.return_value = 10

        mock_session.execute.side_effect = [status_result, source_result, items_result, queue_result]

        stats = await repo.get_job_statistics(
            since=datetime.utcnow() - timedelta(days=30)
        )

        assert stats["total_completed_items"] == 1000
        assert stats["total_failed_items"] == 50
        assert stats["queue_size"] == 10
