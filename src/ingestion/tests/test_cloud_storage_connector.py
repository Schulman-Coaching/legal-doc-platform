"""
Tests for Cloud Storage Connectors
==================================
Comprehensive tests for S3, Azure Blob, and GCS connectors.
"""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from ..connectors.cloud_storage_connector import (
    CloudStorageConnector,
    S3Connector,
    AzureBlobConnector,
    GCSConnector,
    CloudStorageConfig,
    S3Config,
    AzureBlobConfig,
    GCSConfig,
    CloudObject,
    CloudProvider,
    CloudEventType,
)

# Check for optional dependencies
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    import azure.storage.blob
    HAS_AZURE = True
except ImportError:
    HAS_AZURE = False

try:
    import google.cloud.storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False


class TestCloudStorageConfig:
    """Test suite for CloudStorageConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CloudStorageConfig(
            bucket_name="test-bucket",
            provider=CloudProvider.AWS_S3,
        )

        assert config.prefix == ""
        assert config.poll_interval_seconds == 60
        assert "*.pdf" in config.file_patterns
        assert ".*" in config.exclude_patterns
        assert config.max_file_size == 1024 * 1024 * 1024
        assert config.delete_after_processing is False
        assert config.move_after_processing is True
        assert config.processed_prefix == "processed/"
        assert config.error_prefix == "errors/"

    def test_s3_config(self):
        """Test S3-specific configuration."""
        config = S3Config(
            bucket_name="legal-documents",
            prefix="incoming/",
            region_name="eu-west-1",
            aws_access_key_id="AKIATEST",
            aws_secret_access_key="secret123",
            endpoint_url="http://localhost:9000",
        )

        assert config.provider == CloudProvider.AWS_S3
        assert config.region_name == "eu-west-1"
        assert config.aws_access_key_id == "AKIATEST"
        assert config.endpoint_url == "http://localhost:9000"
        assert config.use_ssl is True

    def test_azure_config(self):
        """Test Azure Blob configuration."""
        config = AzureBlobConfig(
            bucket_name="legal-container",
            connection_string="DefaultEndpointsProtocol=https;...",
            prefix="documents/",
        )

        assert config.provider == CloudProvider.AZURE_BLOB
        assert config.connection_string is not None

    def test_gcs_config(self):
        """Test GCS configuration."""
        config = GCSConfig(
            bucket_name="legal-bucket",
            project_id="my-project",
            credentials_path="/path/to/credentials.json",
        )

        assert config.provider == CloudProvider.GOOGLE_CLOUD_STORAGE
        assert config.project_id == "my-project"


class TestCloudObject:
    """Test suite for CloudObject dataclass."""

    def test_cloud_object_creation(self):
        """Test CloudObject creation."""
        obj = CloudObject(
            key="documents/contract.pdf",
            bucket="legal-bucket",
            provider=CloudProvider.AWS_S3,
            size=1024 * 1024,
            last_modified=datetime.utcnow(),
            etag="abc123",
            content_type="application/pdf",
            metadata={"client_id": "CLIENT-001"},
        )

        assert obj.key == "documents/contract.pdf"
        assert obj.bucket == "legal-bucket"
        assert obj.provider == CloudProvider.AWS_S3
        assert obj.size == 1024 * 1024
        assert obj.etag == "abc123"

    def test_cloud_object_defaults(self):
        """Test CloudObject default values."""
        obj = CloudObject(
            key="file.pdf",
            bucket="bucket",
            provider=CloudProvider.AZURE_BLOB,
            size=100,
            last_modified=datetime.utcnow(),
        )

        assert obj.etag is None
        assert obj.content_type is None
        assert obj.metadata == {}
        assert obj.checksum is None
        assert obj.local_path is None


class TestS3Connector:
    """Test suite for S3Connector."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def s3_config(self):
        """Create test S3 configuration."""
        return S3Config(
            bucket_name="test-legal-bucket",
            prefix="incoming/",
            region_name="us-east-1",
            aws_access_key_id="AKIATEST",
            aws_secret_access_key="secret123",
            file_patterns=["*.pdf", "*.docx"],
            exclude_patterns=[".*", "*.tmp"],
        )

    @pytest.fixture
    def connector(self, s3_config, temp_storage):
        """Create S3Connector instance."""
        return S3Connector(
            config=s3_config,
            storage_path=temp_storage,
        )

    @pytest.fixture
    def mock_s3_client(self):
        """Create mock S3 client."""
        mock = MagicMock()
        return mock

    # ===================
    # Client Creation Tests
    # ===================

    @pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
    def test_get_client_with_credentials(self, connector):
        """Test S3 client creation with explicit credentials."""
        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_client = MagicMock()
            mock_session.client.return_value = mock_client
            mock_session_class.return_value = mock_session

            client = connector._get_client()

            mock_session_class.assert_called_once_with(
                aws_access_key_id="AKIATEST",
                aws_secret_access_key="secret123",
            )
            mock_session.client.assert_called_once()

    @pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
    def test_get_client_with_endpoint(self, temp_storage):
        """Test S3 client with custom endpoint (S3-compatible)."""
        config = S3Config(
            bucket_name="test-bucket",
            endpoint_url="http://localhost:9000",
            use_ssl=False,
        )
        connector = S3Connector(config=config, storage_path=temp_storage)

        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_client = MagicMock()
            mock_session.client.return_value = mock_client
            mock_session_class.return_value = mock_session

            connector._get_client()

            call_kwargs = mock_session.client.call_args[1]
            assert call_kwargs["endpoint_url"] == "http://localhost:9000"
            assert call_kwargs["use_ssl"] is False

    # ===================
    # List Objects Tests
    # ===================

    @pytest.mark.asyncio
    async def test_list_objects(self, connector, mock_s3_client):
        """Test listing S3 objects."""
        # Setup paginator mock
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "incoming/document1.pdf",
                        "Size": 1024,
                        "LastModified": datetime.utcnow(),
                        "ETag": '"abc123"',
                    },
                    {
                        "Key": "incoming/document2.docx",
                        "Size": 2048,
                        "LastModified": datetime.utcnow(),
                        "ETag": '"def456"',
                    },
                ]
            }
        ]
        mock_s3_client.get_paginator.return_value = mock_paginator

        with patch.object(connector, "_get_client", return_value=mock_s3_client):
            objects = await connector.list_objects("incoming/")

        assert len(objects) == 2
        assert any(o.key == "incoming/document1.pdf" for o in objects)
        assert any(o.key == "incoming/document2.docx" for o in objects)

    @pytest.mark.asyncio
    async def test_list_objects_empty_bucket(self, connector, mock_s3_client):
        """Test listing empty bucket."""
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Contents": []}]
        mock_s3_client.get_paginator.return_value = mock_paginator

        with patch.object(connector, "_get_client", return_value=mock_s3_client):
            objects = await connector.list_objects()

        assert len(objects) == 0

    # ===================
    # Download Object Tests
    # ===================

    @pytest.mark.asyncio
    async def test_download_object(self, connector, mock_s3_client):
        """Test downloading S3 object."""
        mock_body = MagicMock()
        mock_body.read.return_value = b"%PDF-1.4\nTest content"

        mock_s3_client.get_object.return_value = {
            "Body": mock_body,
            "Metadata": {"client_id": "CLIENT-001"},
        }

        with patch.object(connector, "_get_client", return_value=mock_s3_client):
            content, metadata = await connector.download_object("incoming/document.pdf")

        assert b"%PDF-1.4" in content
        assert metadata["client_id"] == "CLIENT-001"
        mock_s3_client.get_object.assert_called_once_with(
            Bucket="test-legal-bucket",
            Key="incoming/document.pdf",
        )

    # ===================
    # Move Object Tests
    # ===================

    @pytest.mark.asyncio
    async def test_move_object(self, connector, mock_s3_client):
        """Test moving S3 object."""
        with patch.object(connector, "_get_client", return_value=mock_s3_client):
            result = await connector.move_object(
                "incoming/document.pdf",
                "processed/document.pdf",
            )

        assert result is True
        mock_s3_client.copy_object.assert_called_once()
        mock_s3_client.delete_object.assert_called_once()

    # ===================
    # Delete Object Tests
    # ===================

    @pytest.mark.asyncio
    async def test_delete_object(self, connector, mock_s3_client):
        """Test deleting S3 object."""
        with patch.object(connector, "_get_client", return_value=mock_s3_client):
            result = await connector.delete_object("incoming/document.pdf")

        assert result is True
        mock_s3_client.delete_object.assert_called_once_with(
            Bucket="test-legal-bucket",
            Key="incoming/document.pdf",
        )

    # ===================
    # Connection Test
    # ===================

    @pytest.mark.asyncio
    async def test_test_connection_success(self, connector):
        """Test successful connection test."""
        with patch.object(connector, "list_objects") as mock_list:
            mock_list.return_value = [
                CloudObject(
                    key="file1.pdf",
                    bucket="test-bucket",
                    provider=CloudProvider.AWS_S3,
                    size=1024,
                    last_modified=datetime.utcnow(),
                ),
            ]

            success, message = await connector.test_connection()

        assert success is True
        assert "1 objects" in message

    @pytest.mark.asyncio
    async def test_test_connection_failure(self, connector):
        """Test failed connection test."""
        with patch.object(connector, "list_objects") as mock_list:
            mock_list.side_effect = Exception("Access denied")

            success, message = await connector.test_connection()

        assert success is False
        assert "Access denied" in message


class TestAzureBlobConnector:
    """Test suite for AzureBlobConnector."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def azure_config(self):
        """Create test Azure Blob configuration."""
        return AzureBlobConfig(
            bucket_name="legal-container",
            connection_string="DefaultEndpointsProtocol=https;AccountName=test;...",
            prefix="incoming/",
            file_patterns=["*.pdf", "*.docx"],
        )

    @pytest.fixture
    def connector(self, azure_config, temp_storage):
        """Create AzureBlobConnector instance."""
        return AzureBlobConnector(
            config=azure_config,
            storage_path=temp_storage,
        )

    @pytest.fixture
    def mock_container_client(self):
        """Create mock Azure container client."""
        return MagicMock()

    # ===================
    # Client Creation Tests
    # ===================

    @pytest.mark.skipif(not HAS_AZURE, reason="azure-storage-blob not installed")
    def test_get_container_client_connection_string(self, connector):
        """Test container client creation with connection string."""
        with patch("azure.storage.blob.ContainerClient") as mock_class:
            mock_client = MagicMock()
            mock_class.from_connection_string.return_value = mock_client

            client = connector._get_container_client()

            mock_class.from_connection_string.assert_called_once()

    @pytest.mark.skipif(not HAS_AZURE, reason="azure-storage-blob not installed")
    def test_get_container_client_sas_token(self, temp_storage):
        """Test container client creation with SAS token."""
        config = AzureBlobConfig(
            bucket_name="legal-container",
            account_name="testaccount",
            sas_token="?sv=2020-08-04&sig=xxx",
        )
        connector = AzureBlobConnector(config=config, storage_path=temp_storage)

        with patch("azure.storage.blob.BlobServiceClient") as mock_service:
            mock_service_instance = MagicMock()
            mock_container = MagicMock()
            mock_service_instance.get_container_client.return_value = mock_container
            mock_service.return_value = mock_service_instance

            client = connector._get_container_client()

            mock_service.assert_called_once()

    # ===================
    # List Objects Tests
    # ===================

    @pytest.mark.asyncio
    async def test_list_objects(self, connector, mock_container_client):
        """Test listing Azure blobs."""
        mock_blob1 = MagicMock()
        mock_blob1.name = "incoming/document1.pdf"
        mock_blob1.size = 1024
        mock_blob1.last_modified = datetime.utcnow()
        mock_blob1.etag = "abc123"
        mock_blob1.content_settings = MagicMock()
        mock_blob1.content_settings.content_type = "application/pdf"

        mock_blob2 = MagicMock()
        mock_blob2.name = "incoming/document2.docx"
        mock_blob2.size = 2048
        mock_blob2.last_modified = datetime.utcnow()
        mock_blob2.etag = "def456"
        mock_blob2.content_settings = MagicMock()
        mock_blob2.content_settings.content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

        mock_container_client.list_blobs.return_value = [mock_blob1, mock_blob2]

        with patch.object(connector, "_get_container_client", return_value=mock_container_client):
            objects = await connector.list_objects("incoming/")

        assert len(objects) == 2
        assert objects[0].provider == CloudProvider.AZURE_BLOB

    # ===================
    # Download Object Tests
    # ===================

    @pytest.mark.asyncio
    async def test_download_object(self, connector, mock_container_client):
        """Test downloading Azure blob."""
        mock_blob_client = MagicMock()
        mock_download = MagicMock()
        mock_download.readall.return_value = b"%PDF-1.4\nTest content"
        mock_blob_client.download_blob.return_value = mock_download

        mock_properties = MagicMock()
        mock_properties.metadata = {"client_id": "CLIENT-001"}
        mock_blob_client.get_blob_properties.return_value = mock_properties

        mock_container_client.get_blob_client.return_value = mock_blob_client

        with patch.object(connector, "_get_container_client", return_value=mock_container_client):
            content, metadata = await connector.download_object("incoming/document.pdf")

        assert b"%PDF-1.4" in content
        assert metadata["client_id"] == "CLIENT-001"

    # ===================
    # Move Object Tests
    # ===================

    @pytest.mark.asyncio
    async def test_move_object(self, connector, mock_container_client):
        """Test moving Azure blob."""
        mock_source = MagicMock()
        mock_source.url = "https://test.blob.core.windows.net/container/source.pdf"

        mock_dest = MagicMock()
        mock_dest_props = MagicMock()
        mock_dest_props.copy.status = "success"
        mock_dest.get_blob_properties.return_value = mock_dest_props

        def get_blob_client_side_effect(key):
            if key == "incoming/document.pdf":
                return mock_source
            return mock_dest

        mock_container_client.get_blob_client.side_effect = get_blob_client_side_effect

        with patch.object(connector, "_get_container_client", return_value=mock_container_client):
            result = await connector.move_object(
                "incoming/document.pdf",
                "processed/document.pdf",
            )

        assert result is True

    # ===================
    # Delete Object Tests
    # ===================

    @pytest.mark.asyncio
    async def test_delete_object(self, connector, mock_container_client):
        """Test deleting Azure blob."""
        mock_blob_client = MagicMock()
        mock_container_client.get_blob_client.return_value = mock_blob_client

        with patch.object(connector, "_get_container_client", return_value=mock_container_client):
            result = await connector.delete_object("incoming/document.pdf")

        assert result is True
        mock_blob_client.delete_blob.assert_called_once()

    # ===================
    # Connection Test
    # ===================

    @pytest.mark.asyncio
    async def test_test_connection_success(self, connector):
        """Test successful Azure connection test."""
        with patch.object(connector, "list_objects") as mock_list:
            mock_list.return_value = [
                CloudObject(
                    key="file1.pdf",
                    bucket="test-container",
                    provider=CloudProvider.AZURE_BLOB,
                    size=1024,
                    last_modified=datetime.utcnow(),
                ),
            ]

            success, message = await connector.test_connection()

        assert success is True
        assert "1 blobs" in message


class TestGCSConnector:
    """Test suite for GCSConnector."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def gcs_config(self):
        """Create test GCS configuration."""
        return GCSConfig(
            bucket_name="legal-bucket",
            project_id="test-project",
            prefix="incoming/",
            file_patterns=["*.pdf", "*.docx"],
        )

    @pytest.fixture
    def connector(self, gcs_config, temp_storage):
        """Create GCSConnector instance."""
        return GCSConnector(
            config=gcs_config,
            storage_path=temp_storage,
        )

    @pytest.fixture
    def mock_bucket(self):
        """Create mock GCS bucket."""
        return MagicMock()

    # ===================
    # Bucket Creation Tests
    # ===================

    @pytest.mark.skipif(not HAS_GCS, reason="google-cloud-storage not installed")
    def test_get_bucket_with_credentials(self, connector):
        """Test bucket creation with service account credentials."""
        connector.gcs_config.credentials_path = "/path/to/credentials.json"

        with patch("google.cloud.storage.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_bucket = MagicMock()
            mock_client.bucket.return_value = mock_bucket
            mock_client_class.from_service_account_json.return_value = mock_client

            bucket = connector._get_bucket()

            mock_client_class.from_service_account_json.assert_called_once()
            mock_client.bucket.assert_called_with("legal-bucket")

    @pytest.mark.skipif(not HAS_GCS, reason="google-cloud-storage not installed")
    def test_get_bucket_default_credentials(self, temp_storage):
        """Test bucket creation with default credentials."""
        config = GCSConfig(
            bucket_name="legal-bucket",
            project_id="test-project",
        )
        connector = GCSConnector(config=config, storage_path=temp_storage)

        with patch("google.cloud.storage.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_bucket = MagicMock()
            mock_client.bucket.return_value = mock_bucket
            mock_client_class.return_value = mock_client

            bucket = connector._get_bucket()

            mock_client_class.assert_called_once_with(project="test-project")

    # ===================
    # List Objects Tests
    # ===================

    @pytest.mark.asyncio
    async def test_list_objects(self, connector, mock_bucket):
        """Test listing GCS objects."""
        mock_blob1 = MagicMock()
        mock_blob1.name = "incoming/document1.pdf"
        mock_blob1.size = 1024
        mock_blob1.updated = datetime.utcnow()
        mock_blob1.etag = "abc123"
        mock_blob1.content_type = "application/pdf"
        mock_blob1.metadata = {}

        mock_blob2 = MagicMock()
        mock_blob2.name = "incoming/document2.docx"
        mock_blob2.size = 2048
        mock_blob2.updated = datetime.utcnow()
        mock_blob2.etag = "def456"
        mock_blob2.content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        mock_blob2.metadata = {"client": "ACME"}

        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]

        with patch.object(connector, "_get_bucket", return_value=mock_bucket):
            objects = await connector.list_objects("incoming/")

        assert len(objects) == 2
        assert objects[0].provider == CloudProvider.GOOGLE_CLOUD_STORAGE

    # ===================
    # Download Object Tests
    # ===================

    @pytest.mark.asyncio
    async def test_download_object(self, connector, mock_bucket):
        """Test downloading GCS object."""
        mock_blob = MagicMock()
        mock_blob.download_as_bytes.return_value = b"%PDF-1.4\nTest content"
        mock_blob.metadata = {"client_id": "CLIENT-001"}

        mock_bucket.blob.return_value = mock_blob

        with patch.object(connector, "_get_bucket", return_value=mock_bucket):
            content, metadata = await connector.download_object("incoming/document.pdf")

        assert b"%PDF-1.4" in content
        assert metadata["client_id"] == "CLIENT-001"

    # ===================
    # Move Object Tests
    # ===================

    @pytest.mark.asyncio
    async def test_move_object(self, connector, mock_bucket):
        """Test moving GCS object."""
        mock_source_blob = MagicMock()
        mock_bucket.blob.return_value = mock_source_blob

        with patch.object(connector, "_get_bucket", return_value=mock_bucket):
            result = await connector.move_object(
                "incoming/document.pdf",
                "processed/document.pdf",
            )

        assert result is True
        mock_bucket.copy_blob.assert_called_once()
        mock_source_blob.delete.assert_called_once()

    # ===================
    # Delete Object Tests
    # ===================

    @pytest.mark.asyncio
    async def test_delete_object(self, connector, mock_bucket):
        """Test deleting GCS object."""
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        with patch.object(connector, "_get_bucket", return_value=mock_bucket):
            result = await connector.delete_object("incoming/document.pdf")

        assert result is True
        mock_blob.delete.assert_called_once()

    # ===================
    # Connection Test
    # ===================

    @pytest.mark.asyncio
    async def test_test_connection_success(self, connector):
        """Test successful GCS connection test."""
        with patch.object(connector, "list_objects") as mock_list:
            mock_list.return_value = [
                CloudObject(
                    key="file1.pdf",
                    bucket="test-bucket",
                    provider=CloudProvider.GOOGLE_CLOUD_STORAGE,
                    size=1024,
                    last_modified=datetime.utcnow(),
                ),
            ]

            success, message = await connector.test_connection()

        assert success is True
        assert "1 objects" in message


class TestCloudStorageConnectorBase:
    """Test suite for CloudStorageConnector base class methods."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def s3_connector(self, temp_storage):
        """Create S3Connector for base class testing."""
        config = S3Config(
            bucket_name="test-bucket",
            file_patterns=["*.pdf", "*.docx"],
            exclude_patterns=[".*", "*.tmp", "temp_*"],
        )
        return S3Connector(config=config, storage_path=temp_storage)

    # ===================
    # Pattern Matching Tests
    # ===================

    def test_matches_patterns_pdf(self, s3_connector):
        """Test pattern matching for PDF files."""
        assert s3_connector._matches_patterns("documents/contract.pdf", ["*.pdf"]) is True
        assert s3_connector._matches_patterns("docs/Report.PDF", ["*.pdf"]) is True

    def test_matches_patterns_no_match(self, s3_connector):
        """Test pattern matching with no match."""
        assert s3_connector._matches_patterns("images/photo.png", ["*.pdf", "*.docx"]) is False

    def test_matches_patterns_hidden(self, s3_connector):
        """Test pattern matching for hidden files."""
        assert s3_connector._matches_patterns("folder/.hidden", [".*"]) is True

    def test_matches_patterns_temp(self, s3_connector):
        """Test pattern matching for temp files."""
        assert s3_connector._matches_patterns("temp_upload.pdf", ["temp_*"]) is True
        assert s3_connector._matches_patterns("cache.tmp", ["*.tmp"]) is True

    # ===================
    # Post-Processing Tests
    # ===================

    @pytest.mark.asyncio
    async def test_post_process_delete(self, temp_storage):
        """Test delete post-processing."""
        config = S3Config(
            bucket_name="test-bucket",
            delete_after_processing=True,
            move_after_processing=False,
        )
        connector = S3Connector(config=config, storage_path=temp_storage)

        obj = CloudObject(
            key="incoming/document.pdf",
            bucket="test-bucket",
            provider=CloudProvider.AWS_S3,
            size=1024,
            last_modified=datetime.utcnow(),
        )

        with patch.object(connector, "delete_object") as mock_delete:
            mock_delete.return_value = True
            await connector._post_process(obj)
            mock_delete.assert_called_once_with("incoming/document.pdf")

    @pytest.mark.asyncio
    async def test_post_process_move(self, temp_storage):
        """Test move post-processing."""
        config = S3Config(
            bucket_name="test-bucket",
            delete_after_processing=False,
            move_after_processing=True,
            processed_prefix="processed/",
        )
        connector = S3Connector(config=config, storage_path=temp_storage)

        obj = CloudObject(
            key="incoming/document.pdf",
            bucket="test-bucket",
            provider=CloudProvider.AWS_S3,
            size=1024,
            last_modified=datetime.utcnow(),
        )

        with patch.object(connector, "move_object") as mock_move:
            mock_move.return_value = True
            await connector._post_process(obj)
            mock_move.assert_called_once_with(
                "incoming/document.pdf",
                "processed/incoming/document.pdf",
            )

    # ===================
    # Error Handling Tests
    # ===================

    @pytest.mark.asyncio
    async def test_handle_error(self, s3_connector):
        """Test error handling moves to error prefix."""
        obj = CloudObject(
            key="incoming/bad_file.pdf",
            bucket="test-bucket",
            provider=CloudProvider.AWS_S3,
            size=1024,
            last_modified=datetime.utcnow(),
        )

        with patch.object(s3_connector, "move_object") as mock_move:
            mock_move.return_value = True
            await s3_connector._handle_error(obj)

            mock_move.assert_called_once_with(
                "incoming/bad_file.pdf",
                "errors/incoming/bad_file.pdf",
            )

        # Object should be marked as processed
        assert "incoming/bad_file.pdf" in s3_connector._processed_keys

    # ===================
    # Lifecycle Tests
    # ===================

    @pytest.mark.asyncio
    async def test_stop(self, s3_connector):
        """Test connector stop."""
        s3_connector._running = True
        await s3_connector.stop()
        assert s3_connector._running is False

    # ===================
    # Process Object Tests
    # ===================

    @pytest.mark.asyncio
    async def test_process_object_skips_processed(self, s3_connector):
        """Test that already processed objects are skipped."""
        obj = CloudObject(
            key="incoming/document.pdf",
            bucket="test-bucket",
            provider=CloudProvider.AWS_S3,
            size=1024,
            last_modified=datetime.utcnow(),
        )

        s3_connector._processed_keys.add("incoming/document.pdf")

        with patch.object(s3_connector, "download_object") as mock_download:
            await s3_connector._process_object(obj)
            mock_download.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_object_skips_excluded(self, s3_connector):
        """Test that excluded patterns are skipped."""
        obj = CloudObject(
            key="incoming/.hidden.pdf",
            bucket="test-bucket",
            provider=CloudProvider.AWS_S3,
            size=1024,
            last_modified=datetime.utcnow(),
        )

        with patch.object(s3_connector, "download_object") as mock_download:
            await s3_connector._process_object(obj)
            mock_download.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_object_skips_not_matching(self, s3_connector):
        """Test that non-matching patterns are skipped."""
        obj = CloudObject(
            key="incoming/image.png",
            bucket="test-bucket",
            provider=CloudProvider.AWS_S3,
            size=1024,
            last_modified=datetime.utcnow(),
        )

        with patch.object(s3_connector, "download_object") as mock_download:
            await s3_connector._process_object(obj)
            mock_download.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_object_skips_too_large(self, temp_storage):
        """Test that oversized files are skipped."""
        config = S3Config(
            bucket_name="test-bucket",
            max_file_size=1024 * 1024,  # 1MB
        )
        connector = S3Connector(config=config, storage_path=temp_storage)

        obj = CloudObject(
            key="incoming/huge.pdf",
            bucket="test-bucket",
            provider=CloudProvider.AWS_S3,
            size=100 * 1024 * 1024,  # 100MB
            last_modified=datetime.utcnow(),
        )

        with patch.object(connector, "download_object") as mock_download:
            await connector._process_object(obj)
            mock_download.assert_not_called()
