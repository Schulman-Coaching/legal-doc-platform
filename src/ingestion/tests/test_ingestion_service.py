"""
Integration Tests for Ingestion Service
=======================================
Tests for the main IngestionService, APIUploadConnector,
ChunkedUploadManager, and end-to-end document workflows.
"""

import asyncio
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from ..document_ingestion_service import (
    IngestionService,
    IngestionRequest,
    IngestionResponse,
    DocumentStatus,
    DocumentMetadata,
    DocumentValidator,
    DocumentEncryption,
    APIUploadConnector,
    ChunkedUploadManager,
    KafkaDocumentPublisher,
    IngestionSource,
    SecurityClassification,
)


class TestDocumentEncryption:
    """Test suite for DocumentEncryption."""

    @pytest.fixture
    def encryptor(self):
        """Create encryption instance."""
        return DocumentEncryption()

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_encrypt_decrypt_roundtrip(self, encryptor, temp_dir):
        """Test encryption and decryption produce original content."""
        # Create test file
        original_content = b"This is sensitive legal document content"
        input_path = temp_dir / "original.txt"
        encrypted_path = temp_dir / "encrypted.enc"
        decrypted_path = temp_dir / "decrypted.txt"

        input_path.write_bytes(original_content)

        # Encrypt
        key_id = await encryptor.encrypt_file(input_path, encrypted_path)

        assert key_id is not None
        assert len(key_id) == 16
        assert encrypted_path.exists()

        # Encrypted content should be different
        encrypted_content = encrypted_path.read_bytes()
        assert encrypted_content != original_content

        # Decrypt
        await encryptor.decrypt_file(encrypted_path, decrypted_path)

        # Should match original
        decrypted_content = decrypted_path.read_bytes()
        assert decrypted_content == original_content

    @pytest.mark.asyncio
    async def test_encrypt_produces_different_output(self, encryptor, temp_dir):
        """Test that encrypting same content twice produces different ciphertext."""
        content = b"Test content"
        input_path = temp_dir / "input.txt"
        encrypted_path1 = temp_dir / "encrypted1.enc"
        encrypted_path2 = temp_dir / "encrypted2.enc"

        input_path.write_bytes(content)

        await encryptor.encrypt_file(input_path, encrypted_path1)
        await encryptor.encrypt_file(input_path, encrypted_path2)

        # Fernet encryption includes random IV, so outputs should differ
        enc1 = encrypted_path1.read_bytes()
        enc2 = encrypted_path2.read_bytes()

        # Both should decrypt to same content, but ciphertext differs
        assert enc1 != enc2

    @pytest.mark.asyncio
    async def test_key_id_consistent(self, encryptor, temp_dir):
        """Test that key_id is consistent for same encryptor."""
        content = b"Test content"
        input_path = temp_dir / "input.txt"
        encrypted_path1 = temp_dir / "encrypted1.enc"
        encrypted_path2 = temp_dir / "encrypted2.enc"

        input_path.write_bytes(content)

        key_id1 = await encryptor.encrypt_file(input_path, encrypted_path1)
        key_id2 = await encryptor.encrypt_file(input_path, encrypted_path2)

        assert key_id1 == key_id2


class TestAPIUploadConnector:
    """Test suite for APIUploadConnector."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def connector(self, temp_storage):
        """Create APIUploadConnector instance."""
        validator = DocumentValidator(enable_malware_scan=False)
        encryption = DocumentEncryption()
        return APIUploadConnector(
            validator=validator,
            encryption=encryption,
            storage_path=temp_storage,
        )

    @pytest.fixture
    def valid_request(self):
        """Create valid ingestion request."""
        return IngestionRequest(
            filename="contract.pdf",
            content_type="application/pdf",
            source=IngestionSource.API_UPLOAD,
            user_id="user-123",
            client_id="CLIENT-001",
            matter_id="MATTER-001",
            classification=SecurityClassification.CONFIDENTIAL,
            tags=["contract", "legal"],
        )

    @pytest.fixture
    def valid_pdf_content(self):
        """Create valid PDF content."""
        return b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"

    @pytest.mark.asyncio
    async def test_ingest_valid_document(self, connector, valid_request, valid_pdf_content):
        """Test successful document ingestion."""
        response = await connector.ingest(valid_request, valid_pdf_content)

        assert response.status == DocumentStatus.VALIDATED
        assert response.document_id is not None
        assert response.checksum is not None
        assert response.bytes_received == len(valid_pdf_content)
        assert "successfully" in response.message.lower()

    @pytest.mark.asyncio
    async def test_ingest_creates_encrypted_file(self, connector, valid_request, valid_pdf_content, temp_storage):
        """Test that ingestion creates encrypted file."""
        response = await connector.ingest(valid_request, valid_pdf_content)

        encrypted_path = temp_storage / f"{response.document_id}.enc"
        assert encrypted_path.exists()

        # Content should be encrypted (different from original)
        encrypted_content = encrypted_path.read_bytes()
        assert encrypted_content != valid_pdf_content

    @pytest.mark.asyncio
    async def test_ingest_computes_checksums(self, connector, valid_request, valid_pdf_content):
        """Test that checksums are computed correctly."""
        import hashlib

        response = await connector.ingest(valid_request, valid_pdf_content)

        expected_sha256 = hashlib.sha256(valid_pdf_content).hexdigest()
        assert response.checksum == expected_sha256

    @pytest.mark.asyncio
    async def test_ingest_invalid_mime_type(self, connector, valid_pdf_content):
        """Test rejection of invalid MIME type."""
        request = IngestionRequest(
            filename="malware.exe",
            content_type="application/x-msdownload",
            user_id="user-123",
        )

        response = await connector.ingest(request, b"MZ" + valid_pdf_content)

        assert response.status == DocumentStatus.QUARANTINED
        assert "not allowed" in response.message.lower()

    @pytest.mark.asyncio
    async def test_ingest_dangerous_content(self, connector):
        """Test rejection of document with dangerous content."""
        request = IngestionRequest(
            filename="document.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            user_id="user-123",
        )

        # DOCX with macro indicator
        dangerous_content = b"PK\x03\x04" + b"\x00" * 100 + b"AutoOpen" + b"\x00" * 100

        response = await connector.ingest(request, dangerous_content)

        assert response.status == DocumentStatus.QUARANTINED
        assert "dangerous" in response.message.lower()

    @pytest.mark.asyncio
    async def test_ingest_cleans_temp_file_on_success(self, connector, valid_request, valid_pdf_content, temp_storage):
        """Test that temporary files are cleaned up after successful ingestion."""
        response = await connector.ingest(valid_request, valid_pdf_content)

        # Temp file should be deleted
        temp_path = temp_storage / f"temp_{response.document_id}"
        assert not temp_path.exists()

    @pytest.mark.asyncio
    async def test_ingest_cleans_temp_file_on_failure(self, connector, temp_storage):
        """Test that temporary files are cleaned up after failed ingestion."""
        request = IngestionRequest(
            filename="malware.exe",
            content_type="application/x-msdownload",
            user_id="user-123",
        )

        response = await connector.ingest(request, b"MZ\x00\x00")

        # Temp file should be deleted even on failure
        temp_files = list(temp_storage.glob("temp_*"))
        assert len(temp_files) == 0


class TestChunkedUploadManager:
    """Test suite for ChunkedUploadManager."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_storage):
        """Create ChunkedUploadManager instance."""
        return ChunkedUploadManager(temp_storage)

    @pytest.mark.asyncio
    async def test_initialize_upload(self, manager):
        """Test upload initialization."""
        upload_id = await manager.initialize_upload(
            filename="large_document.pdf",
            total_chunks=5,
            total_size=5 * 1024 * 1024,
            user_id="user-123",
        )

        assert upload_id is not None
        assert upload_id in manager._active_uploads
        assert manager._active_uploads[upload_id]["total_chunks"] == 5

    @pytest.mark.asyncio
    async def test_upload_chunks_sequentially(self, manager):
        """Test uploading chunks in order."""
        upload_id = await manager.initialize_upload(
            filename="document.pdf",
            total_chunks=3,
            total_size=3000,
            user_id="user-123",
        )

        # Upload chunks
        for i in range(3):
            success, message = await manager.upload_chunk(
                upload_id=upload_id,
                chunk_index=i,
                data=f"chunk_{i}_data".encode(),
            )
            assert success is True

        # Should be complete
        assert await manager.is_complete(upload_id) is True

    @pytest.mark.asyncio
    async def test_upload_chunks_out_of_order(self, manager):
        """Test uploading chunks out of order."""
        upload_id = await manager.initialize_upload(
            filename="document.pdf",
            total_chunks=3,
            total_size=3000,
            user_id="user-123",
        )

        # Upload in reverse order
        for i in [2, 0, 1]:
            success, _ = await manager.upload_chunk(upload_id, i, f"chunk_{i}".encode())
            assert success is True

        assert await manager.is_complete(upload_id) is True

    @pytest.mark.asyncio
    async def test_upload_duplicate_chunk(self, manager):
        """Test that duplicate chunks are handled gracefully."""
        upload_id = await manager.initialize_upload(
            filename="document.pdf",
            total_chunks=2,
            total_size=2000,
            user_id="user-123",
        )

        # Upload same chunk twice
        await manager.upload_chunk(upload_id, 0, b"chunk_0")
        success, message = await manager.upload_chunk(upload_id, 0, b"chunk_0_duplicate")

        assert success is True
        assert "already received" in message.lower()

    @pytest.mark.asyncio
    async def test_upload_invalid_chunk_index(self, manager):
        """Test rejection of invalid chunk index."""
        upload_id = await manager.initialize_upload(
            filename="document.pdf",
            total_chunks=2,
            total_size=2000,
            user_id="user-123",
        )

        success, message = await manager.upload_chunk(upload_id, 10, b"invalid")

        assert success is False
        assert "invalid" in message.lower()

    @pytest.mark.asyncio
    async def test_upload_invalid_session(self, manager):
        """Test rejection of unknown upload session."""
        success, message = await manager.upload_chunk(
            "nonexistent-upload-id",
            0,
            b"data",
        )

        assert success is False
        assert "not found" in message.lower()

    @pytest.mark.asyncio
    async def test_assemble_file(self, manager):
        """Test file assembly from chunks."""
        upload_id = await manager.initialize_upload(
            filename="document.txt",
            total_chunks=3,
            total_size=15,
            user_id="user-123",
        )

        # Upload chunks
        await manager.upload_chunk(upload_id, 0, b"Hello")
        await manager.upload_chunk(upload_id, 1, b" Wor")
        await manager.upload_chunk(upload_id, 2, b"ld!")

        # Assemble
        assembled_path = await manager.assemble_file(upload_id)

        # Verify content
        assembled_content = assembled_path.read_bytes()
        assert assembled_content == b"Hello World!"

    @pytest.mark.asyncio
    async def test_assemble_incomplete_upload(self, manager):
        """Test that assembling incomplete upload raises error."""
        upload_id = await manager.initialize_upload(
            filename="document.pdf",
            total_chunks=3,
            total_size=3000,
            user_id="user-123",
        )

        await manager.upload_chunk(upload_id, 0, b"chunk_0")
        # Only 1 of 3 chunks uploaded

        with pytest.raises(ValueError, match="not complete"):
            await manager.assemble_file(upload_id)

    @pytest.mark.asyncio
    async def test_cleanup_upload(self, manager, temp_storage):
        """Test cleanup of upload session."""
        upload_id = await manager.initialize_upload(
            filename="document.pdf",
            total_chunks=2,
            total_size=2000,
            user_id="user-123",
        )

        await manager.upload_chunk(upload_id, 0, b"chunk_0")
        await manager.upload_chunk(upload_id, 1, b"chunk_1")

        # Verify chunks exist
        chunks_dir = temp_storage / "chunks" / upload_id
        assert chunks_dir.exists()

        # Cleanup
        await manager.cleanup_upload(upload_id)

        # Session and files should be gone
        assert upload_id not in manager._active_uploads
        assert not chunks_dir.exists()

    @pytest.mark.asyncio
    async def test_is_complete_false(self, manager):
        """Test is_complete returns False for incomplete upload."""
        upload_id = await manager.initialize_upload(
            filename="document.pdf",
            total_chunks=3,
            total_size=3000,
            user_id="user-123",
        )

        await manager.upload_chunk(upload_id, 0, b"chunk_0")

        assert await manager.is_complete(upload_id) is False

    @pytest.mark.asyncio
    async def test_is_complete_unknown_session(self, manager):
        """Test is_complete returns False for unknown session."""
        assert await manager.is_complete("nonexistent") is False


class TestIngestionService:
    """Integration tests for IngestionService."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_kafka_publisher(self):
        """Create mock Kafka publisher."""
        publisher = MagicMock(spec=KafkaDocumentPublisher)
        publisher.start = AsyncMock()
        publisher.stop = AsyncMock()
        publisher.publish_document = AsyncMock()
        return publisher

    @pytest.fixture
    def service(self, temp_storage, mock_kafka_publisher):
        """Create IngestionService instance with mocked Kafka."""
        service = IngestionService(
            storage_path=temp_storage,
            kafka_servers="localhost:9092",
            enable_malware_scan=False,
        )
        service.kafka_publisher = mock_kafka_publisher
        return service

    @pytest.fixture
    def valid_request(self):
        """Create valid ingestion request."""
        return IngestionRequest(
            filename="contract.pdf",
            content_type="application/pdf",
            source=IngestionSource.API_UPLOAD,
            user_id="user-123",
            client_id="CLIENT-001",
            matter_id="MATTER-001",
        )

    @pytest.fixture
    def valid_pdf_content(self):
        """Create valid PDF content."""
        return b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"

    @pytest.mark.asyncio
    async def test_ingest_document_success(self, service, valid_request, valid_pdf_content, mock_kafka_publisher):
        """Test successful document ingestion through service."""
        response = await service.ingest_document(valid_request, valid_pdf_content)

        assert response.status == DocumentStatus.VALIDATED
        assert response.document_id is not None

        # Should publish to Kafka
        mock_kafka_publisher.publish_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_document_quarantined_no_kafka(self, service, mock_kafka_publisher):
        """Test that quarantined documents don't publish to Kafka."""
        request = IngestionRequest(
            filename="malware.exe",
            content_type="application/x-msdownload",
            user_id="user-123",
        )

        response = await service.ingest_document(request, b"MZ\x00\x00")

        assert response.status == DocumentStatus.QUARANTINED

        # Should NOT publish to Kafka
        mock_kafka_publisher.publish_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_chunked_upload_workflow(self, service, mock_kafka_publisher):
        """Test complete chunked upload workflow."""
        # Simulate 3-chunk upload
        chunk_size = 100
        full_content = b"%PDF-1.4\n" + b"x" * 300 + b"\n%%EOF"

        chunks = [
            full_content[:chunk_size],
            full_content[chunk_size:2*chunk_size],
            full_content[2*chunk_size:],
        ]

        # First chunk - initialize
        request1 = IngestionRequest(
            filename="large.pdf",
            content_type="application/pdf",
            user_id="user-123",
            chunk_index=0,
            total_chunks=3,
        )

        response1 = await service.ingest_document(request1, chunks[0])
        assert response1.status == DocumentStatus.RECEIVED
        upload_id = response1.upload_id
        assert upload_id is not None

        # Second chunk
        request2 = IngestionRequest(
            filename="large.pdf",
            content_type="application/pdf",
            user_id="user-123",
            chunk_index=1,
            total_chunks=3,
            upload_id=upload_id,
        )

        response2 = await service.ingest_document(request2, chunks[1])
        assert response2.status == DocumentStatus.RECEIVED

        # Third chunk - completes upload
        request3 = IngestionRequest(
            filename="large.pdf",
            content_type="application/pdf",
            user_id="user-123",
            chunk_index=2,
            total_chunks=3,
            upload_id=upload_id,
        )

        response3 = await service.ingest_document(request3, chunks[2])
        assert response3.status == DocumentStatus.VALIDATED

    @pytest.mark.asyncio
    async def test_chunked_upload_missing_upload_id(self, service):
        """Test chunked upload without upload_id fails."""
        request = IngestionRequest(
            filename="document.pdf",
            user_id="user-123",
            chunk_index=1,  # Not first chunk
            total_chunks=3,
            # No upload_id
        )

        response = await service.ingest_document(request, b"chunk_data")

        assert response.status == DocumentStatus.FAILED
        assert "upload id" in response.message.lower()

    @pytest.mark.asyncio
    async def test_start_stop(self, service, mock_kafka_publisher):
        """Test service start and stop."""
        await service.start()
        mock_kafka_publisher.start.assert_called_once()

        await service.stop()
        mock_kafka_publisher.stop.assert_called_once()


class TestDocumentMetadata:
    """Test suite for DocumentMetadata."""

    def test_metadata_defaults(self):
        """Test default metadata values."""
        metadata = DocumentMetadata()

        assert metadata.document_id is not None
        assert metadata.original_filename == ""
        assert metadata.file_size == 0
        assert metadata.source == IngestionSource.MANUAL
        assert metadata.classification == SecurityClassification.INTERNAL
        assert metadata.tags == []
        assert metadata.custom_metadata == {}
        assert metadata.legal_hold is False

    def test_metadata_with_values(self):
        """Test metadata with custom values."""
        metadata = DocumentMetadata(
            original_filename="contract.pdf",
            file_size=1024,
            mime_type="application/pdf",
            checksum_sha256="abc123",
            source=IngestionSource.EMAIL,
            client_id="CLIENT-001",
            matter_id="MATTER-001",
            classification=SecurityClassification.ATTORNEY_CLIENT_PRIVILEGED,
            tags=["privileged", "urgent"],
            legal_hold=True,
        )

        assert metadata.original_filename == "contract.pdf"
        assert metadata.source == IngestionSource.EMAIL
        assert metadata.classification == SecurityClassification.ATTORNEY_CLIENT_PRIVILEGED
        assert "privileged" in metadata.tags
        assert metadata.legal_hold is True


class TestIngestionRequest:
    """Test suite for IngestionRequest."""

    def test_request_defaults(self):
        """Test default request values."""
        request = IngestionRequest(
            filename="document.pdf",
            user_id="user-123",
        )

        assert request.filename == "document.pdf"
        assert request.source == IngestionSource.API_UPLOAD
        assert request.classification == SecurityClassification.INTERNAL
        assert request.tags == []
        assert request.custom_metadata == {}
        assert request.chunk_index is None
        assert request.total_chunks is None

    def test_request_chunked_upload(self):
        """Test request for chunked upload."""
        request = IngestionRequest(
            filename="large.pdf",
            user_id="user-123",
            chunk_index=0,
            total_chunks=10,
            upload_id="upload-abc",
        )

        assert request.chunk_index == 0
        assert request.total_chunks == 10
        assert request.upload_id == "upload-abc"


class TestIngestionResponse:
    """Test suite for IngestionResponse."""

    def test_response_success(self):
        """Test successful response."""
        response = IngestionResponse(
            document_id="doc-123",
            status=DocumentStatus.VALIDATED,
            message="Document ingested successfully",
            checksum="abc123",
            bytes_received=1024,
        )

        assert response.document_id == "doc-123"
        assert response.status == DocumentStatus.VALIDATED

    def test_response_failure(self):
        """Test failure response."""
        response = IngestionResponse(
            document_id="doc-123",
            status=DocumentStatus.FAILED,
            message="Validation failed: Invalid MIME type",
        )

        assert response.status == DocumentStatus.FAILED
        assert "failed" in response.message.lower()


class TestKafkaDocumentPublisher:
    """Test suite for KafkaDocumentPublisher."""

    def test_publisher_config(self):
        """Test publisher configuration."""
        publisher = KafkaDocumentPublisher(
            bootstrap_servers="kafka1:9092,kafka2:9092",
            topic="custom-topic",
        )

        assert publisher.bootstrap_servers == "kafka1:9092,kafka2:9092"
        assert publisher.topic == "custom-topic"

    @pytest.mark.asyncio
    async def test_publish_without_start_raises(self):
        """Test that publishing without starting raises error."""
        publisher = KafkaDocumentPublisher()

        metadata = DocumentMetadata(
            document_id="doc-123",
            original_filename="test.pdf",
        )

        with pytest.raises(RuntimeError, match="not started"):
            await publisher.publish_document(metadata)
