"""
Legal Document Ingestion Service
================================
Handles secure document ingestion from multiple sources including
email, API uploads, SFTP, scanners, and cloud storage.
"""

import asyncio
import hashlib
import logging
import mimetypes
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import aiofiles
from aiokafka import AIOKafkaProducer
from cryptography.fernet import Fernet
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for improved maintainability
DEFAULT_CHUNK_TIMEOUT = 3600  # 1 hour
MAX_FILE_SIZE_DEFAULT = 100 * 1024 * 1024  # 100 MB
MAX_FILE_SIZE_PDF = 500 * 1024 * 1024  # 500 MB
MAX_FILE_SIZE_TIFF = 1024 * 1024 * 1024  # 1 GB
VALIDATION_SAMPLE_SIZE = 1024 * 1024  # 1 MB for dangerous content check


# Custom exceptions for better error handling
class IngestionError(Exception):
    """Base exception for ingestion errors."""
    pass


class ValidationError(IngestionError):
    """Raised when document validation fails."""
    pass


class EncryptionError(IngestionError):
    """Raised when encryption/decryption fails."""
    pass


class StorageError(IngestionError):
    """Raised when storage operations fail."""
    pass


class ConfigurationError(IngestionError):
    """Raised when configuration is invalid."""
    pass


class IngestionSource(str, Enum):
    """Supported document ingestion sources."""
    EMAIL = "email"
    API_UPLOAD = "api_upload"
    SFTP = "sftp"
    SCANNER = "scanner"
    CLOUD_STORAGE = "cloud_storage"
    MANUAL = "manual"


class DocumentStatus(str, Enum):
    """Document processing status."""
    RECEIVED = "received"
    VALIDATING = "validating"
    VALIDATED = "validated"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    QUARANTINED = "quarantined"


class SecurityClassification(str, Enum):
    """Document security classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    ATTORNEY_CLIENT_PRIVILEGED = "attorney_client_privileged"


@dataclass
class DocumentMetadata:
    """Metadata for ingested documents."""
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_filename: str = ""
    file_size: int = 0
    mime_type: str = ""
    checksum_sha256: str = ""
    checksum_md5: str = ""
    source: IngestionSource = IngestionSource.MANUAL
    source_identifier: str = ""
    received_at: datetime = field(default_factory=datetime.utcnow)
    client_id: Optional[str] = None
    matter_id: Optional[str] = None
    user_id: Optional[str] = None
    classification: SecurityClassification = SecurityClassification.INTERNAL
    tags: list[str] = field(default_factory=list)
    custom_metadata: dict[str, Any] = field(default_factory=dict)
    encryption_key_id: Optional[str] = None
    retention_policy: str = "standard"
    legal_hold: bool = False


class IngestionRequest(BaseModel):
    """Request model for document ingestion."""
    filename: str
    content_type: Optional[str] = None
    source: IngestionSource = IngestionSource.API_UPLOAD
    source_identifier: Optional[str] = None
    client_id: Optional[str] = None
    matter_id: Optional[str] = None
    user_id: str
    classification: SecurityClassification = SecurityClassification.INTERNAL
    tags: list[str] = Field(default_factory=list)
    custom_metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    upload_id: Optional[str] = None


class IngestionResponse(BaseModel):
    """Response model for document ingestion."""
    document_id: str
    status: DocumentStatus
    message: str
    checksum: Optional[str] = None
    upload_id: Optional[str] = None
    bytes_received: int = 0


class DocumentValidator:
    """Validates incoming documents for security and compliance."""

    # Allowed MIME types for legal documents
    ALLOWED_MIME_TYPES = {
        # Documents
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/rtf',
        'text/plain',
        'text/csv',
        'application/vnd.oasis.opendocument.text',
        # Spreadsheets
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.oasis.opendocument.spreadsheet',
        # Images (for scanned documents)
        'image/png',
        'image/jpeg',
        'image/tiff',
        'image/bmp',
        # Archives
        'application/zip',
        'application/x-7z-compressed',
        'application/gzip',
        # Email
        'message/rfc822',
        'application/vnd.ms-outlook',
    }

    # Maximum file sizes by type (in bytes) - using constants for maintainability
    MAX_FILE_SIZES = {
        'application/pdf': MAX_FILE_SIZE_PDF,
        'image/tiff': MAX_FILE_SIZE_TIFF,
        'default': MAX_FILE_SIZE_DEFAULT,
    }

    # File signatures (magic bytes) for validation
    FILE_SIGNATURES = {
        b'%PDF': 'application/pdf',
        b'PK\x03\x04': 'application/zip',  # Also DOCX, XLSX
        b'\x89PNG': 'image/png',
        b'\xff\xd8\xff': 'image/jpeg',
        b'II*\x00': 'image/tiff',  # Little-endian TIFF
        b'MM\x00*': 'image/tiff',  # Big-endian TIFF
    }

    def __init__(self, enable_malware_scan: bool = True):
        self.enable_malware_scan = enable_malware_scan

    async def validate(self, file_path: Path, metadata: DocumentMetadata) -> None:
        """
        Validate a document for ingestion.

        Raises:
            ValidationError: If validation fails
        """
        # Check file exists
        if not file_path.exists():
            raise ValidationError("File does not exist")

        # Check file size
        file_size = file_path.stat().st_size
        max_size = self.MAX_FILE_SIZES.get(
            metadata.mime_type,
            self.MAX_FILE_SIZES['default']
        )
        if file_size > max_size:
            raise ValidationError(f"File size {file_size} exceeds maximum {max_size}")

        # Validate MIME type
        if metadata.mime_type not in self.ALLOWED_MIME_TYPES:
            raise ValidationError(f"MIME type {metadata.mime_type} not allowed")

        # Validate file signature matches claimed type
        signature_valid = await self._validate_file_signature(file_path, metadata.mime_type)
        if not signature_valid:
            raise ValidationError("File signature does not match claimed type")

        # Check for potentially dangerous content
        dangerous = await self._check_dangerous_content(file_path)
        if dangerous:
            raise ValidationError("File contains potentially dangerous content")

        # Malware scan (if enabled)
        if self.enable_malware_scan:
            malware_found = await self._scan_for_malware(file_path)
            if malware_found:
                raise ValidationError("Malware detected in file")

    async def _validate_file_signature(self, file_path: Path, claimed_type: str) -> bool:
        """Validate file magic bytes match claimed MIME type."""
        async with aiofiles.open(file_path, 'rb') as f:
            header = await f.read(16)

        for signature, mime_type in self.FILE_SIGNATURES.items():
            if header.startswith(signature):
                # ZIP-based formats (DOCX, XLSX) need special handling
                if mime_type == 'application/zip' and claimed_type in {
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                }:
                    return True
                return mime_type == claimed_type

        # No signature match - allow if no signature defined for type
        return True

    async def _check_dangerous_content(self, file_path: Path) -> bool:
        """Check for potentially dangerous content like macros."""
        # Simple check for common dangerous patterns
        dangerous_patterns = [
            b'<script',
            b'javascript:',
            b'vbscript:',
            b'AutoOpen',  # Word macro
            b'Auto_Open',  # Excel macro
        ]

        async with aiofiles.open(file_path, 'rb') as f:
            content = await f.read(VALIDATION_SAMPLE_SIZE)  # Check first sample size

        for pattern in dangerous_patterns:
            if pattern.lower() in content.lower():
                return True

        return False

    async def _scan_for_malware(self, file_path: Path) -> bool:
        """
        Scan file for malware using ClamAV or similar.
        This is a placeholder - implement actual scanning in production.
        """
        # In production, integrate with ClamAV daemon or cloud-based scanning
        # Example: clamd.instream(file_content)
        logger.info(f"Malware scan placeholder for {file_path}")
        return False


class DocumentEncryption:
    """Handles document encryption at rest."""

    def __init__(self, master_key: Optional[bytes] = None):
        self.master_key = master_key or Fernet.generate_key()
        self._fernet = Fernet(self.master_key)

    async def encrypt_file(self, input_path: Path, output_path: Path) -> str:
        """
        Encrypt a file and return the key ID.
        In production, use envelope encryption with KMS.
        """
        async with aiofiles.open(input_path, 'rb') as f:
            plaintext = await f.read()

        ciphertext = self._fernet.encrypt(plaintext)

        async with aiofiles.open(output_path, 'wb') as f:
            await f.write(ciphertext)

        # Return key ID (in production, this would be from KMS)
        key_id = hashlib.sha256(self.master_key).hexdigest()[:16]
        return key_id

    async def decrypt_file(self, input_path: Path, output_path: Path) -> None:
        """Decrypt a file."""
        async with aiofiles.open(input_path, 'rb') as f:
            ciphertext = await f.read()

        plaintext = self._fernet.decrypt(ciphertext)

        async with aiofiles.open(output_path, 'wb') as f:
            await f.write(plaintext)


class BaseIngestionConnector(ABC):
    """Base class for document ingestion connectors."""

    def __init__(
        self,
        validator: DocumentValidator,
        encryption: DocumentEncryption,
        storage_path: Path,
    ):
        self.validator = validator
        self.encryption = encryption
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def ingest(self, request: IngestionRequest, data: bytes) -> IngestionResponse:
        """Ingest a document from the specific source."""
        pass

    async def _compute_checksums(self, data: bytes) -> tuple[str, str]:
        """Compute SHA-256 and MD5 checksums."""
        sha256 = hashlib.sha256(data).hexdigest()
        md5 = hashlib.md5(data).hexdigest()
        return sha256, md5

    async def _detect_mime_type(self, filename: str, data: bytes) -> str:
        """Detect MIME type from filename and content."""
        # First try by extension
        mime_type, _ = mimetypes.guess_type(filename)

        # Validate against content
        if not mime_type:
            # Check magic bytes
            for signature, detected_type in DocumentValidator.FILE_SIGNATURES.items():
                if data.startswith(signature):
                    mime_type = detected_type
                    break

        return mime_type or 'application/octet-stream'


class APIUploadConnector(BaseIngestionConnector):
    """Handles document uploads via REST API."""

    async def ingest(self, request: IngestionRequest, data: bytes) -> IngestionResponse:
        """Process an API upload request."""
        # Edge case: empty file
        if not data:
            return IngestionResponse(
                document_id="",
                status=DocumentStatus.FAILED,
                message="Empty file not allowed",
                bytes_received=0,
            )

        document_id = str(uuid.uuid4())

        try:
            # Compute checksums
            sha256, md5 = await self._compute_checksums(data)

            # Detect MIME type
            mime_type = await self._detect_mime_type(request.filename, data)
            if request.content_type:
                mime_type = request.content_type

            # Create metadata
            metadata = DocumentMetadata(
                document_id=document_id,
                original_filename=request.filename,
                file_size=len(data),
                mime_type=mime_type,
                checksum_sha256=sha256,
                checksum_md5=md5,
                source=IngestionSource.API_UPLOAD,
                source_identifier=request.source_identifier or "",
                client_id=request.client_id,
                matter_id=request.matter_id,
                user_id=request.user_id,
                classification=request.classification,
                tags=request.tags,
                custom_metadata=request.custom_metadata,
            )

            # Save temporarily for validation
            temp_path = self.storage_path / f"temp_{document_id}"
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(data)

            # Validate document
            await self.validator.validate(temp_path, metadata)

            # Encrypt and store
            encrypted_path = self.storage_path / f"{document_id}.enc"
            key_id = await self.encryption.encrypt_file(temp_path, encrypted_path)
            metadata.encryption_key_id = key_id

            # Clean up temp file
            temp_path.unlink(missing_ok=True)

            logger.info(f"Document {document_id} ingested successfully")

            return IngestionResponse(
                document_id=document_id,
                status=DocumentStatus.VALIDATED,
                message="Document ingested successfully",
                checksum=sha256,
                bytes_received=len(data),
            )

        except ValidationError as e:
            logger.warning(f"Validation failed for {document_id}: {e}")
            return IngestionResponse(
                document_id=document_id,
                status=DocumentStatus.QUARANTINED,
                message=f"Validation failed: {str(e)}",
                bytes_received=len(data),
            )
        except (StorageError, EncryptionError) as e:
            logger.error(f"Storage/encryption failed for {document_id}: {e}")
            return IngestionResponse(
                document_id=document_id,
                status=DocumentStatus.FAILED,
                message=f"Processing failed: {str(e)}",
                bytes_received=len(data),
            )
        except Exception as e:
            logger.error(f"Unexpected error for {document_id}: {e}")
            return IngestionResponse(
                document_id=document_id,
                status=DocumentStatus.FAILED,
                message=f"Unexpected error: {str(e)}",
                bytes_received=len(data),
            )


class ChunkedUploadManager:
    """Manages chunked uploads for large files."""

    def __init__(self, storage_path: Path, chunk_timeout_seconds: int = DEFAULT_CHUNK_TIMEOUT):
        self.storage_path = storage_path / "chunks"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.chunk_timeout = chunk_timeout_seconds
        self._active_uploads: dict[str, dict] = {}

    async def initialize_upload(
        self,
        filename: str,
        total_chunks: int,
        total_size: int,
        user_id: str,
    ) -> str:
        """Initialize a new chunked upload session."""
        upload_id = str(uuid.uuid4())

        self._active_uploads[upload_id] = {
            'filename': filename,
            'total_chunks': total_chunks,
            'total_size': total_size,
            'user_id': user_id,
            'received_chunks': set(),
            'created_at': datetime.utcnow(),
        }

        # Create upload directory
        upload_dir = self.storage_path / upload_id
        upload_dir.mkdir(exist_ok=True)

        return upload_id

    async def upload_chunk(
        self,
        upload_id: str,
        chunk_index: int,
        data: bytes,
    ) -> tuple[bool, str]:
        """
        Upload a single chunk.

        Returns:
            Tuple of (success, message)
        """
        if upload_id not in self._active_uploads:
            return False, "Upload session not found"

        upload = self._active_uploads[upload_id]

        if chunk_index >= upload['total_chunks']:
            return False, "Invalid chunk index"

        if chunk_index in upload['received_chunks']:
            return True, "Chunk already received"

        # Save chunk
        chunk_path = self.storage_path / upload_id / f"chunk_{chunk_index:08d}"
        async with aiofiles.open(chunk_path, 'wb') as f:
            await f.write(data)

        upload['received_chunks'].add(chunk_index)

        return True, "Chunk uploaded successfully"

    async def is_complete(self, upload_id: str) -> bool:
        """Check if all chunks have been received."""
        if upload_id not in self._active_uploads:
            return False

        upload = self._active_uploads[upload_id]
        return len(upload['received_chunks']) == upload['total_chunks']

    async def assemble_file(self, upload_id: str) -> Path:
        """Assemble all chunks into final file."""
        if upload_id not in self._active_uploads:
            raise ValueError("Upload session not found")

        upload = self._active_uploads[upload_id]

        if not await self.is_complete(upload_id):
            raise ValueError("Upload not complete")

        # Assemble file
        output_path = self.storage_path / f"assembled_{upload_id}"
        upload_dir = self.storage_path / upload_id

        async with aiofiles.open(output_path, 'wb') as outfile:
            for i in range(upload['total_chunks']):
                chunk_path = upload_dir / f"chunk_{i:08d}"
                async with aiofiles.open(chunk_path, 'rb') as chunk:
                    data = await chunk.read()
                    await outfile.write(data)

        return output_path

    async def cleanup_upload(self, upload_id: str) -> None:
        """Clean up upload session and temporary files."""
        if upload_id in self._active_uploads:
            del self._active_uploads[upload_id]

        upload_dir = self.storage_path / upload_id
        if upload_dir.exists():
            for file in upload_dir.iterdir():
                file.unlink()
            upload_dir.rmdir()


class KafkaDocumentPublisher:
    """Publishes ingested documents to Kafka for processing."""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "legal-documents-ingested",
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self._producer: Optional[AIOKafkaProducer] = None

    async def start(self) -> None:
        """Start the Kafka producer."""
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: v.encode('utf-8') if isinstance(v, str) else v,
            compression_type='lz4',
            acks='all',  # Wait for all replicas
        )
        await self._producer.start()
        logger.info("Kafka producer started")

    async def stop(self) -> None:
        """Stop the Kafka producer."""
        if self._producer:
            await self._producer.stop()
            logger.info("Kafka producer stopped")

    async def publish_document(self, metadata: DocumentMetadata) -> None:
        """Publish document metadata to Kafka."""
        if not self._producer:
            raise RuntimeError("Producer not started")

        import json

        message = {
            'document_id': metadata.document_id,
            'original_filename': metadata.original_filename,
            'file_size': metadata.file_size,
            'mime_type': metadata.mime_type,
            'checksum_sha256': metadata.checksum_sha256,
            'source': metadata.source.value,
            'received_at': metadata.received_at.isoformat(),
            'client_id': metadata.client_id,
            'matter_id': metadata.matter_id,
            'user_id': metadata.user_id,
            'classification': metadata.classification.value,
            'tags': metadata.tags,
            'encryption_key_id': metadata.encryption_key_id,
        }

        # Use document_id as key for partitioning
        await self._producer.send_and_wait(
            self.topic,
            key=metadata.document_id.encode('utf-8'),
            value=json.dumps(message),
        )

        logger.info(f"Published document {metadata.document_id} to Kafka")


class IngestionService:
    """Main ingestion service orchestrating all components."""

    def __init__(
        self,
        storage_path: Path,
        kafka_servers: str = "localhost:9092",
        enable_malware_scan: bool = True,
    ):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.validator = DocumentValidator(enable_malware_scan=enable_malware_scan)
        self.encryption = DocumentEncryption()
        self.chunk_manager = ChunkedUploadManager(storage_path)
        self.kafka_publisher = KafkaDocumentPublisher(kafka_servers)

        # Initialize connectors
        self.api_connector = APIUploadConnector(
            validator=self.validator,
            encryption=self.encryption,
            storage_path=storage_path / "documents",
        )

    async def start(self) -> None:
        """Start the ingestion service."""
        await self.kafka_publisher.start()
        logger.info("Ingestion service started")

    async def stop(self) -> None:
        """Stop the ingestion service."""
        await self.kafka_publisher.stop()
        logger.info("Ingestion service stopped")

    async def ingest_document(
        self,
        request: IngestionRequest,
        data: bytes,
    ) -> IngestionResponse:
        """
        Ingest a document through the API.

        Args:
            request: Ingestion request details
            data: Document binary data

        Returns:
            IngestionResponse with status
        """
        # Handle chunked uploads
        if request.chunk_index is not None and request.total_chunks:
            return await self._handle_chunked_upload(request, data)

        # Regular upload
        response = await self.api_connector.ingest(request, data)

        # If successful, publish to Kafka
        if response.status == DocumentStatus.VALIDATED:
            metadata = DocumentMetadata(
                document_id=response.document_id,
                original_filename=request.filename,
                file_size=len(data),
                checksum_sha256=response.checksum or "",
                source=request.source,
                client_id=request.client_id,
                matter_id=request.matter_id,
                user_id=request.user_id,
                classification=request.classification,
                tags=request.tags,
            )
            await self.kafka_publisher.publish_document(metadata)

        return response

    async def _handle_chunked_upload(
        self,
        request: IngestionRequest,
        data: bytes,
    ) -> IngestionResponse:
        """Handle chunked upload requests."""
        # Initialize upload if first chunk
        if request.chunk_index == 0 and not request.upload_id:
            upload_id = await self.chunk_manager.initialize_upload(
                filename=request.filename,
                total_chunks=request.total_chunks or 1,
                total_size=len(data) * (request.total_chunks or 1),  # Estimate
                user_id=request.user_id,
            )
            request.upload_id = upload_id

        if not request.upload_id:
            return IngestionResponse(
                document_id="",
                status=DocumentStatus.FAILED,
                message="Upload ID required for chunk upload",
            )

        # Upload chunk
        success, message = await self.chunk_manager.upload_chunk(
            upload_id=request.upload_id,
            chunk_index=request.chunk_index or 0,
            data=data,
        )

        if not success:
            return IngestionResponse(
                document_id="",
                status=DocumentStatus.FAILED,
                message=message,
                upload_id=request.upload_id,
            )

        # Check if complete
        if await self.chunk_manager.is_complete(request.upload_id):
            # Assemble and process
            assembled_path = await self.chunk_manager.assemble_file(request.upload_id)

            async with aiofiles.open(assembled_path, 'rb') as f:
                full_data = await f.read()

            # Process as regular upload
            response = await self.api_connector.ingest(request, full_data)

            # Cleanup
            assembled_path.unlink(missing_ok=True)
            await self.chunk_manager.cleanup_upload(request.upload_id)

            return response

        # More chunks expected
        return IngestionResponse(
            document_id="",
            status=DocumentStatus.RECEIVED,
            message=f"Chunk {request.chunk_index} received",
            upload_id=request.upload_id,
            bytes_received=len(data),
        )


# FastAPI Application
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import json

app = FastAPI(
    title="Legal Document Ingestion Service",
    description="Secure document ingestion for legal document processing platform",
    version="1.0.0",
)

security = HTTPBearer()
ingestion_service: Optional[IngestionService] = None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Validate JWT token and return user info."""
    # In production, validate JWT with Keycloak
    # This is a placeholder implementation
    return {"user_id": "demo-user", "roles": ["uploader"]}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global ingestion_service
    ingestion_service = IngestionService(
        storage_path=Path("/data/ingestion"),
        kafka_servers=os.getenv("KAFKA_SERVERS", "localhost:9092"),
    )
    await ingestion_service.start()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global ingestion_service
    if ingestion_service:
        await ingestion_service.stop()


@app.post("/api/v1/documents/upload", response_model=IngestionResponse)
async def upload_document(
    file: UploadFile = File(...),
    client_id: Optional[str] = Form(None),
    matter_id: Optional[str] = Form(None),
    classification: str = Form("internal"),
    tags: str = Form("[]"),
    custom_metadata: str = Form("{}"),
    user: dict = Depends(get_current_user),
):
    """
    Upload a single document.

    - **file**: Document file to upload
    - **client_id**: Optional client identifier
    - **matter_id**: Optional matter/case identifier
    - **classification**: Security classification level
    - **tags**: JSON array of tags
    - **custom_metadata**: JSON object of custom metadata
    """
    if not ingestion_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Read file content
    content = await file.read()

    # Parse tags and metadata
    try:
        tags_list = json.loads(tags) if tags else []
        metadata_dict = json.loads(custom_metadata) if custom_metadata else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in tags or metadata")

    # Create request
    request = IngestionRequest(
        filename=file.filename or "unknown",
        content_type=file.content_type,
        source=IngestionSource.API_UPLOAD,
        client_id=client_id,
        matter_id=matter_id,
        user_id=user["user_id"],
        classification=SecurityClassification(classification),
        tags=tags_list,
        custom_metadata=metadata_dict,
    )

    # Process upload
    response = await ingestion_service.ingest_document(request, content)

    if response.status in [DocumentStatus.FAILED, DocumentStatus.QUARANTINED]:
        raise HTTPException(status_code=400, detail=response.message)

    return response


@app.post("/api/v1/documents/upload/chunked/init")
async def init_chunked_upload(
    filename: str = Form(...),
    total_chunks: int = Form(...),
    total_size: int = Form(...),
    user: dict = Depends(get_current_user),
):
    """Initialize a chunked upload session."""
    if not ingestion_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    upload_id = await ingestion_service.chunk_manager.initialize_upload(
        filename=filename,
        total_chunks=total_chunks,
        total_size=total_size,
        user_id=user["user_id"],
    )

    return {"upload_id": upload_id, "message": "Upload session initialized"}


@app.post("/api/v1/documents/upload/chunked/{upload_id}")
async def upload_chunk(
    upload_id: str,
    chunk_index: int = Form(...),
    chunk: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """Upload a single chunk of a chunked upload."""
    if not ingestion_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    content = await chunk.read()

    success, message = await ingestion_service.chunk_manager.upload_chunk(
        upload_id=upload_id,
        chunk_index=chunk_index,
        data=content,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    # Check if complete
    is_complete = await ingestion_service.chunk_manager.is_complete(upload_id)

    return {
        "message": message,
        "is_complete": is_complete,
    }


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ingestion"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
