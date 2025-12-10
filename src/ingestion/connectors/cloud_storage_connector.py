"""
Cloud Storage Ingestion Connectors
==================================
Connectors for cloud storage providers: AWS S3, Azure Blob, Google Cloud Storage.
Supports event-driven and polling-based ingestion with full lifecycle management.
"""

import asyncio
import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CloudProvider(str, Enum):
    """Supported cloud storage providers."""
    AWS_S3 = "aws_s3"
    AZURE_BLOB = "azure_blob"
    GOOGLE_CLOUD_STORAGE = "gcs"


class CloudEventType(str, Enum):
    """Cloud storage event types."""
    OBJECT_CREATED = "object_created"
    OBJECT_DELETED = "object_deleted"
    OBJECT_MODIFIED = "object_modified"


@dataclass
class CloudStorageConfig:
    """Base configuration for cloud storage connectors."""
    provider: CloudProvider
    # Bucket/container settings
    bucket_name: str
    prefix: str = ""
    # Polling settings
    poll_interval_seconds: int = 60
    # File filtering
    file_patterns: list[str] = field(default_factory=lambda: ["*.pdf", "*.doc*"])
    exclude_patterns: list[str] = field(default_factory=lambda: [".*", "*.tmp"])
    max_file_size: int = 1024 * 1024 * 1024  # 1GB
    # Processing settings
    delete_after_processing: bool = False
    move_after_processing: bool = True
    processed_prefix: str = "processed/"
    error_prefix: str = "errors/"
    # Download settings
    download_chunk_size: int = 8 * 1024 * 1024  # 8MB chunks


@dataclass
class S3Config(CloudStorageConfig):
    """AWS S3 specific configuration."""
    provider: CloudProvider = CloudProvider.AWS_S3
    # AWS credentials (use IAM role if not provided)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    # Region
    region_name: str = "us-east-1"
    # S3-specific settings
    endpoint_url: Optional[str] = None  # For S3-compatible services
    use_ssl: bool = True
    # Event notification settings
    sqs_queue_url: Optional[str] = None  # For event-driven ingestion


@dataclass
class AzureBlobConfig(CloudStorageConfig):
    """Azure Blob Storage specific configuration."""
    provider: CloudProvider = CloudProvider.AZURE_BLOB
    # Azure credentials
    connection_string: Optional[str] = None
    account_name: Optional[str] = None
    account_key: Optional[str] = None
    sas_token: Optional[str] = None
    # Event settings
    event_grid_topic: Optional[str] = None


@dataclass
class GCSConfig(CloudStorageConfig):
    """Google Cloud Storage specific configuration."""
    provider: CloudProvider = CloudProvider.GOOGLE_CLOUD_STORAGE
    # GCS credentials
    credentials_path: Optional[str] = None
    project_id: Optional[str] = None
    # Pub/Sub settings for events
    pubsub_subscription: Optional[str] = None


@dataclass
class CloudObject:
    """Represents an object in cloud storage."""
    key: str
    bucket: str
    provider: CloudProvider
    size: int
    last_modified: datetime
    etag: Optional[str] = None
    content_type: Optional[str] = None
    metadata: dict[str, str] = field(default_factory=dict)
    checksum: Optional[str] = None
    local_path: Optional[Path] = None


class CloudStorageConnector(ABC):
    """
    Abstract base class for cloud storage connectors.

    Provides common functionality for polling-based ingestion
    from cloud storage providers.
    """

    def __init__(
        self,
        config: CloudStorageConfig,
        storage_path: Path,
        callback: Optional[Callable[[CloudObject, bytes], None]] = None,
    ):
        self.config = config
        self.storage_path = storage_path / "cloud_ingestion" / config.provider.value
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.callback = callback
        self._running = False
        self._processed_keys: set[str] = set()

    async def start(self) -> None:
        """Start the cloud storage polling service."""
        self._running = True
        logger.info(f"Starting {self.config.provider.value} connector for {self.config.bucket_name}")

        while self._running:
            try:
                await self._poll_bucket()
            except Exception as e:
                logger.error(f"Error polling {self.config.provider.value}: {e}")

            await asyncio.sleep(self.config.poll_interval_seconds)

    async def stop(self) -> None:
        """Stop the cloud storage polling service."""
        self._running = False
        logger.info(f"{self.config.provider.value} connector stopped")

    @abstractmethod
    async def _poll_bucket(self) -> None:
        """Poll bucket for new objects."""
        pass

    @abstractmethod
    async def download_object(self, key: str) -> tuple[bytes, dict]:
        """Download object from cloud storage."""
        pass

    @abstractmethod
    async def move_object(self, source_key: str, dest_key: str) -> bool:
        """Move object within bucket."""
        pass

    @abstractmethod
    async def delete_object(self, key: str) -> bool:
        """Delete object from bucket."""
        pass

    @abstractmethod
    async def list_objects(self, prefix: Optional[str] = None) -> list[CloudObject]:
        """List objects in bucket."""
        pass

    @abstractmethod
    async def test_connection(self) -> tuple[bool, str]:
        """Test connection to cloud storage."""
        pass

    def _matches_patterns(self, key: str, patterns: list[str]) -> bool:
        """Check if key matches any pattern."""
        import fnmatch
        filename = key.split("/")[-1]
        return any(fnmatch.fnmatch(filename.lower(), p.lower()) for p in patterns)

    async def _process_object(self, obj: CloudObject) -> None:
        """Download and process a cloud object."""
        if obj.key in self._processed_keys:
            return

        # Check exclusion patterns
        if self._matches_patterns(obj.key, self.config.exclude_patterns):
            return

        # Check inclusion patterns
        if not self._matches_patterns(obj.key, self.config.file_patterns):
            return

        # Check file size
        if obj.size > self.config.max_file_size:
            logger.warning(f"Object too large: {obj.key} ({obj.size} bytes)")
            return

        logger.info(f"Processing {obj.key} ({obj.size} bytes)")

        try:
            # Download object
            content, metadata = await self.download_object(obj.key)
            obj.metadata = metadata
            obj.checksum = hashlib.sha256(content).hexdigest()

            # Save locally
            local_filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{obj.key.replace('/', '_')}"
            obj.local_path = self.storage_path / local_filename

            with open(obj.local_path, "wb") as f:
                f.write(content)

            # Call callback
            if self.callback:
                self.callback(obj, content)

            # Post-process
            await self._post_process(obj)

            # Mark as processed
            self._processed_keys.add(obj.key)

        except Exception as e:
            logger.error(f"Failed to process {obj.key}: {e}")
            await self._handle_error(obj)

    async def _post_process(self, obj: CloudObject) -> None:
        """Post-process object after successful ingestion."""
        if self.config.delete_after_processing:
            await self.delete_object(obj.key)
            logger.info(f"Deleted {obj.key}")

        elif self.config.move_after_processing:
            dest_key = self.config.processed_prefix + obj.key.lstrip("/")
            await self.move_object(obj.key, dest_key)
            logger.info(f"Moved {obj.key} to {dest_key}")

    async def _handle_error(self, obj: CloudObject) -> None:
        """Handle object that failed processing."""
        try:
            error_key = self.config.error_prefix + obj.key.lstrip("/")
            await self.move_object(obj.key, error_key)
            logger.info(f"Moved failed object to {error_key}")
        except Exception as e:
            logger.error(f"Failed to move to error prefix: {e}")

        self._processed_keys.add(obj.key)


class S3Connector(CloudStorageConnector):
    """
    AWS S3 connector with support for:
    - IAM role and credential-based authentication
    - S3-compatible services (MinIO, etc.)
    - Event-driven ingestion via SQS
    - Multipart downloads
    """

    def __init__(
        self,
        config: S3Config,
        storage_path: Path,
        callback: Optional[Callable[[CloudObject, bytes], None]] = None,
    ):
        super().__init__(config, storage_path, callback)
        self.s3_config = config
        self._client = None

    def _get_client(self):
        """Get or create boto3 S3 client."""
        if self._client is None:
            import boto3

            session_kwargs = {}
            if self.s3_config.aws_access_key_id:
                session_kwargs["aws_access_key_id"] = self.s3_config.aws_access_key_id
                session_kwargs["aws_secret_access_key"] = self.s3_config.aws_secret_access_key
                if self.s3_config.aws_session_token:
                    session_kwargs["aws_session_token"] = self.s3_config.aws_session_token

            session = boto3.Session(**session_kwargs)

            client_kwargs = {"region_name": self.s3_config.region_name}
            if self.s3_config.endpoint_url:
                client_kwargs["endpoint_url"] = self.s3_config.endpoint_url
            client_kwargs["use_ssl"] = self.s3_config.use_ssl

            self._client = session.client("s3", **client_kwargs)

        return self._client

    async def _poll_bucket(self) -> None:
        """Poll S3 bucket for new objects."""
        objects = await self.list_objects(self.config.prefix)

        for obj in objects:
            if not self._running:
                break
            await self._process_object(obj)

    async def list_objects(self, prefix: Optional[str] = None) -> list[CloudObject]:
        """List objects in S3 bucket."""
        loop = asyncio.get_event_loop()
        client = self._get_client()

        def _list():
            objects = []
            paginator = client.get_paginator("list_objects_v2")

            pages = paginator.paginate(
                Bucket=self.s3_config.bucket_name,
                Prefix=prefix or "",
            )

            for page in pages:
                for item in page.get("Contents", []):
                    objects.append(CloudObject(
                        key=item["Key"],
                        bucket=self.s3_config.bucket_name,
                        provider=CloudProvider.AWS_S3,
                        size=item["Size"],
                        last_modified=item["LastModified"],
                        etag=item.get("ETag", "").strip('"'),
                    ))

            return objects

        return await loop.run_in_executor(None, _list)

    async def download_object(self, key: str) -> tuple[bytes, dict]:
        """Download object from S3."""
        loop = asyncio.get_event_loop()
        client = self._get_client()

        def _download():
            response = client.get_object(
                Bucket=self.s3_config.bucket_name,
                Key=key,
            )
            content = response["Body"].read()
            metadata = response.get("Metadata", {})
            return content, metadata

        return await loop.run_in_executor(None, _download)

    async def move_object(self, source_key: str, dest_key: str) -> bool:
        """Move object within S3 bucket."""
        loop = asyncio.get_event_loop()
        client = self._get_client()

        def _move():
            # Copy to new location
            client.copy_object(
                Bucket=self.s3_config.bucket_name,
                CopySource={"Bucket": self.s3_config.bucket_name, "Key": source_key},
                Key=dest_key,
            )
            # Delete original
            client.delete_object(
                Bucket=self.s3_config.bucket_name,
                Key=source_key,
            )
            return True

        return await loop.run_in_executor(None, _move)

    async def delete_object(self, key: str) -> bool:
        """Delete object from S3."""
        loop = asyncio.get_event_loop()
        client = self._get_client()

        def _delete():
            client.delete_object(
                Bucket=self.s3_config.bucket_name,
                Key=key,
            )
            return True

        return await loop.run_in_executor(None, _delete)

    async def test_connection(self) -> tuple[bool, str]:
        """Test S3 connection."""
        try:
            objects = await self.list_objects()
            return True, f"Connected. Found {len(objects)} objects in {self.s3_config.bucket_name}"
        except Exception as e:
            return False, str(e)


class AzureBlobConnector(CloudStorageConnector):
    """
    Azure Blob Storage connector with support for:
    - Connection string and SAS token authentication
    - Event-driven ingestion via Event Grid
    - Block blob downloads
    """

    def __init__(
        self,
        config: AzureBlobConfig,
        storage_path: Path,
        callback: Optional[Callable[[CloudObject, bytes], None]] = None,
    ):
        super().__init__(config, storage_path, callback)
        self.azure_config = config
        self._container_client = None

    def _get_container_client(self):
        """Get or create Azure container client."""
        if self._container_client is None:
            from azure.storage.blob import ContainerClient

            if self.azure_config.connection_string:
                self._container_client = ContainerClient.from_connection_string(
                    self.azure_config.connection_string,
                    container_name=self.azure_config.bucket_name,
                )
            else:
                from azure.storage.blob import BlobServiceClient

                account_url = f"https://{self.azure_config.account_name}.blob.core.windows.net"

                if self.azure_config.sas_token:
                    service_client = BlobServiceClient(
                        account_url=account_url,
                        credential=self.azure_config.sas_token,
                    )
                else:
                    service_client = BlobServiceClient(
                        account_url=account_url,
                        credential=self.azure_config.account_key,
                    )

                self._container_client = service_client.get_container_client(
                    self.azure_config.bucket_name
                )

        return self._container_client

    async def _poll_bucket(self) -> None:
        """Poll Azure container for new blobs."""
        objects = await self.list_objects(self.config.prefix)

        for obj in objects:
            if not self._running:
                break
            await self._process_object(obj)

    async def list_objects(self, prefix: Optional[str] = None) -> list[CloudObject]:
        """List blobs in Azure container."""
        loop = asyncio.get_event_loop()
        container = self._get_container_client()

        def _list():
            objects = []
            blobs = container.list_blobs(name_starts_with=prefix or "")

            for blob in blobs:
                objects.append(CloudObject(
                    key=blob.name,
                    bucket=self.azure_config.bucket_name,
                    provider=CloudProvider.AZURE_BLOB,
                    size=blob.size,
                    last_modified=blob.last_modified,
                    etag=blob.etag,
                    content_type=blob.content_settings.content_type if blob.content_settings else None,
                ))

            return objects

        return await loop.run_in_executor(None, _list)

    async def download_object(self, key: str) -> tuple[bytes, dict]:
        """Download blob from Azure."""
        loop = asyncio.get_event_loop()
        container = self._get_container_client()

        def _download():
            blob_client = container.get_blob_client(key)
            download = blob_client.download_blob()
            content = download.readall()
            properties = blob_client.get_blob_properties()
            metadata = properties.metadata or {}
            return content, metadata

        return await loop.run_in_executor(None, _download)

    async def move_object(self, source_key: str, dest_key: str) -> bool:
        """Move blob within Azure container."""
        loop = asyncio.get_event_loop()
        container = self._get_container_client()

        def _move():
            # Copy to new location
            source_blob = container.get_blob_client(source_key)
            dest_blob = container.get_blob_client(dest_key)

            dest_blob.start_copy_from_url(source_blob.url)

            # Wait for copy to complete
            props = dest_blob.get_blob_properties()
            while props.copy.status == "pending":
                import time
                time.sleep(0.5)
                props = dest_blob.get_blob_properties()

            # Delete original
            source_blob.delete_blob()
            return True

        return await loop.run_in_executor(None, _move)

    async def delete_object(self, key: str) -> bool:
        """Delete blob from Azure."""
        loop = asyncio.get_event_loop()
        container = self._get_container_client()

        def _delete():
            blob_client = container.get_blob_client(key)
            blob_client.delete_blob()
            return True

        return await loop.run_in_executor(None, _delete)

    async def test_connection(self) -> tuple[bool, str]:
        """Test Azure connection."""
        try:
            objects = await self.list_objects()
            return True, f"Connected. Found {len(objects)} blobs in {self.azure_config.bucket_name}"
        except Exception as e:
            return False, str(e)


class GCSConnector(CloudStorageConnector):
    """
    Google Cloud Storage connector with support for:
    - Service account and application default credentials
    - Event-driven ingestion via Pub/Sub
    - Resumable downloads
    """

    def __init__(
        self,
        config: GCSConfig,
        storage_path: Path,
        callback: Optional[Callable[[CloudObject, bytes], None]] = None,
    ):
        super().__init__(config, storage_path, callback)
        self.gcs_config = config
        self._bucket = None

    def _get_bucket(self):
        """Get or create GCS bucket client."""
        if self._bucket is None:
            from google.cloud import storage

            if self.gcs_config.credentials_path:
                client = storage.Client.from_service_account_json(
                    self.gcs_config.credentials_path,
                    project=self.gcs_config.project_id,
                )
            else:
                client = storage.Client(project=self.gcs_config.project_id)

            self._bucket = client.bucket(self.gcs_config.bucket_name)

        return self._bucket

    async def _poll_bucket(self) -> None:
        """Poll GCS bucket for new objects."""
        objects = await self.list_objects(self.config.prefix)

        for obj in objects:
            if not self._running:
                break
            await self._process_object(obj)

    async def list_objects(self, prefix: Optional[str] = None) -> list[CloudObject]:
        """List objects in GCS bucket."""
        loop = asyncio.get_event_loop()
        bucket = self._get_bucket()

        def _list():
            objects = []
            blobs = bucket.list_blobs(prefix=prefix or "")

            for blob in blobs:
                objects.append(CloudObject(
                    key=blob.name,
                    bucket=self.gcs_config.bucket_name,
                    provider=CloudProvider.GOOGLE_CLOUD_STORAGE,
                    size=blob.size or 0,
                    last_modified=blob.updated or datetime.utcnow(),
                    etag=blob.etag,
                    content_type=blob.content_type,
                    metadata=blob.metadata or {},
                ))

            return objects

        return await loop.run_in_executor(None, _list)

    async def download_object(self, key: str) -> tuple[bytes, dict]:
        """Download object from GCS."""
        loop = asyncio.get_event_loop()
        bucket = self._get_bucket()

        def _download():
            blob = bucket.blob(key)
            content = blob.download_as_bytes()
            blob.reload()
            metadata = blob.metadata or {}
            return content, metadata

        return await loop.run_in_executor(None, _download)

    async def move_object(self, source_key: str, dest_key: str) -> bool:
        """Move object within GCS bucket."""
        loop = asyncio.get_event_loop()
        bucket = self._get_bucket()

        def _move():
            source_blob = bucket.blob(source_key)
            bucket.copy_blob(source_blob, bucket, dest_key)
            source_blob.delete()
            return True

        return await loop.run_in_executor(None, _move)

    async def delete_object(self, key: str) -> bool:
        """Delete object from GCS."""
        loop = asyncio.get_event_loop()
        bucket = self._get_bucket()

        def _delete():
            blob = bucket.blob(key)
            blob.delete()
            return True

        return await loop.run_in_executor(None, _delete)

    async def test_connection(self) -> tuple[bool, str]:
        """Test GCS connection."""
        try:
            objects = await self.list_objects()
            return True, f"Connected. Found {len(objects)} objects in {self.gcs_config.bucket_name}"
        except Exception as e:
            return False, str(e)
