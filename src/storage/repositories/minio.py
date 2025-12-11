"""
MinIO/S3 Repository
===================
Object storage repository using MinIO for document file storage.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import partial
from typing import Any, AsyncIterator, Optional

from minio import Minio
from minio.error import S3Error
from minio.commonconfig import CopySource, Filter
from minio.lifecycleconfig import LifecycleConfig, Rule, Transition

from ..config import MinIOConfig
from ..models import DocumentStorageClass

logger = logging.getLogger(__name__)


class MinIORepository:
    """
    MinIO repository for document file storage.

    Supports multiple storage classes, versioning, and lifecycle management.
    """

    # Storage class prefixes
    STORAGE_PREFIXES = {
        DocumentStorageClass.HOT: "hot",
        DocumentStorageClass.WARM: "warm",
        DocumentStorageClass.COLD: "cold",
        DocumentStorageClass.GLACIER: "glacier",
    }

    def __init__(self, config: MinIOConfig):
        self.config = config
        self._client: Optional[Minio] = None
        self._executor = ThreadPoolExecutor(max_workers=10)

    async def connect(self) -> None:
        """Initialize MinIO client."""
        try:
            self._client = Minio(
                self.config.endpoint,
                access_key=self.config.access_key,
                secret_key=self.config.secret_key,
                secure=self.config.secure,
                region=self.config.region,
            )

            # Test connection by listing buckets
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self._client.list_buckets)

            logger.info(f"Connected to MinIO at {self.config.endpoint}")

        except Exception as e:
            logger.error(f"Failed to connect to MinIO: {e}")
            raise

    async def disconnect(self) -> None:
        """Close MinIO connection."""
        if self._executor:
            self._executor.shutdown(wait=False)
        self._client = None
        logger.info("Disconnected from MinIO")

    async def _run_sync(self, func, *args, **kwargs):
        """Run synchronous MinIO operations in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(func, *args, **kwargs)
        )

    async def create_bucket(self, bucket_name: Optional[str] = None) -> bool:
        """Create storage bucket if it doesn't exist."""
        if not self._client:
            raise RuntimeError("MinIO client not connected")

        bucket = bucket_name or self.config.bucket_name

        try:
            exists = await self._run_sync(self._client.bucket_exists, bucket)
            if not exists:
                await self._run_sync(
                    self._client.make_bucket,
                    bucket,
                    location=self.config.region
                )
                logger.info(f"Created bucket {bucket}")

                # Set up lifecycle rules for storage tiering
                await self._setup_lifecycle_rules(bucket)

            return True

        except S3Error as e:
            logger.error(f"Failed to create bucket {bucket}: {e}")
            return False

    async def _setup_lifecycle_rules(self, bucket: str) -> None:
        """Set up lifecycle rules for automatic tiering."""
        try:
            # Move to warm after 30 days, cold after 90 days
            config = LifecycleConfig(
                [
                    Rule(
                        "warm-transition",
                        status="Enabled",
                        rule_filter=Filter(prefix="hot/"),
                        transition=Transition(days=30, storage_class="WARM"),
                    ),
                    Rule(
                        "cold-transition",
                        status="Enabled",
                        rule_filter=Filter(prefix="warm/"),
                        transition=Transition(days=60, storage_class="COLD"),
                    ),
                ]
            )
            await self._run_sync(
                self._client.set_bucket_lifecycle,
                bucket,
                config
            )
            logger.debug(f"Set lifecycle rules for bucket {bucket}")
        except Exception as e:
            logger.warning(f"Could not set lifecycle rules: {e}")

    # File Operations

    async def upload_file(
        self,
        document_id: str,
        data: bytes,
        content_type: str,
        metadata: Optional[dict[str, str]] = None,
        storage_class: DocumentStorageClass = DocumentStorageClass.HOT,
        bucket_name: Optional[str] = None,
    ) -> str:
        """
        Upload file to object storage.

        Args:
            document_id: Unique document identifier
            data: File bytes
            content_type: MIME type
            metadata: Optional custom metadata
            storage_class: Storage tier
            bucket_name: Optional bucket override

        Returns:
            Object path (key)
        """
        if not self._client:
            raise RuntimeError("MinIO client not connected")

        bucket = bucket_name or self.config.bucket_name
        prefix = self.STORAGE_PREFIXES.get(storage_class, "hot")
        object_name = f"{prefix}/{document_id}"

        # Add checksum to metadata
        checksum = hashlib.sha256(data).hexdigest()
        meta = metadata or {}
        meta["x-amz-meta-checksum-sha256"] = checksum
        meta["x-amz-meta-storage-class"] = storage_class.value

        try:
            data_stream = io.BytesIO(data)

            await self._run_sync(
                self._client.put_object,
                bucket,
                object_name,
                data_stream,
                length=len(data),
                content_type=content_type,
                metadata=meta,
                part_size=self.config.part_size,
            )

            logger.info(f"Uploaded {object_name} ({len(data)} bytes) to {bucket}")
            return object_name

        except S3Error as e:
            logger.error(f"Failed to upload {object_name}: {e}")
            raise

    async def upload_file_stream(
        self,
        document_id: str,
        stream: io.IOBase,
        length: int,
        content_type: str,
        metadata: Optional[dict[str, str]] = None,
        storage_class: DocumentStorageClass = DocumentStorageClass.HOT,
        bucket_name: Optional[str] = None,
    ) -> str:
        """Upload file from stream (for large files)."""
        if not self._client:
            raise RuntimeError("MinIO client not connected")

        bucket = bucket_name or self.config.bucket_name
        prefix = self.STORAGE_PREFIXES.get(storage_class, "hot")
        object_name = f"{prefix}/{document_id}"

        meta = metadata or {}
        meta["x-amz-meta-storage-class"] = storage_class.value

        try:
            await self._run_sync(
                self._client.put_object,
                bucket,
                object_name,
                stream,
                length=length,
                content_type=content_type,
                metadata=meta,
                part_size=self.config.part_size,
            )

            logger.info(f"Uploaded stream {object_name} ({length} bytes)")
            return object_name

        except S3Error as e:
            logger.error(f"Failed to upload stream {object_name}: {e}")
            raise

    async def download_file(
        self,
        object_name: str,
        bucket_name: Optional[str] = None,
    ) -> bytes:
        """Download file from object storage."""
        if not self._client:
            raise RuntimeError("MinIO client not connected")

        bucket = bucket_name or self.config.bucket_name

        try:
            response = await self._run_sync(
                self._client.get_object,
                bucket,
                object_name
            )

            try:
                data = response.read()
            finally:
                response.close()
                response.release_conn()

            logger.debug(f"Downloaded {object_name} ({len(data)} bytes)")
            return data

        except S3Error as e:
            logger.error(f"Failed to download {object_name}: {e}")
            raise

    async def download_file_stream(
        self,
        object_name: str,
        bucket_name: Optional[str] = None,
    ) -> AsyncIterator[bytes]:
        """Download file as async stream."""
        if not self._client:
            raise RuntimeError("MinIO client not connected")

        bucket = bucket_name or self.config.bucket_name

        try:
            response = await self._run_sync(
                self._client.get_object,
                bucket,
                object_name
            )

            try:
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
            finally:
                response.close()
                response.release_conn()

        except S3Error as e:
            logger.error(f"Failed to stream {object_name}: {e}")
            raise

    async def delete_file(
        self,
        object_name: str,
        bucket_name: Optional[str] = None,
    ) -> bool:
        """Delete file from object storage."""
        if not self._client:
            raise RuntimeError("MinIO client not connected")

        bucket = bucket_name or self.config.bucket_name

        try:
            await self._run_sync(
                self._client.remove_object,
                bucket,
                object_name
            )
            logger.info(f"Deleted {object_name} from {bucket}")
            return True

        except S3Error as e:
            logger.error(f"Failed to delete {object_name}: {e}")
            return False

    async def delete_files(
        self,
        object_names: list[str],
        bucket_name: Optional[str] = None,
    ) -> dict[str, int]:
        """Bulk delete multiple files."""
        if not self._client:
            raise RuntimeError("MinIO client not connected")

        bucket = bucket_name or self.config.bucket_name

        from minio.deleteobjects import DeleteObject

        delete_objects = [DeleteObject(name) for name in object_names]

        try:
            errors = list(await self._run_sync(
                self._client.remove_objects,
                bucket,
                delete_objects
            ))

            success = len(object_names) - len(errors)
            logger.info(f"Bulk deleted {success}/{len(object_names)} objects")

            return {"success": success, "failed": len(errors)}

        except Exception as e:
            logger.error(f"Bulk delete failed: {e}")
            return {"success": 0, "failed": len(object_names)}

    async def file_exists(
        self,
        object_name: str,
        bucket_name: Optional[str] = None,
    ) -> bool:
        """Check if file exists."""
        if not self._client:
            raise RuntimeError("MinIO client not connected")

        bucket = bucket_name or self.config.bucket_name

        try:
            await self._run_sync(
                self._client.stat_object,
                bucket,
                object_name
            )
            return True
        except S3Error:
            return False

    async def get_file_info(
        self,
        object_name: str,
        bucket_name: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Get file metadata and info."""
        if not self._client:
            raise RuntimeError("MinIO client not connected")

        bucket = bucket_name or self.config.bucket_name

        try:
            stat = await self._run_sync(
                self._client.stat_object,
                bucket,
                object_name
            )

            return {
                "object_name": stat.object_name,
                "size": stat.size,
                "etag": stat.etag,
                "content_type": stat.content_type,
                "last_modified": stat.last_modified,
                "metadata": stat.metadata,
                "version_id": stat.version_id,
            }

        except S3Error:
            return None

    # Presigned URLs

    async def get_presigned_url(
        self,
        object_name: str,
        expires: timedelta = timedelta(hours=1),
        bucket_name: Optional[str] = None,
    ) -> str:
        """Generate presigned URL for download."""
        if not self._client:
            raise RuntimeError("MinIO client not connected")

        bucket = bucket_name or self.config.bucket_name

        url = await self._run_sync(
            self._client.presigned_get_object,
            bucket,
            object_name,
            expires=expires
        )

        return url

    async def get_presigned_upload_url(
        self,
        object_name: str,
        expires: timedelta = timedelta(hours=1),
        bucket_name: Optional[str] = None,
    ) -> str:
        """Generate presigned URL for upload."""
        if not self._client:
            raise RuntimeError("MinIO client not connected")

        bucket = bucket_name or self.config.bucket_name

        url = await self._run_sync(
            self._client.presigned_put_object,
            bucket,
            object_name,
            expires=expires
        )

        return url

    # Storage Class Management

    async def move_storage_class(
        self,
        object_name: str,
        new_class: DocumentStorageClass,
        bucket_name: Optional[str] = None,
    ) -> str:
        """Move file to different storage class."""
        if not self._client:
            raise RuntimeError("MinIO client not connected")

        bucket = bucket_name or self.config.bucket_name

        # Extract document ID from path
        parts = object_name.split("/")
        document_id = parts[-1] if len(parts) > 1 else object_name

        # New object name
        new_prefix = self.STORAGE_PREFIXES.get(new_class, "hot")
        new_object_name = f"{new_prefix}/{document_id}"

        if object_name == new_object_name:
            return object_name

        try:
            # Copy to new location
            await self._run_sync(
                self._client.copy_object,
                bucket,
                new_object_name,
                CopySource(bucket, object_name)
            )

            # Update metadata
            stat = await self._run_sync(
                self._client.stat_object,
                bucket,
                new_object_name
            )

            # Delete old object
            await self._run_sync(
                self._client.remove_object,
                bucket,
                object_name
            )

            logger.info(f"Moved {object_name} to {new_object_name}")
            return new_object_name

        except S3Error as e:
            logger.error(f"Failed to move {object_name}: {e}")
            raise

    # Listing

    async def list_objects(
        self,
        prefix: Optional[str] = None,
        recursive: bool = True,
        bucket_name: Optional[str] = None,
        max_keys: int = 1000,
    ) -> list[dict[str, Any]]:
        """List objects in bucket."""
        if not self._client:
            raise RuntimeError("MinIO client not connected")

        bucket = bucket_name or self.config.bucket_name

        try:
            objects = await self._run_sync(
                lambda: list(self._client.list_objects(
                    bucket,
                    prefix=prefix,
                    recursive=recursive,
                ))
            )

            results = []
            for obj in objects[:max_keys]:
                results.append({
                    "object_name": obj.object_name,
                    "size": obj.size,
                    "etag": obj.etag,
                    "last_modified": obj.last_modified,
                    "is_dir": obj.is_dir,
                })

            return results

        except S3Error as e:
            logger.error(f"Failed to list objects: {e}")
            return []

    async def get_bucket_size(
        self,
        prefix: Optional[str] = None,
        bucket_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get total bucket/prefix size."""
        objects = await self.list_objects(
            prefix=prefix,
            bucket_name=bucket_name,
            max_keys=100000,
        )

        total_size = sum(obj["size"] or 0 for obj in objects if not obj["is_dir"])
        object_count = sum(1 for obj in objects if not obj["is_dir"])

        return {
            "total_size_bytes": total_size,
            "object_count": object_count,
        }

    # Versioning

    async def enable_versioning(self, bucket_name: Optional[str] = None) -> bool:
        """Enable bucket versioning."""
        if not self._client:
            raise RuntimeError("MinIO client not connected")

        bucket = bucket_name or self.config.bucket_name

        try:
            from minio.versioningconfig import VersioningConfig

            await self._run_sync(
                self._client.set_bucket_versioning,
                bucket,
                VersioningConfig("Enabled")
            )
            logger.info(f"Enabled versioning for bucket {bucket}")
            return True

        except Exception as e:
            logger.error(f"Failed to enable versioning: {e}")
            return False

    async def get_object_versions(
        self,
        object_name: str,
        bucket_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get all versions of an object."""
        if not self._client:
            raise RuntimeError("MinIO client not connected")

        bucket = bucket_name or self.config.bucket_name

        try:
            versions = await self._run_sync(
                lambda: list(self._client.list_objects(
                    bucket,
                    prefix=object_name,
                    include_version=True,
                ))
            )

            return [
                {
                    "version_id": v.version_id,
                    "size": v.size,
                    "last_modified": v.last_modified,
                    "is_latest": v.is_latest,
                }
                for v in versions
                if v.object_name == object_name
            ]

        except Exception as e:
            logger.error(f"Failed to get versions: {e}")
            return []

    # Health Check

    async def health_check(self) -> dict[str, Any]:
        """Check MinIO health."""
        if not self._client:
            return {"status": "disconnected"}

        try:
            buckets = await self._run_sync(self._client.list_buckets)
            bucket_exists = any(
                b.name == self.config.bucket_name for b in buckets
            )

            bucket_info = {}
            if bucket_exists:
                bucket_info = await self.get_bucket_size()

            return {
                "status": "healthy",
                "endpoint": self.config.endpoint,
                "bucket_exists": bucket_exists,
                "bucket_name": self.config.bucket_name,
                **bucket_info,
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
