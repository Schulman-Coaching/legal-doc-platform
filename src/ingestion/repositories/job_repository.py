"""
Ingestion Job Repository
========================
Repository for managing ingestion jobs and items.
"""

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.ingestion_job import (
    IngestionJob,
    IngestionJobItem,
    IngestionQueue,
    IngestionSource,
    IngestionStatus,
    ItemStatus,
)

logger = logging.getLogger(__name__)


class IngestionJobRepository:
    """Repository for ingestion job operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    # Job operations

    async def create_job(
        self,
        source: IngestionSource,
        created_by: str,
        name: Optional[str] = None,
        client_id: Optional[str] = None,
        matter_id: Optional[str] = None,
        priority: int = 0,
        options: Optional[dict] = None,
    ) -> IngestionJob:
        """Create a new ingestion job."""
        job = IngestionJob(
            name=name or f"{source.value} ingestion",
            source=source,
            created_by=created_by,
            client_id=client_id,
            matter_id=matter_id,
            priority=priority,
            options=options or {},
        )

        self.session.add(job)
        await self.session.flush()

        logger.info(f"Created ingestion job {job.id}")
        return job

    async def get_job(
        self,
        job_id: UUID,
        include_items: bool = False,
    ) -> Optional[IngestionJob]:
        """Get job by ID."""
        query = select(IngestionJob).where(IngestionJob.id == job_id)

        if include_items:
            query = query.options(selectinload(IngestionJob.items))

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def list_jobs(
        self,
        status: Optional[IngestionStatus] = None,
        source: Optional[IngestionSource] = None,
        created_by: Optional[str] = None,
        client_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[IngestionJob]:
        """List jobs with filters."""
        query = select(IngestionJob)

        if status:
            query = query.where(IngestionJob.status == status)
        if source:
            query = query.where(IngestionJob.source == source)
        if created_by:
            query = query.where(IngestionJob.created_by == created_by)
        if client_id:
            query = query.where(IngestionJob.client_id == client_id)
        if since:
            query = query.where(IngestionJob.created_at >= since)

        query = query.order_by(IngestionJob.created_at.desc())
        query = query.offset(offset).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def update_job_status(
        self,
        job_id: UUID,
        status: IngestionStatus,
        error: Optional[str] = None,
    ) -> bool:
        """Update job status."""
        values = {
            "status": status,
            "last_error": error,
        }

        if status == IngestionStatus.RUNNING:
            values["started_at"] = datetime.utcnow()
        elif status in [IngestionStatus.COMPLETED, IngestionStatus.FAILED, IngestionStatus.CANCELLED]:
            values["completed_at"] = datetime.utcnow()

        stmt = (
            update(IngestionJob)
            .where(IngestionJob.id == job_id)
            .values(**values)
        )

        result = await self.session.execute(stmt)
        return result.rowcount > 0

    async def update_job_progress(
        self,
        job_id: UUID,
        completed: int = 0,
        failed: int = 0,
        skipped: int = 0,
        bytes_processed: int = 0,
    ) -> bool:
        """Increment job progress counters."""
        stmt = (
            update(IngestionJob)
            .where(IngestionJob.id == job_id)
            .values(
                completed_items=IngestionJob.completed_items + completed,
                failed_items=IngestionJob.failed_items + failed,
                skipped_items=IngestionJob.skipped_items + skipped,
                processed_bytes=IngestionJob.processed_bytes + bytes_processed,
            )
        )

        result = await self.session.execute(stmt)
        return result.rowcount > 0

    # Item operations

    async def add_item(
        self,
        job_id: UUID,
        original_filename: str,
        file_size: int = 0,
        mime_type: Optional[str] = None,
        source_path: Optional[str] = None,
        checksum_sha256: Optional[str] = None,
        source_metadata: Optional[dict] = None,
    ) -> IngestionJobItem:
        """Add item to job."""
        item = IngestionJobItem(
            job_id=job_id,
            original_filename=original_filename,
            file_size=file_size,
            mime_type=mime_type,
            source_path=source_path,
            checksum_sha256=checksum_sha256,
            source_metadata=source_metadata or {},
        )

        self.session.add(item)
        await self.session.flush()

        # Update job total
        stmt = (
            update(IngestionJob)
            .where(IngestionJob.id == job_id)
            .values(
                total_items=IngestionJob.total_items + 1,
                total_bytes=IngestionJob.total_bytes + file_size,
            )
        )
        await self.session.execute(stmt)

        return item

    async def add_items_batch(
        self,
        job_id: UUID,
        items: list[dict],
    ) -> list[IngestionJobItem]:
        """Add multiple items to job efficiently."""
        created_items = []
        total_size = 0

        for item_data in items:
            item = IngestionJobItem(
                job_id=job_id,
                original_filename=item_data["filename"],
                file_size=item_data.get("size", 0),
                mime_type=item_data.get("mime_type"),
                source_path=item_data.get("source_path"),
                checksum_sha256=item_data.get("checksum"),
                source_metadata=item_data.get("metadata", {}),
            )
            self.session.add(item)
            created_items.append(item)
            total_size += item.file_size

        await self.session.flush()

        # Update job totals
        stmt = (
            update(IngestionJob)
            .where(IngestionJob.id == job_id)
            .values(
                total_items=IngestionJob.total_items + len(items),
                total_bytes=IngestionJob.total_bytes + total_size,
            )
        )
        await self.session.execute(stmt)

        return created_items

    async def get_item(self, item_id: UUID) -> Optional[IngestionJobItem]:
        """Get item by ID."""
        query = select(IngestionJobItem).where(IngestionJobItem.id == item_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_pending_items(
        self,
        job_id: UUID,
        limit: int = 100,
    ) -> list[IngestionJobItem]:
        """Get pending items for a job."""
        query = (
            select(IngestionJobItem)
            .where(
                and_(
                    IngestionJobItem.job_id == job_id,
                    IngestionJobItem.status == ItemStatus.PENDING,
                )
            )
            .limit(limit)
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def update_item_status(
        self,
        item_id: UUID,
        status: ItemStatus,
        document_id: Optional[UUID] = None,
        error_message: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
    ) -> bool:
        """Update item status."""
        values = {
            "status": status,
            "error_message": error_message,
        }

        if status == ItemStatus.PROCESSING:
            values["started_at"] = datetime.utcnow()
        elif status in [ItemStatus.COMPLETED, ItemStatus.FAILED, ItemStatus.SKIPPED]:
            values["completed_at"] = datetime.utcnow()

        if document_id:
            values["document_id"] = document_id
        if processing_time_ms:
            values["processing_time_ms"] = processing_time_ms

        stmt = (
            update(IngestionJobItem)
            .where(IngestionJobItem.id == item_id)
            .values(**values)
        )

        result = await self.session.execute(stmt)
        return result.rowcount > 0

    async def increment_retry(self, item_id: UUID) -> int:
        """Increment retry count and return new value."""
        stmt = (
            update(IngestionJobItem)
            .where(IngestionJobItem.id == item_id)
            .values(retries=IngestionJobItem.retries + 1)
            .returning(IngestionJobItem.retries)
        )

        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def find_by_checksum(
        self,
        job_id: UUID,
        checksum: str,
    ) -> Optional[IngestionJobItem]:
        """Find item by checksum within job."""
        query = select(IngestionJobItem).where(
            and_(
                IngestionJobItem.job_id == job_id,
                IngestionJobItem.checksum_sha256 == checksum,
            )
        )

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    # Queue operations

    async def enqueue_job(
        self,
        job_id: UUID,
        priority: int = 0,
        not_before: Optional[datetime] = None,
        rate_limit_key: Optional[str] = None,
    ) -> IngestionQueue:
        """Add job to processing queue."""
        queue_item = IngestionQueue(
            job_id=job_id,
            priority=-priority,  # Negative for max-heap
            not_before=not_before,
            rate_limit_key=rate_limit_key,
        )

        self.session.add(queue_item)
        await self.session.flush()

        return queue_item

    async def dequeue_job(
        self,
        worker_id: str,
        rate_limit_keys: Optional[list[str]] = None,
    ) -> Optional[IngestionQueue]:
        """Get next job from queue and lock it."""
        query = (
            select(IngestionQueue)
            .where(
                and_(
                    IngestionQueue.is_processing == False,
                    IngestionQueue.not_before <= datetime.utcnow(),
                )
            )
            .order_by(IngestionQueue.priority, IngestionQueue.scheduled_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )

        if rate_limit_keys:
            query = query.where(
                IngestionQueue.rate_limit_key.not_in(rate_limit_keys)
            )

        result = await self.session.execute(query)
        queue_item = result.scalar_one_or_none()

        if queue_item:
            queue_item.is_processing = True
            queue_item.locked_by = worker_id
            queue_item.locked_at = datetime.utcnow()
            await self.session.flush()

        return queue_item

    async def release_queue_item(
        self,
        queue_id: UUID,
        reschedule: bool = False,
    ) -> bool:
        """Release queue item lock."""
        if reschedule:
            stmt = (
                update(IngestionQueue)
                .where(IngestionQueue.id == queue_id)
                .values(
                    is_processing=False,
                    locked_by=None,
                    locked_at=None,
                )
            )
        else:
            from sqlalchemy import delete
            stmt = delete(IngestionQueue).where(IngestionQueue.id == queue_id)

        result = await self.session.execute(stmt)
        return result.rowcount > 0

    # Statistics

    async def get_job_statistics(
        self,
        since: Optional[datetime] = None,
    ) -> dict:
        """Get job statistics."""
        base_filter = True
        if since:
            base_filter = IngestionJob.created_at >= since

        # Count by status
        status_query = (
            select(IngestionJob.status, func.count(IngestionJob.id))
            .where(base_filter)
            .group_by(IngestionJob.status)
        )
        status_result = await self.session.execute(status_query)
        by_status = {row[0].value: row[1] for row in status_result.all()}

        # Count by source
        source_query = (
            select(IngestionJob.source, func.count(IngestionJob.id))
            .where(base_filter)
            .group_by(IngestionJob.source)
        )
        source_result = await self.session.execute(source_query)
        by_source = {row[0].value: row[1] for row in source_result.all()}

        # Total items processed
        items_query = select(
            func.sum(IngestionJob.completed_items),
            func.sum(IngestionJob.failed_items),
            func.sum(IngestionJob.processed_bytes),
        ).where(base_filter)
        items_result = await self.session.execute(items_query)
        items_row = items_result.one()

        # Queue size
        queue_query = select(func.count(IngestionQueue.id)).where(
            IngestionQueue.is_processing == False
        )
        queue_result = await self.session.execute(queue_query)
        queue_size = queue_result.scalar_one()

        return {
            "by_status": by_status,
            "by_source": by_source,
            "total_completed_items": items_row[0] or 0,
            "total_failed_items": items_row[1] or 0,
            "total_bytes_processed": items_row[2] or 0,
            "queue_size": queue_size,
        }
