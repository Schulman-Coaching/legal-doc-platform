"""
Batch Processing Service
========================
Handles batch document ingestion with parallel processing,
progress tracking, and failure recovery.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class BatchStatus(str, Enum):
    """Status of batch processing."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"  # Some items failed


class ItemStatus(str, Enum):
    """Status of individual batch item."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class BatchItem:
    """Individual item in a batch."""
    item_id: str
    filename: str
    size: int
    status: ItemStatus = ItemStatus.PENDING
    document_id: Optional[str] = None
    error_message: Optional[str] = None
    retries: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchProgress:
    """Progress information for a batch."""
    total: int = 0
    pending: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    bytes_processed: int = 0
    bytes_total: int = 0
    estimated_remaining_seconds: Optional[float] = None
    current_rate: float = 0.0  # items per second


@dataclass
class BatchJob:
    """Batch processing job."""
    batch_id: str
    name: str
    status: BatchStatus = BatchStatus.PENDING
    items: list[BatchItem] = field(default_factory=list)
    progress: BatchProgress = field(default_factory=BatchProgress)
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: str = ""
    priority: int = 0  # Higher = more priority
    # Configuration
    max_concurrent: int = 5
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    stop_on_error: bool = False
    # Results
    total_processing_time: float = 0.0
    error_summary: dict[str, int] = field(default_factory=dict)


class BatchProcessor:
    """
    Batch document processing service.

    Features:
    - Parallel processing with configurable concurrency
    - Progress tracking and estimation
    - Automatic retries with backoff
    - Pause/resume/cancel support
    - Priority-based scheduling
    - Failure recovery
    """

    def __init__(
        self,
        process_callback: Callable[[bytes, str, dict], str],
        max_concurrent_batches: int = 3,
        storage_path: Optional[Path] = None,
    ):
        """
        Initialize batch processor.

        Args:
            process_callback: Async function(content, filename, metadata) -> document_id
            max_concurrent_batches: Max batches to process simultaneously
            storage_path: Path for temporary storage
        """
        self.process_callback = process_callback
        self.max_concurrent_batches = max_concurrent_batches
        self.storage_path = storage_path or Path("/tmp/batch_processing")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._batches: dict[str, BatchJob] = {}
        self._batch_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._active_batches: set[str] = set()
        self._running = False
        self._workers: list[asyncio.Task] = []

    async def start(self) -> None:
        """Start the batch processor workers."""
        self._running = True

        # Start worker tasks
        for i in range(self.max_concurrent_batches):
            worker = asyncio.create_task(self._batch_worker(i))
            self._workers.append(worker)

        logger.info(f"Batch processor started with {self.max_concurrent_batches} workers")

    async def stop(self) -> None:
        """Stop the batch processor."""
        self._running = False

        # Cancel workers
        for worker in self._workers:
            worker.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        logger.info("Batch processor stopped")

    async def create_batch(
        self,
        name: str,
        files: list[tuple[bytes, str, dict]],  # (content, filename, metadata)
        user_id: str,
        priority: int = 0,
        max_concurrent: int = 5,
        max_retries: int = 3,
        stop_on_error: bool = False,
    ) -> BatchJob:
        """
        Create a new batch job.

        Args:
            name: Batch name/description
            files: List of (content, filename, metadata) tuples
            user_id: User creating the batch
            priority: Processing priority (higher = sooner)
            max_concurrent: Max concurrent items within batch
            max_retries: Max retries per item
            stop_on_error: Stop batch on first error

        Returns:
            Created BatchJob
        """
        batch_id = str(uuid.uuid4())

        # Create batch items
        items = []
        total_size = 0

        for content, filename, metadata in files:
            item = BatchItem(
                item_id=str(uuid.uuid4()),
                filename=filename,
                size=len(content),
                metadata={
                    **metadata,
                    "_content_hash": hash(content),  # For dedup
                },
            )
            items.append(item)
            total_size += len(content)

            # Store content temporarily
            item_path = self.storage_path / batch_id / item.item_id
            item_path.parent.mkdir(parents=True, exist_ok=True)
            item_path.write_bytes(content)

        # Create batch
        batch = BatchJob(
            batch_id=batch_id,
            name=name,
            items=items,
            created_by=user_id,
            priority=priority,
            max_concurrent=max_concurrent,
            max_retries=max_retries,
            stop_on_error=stop_on_error,
            progress=BatchProgress(
                total=len(items),
                pending=len(items),
                bytes_total=total_size,
            ),
        )

        self._batches[batch_id] = batch

        # Queue for processing
        await self._batch_queue.put((-priority, batch_id))  # Negative for max-heap

        logger.info(f"Created batch {batch_id} with {len(items)} items")
        return batch

    async def get_batch(self, batch_id: str) -> Optional[BatchJob]:
        """Get batch job by ID."""
        return self._batches.get(batch_id)

    async def get_batch_progress(self, batch_id: str) -> Optional[BatchProgress]:
        """Get batch progress."""
        batch = self._batches.get(batch_id)
        if batch:
            return batch.progress
        return None

    async def pause_batch(self, batch_id: str) -> bool:
        """Pause a running batch."""
        batch = self._batches.get(batch_id)
        if batch and batch.status == BatchStatus.RUNNING:
            batch.status = BatchStatus.PAUSED
            logger.info(f"Paused batch {batch_id}")
            return True
        return False

    async def resume_batch(self, batch_id: str) -> bool:
        """Resume a paused batch."""
        batch = self._batches.get(batch_id)
        if batch and batch.status == BatchStatus.PAUSED:
            batch.status = BatchStatus.RUNNING
            await self._batch_queue.put((-batch.priority, batch_id))
            logger.info(f"Resumed batch {batch_id}")
            return True
        return False

    async def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a batch."""
        batch = self._batches.get(batch_id)
        if batch and batch.status in [BatchStatus.PENDING, BatchStatus.RUNNING, BatchStatus.PAUSED]:
            batch.status = BatchStatus.CANCELLED
            batch.completed_at = datetime.utcnow()

            # Mark pending items as skipped
            for item in batch.items:
                if item.status == ItemStatus.PENDING:
                    item.status = ItemStatus.SKIPPED

            logger.info(f"Cancelled batch {batch_id}")
            return True
        return False

    async def retry_failed(self, batch_id: str) -> bool:
        """Retry failed items in a batch."""
        batch = self._batches.get(batch_id)
        if not batch:
            return False

        # Reset failed items
        retry_count = 0
        for item in batch.items:
            if item.status == ItemStatus.FAILED:
                item.status = ItemStatus.PENDING
                item.retries = 0
                item.error_message = None
                retry_count += 1

        if retry_count > 0:
            batch.status = BatchStatus.PENDING
            batch.progress.pending += retry_count
            batch.progress.failed -= retry_count
            await self._batch_queue.put((-batch.priority, batch_id))
            logger.info(f"Queued {retry_count} items for retry in batch {batch_id}")
            return True

        return False

    async def list_batches(
        self,
        status: Optional[BatchStatus] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[BatchJob]:
        """List batches with optional filtering."""
        batches = list(self._batches.values())

        if status:
            batches = [b for b in batches if b.status == status]

        if user_id:
            batches = [b for b in batches if b.created_by == user_id]

        # Sort by created_at descending
        batches.sort(key=lambda b: b.created_at, reverse=True)

        return batches[:limit]

    async def _batch_worker(self, worker_id: int) -> None:
        """Worker that processes batches from queue."""
        logger.debug(f"Batch worker {worker_id} started")

        while self._running:
            try:
                # Get next batch (with timeout to allow shutdown)
                try:
                    _, batch_id = await asyncio.wait_for(
                        self._batch_queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                batch = self._batches.get(batch_id)
                if not batch:
                    continue

                # Skip if cancelled or already processing
                if batch.status in [BatchStatus.CANCELLED, BatchStatus.COMPLETED]:
                    continue

                if batch_id in self._active_batches:
                    continue

                self._active_batches.add(batch_id)

                try:
                    await self._process_batch(batch)
                finally:
                    self._active_batches.discard(batch_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.debug(f"Batch worker {worker_id} stopped")

    async def _process_batch(self, batch: BatchJob) -> None:
        """Process a single batch."""
        batch.status = BatchStatus.RUNNING
        batch.started_at = datetime.utcnow()
        start_time = asyncio.get_event_loop().time()

        logger.info(f"Processing batch {batch.batch_id}")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(batch.max_concurrent)

        # Process items
        pending_items = [i for i in batch.items if i.status == ItemStatus.PENDING]
        tasks = []

        for item in pending_items:
            task = asyncio.create_task(
                self._process_item(batch, item, semaphore)
            )
            tasks.append(task)

        # Wait for all items
        await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate final status
        batch.completed_at = datetime.utcnow()
        batch.total_processing_time = asyncio.get_event_loop().time() - start_time

        if batch.progress.failed > 0:
            if batch.progress.completed > 0:
                batch.status = BatchStatus.PARTIAL
            else:
                batch.status = BatchStatus.FAILED
        else:
            batch.status = BatchStatus.COMPLETED

        # Cleanup temporary files
        batch_path = self.storage_path / batch.batch_id
        if batch_path.exists():
            import shutil
            shutil.rmtree(batch_path, ignore_errors=True)

        logger.info(
            f"Batch {batch.batch_id} {batch.status}: "
            f"{batch.progress.completed}/{batch.progress.total} completed, "
            f"{batch.progress.failed} failed"
        )

    async def _process_item(
        self,
        batch: BatchJob,
        item: BatchItem,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Process a single item with retry logic."""
        async with semaphore:
            # Check if batch was cancelled/paused
            if batch.status in [BatchStatus.CANCELLED, BatchStatus.PAUSED]:
                return

            item.status = ItemStatus.PROCESSING
            item.started_at = datetime.utcnow()
            batch.progress.pending -= 1
            batch.progress.processing += 1

            while item.retries <= batch.max_retries:
                try:
                    # Load content
                    item_path = self.storage_path / batch.batch_id / item.item_id
                    content = item_path.read_bytes()

                    # Process
                    document_id = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.process_callback,
                        content,
                        item.filename,
                        item.metadata,
                    )

                    # Success
                    item.document_id = document_id
                    item.status = ItemStatus.COMPLETED
                    item.completed_at = datetime.utcnow()

                    batch.progress.processing -= 1
                    batch.progress.completed += 1
                    batch.progress.bytes_processed += item.size

                    # Update rate
                    elapsed = (datetime.utcnow() - batch.started_at).total_seconds()
                    if elapsed > 0:
                        batch.progress.current_rate = batch.progress.completed / elapsed
                        remaining = batch.progress.pending + batch.progress.processing
                        if batch.progress.current_rate > 0:
                            batch.progress.estimated_remaining_seconds = remaining / batch.progress.current_rate

                    return

                except Exception as e:
                    item.retries += 1
                    error_msg = str(e)

                    if item.retries <= batch.max_retries:
                        item.status = ItemStatus.RETRYING
                        logger.warning(
                            f"Retry {item.retries}/{batch.max_retries} for {item.filename}: {error_msg}"
                        )
                        await asyncio.sleep(batch.retry_delay_seconds * item.retries)
                    else:
                        # Final failure
                        item.status = ItemStatus.FAILED
                        item.error_message = error_msg
                        item.completed_at = datetime.utcnow()

                        batch.progress.processing -= 1
                        batch.progress.failed += 1

                        # Track error types
                        error_type = type(e).__name__
                        batch.error_summary[error_type] = batch.error_summary.get(error_type, 0) + 1

                        logger.error(f"Failed to process {item.filename}: {error_msg}")

                        # Stop on error if configured
                        if batch.stop_on_error:
                            batch.status = BatchStatus.FAILED
                            return

    def get_stats(self) -> dict:
        """Get batch processor statistics."""
        total_batches = len(self._batches)
        status_counts = {}
        total_items = 0
        total_completed = 0
        total_failed = 0

        for batch in self._batches.values():
            status_counts[batch.status.value] = status_counts.get(batch.status.value, 0) + 1
            total_items += batch.progress.total
            total_completed += batch.progress.completed
            total_failed += batch.progress.failed

        return {
            "total_batches": total_batches,
            "active_batches": len(self._active_batches),
            "queued_batches": self._batch_queue.qsize(),
            "status_counts": status_counts,
            "total_items": total_items,
            "total_completed": total_completed,
            "total_failed": total_failed,
            "workers": len(self._workers),
        }
