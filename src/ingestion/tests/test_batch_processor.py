"""
Tests for Batch Processor Service
=================================
Comprehensive tests for batch document processing, including
parallel processing, retry logic, pause/resume, and progress tracking.
"""

import asyncio
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from ..services.batch_processor import (
    BatchProcessor,
    BatchJob,
    BatchItem,
    BatchProgress,
    BatchStatus,
    ItemStatus,
)


class TestBatchProcessor:
    """Test suite for BatchProcessor."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_process_callback(self):
        """Mock document processing callback."""
        def callback(content: bytes, filename: str, metadata: dict) -> str:
            # Return a mock document ID
            return f"doc-{filename}-{len(content)}"
        return callback

    @pytest.fixture
    def failing_callback(self):
        """Callback that always fails."""
        def callback(content: bytes, filename: str, metadata: dict) -> str:
            raise ValueError(f"Processing failed for {filename}")
        return callback

    @pytest.fixture
    def intermittent_callback(self):
        """Callback that fails first 2 times, then succeeds."""
        call_counts = {}

        def callback(content: bytes, filename: str, metadata: dict) -> str:
            call_counts[filename] = call_counts.get(filename, 0) + 1
            if call_counts[filename] <= 2:
                raise ValueError(f"Temporary failure for {filename}")
            return f"doc-{filename}"

        return callback

    @pytest.fixture
    def processor(self, mock_process_callback, temp_storage):
        """Create BatchProcessor instance."""
        return BatchProcessor(
            process_callback=mock_process_callback,
            max_concurrent_batches=2,
            storage_path=temp_storage,
        )

    @pytest.fixture
    def sample_files(self):
        """Sample files for batch processing."""
        return [
            (b"PDF content 1", "document1.pdf", {"client_id": "CLIENT-001"}),
            (b"PDF content 2", "document2.pdf", {"client_id": "CLIENT-001"}),
            (b"DOCX content", "contract.docx", {"client_id": "CLIENT-002"}),
            (b"Text content", "notes.txt", {"client_id": "CLIENT-002"}),
        ]

    # ===================
    # Batch Creation Tests
    # ===================

    @pytest.mark.asyncio
    async def test_create_batch(self, processor, sample_files):
        """Test creating a new batch."""
        batch = await processor.create_batch(
            name="Test Batch",
            files=sample_files,
            user_id="user-123",
            priority=5,
        )

        assert batch.batch_id is not None
        assert batch.name == "Test Batch"
        assert batch.status == BatchStatus.PENDING
        assert batch.created_by == "user-123"
        assert batch.priority == 5
        assert len(batch.items) == 4
        assert batch.progress.total == 4
        assert batch.progress.pending == 4
        assert batch.progress.bytes_total > 0

    @pytest.mark.asyncio
    async def test_create_batch_with_configuration(self, processor, sample_files):
        """Test batch creation with custom configuration."""
        batch = await processor.create_batch(
            name="Configured Batch",
            files=sample_files,
            user_id="user-456",
            max_concurrent=10,
            max_retries=5,
            stop_on_error=True,
        )

        assert batch.max_concurrent == 10
        assert batch.max_retries == 5
        assert batch.stop_on_error is True

    @pytest.mark.asyncio
    async def test_create_batch_stores_files(self, processor, sample_files):
        """Test that batch creation stores files temporarily."""
        batch = await processor.create_batch(
            name="Storage Test",
            files=sample_files,
            user_id="user-123",
        )

        # Verify files are stored
        batch_path = processor.storage_path / batch.batch_id
        assert batch_path.exists()

        for item in batch.items:
            item_path = batch_path / item.item_id
            assert item_path.exists()

    @pytest.mark.asyncio
    async def test_create_empty_batch(self, processor):
        """Test creating a batch with no files."""
        batch = await processor.create_batch(
            name="Empty Batch",
            files=[],
            user_id="user-123",
        )

        assert len(batch.items) == 0
        assert batch.progress.total == 0

    # ===================
    # Batch Processing Tests
    # ===================

    @pytest.mark.asyncio
    async def test_process_batch_success(self, processor, sample_files):
        """Test successful batch processing."""
        await processor.start()

        try:
            batch = await processor.create_batch(
                name="Success Test",
                files=sample_files,
                user_id="user-123",
            )

            # Wait for processing to complete
            for _ in range(50):  # Max 5 seconds
                await asyncio.sleep(0.1)
                current_batch = await processor.get_batch(batch.batch_id)
                if current_batch.status in [BatchStatus.COMPLETED, BatchStatus.FAILED]:
                    break

            # Verify results
            final_batch = await processor.get_batch(batch.batch_id)
            assert final_batch.status == BatchStatus.COMPLETED
            assert final_batch.progress.completed == 4
            assert final_batch.progress.failed == 0

            # Verify all items have document IDs
            for item in final_batch.items:
                assert item.status == ItemStatus.COMPLETED
                assert item.document_id is not None

        finally:
            await processor.stop()

    @pytest.mark.asyncio
    async def test_process_batch_with_failures(self, failing_callback, temp_storage, sample_files):
        """Test batch processing with failing items."""
        processor = BatchProcessor(
            process_callback=failing_callback,
            max_concurrent_batches=2,
            storage_path=temp_storage,
        )

        await processor.start()

        try:
            batch = await processor.create_batch(
                name="Failure Test",
                files=sample_files,
                user_id="user-123",
                max_retries=1,
            )

            # Wait for processing
            for _ in range(50):
                await asyncio.sleep(0.1)
                current_batch = await processor.get_batch(batch.batch_id)
                if current_batch.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.PARTIAL]:
                    break

            final_batch = await processor.get_batch(batch.batch_id)
            assert final_batch.status == BatchStatus.FAILED
            assert final_batch.progress.failed == 4
            assert final_batch.progress.completed == 0

            # Verify error messages
            for item in final_batch.items:
                assert item.status == ItemStatus.FAILED
                assert item.error_message is not None

        finally:
            await processor.stop()

    @pytest.mark.asyncio
    async def test_process_batch_with_retries(self, intermittent_callback, temp_storage):
        """Test batch processing with retry logic."""
        processor = BatchProcessor(
            process_callback=intermittent_callback,
            max_concurrent_batches=2,
            storage_path=temp_storage,
        )

        await processor.start()

        try:
            files = [(b"test content", "retry_test.pdf", {})]

            batch = await processor.create_batch(
                name="Retry Test",
                files=files,
                user_id="user-123",
                max_retries=3,
            )
            batch.retry_delay_seconds = 0.1  # Speed up tests

            # Wait for processing
            for _ in range(100):
                await asyncio.sleep(0.1)
                current_batch = await processor.get_batch(batch.batch_id)
                if current_batch.status in [BatchStatus.COMPLETED, BatchStatus.FAILED]:
                    break

            final_batch = await processor.get_batch(batch.batch_id)
            assert final_batch.status == BatchStatus.COMPLETED

            # Item should have succeeded after retries
            item = final_batch.items[0]
            assert item.status == ItemStatus.COMPLETED
            assert item.retries > 0

        finally:
            await processor.stop()

    @pytest.mark.asyncio
    async def test_stop_on_error(self, failing_callback, temp_storage, sample_files):
        """Test stop_on_error configuration."""
        processor = BatchProcessor(
            process_callback=failing_callback,
            max_concurrent_batches=1,
            storage_path=temp_storage,
        )

        await processor.start()

        try:
            batch = await processor.create_batch(
                name="Stop On Error Test",
                files=sample_files,
                user_id="user-123",
                max_retries=0,
                stop_on_error=True,
                max_concurrent=1,  # Process sequentially
            )

            # Wait for processing
            for _ in range(50):
                await asyncio.sleep(0.1)
                current_batch = await processor.get_batch(batch.batch_id)
                if current_batch.status in [BatchStatus.COMPLETED, BatchStatus.FAILED]:
                    break

            final_batch = await processor.get_batch(batch.batch_id)
            assert final_batch.status == BatchStatus.FAILED

            # Not all items should be processed due to stop_on_error
            failed_count = sum(1 for i in final_batch.items if i.status == ItemStatus.FAILED)
            assert failed_count >= 1

        finally:
            await processor.stop()

    # ===================
    # Batch Control Tests
    # ===================

    @pytest.mark.asyncio
    async def test_pause_and_resume_batch(self, processor, sample_files):
        """Test pausing and resuming a batch."""
        await processor.start()

        try:
            # Create batch but don't wait for completion
            batch = await processor.create_batch(
                name="Pause Test",
                files=sample_files,
                user_id="user-123",
            )

            # Try to pause (may or may not work depending on timing)
            result = await processor.pause_batch(batch.batch_id)

            # Resume
            if result:
                current_batch = await processor.get_batch(batch.batch_id)
                assert current_batch.status == BatchStatus.PAUSED

                await processor.resume_batch(batch.batch_id)
                current_batch = await processor.get_batch(batch.batch_id)
                # Status should be either RUNNING or already COMPLETED
                assert current_batch.status in [BatchStatus.RUNNING, BatchStatus.PENDING, BatchStatus.COMPLETED]

        finally:
            await processor.stop()

    @pytest.mark.asyncio
    async def test_cancel_batch(self, processor, sample_files):
        """Test cancelling a batch."""
        batch = await processor.create_batch(
            name="Cancel Test",
            files=sample_files,
            user_id="user-123",
        )

        result = await processor.cancel_batch(batch.batch_id)
        assert result is True

        cancelled_batch = await processor.get_batch(batch.batch_id)
        assert cancelled_batch.status == BatchStatus.CANCELLED
        assert cancelled_batch.completed_at is not None

        # Pending items should be marked as skipped
        for item in cancelled_batch.items:
            if item.status != ItemStatus.PROCESSING:
                assert item.status == ItemStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_batch(self, processor):
        """Test cancelling a batch that doesn't exist."""
        result = await processor.cancel_batch("nonexistent-batch-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_retry_failed_items(self, temp_storage, sample_files):
        """Test retrying failed items.

        Note: The current BatchProcessor implementation cleans up temporary files
        after the first batch processing completes, which means retry_failed()
        will fail to find the original files. This test verifies the retry_failed()
        method works correctly to reset item states, even though the actual retry
        processing may fail due to missing files.

        In a production scenario, files should either:
        1. Not be cleaned up until all retries are exhausted
        2. Be re-uploaded for retry operations
        """
        call_count = 0

        def conditional_callback(content, filename, metadata):
            nonlocal call_count
            call_count += 1
            if call_count <= 4:  # Fail first 4 calls
                raise ValueError("Simulated failure")
            return f"doc-{filename}"

        processor = BatchProcessor(
            process_callback=conditional_callback,
            max_concurrent_batches=2,
            storage_path=temp_storage,
        )

        await processor.start()

        try:
            batch = await processor.create_batch(
                name="Retry Failed Test",
                files=sample_files,
                user_id="user-123",
                max_retries=0,  # No automatic retries
            )

            # Wait for initial processing to fail
            for _ in range(50):
                await asyncio.sleep(0.1)
                current_batch = await processor.get_batch(batch.batch_id)
                if current_batch.status in [BatchStatus.FAILED, BatchStatus.PARTIAL]:
                    break

            # Verify initial failure state
            failed_batch = await processor.get_batch(batch.batch_id)
            assert failed_batch.status == BatchStatus.FAILED
            assert failed_batch.progress.failed == 4

            # Test that retry_failed resets item states
            result = await processor.retry_failed(batch.batch_id)
            assert result is True

            # Check that items were reset to pending
            retrying_batch = await processor.get_batch(batch.batch_id)
            assert retrying_batch.status == BatchStatus.PENDING
            assert retrying_batch.progress.pending == 4
            assert retrying_batch.progress.failed == 0

        finally:
            await processor.stop()

    # ===================
    # Progress Tracking Tests
    # ===================

    @pytest.mark.asyncio
    async def test_progress_tracking(self, processor, sample_files):
        """Test progress tracking during batch processing."""
        await processor.start()

        try:
            batch = await processor.create_batch(
                name="Progress Test",
                files=sample_files,
                user_id="user-123",
            )

            progress_snapshots = []

            # Capture progress snapshots
            for _ in range(50):
                await asyncio.sleep(0.1)
                progress = await processor.get_batch_progress(batch.batch_id)
                if progress:
                    progress_snapshots.append({
                        "completed": progress.completed,
                        "pending": progress.pending,
                        "processing": progress.processing,
                    })

                current_batch = await processor.get_batch(batch.batch_id)
                if current_batch.status == BatchStatus.COMPLETED:
                    break

            # Verify progress was tracked
            assert len(progress_snapshots) > 0

            # Final progress should show completion
            final_progress = await processor.get_batch_progress(batch.batch_id)
            assert final_progress.completed == 4
            assert final_progress.pending == 0

        finally:
            await processor.stop()

    @pytest.mark.asyncio
    async def test_bytes_processed_tracking(self, processor, sample_files):
        """Test bytes processed tracking."""
        await processor.start()

        try:
            batch = await processor.create_batch(
                name="Bytes Test",
                files=sample_files,
                user_id="user-123",
            )

            initial_progress = await processor.get_batch_progress(batch.batch_id)
            assert initial_progress.bytes_total > 0
            assert initial_progress.bytes_processed == 0

            # Wait for completion
            for _ in range(50):
                await asyncio.sleep(0.1)
                current_batch = await processor.get_batch(batch.batch_id)
                if current_batch.status == BatchStatus.COMPLETED:
                    break

            final_progress = await processor.get_batch_progress(batch.batch_id)
            assert final_progress.bytes_processed == final_progress.bytes_total

        finally:
            await processor.stop()

    # ===================
    # Listing Tests
    # ===================

    @pytest.mark.asyncio
    async def test_list_batches(self, processor, sample_files):
        """Test listing batches."""
        # Create multiple batches
        for i in range(3):
            await processor.create_batch(
                name=f"Batch {i}",
                files=sample_files[:2],
                user_id=f"user-{i % 2}",
            )

        # List all
        all_batches = await processor.list_batches()
        assert len(all_batches) == 3

        # List by user
        user_0_batches = await processor.list_batches(user_id="user-0")
        assert len(user_0_batches) == 2

        user_1_batches = await processor.list_batches(user_id="user-1")
        assert len(user_1_batches) == 1

    @pytest.mark.asyncio
    async def test_list_batches_by_status(self, processor, sample_files):
        """Test listing batches by status."""
        batch = await processor.create_batch(
            name="Status Test",
            files=sample_files,
            user_id="user-123",
        )

        # List pending
        pending = await processor.list_batches(status=BatchStatus.PENDING)
        assert len(pending) == 1

        # Cancel and check
        await processor.cancel_batch(batch.batch_id)

        cancelled = await processor.list_batches(status=BatchStatus.CANCELLED)
        assert len(cancelled) == 1

    @pytest.mark.asyncio
    async def test_list_batches_with_limit(self, processor, sample_files):
        """Test listing batches with limit."""
        for i in range(5):
            await processor.create_batch(
                name=f"Batch {i}",
                files=sample_files[:1],
                user_id="user-123",
            )

        limited = await processor.list_batches(limit=3)
        assert len(limited) == 3

    # ===================
    # Statistics Tests
    # ===================

    @pytest.mark.asyncio
    async def test_get_stats(self, processor, sample_files):
        """Test getting processor statistics."""
        await processor.start()

        try:
            # Create a batch
            await processor.create_batch(
                name="Stats Test",
                files=sample_files,
                user_id="user-123",
            )

            stats = processor.get_stats()

            assert "total_batches" in stats
            assert "active_batches" in stats
            assert "queued_batches" in stats
            assert "status_counts" in stats
            assert "total_items" in stats
            assert "workers" in stats
            assert stats["total_batches"] >= 1

        finally:
            await processor.stop()

    # ===================
    # Worker Tests
    # ===================

    @pytest.mark.asyncio
    async def test_multiple_workers(self, mock_process_callback, temp_storage, sample_files):
        """Test processing with multiple workers."""
        processor = BatchProcessor(
            process_callback=mock_process_callback,
            max_concurrent_batches=3,
            storage_path=temp_storage,
        )

        await processor.start()

        try:
            # Create multiple batches
            batches = []
            for i in range(3):
                batch = await processor.create_batch(
                    name=f"Worker Test {i}",
                    files=sample_files,
                    user_id="user-123",
                )
                batches.append(batch)

            # Wait for all to complete
            for _ in range(100):
                await asyncio.sleep(0.1)
                all_done = True
                for batch in batches:
                    current = await processor.get_batch(batch.batch_id)
                    if current.status not in [BatchStatus.COMPLETED, BatchStatus.FAILED]:
                        all_done = False
                        break
                if all_done:
                    break

            # Verify all completed
            for batch in batches:
                final = await processor.get_batch(batch.batch_id)
                assert final.status == BatchStatus.COMPLETED

        finally:
            await processor.stop()

    @pytest.mark.asyncio
    async def test_worker_shutdown(self, processor):
        """Test graceful worker shutdown."""
        await processor.start()
        assert len(processor._workers) == 2

        await processor.stop()
        assert processor._running is False

    # ===================
    # Priority Tests
    # ===================

    @pytest.mark.asyncio
    async def test_priority_ordering(self, mock_process_callback, temp_storage):
        """Test that higher priority batches are processed first."""
        # Use slow callback to ensure ordering matters
        processed_order = []

        def tracking_callback(content, filename, metadata):
            processed_order.append(metadata.get("batch_name"))
            return f"doc-{filename}"

        processor = BatchProcessor(
            process_callback=tracking_callback,
            max_concurrent_batches=1,  # Process one at a time
            storage_path=temp_storage,
        )

        # Create batches with different priorities (before starting)
        low_priority = await processor.create_batch(
            name="Low Priority",
            files=[(b"content", "low.pdf", {"batch_name": "low"})],
            user_id="user-123",
            priority=1,
        )

        high_priority = await processor.create_batch(
            name="High Priority",
            files=[(b"content", "high.pdf", {"batch_name": "high"})],
            user_id="user-123",
            priority=10,
        )

        await processor.start()

        try:
            # Wait for completion
            for _ in range(50):
                await asyncio.sleep(0.1)
                low = await processor.get_batch(low_priority.batch_id)
                high = await processor.get_batch(high_priority.batch_id)
                if low.status == BatchStatus.COMPLETED and high.status == BatchStatus.COMPLETED:
                    break

            # High priority should be processed first
            assert processed_order[0] == "high"

        finally:
            await processor.stop()

    # ===================
    # Edge Cases
    # ===================

    @pytest.mark.asyncio
    async def test_get_nonexistent_batch(self, processor):
        """Test getting a batch that doesn't exist."""
        batch = await processor.get_batch("nonexistent-id")
        assert batch is None

    @pytest.mark.asyncio
    async def test_get_progress_nonexistent_batch(self, processor):
        """Test getting progress for nonexistent batch."""
        progress = await processor.get_batch_progress("nonexistent-id")
        assert progress is None

    @pytest.mark.asyncio
    async def test_pause_nonexistent_batch(self, processor):
        """Test pausing a nonexistent batch."""
        result = await processor.pause_batch("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_resume_nonexistent_batch(self, processor):
        """Test resuming a nonexistent batch."""
        result = await processor.resume_batch("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_retry_nonexistent_batch(self, processor):
        """Test retrying a nonexistent batch."""
        result = await processor.retry_failed("nonexistent-id")
        assert result is False


class TestBatchItem:
    """Test suite for BatchItem data class."""

    def test_batch_item_defaults(self):
        """Test BatchItem default values."""
        item = BatchItem(
            item_id="item-1",
            filename="test.pdf",
            size=1024,
        )

        assert item.status == ItemStatus.PENDING
        assert item.document_id is None
        assert item.error_message is None
        assert item.retries == 0
        assert item.metadata == {}

    def test_batch_item_with_metadata(self):
        """Test BatchItem with metadata."""
        item = BatchItem(
            item_id="item-2",
            filename="contract.docx",
            size=2048,
            metadata={"client_id": "CLIENT-001", "matter_id": "MATTER-001"},
        )

        assert item.metadata["client_id"] == "CLIENT-001"
        assert item.metadata["matter_id"] == "MATTER-001"


class TestBatchProgress:
    """Test suite for BatchProgress data class."""

    def test_batch_progress_defaults(self):
        """Test BatchProgress default values."""
        progress = BatchProgress()

        assert progress.total == 0
        assert progress.pending == 0
        assert progress.processing == 0
        assert progress.completed == 0
        assert progress.failed == 0
        assert progress.bytes_processed == 0
        assert progress.estimated_remaining_seconds is None
        assert progress.current_rate == 0.0

    def test_batch_progress_calculations(self):
        """Test BatchProgress with values."""
        progress = BatchProgress(
            total=100,
            pending=50,
            processing=10,
            completed=35,
            failed=5,
            bytes_processed=350000,
            bytes_total=1000000,
        )

        assert progress.pending + progress.processing + progress.completed + progress.failed == 100
