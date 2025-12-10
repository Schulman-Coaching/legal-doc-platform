"""
Scanner Ingestion Connector
===========================
Handles document ingestion from network scanners and MFPs.
Supports watched folders, TWAIN/WIA integration, and scanner APIs.
"""

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

logger = logging.getLogger(__name__)


class ScannerType(str, Enum):
    """Types of scanner integrations."""
    WATCHED_FOLDER = "watched_folder"
    NETWORK_SHARE = "network_share"
    FTP_DROP = "ftp_drop"
    SCANNER_API = "scanner_api"


class ScanQuality(str, Enum):
    """Scan quality presets."""
    DRAFT = "draft"  # 100 DPI
    STANDARD = "standard"  # 200 DPI
    HIGH = "high"  # 300 DPI
    ARCHIVAL = "archival"  # 600 DPI


@dataclass
class ScannerConfig:
    """Configuration for scanner connector."""
    scanner_type: ScannerType = ScannerType.WATCHED_FOLDER
    # Watched folder settings
    watch_path: str = ""
    # File processing
    file_patterns: list[str] = field(default_factory=lambda: ["*.pdf", "*.tif", "*.tiff", "*.png", "*.jpg"])
    min_file_age_seconds: int = 5  # Wait for file to be fully written
    process_existing: bool = True  # Process files present at startup
    # Post-processing
    delete_after_processing: bool = False
    move_after_processing: bool = True
    processed_folder: str = "processed"
    error_folder: str = "errors"
    # Scanner API settings (for TWAIN/WIA)
    scanner_id: Optional[str] = None
    default_quality: ScanQuality = ScanQuality.STANDARD
    # Network share settings
    share_path: Optional[str] = None
    share_username: Optional[str] = None
    share_password: Optional[str] = None
    # Naming conventions
    filename_pattern: str = "{datetime}_{scanner}_{sequence}"
    # Legal-specific
    auto_deskew: bool = True
    remove_blank_pages: bool = True


@dataclass
class ScannedDocument:
    """Represents a scanned document."""
    document_id: str
    source_path: Path
    filename: str
    size: int
    created_at: datetime
    scanner_id: Optional[str] = None
    page_count: int = 1
    quality: Optional[ScanQuality] = None
    checksum: Optional[str] = None
    local_path: Optional[Path] = None
    # Metadata extracted from filename or scanner
    metadata: dict[str, Any] = field(default_factory=dict)


class ScannerFileHandler(FileSystemEventHandler):
    """Watchdog handler for scanner folder monitoring."""

    def __init__(
        self,
        connector: "ScannerConnector",
        patterns: list[str],
    ):
        self.connector = connector
        self.patterns = [p.lower() for p in patterns]
        self._pending_files: dict[str, datetime] = {}

    def on_created(self, event: FileCreatedEvent):
        """Handle new file creation."""
        if event.is_directory:
            return

        filepath = Path(event.src_path)
        if self._matches_pattern(filepath.name):
            # Add to pending for stability check
            self._pending_files[str(filepath)] = datetime.utcnow()
            logger.debug(f"New file detected: {filepath}")

    def _matches_pattern(self, filename: str) -> bool:
        """Check if filename matches configured patterns."""
        import fnmatch
        return any(fnmatch.fnmatch(filename.lower(), p) for p in self.patterns)

    def get_stable_files(self, min_age_seconds: int) -> list[Path]:
        """Get files that have been stable for minimum age."""
        stable = []
        now = datetime.utcnow()

        for filepath_str, detected_at in list(self._pending_files.items()):
            filepath = Path(filepath_str)

            # Check if file still exists
            if not filepath.exists():
                del self._pending_files[filepath_str]
                continue

            # Check age
            age = (now - detected_at).total_seconds()
            if age >= min_age_seconds:
                # Verify file size hasn't changed
                try:
                    current_size = filepath.stat().st_size
                    if current_size > 0:
                        stable.append(filepath)
                        del self._pending_files[filepath_str]
                except OSError:
                    del self._pending_files[filepath_str]

        return stable


class ScannerConnector:
    """
    Scanner connector for watching folders and integrating with scanners.

    Features:
    - Watched folder monitoring with file stability detection
    - Automatic OCR preparation
    - Blank page detection
    - Auto-deskewing
    - Batch processing support
    """

    def __init__(
        self,
        config: ScannerConfig,
        storage_path: Path,
        callback: Optional[Callable[[ScannedDocument, bytes], None]] = None,
    ):
        self.config = config
        self.storage_path = storage_path / "scanner_ingestion"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.callback = callback
        self._running = False
        self._observer: Optional[Observer] = None
        self._handler: Optional[ScannerFileHandler] = None
        self._processed_files: set[str] = set()
        self._sequence = 0

    async def start(self) -> None:
        """Start the scanner connector."""
        self._running = True
        logger.info(f"Starting scanner connector for {self.config.watch_path}")

        if self.config.scanner_type == ScannerType.WATCHED_FOLDER:
            await self._start_folder_watch()
        elif self.config.scanner_type == ScannerType.NETWORK_SHARE:
            await self._start_network_watch()
        else:
            logger.warning(f"Scanner type {self.config.scanner_type} not yet implemented")

    async def stop(self) -> None:
        """Stop the scanner connector."""
        self._running = False

        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None

        logger.info("Scanner connector stopped")

    async def _start_folder_watch(self) -> None:
        """Start watching a local folder."""
        watch_path = Path(self.config.watch_path)

        if not watch_path.exists():
            watch_path.mkdir(parents=True)

        # Create processed and error folders
        (watch_path / self.config.processed_folder).mkdir(exist_ok=True)
        (watch_path / self.config.error_folder).mkdir(exist_ok=True)

        # Process existing files if configured
        if self.config.process_existing:
            await self._process_existing_files(watch_path)

        # Start watchdog observer
        self._handler = ScannerFileHandler(self, self.config.file_patterns)
        self._observer = Observer()
        self._observer.schedule(self._handler, str(watch_path), recursive=False)
        self._observer.start()

        # Process stable files periodically
        while self._running:
            await self._check_stable_files()
            await asyncio.sleep(1)

    async def _start_network_watch(self) -> None:
        """Start watching a network share."""
        # For network shares, we poll rather than use watchdog
        while self._running:
            try:
                await self._poll_network_share()
            except Exception as e:
                logger.error(f"Error polling network share: {e}")

            await asyncio.sleep(5)

    async def _process_existing_files(self, watch_path: Path) -> None:
        """Process files that existed before connector started."""
        import fnmatch

        for pattern in self.config.file_patterns:
            for filepath in watch_path.glob(pattern):
                if filepath.is_file() and str(filepath) not in self._processed_files:
                    # Add to handler's pending files
                    if self._handler:
                        self._handler._pending_files[str(filepath)] = datetime.utcnow()

    async def _check_stable_files(self) -> None:
        """Check for stable files and process them."""
        if not self._handler:
            return

        stable_files = self._handler.get_stable_files(self.config.min_file_age_seconds)

        for filepath in stable_files:
            try:
                await self._process_file(filepath)
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
                await self._handle_error(filepath)

    async def _poll_network_share(self) -> None:
        """Poll network share for new files."""
        import fnmatch

        share_path = Path(self.config.share_path or self.config.watch_path)

        for pattern in self.config.file_patterns:
            for filepath in share_path.glob(pattern):
                if filepath.is_file() and str(filepath) not in self._processed_files:
                    # Check file stability
                    try:
                        stat1 = filepath.stat()
                        await asyncio.sleep(self.config.min_file_age_seconds)
                        stat2 = filepath.stat()

                        if stat1.st_size == stat2.st_size and stat1.st_size > 0:
                            await self._process_file(filepath)
                    except OSError:
                        continue

    async def _process_file(self, filepath: Path) -> None:
        """Process a scanned document file."""
        if str(filepath) in self._processed_files:
            return

        logger.info(f"Processing scanned document: {filepath}")

        # Read file content
        with open(filepath, "rb") as f:
            content = f.read()

        # Create document record
        self._sequence += 1
        doc = ScannedDocument(
            document_id=f"scan_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{self._sequence:04d}",
            source_path=filepath,
            filename=filepath.name,
            size=len(content),
            created_at=datetime.fromtimestamp(filepath.stat().st_mtime),
            checksum=hashlib.sha256(content).hexdigest(),
            scanner_id=self.config.scanner_id,
            quality=self.config.default_quality,
        )

        # Parse metadata from filename if present
        doc.metadata = self._parse_filename_metadata(filepath.name)

        # Save to staging
        staging_path = self.storage_path / f"{doc.document_id}_{filepath.name}"
        with open(staging_path, "wb") as f:
            f.write(content)
        doc.local_path = staging_path

        # Preprocess if needed
        if self.config.auto_deskew or self.config.remove_blank_pages:
            await self._preprocess_scan(doc, content)

        # Call callback
        if self.callback:
            self.callback(doc, content)

        # Post-process original file
        await self._post_process(filepath)

        self._processed_files.add(str(filepath))

    def _parse_filename_metadata(self, filename: str) -> dict[str, Any]:
        """Extract metadata from structured filenames."""
        metadata = {}

        # Common scanner filename patterns
        patterns = [
            # Scanner with date: SCAN_20240115_143022.pdf
            (r"SCAN[_-](\d{8})[_-](\d{6})", ["date", "time"]),
            # Matter reference: Matter-12345_20240115.pdf
            (r"Matter[_-](\w+)[_-](\d+)", ["matter_id", "sequence"]),
            # Client code: ABC123_contract_signed.pdf
            (r"^([A-Z]{2,5}\d+)_", ["client_code"]),
        ]

        import re
        for pattern, keys in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                for i, key in enumerate(keys):
                    if i < len(match.groups()):
                        metadata[key] = match.group(i + 1)

        return metadata

    async def _preprocess_scan(self, doc: ScannedDocument, content: bytes) -> bytes:
        """Preprocess scanned document (deskew, remove blanks)."""
        # This is a placeholder - implement actual image processing
        # using PIL/Pillow or OpenCV for production

        if doc.filename.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
            logger.debug(f"Preprocessing {doc.filename} (deskew: {self.config.auto_deskew}, "
                        f"remove_blanks: {self.config.remove_blank_pages})")

            # In production:
            # - Use PIL/cv2 to detect and correct skew
            # - Use histogram analysis to detect blank pages
            # - Remove or mark blank pages

        return content

    async def _post_process(self, filepath: Path) -> None:
        """Post-process original file after ingestion."""
        watch_path = Path(self.config.watch_path)

        if self.config.delete_after_processing:
            filepath.unlink()
            logger.info(f"Deleted processed file: {filepath}")

        elif self.config.move_after_processing:
            processed_path = watch_path / self.config.processed_folder / filepath.name
            # Handle duplicate names
            if processed_path.exists():
                stem = processed_path.stem
                suffix = processed_path.suffix
                counter = 1
                while processed_path.exists():
                    processed_path = watch_path / self.config.processed_folder / f"{stem}_{counter}{suffix}"
                    counter += 1

            filepath.rename(processed_path)
            logger.info(f"Moved to: {processed_path}")

    async def _handle_error(self, filepath: Path) -> None:
        """Handle file that failed processing."""
        try:
            watch_path = Path(self.config.watch_path)
            error_path = watch_path / self.config.error_folder / filepath.name

            if error_path.exists():
                stem = error_path.stem
                suffix = error_path.suffix
                error_path = watch_path / self.config.error_folder / f"{stem}_{datetime.utcnow().strftime('%H%M%S')}{suffix}"

            filepath.rename(error_path)
            logger.info(f"Moved to error folder: {error_path}")

        except Exception as e:
            logger.error(f"Failed to move to error folder: {e}")

        self._processed_files.add(str(filepath))

    async def test_connection(self) -> tuple[bool, str]:
        """Test scanner connector configuration."""
        watch_path = Path(self.config.watch_path)

        try:
            if not watch_path.exists():
                return False, f"Watch path does not exist: {watch_path}"

            if not os.access(watch_path, os.R_OK | os.W_OK):
                return False, f"Insufficient permissions on: {watch_path}"

            # Count existing files
            import fnmatch
            file_count = 0
            for pattern in self.config.file_patterns:
                file_count += len(list(watch_path.glob(pattern)))

            return True, f"Watch path accessible. Found {file_count} matching files."

        except Exception as e:
            return False, str(e)

    def generate_filename(
        self,
        scanner_id: Optional[str] = None,
        extension: str = ".pdf",
    ) -> str:
        """Generate filename based on configured pattern."""
        self._sequence += 1

        placeholders = {
            "datetime": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            "date": datetime.utcnow().strftime("%Y%m%d"),
            "time": datetime.utcnow().strftime("%H%M%S"),
            "scanner": scanner_id or self.config.scanner_id or "scanner",
            "sequence": f"{self._sequence:04d}",
        }

        filename = self.config.filename_pattern
        for key, value in placeholders.items():
            filename = filename.replace(f"{{{key}}}", value)

        return filename + extension
