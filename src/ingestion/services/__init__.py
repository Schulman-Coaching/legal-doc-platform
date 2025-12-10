"""
Ingestion Services Package
==========================
Contains services for document validation, scanning, deduplication,
archive extraction, and batch processing.
"""

from .malware_scanner import MalwareScanner, ClamAVScanner, VirusTotalScanner, ScanResult
from .deduplication_service import DeduplicationService, DuplicateResult
from .archive_extractor import ArchiveExtractor, ExtractedFile
from .batch_processor import BatchProcessor, BatchJob, BatchStatus

__all__ = [
    "MalwareScanner",
    "ClamAVScanner",
    "VirusTotalScanner",
    "ScanResult",
    "DeduplicationService",
    "DuplicateResult",
    "ArchiveExtractor",
    "ExtractedFile",
    "BatchProcessor",
    "BatchJob",
    "BatchStatus",
]
