"""
Archive Extraction Service
==========================
Extracts files from archives (ZIP, 7z, tar.gz, RAR) with security checks.
Prevents zip bombs and path traversal attacks.
"""

import asyncio
import gzip
import io
import logging
import os
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)


class ArchiveType(str, Enum):
    """Supported archive types."""
    ZIP = "zip"
    TAR = "tar"
    TAR_GZ = "tar.gz"
    TAR_BZ2 = "tar.bz2"
    GZIP = "gzip"
    SEVEN_ZIP = "7z"
    RAR = "rar"


class ExtractionStatus(str, Enum):
    """Status of extraction."""
    SUCCESS = "success"
    PARTIAL = "partial"  # Some files extracted
    FAILED = "failed"
    SKIPPED = "skipped"
    SECURITY_BLOCKED = "security_blocked"


@dataclass
class ExtractedFile:
    """Represents an extracted file from archive."""
    filename: str
    original_path: str  # Path within archive
    size: int
    compressed_size: Optional[int] = None
    content: Optional[bytes] = None
    extracted_path: Optional[Path] = None
    mime_type: Optional[str] = None
    modified_time: Optional[datetime] = None
    is_encrypted: bool = False
    extraction_status: ExtractionStatus = ExtractionStatus.SUCCESS
    error_message: Optional[str] = None


@dataclass
class ExtractionResult:
    """Result of archive extraction."""
    archive_type: ArchiveType
    status: ExtractionStatus
    files: list[ExtractedFile] = field(default_factory=list)
    total_files: int = 0
    extracted_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    total_size: int = 0
    security_warnings: list[str] = field(default_factory=list)
    error_message: Optional[str] = None


class ArchiveExtractor:
    """
    Secure archive extraction service.

    Security features:
    - Zip bomb detection (ratio limits)
    - Path traversal prevention
    - Maximum file limits
    - Nested archive detection
    - Encrypted file handling
    """

    # File signatures for archive detection
    SIGNATURES = {
        b"PK\x03\x04": ArchiveType.ZIP,
        b"PK\x05\x06": ArchiveType.ZIP,  # Empty archive
        b"\x1f\x8b": ArchiveType.GZIP,
        b"7z\xbc\xaf\x27\x1c": ArchiveType.SEVEN_ZIP,
        b"Rar!\x1a\x07\x00": ArchiveType.RAR,
        b"Rar!\x1a\x07\x01\x00": ArchiveType.RAR,  # RAR 5.0
    }

    def __init__(
        self,
        max_files: int = 1000,
        max_total_size: int = 5 * 1024 * 1024 * 1024,  # 5GB
        max_compression_ratio: float = 100.0,  # Zip bomb protection
        max_nested_depth: int = 3,
        extract_to_memory: bool = True,
        staging_path: Optional[Path] = None,
        allowed_extensions: Optional[set[str]] = None,
    ):
        self.max_files = max_files
        self.max_total_size = max_total_size
        self.max_compression_ratio = max_compression_ratio
        self.max_nested_depth = max_nested_depth
        self.extract_to_memory = extract_to_memory
        self.staging_path = staging_path or Path(tempfile.gettempdir()) / "archive_extract"
        self.staging_path.mkdir(parents=True, exist_ok=True)

        # Default allowed extensions for legal documents
        self.allowed_extensions = allowed_extensions or {
            ".pdf", ".doc", ".docx", ".txt", ".rtf",
            ".xls", ".xlsx", ".csv",
            ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp",
            ".eml", ".msg",
        }

    async def extract(
        self,
        archive_data: bytes,
        filename: str,
        password: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract files from archive.

        Args:
            archive_data: Archive binary content
            filename: Archive filename (for type detection)
            password: Optional password for encrypted archives

        Returns:
            ExtractionResult with extracted files
        """
        # Detect archive type
        archive_type = self._detect_type(archive_data, filename)

        if archive_type is None:
            return ExtractionResult(
                archive_type=ArchiveType.ZIP,
                status=ExtractionStatus.FAILED,
                error_message="Unknown or unsupported archive type",
            )

        # Route to appropriate extractor
        loop = asyncio.get_event_loop()

        if archive_type == ArchiveType.ZIP:
            return await loop.run_in_executor(
                None, self._extract_zip, archive_data, password
            )
        elif archive_type in [ArchiveType.TAR, ArchiveType.TAR_GZ, ArchiveType.TAR_BZ2]:
            return await loop.run_in_executor(
                None, self._extract_tar, archive_data, archive_type
            )
        elif archive_type == ArchiveType.GZIP:
            return await loop.run_in_executor(
                None, self._extract_gzip, archive_data, filename
            )
        elif archive_type == ArchiveType.SEVEN_ZIP:
            return await loop.run_in_executor(
                None, self._extract_7z, archive_data, password
            )
        elif archive_type == ArchiveType.RAR:
            return await loop.run_in_executor(
                None, self._extract_rar, archive_data, password
            )

        return ExtractionResult(
            archive_type=archive_type,
            status=ExtractionStatus.FAILED,
            error_message=f"Extractor for {archive_type} not implemented",
        )

    def _detect_type(self, data: bytes, filename: str) -> Optional[ArchiveType]:
        """Detect archive type from signature and filename."""
        # Check signatures
        for signature, archive_type in self.SIGNATURES.items():
            if data.startswith(signature):
                return archive_type

        # Check by extension
        ext = Path(filename).suffix.lower()
        ext_map = {
            ".zip": ArchiveType.ZIP,
            ".tar": ArchiveType.TAR,
            ".gz": ArchiveType.GZIP,
            ".tgz": ArchiveType.TAR_GZ,
            ".7z": ArchiveType.SEVEN_ZIP,
            ".rar": ArchiveType.RAR,
        }

        if ext in ext_map:
            return ext_map[ext]

        # Check for .tar.gz, .tar.bz2
        if filename.lower().endswith(".tar.gz"):
            return ArchiveType.TAR_GZ
        if filename.lower().endswith(".tar.bz2"):
            return ArchiveType.TAR_BZ2

        return None

    def _extract_zip(
        self,
        data: bytes,
        password: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract ZIP archive."""
        result = ExtractionResult(
            archive_type=ArchiveType.ZIP,
            status=ExtractionStatus.SUCCESS,
        )

        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                # Security check: count files and estimate size
                infos = zf.infolist()
                result.total_files = len(infos)

                if result.total_files > self.max_files:
                    result.status = ExtractionStatus.SECURITY_BLOCKED
                    result.security_warnings.append(
                        f"Archive exceeds max files ({result.total_files} > {self.max_files})"
                    )
                    return result

                # Check total uncompressed size
                total_uncompressed = sum(i.file_size for i in infos)
                total_compressed = sum(i.compress_size for i in infos)

                if total_uncompressed > self.max_total_size:
                    result.status = ExtractionStatus.SECURITY_BLOCKED
                    result.security_warnings.append(
                        f"Archive exceeds max size ({total_uncompressed} bytes)"
                    )
                    return result

                # Check compression ratio (zip bomb detection)
                if total_compressed > 0:
                    ratio = total_uncompressed / total_compressed
                    if ratio > self.max_compression_ratio:
                        result.status = ExtractionStatus.SECURITY_BLOCKED
                        result.security_warnings.append(
                            f"Suspicious compression ratio ({ratio:.1f}x)"
                        )
                        return result

                # Extract files
                pwd = password.encode() if password else None

                for info in infos:
                    extracted = self._extract_zip_member(zf, info, pwd)
                    result.files.append(extracted)

                    if extracted.extraction_status == ExtractionStatus.SUCCESS:
                        result.extracted_count += 1
                        result.total_size += extracted.size
                    elif extracted.extraction_status == ExtractionStatus.SKIPPED:
                        result.skipped_count += 1
                    else:
                        result.failed_count += 1

        except zipfile.BadZipFile as e:
            result.status = ExtractionStatus.FAILED
            result.error_message = f"Invalid ZIP file: {e}"
        except Exception as e:
            result.status = ExtractionStatus.FAILED
            result.error_message = str(e)

        if result.failed_count > 0 and result.extracted_count > 0:
            result.status = ExtractionStatus.PARTIAL

        return result

    def _extract_zip_member(
        self,
        zf: zipfile.ZipFile,
        info: zipfile.ZipInfo,
        password: Optional[bytes] = None,
    ) -> ExtractedFile:
        """Extract single file from ZIP."""
        extracted = ExtractedFile(
            filename=Path(info.filename).name,
            original_path=info.filename,
            size=info.file_size,
            compressed_size=info.compress_size,
        )

        # Skip directories
        if info.is_dir():
            extracted.extraction_status = ExtractionStatus.SKIPPED
            extracted.error_message = "Directory entry"
            return extracted

        # Security: check for path traversal
        if ".." in info.filename or info.filename.startswith("/"):
            extracted.extraction_status = ExtractionStatus.SECURITY_BLOCKED
            extracted.error_message = "Path traversal detected"
            return extracted

        # Check file extension
        ext = Path(info.filename).suffix.lower()
        if ext and ext not in self.allowed_extensions:
            # Check if it's a nested archive
            if ext in {".zip", ".tar", ".gz", ".7z", ".rar"}:
                extracted.extraction_status = ExtractionStatus.SKIPPED
                extracted.error_message = "Nested archive"
            else:
                extracted.extraction_status = ExtractionStatus.SKIPPED
                extracted.error_message = f"Extension {ext} not allowed"
            return extracted

        # Check encryption
        if info.flag_bits & 0x1:
            extracted.is_encrypted = True
            if password is None:
                extracted.extraction_status = ExtractionStatus.FAILED
                extracted.error_message = "File is encrypted, no password provided"
                return extracted

        try:
            # Extract content
            content = zf.read(info.filename, pwd=password)
            extracted.size = len(content)

            if self.extract_to_memory:
                extracted.content = content
            else:
                # Save to staging
                safe_name = self._sanitize_filename(info.filename)
                output_path = self.staging_path / safe_name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(content)
                extracted.extracted_path = output_path

            extracted.extraction_status = ExtractionStatus.SUCCESS

        except RuntimeError as e:
            if "password" in str(e).lower():
                extracted.extraction_status = ExtractionStatus.FAILED
                extracted.error_message = "Wrong password"
            else:
                extracted.extraction_status = ExtractionStatus.FAILED
                extracted.error_message = str(e)
        except Exception as e:
            extracted.extraction_status = ExtractionStatus.FAILED
            extracted.error_message = str(e)

        return extracted

    def _extract_tar(
        self,
        data: bytes,
        archive_type: ArchiveType,
    ) -> ExtractionResult:
        """Extract TAR archive (including .tar.gz, .tar.bz2)."""
        result = ExtractionResult(
            archive_type=archive_type,
            status=ExtractionStatus.SUCCESS,
        )

        try:
            # Determine mode
            if archive_type == ArchiveType.TAR_GZ:
                mode = "r:gz"
            elif archive_type == ArchiveType.TAR_BZ2:
                mode = "r:bz2"
            else:
                mode = "r"

            with tarfile.open(fileobj=io.BytesIO(data), mode=mode) as tf:
                members = tf.getmembers()
                result.total_files = len(members)

                if result.total_files > self.max_files:
                    result.status = ExtractionStatus.SECURITY_BLOCKED
                    result.security_warnings.append(
                        f"Archive exceeds max files ({result.total_files})"
                    )
                    return result

                for member in members:
                    extracted = self._extract_tar_member(tf, member)
                    result.files.append(extracted)

                    if extracted.extraction_status == ExtractionStatus.SUCCESS:
                        result.extracted_count += 1
                        result.total_size += extracted.size
                    elif extracted.extraction_status == ExtractionStatus.SKIPPED:
                        result.skipped_count += 1
                    else:
                        result.failed_count += 1

        except tarfile.TarError as e:
            result.status = ExtractionStatus.FAILED
            result.error_message = f"Invalid TAR file: {e}"
        except Exception as e:
            result.status = ExtractionStatus.FAILED
            result.error_message = str(e)

        return result

    def _extract_tar_member(
        self,
        tf: tarfile.TarFile,
        member: tarfile.TarInfo,
    ) -> ExtractedFile:
        """Extract single file from TAR."""
        extracted = ExtractedFile(
            filename=Path(member.name).name,
            original_path=member.name,
            size=member.size,
        )

        # Skip non-files
        if not member.isfile():
            extracted.extraction_status = ExtractionStatus.SKIPPED
            extracted.error_message = "Not a regular file"
            return extracted

        # Security: check for path traversal
        if ".." in member.name or member.name.startswith("/"):
            extracted.extraction_status = ExtractionStatus.SECURITY_BLOCKED
            extracted.error_message = "Path traversal detected"
            return extracted

        # Check extension
        ext = Path(member.name).suffix.lower()
        if ext and ext not in self.allowed_extensions:
            extracted.extraction_status = ExtractionStatus.SKIPPED
            extracted.error_message = f"Extension {ext} not allowed"
            return extracted

        try:
            f = tf.extractfile(member)
            if f is None:
                extracted.extraction_status = ExtractionStatus.FAILED
                extracted.error_message = "Could not extract file"
                return extracted

            content = f.read()
            extracted.size = len(content)

            if self.extract_to_memory:
                extracted.content = content
            else:
                safe_name = self._sanitize_filename(member.name)
                output_path = self.staging_path / safe_name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(content)
                extracted.extracted_path = output_path

            extracted.extraction_status = ExtractionStatus.SUCCESS

        except Exception as e:
            extracted.extraction_status = ExtractionStatus.FAILED
            extracted.error_message = str(e)

        return extracted

    def _extract_gzip(
        self,
        data: bytes,
        filename: str,
    ) -> ExtractionResult:
        """Extract GZIP file (single file only)."""
        result = ExtractionResult(
            archive_type=ArchiveType.GZIP,
            status=ExtractionStatus.SUCCESS,
            total_files=1,
        )

        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
                content = gz.read()

                # Check size
                if len(content) > self.max_total_size:
                    result.status = ExtractionStatus.SECURITY_BLOCKED
                    result.security_warnings.append("Decompressed size exceeds limit")
                    return result

                # Determine output filename
                output_name = filename
                if output_name.lower().endswith(".gz"):
                    output_name = output_name[:-3]

                extracted = ExtractedFile(
                    filename=output_name,
                    original_path=filename,
                    size=len(content),
                    compressed_size=len(data),
                    extraction_status=ExtractionStatus.SUCCESS,
                )

                if self.extract_to_memory:
                    extracted.content = content
                else:
                    output_path = self.staging_path / self._sanitize_filename(output_name)
                    output_path.write_bytes(content)
                    extracted.extracted_path = output_path

                result.files.append(extracted)
                result.extracted_count = 1
                result.total_size = len(content)

        except Exception as e:
            result.status = ExtractionStatus.FAILED
            result.error_message = str(e)

        return result

    def _extract_7z(
        self,
        data: bytes,
        password: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract 7-Zip archive using py7zr."""
        result = ExtractionResult(
            archive_type=ArchiveType.SEVEN_ZIP,
            status=ExtractionStatus.SUCCESS,
        )

        try:
            import py7zr

            with py7zr.SevenZipFile(io.BytesIO(data), password=password) as sz:
                # Get file list
                file_list = sz.list()
                result.total_files = len(file_list)

                if result.total_files > self.max_files:
                    result.status = ExtractionStatus.SECURITY_BLOCKED
                    result.security_warnings.append("Too many files in archive")
                    return result

                # Extract all
                extracted_dict = sz.readall()

                for filename, content in extracted_dict.items():
                    bio = content
                    data_bytes = bio.read() if hasattr(bio, 'read') else bio

                    extracted = ExtractedFile(
                        filename=Path(filename).name,
                        original_path=filename,
                        size=len(data_bytes),
                        extraction_status=ExtractionStatus.SUCCESS,
                    )

                    # Check extension
                    ext = Path(filename).suffix.lower()
                    if ext and ext not in self.allowed_extensions:
                        extracted.extraction_status = ExtractionStatus.SKIPPED
                        result.skipped_count += 1
                    elif self.extract_to_memory:
                        extracted.content = data_bytes
                        result.extracted_count += 1
                        result.total_size += len(data_bytes)
                    else:
                        output_path = self.staging_path / self._sanitize_filename(filename)
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_path.write_bytes(data_bytes)
                        extracted.extracted_path = output_path
                        result.extracted_count += 1
                        result.total_size += len(data_bytes)

                    result.files.append(extracted)

        except ImportError:
            result.status = ExtractionStatus.FAILED
            result.error_message = "py7zr library not installed"
        except Exception as e:
            result.status = ExtractionStatus.FAILED
            result.error_message = str(e)

        return result

    def _extract_rar(
        self,
        data: bytes,
        password: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract RAR archive using rarfile."""
        result = ExtractionResult(
            archive_type=ArchiveType.RAR,
            status=ExtractionStatus.SUCCESS,
        )

        try:
            import rarfile

            # Write to temp file (rarfile needs file path)
            with tempfile.NamedTemporaryFile(suffix=".rar", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            try:
                with rarfile.RarFile(tmp_path) as rf:
                    if password:
                        rf.setpassword(password)

                    infos = rf.infolist()
                    result.total_files = len(infos)

                    if result.total_files > self.max_files:
                        result.status = ExtractionStatus.SECURITY_BLOCKED
                        return result

                    for info in infos:
                        if info.is_dir():
                            continue

                        extracted = ExtractedFile(
                            filename=Path(info.filename).name,
                            original_path=info.filename,
                            size=info.file_size,
                            compressed_size=info.compress_size,
                            is_encrypted=info.needs_password(),
                        )

                        ext = Path(info.filename).suffix.lower()
                        if ext and ext not in self.allowed_extensions:
                            extracted.extraction_status = ExtractionStatus.SKIPPED
                            result.skipped_count += 1
                        else:
                            try:
                                content = rf.read(info.filename)
                                extracted.size = len(content)

                                if self.extract_to_memory:
                                    extracted.content = content
                                else:
                                    output_path = self.staging_path / self._sanitize_filename(info.filename)
                                    output_path.parent.mkdir(parents=True, exist_ok=True)
                                    output_path.write_bytes(content)
                                    extracted.extracted_path = output_path

                                extracted.extraction_status = ExtractionStatus.SUCCESS
                                result.extracted_count += 1
                                result.total_size += len(content)

                            except Exception as e:
                                extracted.extraction_status = ExtractionStatus.FAILED
                                extracted.error_message = str(e)
                                result.failed_count += 1

                        result.files.append(extracted)

            finally:
                os.unlink(tmp_path)

        except ImportError:
            result.status = ExtractionStatus.FAILED
            result.error_message = "rarfile library not installed"
        except Exception as e:
            result.status = ExtractionStatus.FAILED
            result.error_message = str(e)

        return result

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe extraction."""
        # Remove path traversal
        safe = PurePosixPath(filename).name

        # Replace problematic characters
        for char in ['<', '>', ':', '"', '|', '?', '*']:
            safe = safe.replace(char, '_')

        # Limit length
        if len(safe) > 200:
            name, ext = os.path.splitext(safe)
            safe = name[:200 - len(ext)] + ext

        return safe

    async def is_archive(self, data: bytes, filename: str) -> bool:
        """Check if data is a supported archive."""
        return self._detect_type(data, filename) is not None
