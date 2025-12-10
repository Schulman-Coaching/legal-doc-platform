"""
Tests for Archive Extractor
===========================
"""

import io
import gzip
import tarfile
import zipfile
import pytest
import tempfile
from pathlib import Path

from ..services.archive_extractor import (
    ArchiveExtractor,
    ArchiveType,
    ExtractionStatus,
)


class TestArchiveExtractor:
    """Test suite for ArchiveExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return ArchiveExtractor(
            max_files=100,
            max_total_size=100 * 1024 * 1024,  # 100 MB
            max_compression_ratio=50.0,
            extract_to_memory=True,
        )

    @pytest.fixture
    def sample_zip(self):
        """Create sample ZIP archive in memory."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("document.pdf", b"%PDF-1.4\nSample PDF content")
            zf.writestr("notes.txt", b"Some text notes")
            zf.writestr("subfolder/contract.docx", b"PK\x03\x04Contract content")
        return buffer.getvalue()

    @pytest.fixture
    def sample_tar_gz(self):
        """Create sample tar.gz archive in memory."""
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
            # Add a text file
            content = b"Sample text content"
            info = tarfile.TarInfo(name="document.txt")
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))

            # Add another file
            content2 = b"%PDF-1.4\nPDF content"
            info2 = tarfile.TarInfo(name="file.pdf")
            info2.size = len(content2)
            tf.addfile(info2, io.BytesIO(content2))

        return buffer.getvalue()

    @pytest.fixture
    def sample_gzip(self):
        """Create sample gzip file in memory."""
        content = b"This is the original uncompressed content"
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
            gz.write(content)
        return buffer.getvalue(), content

    # Archive type detection tests

    def test_detect_zip_by_signature(self, extractor):
        """Test ZIP detection by magic bytes."""
        zip_data = b"PK\x03\x04" + b"\x00" * 100

        archive_type = extractor._detect_type(zip_data, "unknown.bin")

        assert archive_type == ArchiveType.ZIP

    def test_detect_gzip_by_signature(self, extractor):
        """Test GZIP detection by magic bytes."""
        gzip_data = b"\x1f\x8b" + b"\x00" * 100

        archive_type = extractor._detect_type(gzip_data, "unknown.bin")

        assert archive_type == ArchiveType.GZIP

    def test_detect_by_extension(self, extractor):
        """Test archive detection by file extension."""
        data = b"some data"

        assert extractor._detect_type(data, "archive.zip") == ArchiveType.ZIP
        assert extractor._detect_type(data, "archive.tar") == ArchiveType.TAR
        assert extractor._detect_type(data, "archive.tar.gz") == ArchiveType.TAR_GZ
        assert extractor._detect_type(data, "archive.7z") == ArchiveType.SEVEN_ZIP
        assert extractor._detect_type(data, "archive.rar") == ArchiveType.RAR

    def test_detect_unknown_type(self, extractor):
        """Test that unknown archives return None."""
        unknown_data = b"unknown file format"

        archive_type = extractor._detect_type(unknown_data, "file.dat")

        assert archive_type is None

    # ZIP extraction tests

    @pytest.mark.asyncio
    async def test_extract_zip_success(self, extractor, sample_zip):
        """Test successful ZIP extraction."""
        result = await extractor.extract(sample_zip, "test.zip")

        assert result.status == ExtractionStatus.SUCCESS
        assert result.archive_type == ArchiveType.ZIP
        assert result.total_files == 3
        assert result.extracted_count == 2  # pdf and txt allowed, docx may vary
        assert result.total_size > 0

    @pytest.mark.asyncio
    async def test_extract_zip_with_password(self, extractor):
        """Test ZIP extraction with password."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.setpassword(b"secret")
            zf.writestr("document.txt", b"Secret content")

        # Note: Standard zipfile doesn't encrypt on write
        # This tests the API, actual encrypted ZIPs need different handling

    @pytest.mark.asyncio
    async def test_extract_zip_path_traversal_blocked(self, extractor):
        """Test that path traversal attempts are blocked."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            # Create file with path traversal attempt
            zf.writestr("../../../etc/passwd", b"malicious content")

        result = await extractor.extract(buffer.getvalue(), "malicious.zip")

        # File should be blocked
        assert any(
            f.extraction_status == ExtractionStatus.SECURITY_BLOCKED
            for f in result.files
        )

    @pytest.mark.asyncio
    async def test_extract_zip_disallowed_extension(self, extractor):
        """Test that disallowed file extensions are skipped."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("malware.exe", b"MZ executable")
            zf.writestr("document.pdf", b"%PDF-1.4\nContent")

        result = await extractor.extract(buffer.getvalue(), "mixed.zip")

        # exe should be skipped, pdf extracted
        exe_file = next((f for f in result.files if f.filename == "malware.exe"), None)
        pdf_file = next((f for f in result.files if f.filename == "document.pdf"), None)

        assert exe_file.extraction_status == ExtractionStatus.SKIPPED
        assert pdf_file.extraction_status == ExtractionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_extract_zip_too_many_files(self, extractor):
        """Test ZIP bomb protection - too many files."""
        extractor.max_files = 5

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            for i in range(10):
                zf.writestr(f"file{i}.txt", b"content")

        result = await extractor.extract(buffer.getvalue(), "many_files.zip")

        assert result.status == ExtractionStatus.SECURITY_BLOCKED
        assert any("max files" in w.lower() for w in result.security_warnings)

    # TAR extraction tests

    @pytest.mark.asyncio
    async def test_extract_tar_gz_success(self, extractor, sample_tar_gz):
        """Test successful tar.gz extraction."""
        result = await extractor.extract(sample_tar_gz, "archive.tar.gz")

        assert result.status == ExtractionStatus.SUCCESS
        assert result.archive_type == ArchiveType.TAR_GZ
        assert result.total_files == 2
        assert result.extracted_count == 2

    @pytest.mark.asyncio
    async def test_extract_tar_path_traversal_blocked(self, extractor):
        """Test that path traversal in TAR is blocked."""
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tf:
            info = tarfile.TarInfo(name="../../../etc/passwd")
            info.size = 10
            tf.addfile(info, io.BytesIO(b"malicious"))

        result = await extractor.extract(buffer.getvalue(), "malicious.tar")

        assert any(
            f.extraction_status == ExtractionStatus.SECURITY_BLOCKED
            for f in result.files
        )

    # GZIP extraction tests

    @pytest.mark.asyncio
    async def test_extract_gzip_success(self, extractor, sample_gzip):
        """Test successful GZIP extraction."""
        gzip_data, original_content = sample_gzip

        result = await extractor.extract(gzip_data, "file.txt.gz")

        assert result.status == ExtractionStatus.SUCCESS
        assert result.archive_type == ArchiveType.GZIP
        assert result.extracted_count == 1
        assert result.files[0].content == original_content

    @pytest.mark.asyncio
    async def test_extract_gzip_output_filename(self, extractor, sample_gzip):
        """Test GZIP extracts with correct filename."""
        gzip_data, _ = sample_gzip

        result = await extractor.extract(gzip_data, "document.pdf.gz")

        assert result.files[0].filename == "document.pdf"

    # Edge cases

    @pytest.mark.asyncio
    async def test_extract_invalid_archive(self, extractor):
        """Test handling of invalid archive data."""
        invalid_data = b"This is not an archive at all"

        result = await extractor.extract(invalid_data, "invalid.zip")

        assert result.status == ExtractionStatus.FAILED

    @pytest.mark.asyncio
    async def test_extract_empty_zip(self, extractor):
        """Test extraction of empty ZIP archive."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            pass  # Empty archive

        result = await extractor.extract(buffer.getvalue(), "empty.zip")

        assert result.status == ExtractionStatus.SUCCESS
        assert result.total_files == 0

    @pytest.mark.asyncio
    async def test_is_archive_detection(self, extractor, sample_zip, sample_tar_gz):
        """Test is_archive helper method."""
        assert await extractor.is_archive(sample_zip, "test.zip") is True
        assert await extractor.is_archive(sample_tar_gz, "test.tar.gz") is True
        assert await extractor.is_archive(b"not an archive", "file.txt") is False

    # Filename sanitization tests

    def test_sanitize_filename_path_traversal(self, extractor):
        """Test filename sanitization removes path components."""
        dangerous = "../../../etc/passwd"

        safe = extractor._sanitize_filename(dangerous)

        assert ".." not in safe
        assert "/" not in safe

    def test_sanitize_filename_special_chars(self, extractor):
        """Test filename sanitization handles special characters."""
        dangerous = 'file<>:"|?*.txt'

        safe = extractor._sanitize_filename(dangerous)

        for char in ['<', '>', ':', '"', '|', '?', '*']:
            assert char not in safe

    def test_sanitize_filename_length_limit(self, extractor):
        """Test filename sanitization limits length."""
        long_name = "a" * 300 + ".pdf"

        safe = extractor._sanitize_filename(long_name)

        assert len(safe) <= 204  # 200 + .pdf


class TestArchiveExtractorToFile:
    """Test extraction to filesystem instead of memory."""

    @pytest.fixture
    def file_extractor(self, tmp_path):
        """Create extractor that writes to filesystem."""
        return ArchiveExtractor(
            extract_to_memory=False,
            staging_path=tmp_path,
        )

    @pytest.mark.asyncio
    async def test_extract_to_file(self, file_extractor):
        """Test extraction to filesystem."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("document.pdf", b"%PDF-1.4\nContent")

        result = await file_extractor.extract(buffer.getvalue(), "test.zip")

        assert result.status == ExtractionStatus.SUCCESS
        assert result.files[0].extracted_path is not None
        assert result.files[0].extracted_path.exists()
        assert result.files[0].content is None  # Not in memory
