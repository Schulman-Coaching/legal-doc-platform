"""
Tests for Document Validator
============================
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock

from ..document_ingestion_service import (
    DocumentValidator,
    DocumentMetadata,
    IngestionSource,
    SecurityClassification,
)


class TestDocumentValidator:
    """Test suite for DocumentValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return DocumentValidator(enable_malware_scan=False)

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_pdf(self, temp_dir):
        """Create sample PDF file."""
        filepath = temp_dir / "sample.pdf"
        # PDF magic bytes + minimal content
        content = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\n"
        filepath.write_bytes(content)
        return filepath

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return DocumentMetadata(
            original_filename="test.pdf",
            file_size=1024,
            mime_type="application/pdf",
            source=IngestionSource.API_UPLOAD,
        )

    @pytest.mark.asyncio
    async def test_validate_valid_pdf(self, validator, sample_pdf, sample_metadata):
        """Test validation of valid PDF file."""
        sample_metadata.file_size = sample_pdf.stat().st_size

        is_valid, errors = await validator.validate(sample_pdf, sample_metadata)

        assert is_valid is True
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_validate_file_not_found(self, validator, temp_dir, sample_metadata):
        """Test validation when file doesn't exist."""
        non_existent = temp_dir / "missing.pdf"

        is_valid, errors = await validator.validate(non_existent, sample_metadata)

        assert is_valid is False
        assert "File does not exist" in errors

    @pytest.mark.asyncio
    async def test_validate_file_too_large(self, validator, temp_dir, sample_metadata):
        """Test validation of oversized file."""
        large_file = temp_dir / "large.pdf"
        large_file.write_bytes(b"%PDF-1.4\n" + b"x" * (101 * 1024 * 1024))  # 101 MB
        sample_metadata.file_size = large_file.stat().st_size
        sample_metadata.mime_type = "application/pdf"

        is_valid, errors = await validator.validate(large_file, sample_metadata)

        assert is_valid is False
        assert any("exceeds maximum" in e for e in errors)

    @pytest.mark.asyncio
    async def test_validate_disallowed_mime_type(self, validator, temp_dir, sample_metadata):
        """Test validation of disallowed MIME type."""
        exe_file = temp_dir / "malware.exe"
        exe_file.write_bytes(b"MZ" + b"\x00" * 100)
        sample_metadata.mime_type = "application/x-msdownload"
        sample_metadata.file_size = exe_file.stat().st_size

        is_valid, errors = await validator.validate(exe_file, sample_metadata)

        assert is_valid is False
        assert any("not allowed" in e for e in errors)

    @pytest.mark.asyncio
    async def test_validate_signature_mismatch(self, validator, temp_dir, sample_metadata):
        """Test validation when file signature doesn't match claimed type."""
        fake_pdf = temp_dir / "fake.pdf"
        fake_pdf.write_bytes(b"This is not a PDF")  # No PDF signature
        sample_metadata.mime_type = "application/pdf"
        sample_metadata.file_size = fake_pdf.stat().st_size

        is_valid, errors = await validator.validate(fake_pdf, sample_metadata)

        assert is_valid is False
        assert any("signature" in e.lower() for e in errors)

    @pytest.mark.asyncio
    async def test_validate_dangerous_content(self, validator, temp_dir, sample_metadata):
        """Test detection of dangerous content."""
        dangerous_file = temp_dir / "macro.docx"
        # PK signature (ZIP/DOCX) with AutoOpen macro indicator
        content = b"PK\x03\x04" + b"\x00" * 100 + b"AutoOpen" + b"\x00" * 100
        dangerous_file.write_bytes(content)
        sample_metadata.mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        sample_metadata.file_size = dangerous_file.stat().st_size

        is_valid, errors = await validator.validate(dangerous_file, sample_metadata)

        assert is_valid is False
        assert any("dangerous" in e.lower() for e in errors)

    def test_allowed_mime_types(self, validator):
        """Test that common legal document types are allowed."""
        allowed = [
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "image/png",
            "image/jpeg",
            "image/tiff",
        ]

        for mime_type in allowed:
            assert mime_type in validator.ALLOWED_MIME_TYPES

    def test_max_file_sizes_defined(self, validator):
        """Test that file size limits are defined."""
        assert "application/pdf" in validator.MAX_FILE_SIZES
        assert "default" in validator.MAX_FILE_SIZES
        assert validator.MAX_FILE_SIZES["application/pdf"] > 0


class TestDocumentValidatorWithMalwareScan:
    """Test malware scanning integration."""

    @pytest.mark.asyncio
    async def test_malware_scan_placeholder(self):
        """Test malware scanning placeholder."""
        validator = DocumentValidator(enable_malware_scan=True)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\ntest content")
            filepath = Path(f.name)

        try:
            # Placeholder should return False (no malware)
            result = await validator._scan_for_malware(filepath)
            assert result is False
        finally:
            filepath.unlink()
