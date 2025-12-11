"""
Tests for SFTP Connector
========================
Comprehensive tests for SFTP ingestion connector including
authentication, file scanning, and post-processing.
"""

import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import stat

from ..connectors.sftp_connector import (
    SFTPConnector,
    SFTPConfig,
    SFTPAuthMethod,
    SFTPPostAction,
    SFTPFile,
)


class TestSFTPConfig:
    """Test suite for SFTPConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SFTPConfig(
            host="sftp.example.com",
            username="user",
        )

        assert config.port == 22
        assert config.auth_method == SFTPAuthMethod.PASSWORD
        assert config.remote_path == "/"
        assert config.processed_path == "/processed"
        assert config.error_path == "/errors"
        assert config.poll_interval_seconds == 300
        assert config.recursive is True
        assert config.max_depth == 10
        assert config.post_action == SFTPPostAction.MOVE
        assert "*.pdf" in config.file_patterns
        assert ".*" in config.exclude_patterns
        assert config.min_age_seconds == 60

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SFTPConfig(
            host="sftp.legal.com",
            port=2222,
            username="legal_user",
            auth_method=SFTPAuthMethod.KEY_FILE,
            key_file="/path/to/key",
            key_passphrase="keypass",
            remote_path="/legal_docs",
            processed_path="/archive",
            file_patterns=["*.pdf", "*.docx", "*.xlsx"],
            exclude_patterns=["temp_*", "*.tmp"],
            max_file_size=500 * 1024 * 1024,
            min_age_seconds=120,
            post_action=SFTPPostAction.DELETE,
        )

        assert config.port == 2222
        assert config.auth_method == SFTPAuthMethod.KEY_FILE
        assert config.key_file == "/path/to/key"
        assert config.remote_path == "/legal_docs"
        assert config.post_action == SFTPPostAction.DELETE
        assert config.min_age_seconds == 120


class TestSFTPFile:
    """Test suite for SFTPFile dataclass."""

    def test_sftp_file_creation(self):
        """Test SFTPFile creation."""
        file_info = SFTPFile(
            remote_path="/legal_docs/contract.pdf",
            filename="contract.pdf",
            size=1024 * 1024,
            mtime=datetime.utcnow(),
            checksum="abc123",
            source_host="sftp.example.com",
        )

        assert file_info.remote_path == "/legal_docs/contract.pdf"
        assert file_info.filename == "contract.pdf"
        assert file_info.size == 1024 * 1024
        assert file_info.checksum == "abc123"

    def test_sftp_file_defaults(self):
        """Test SFTPFile default values."""
        file_info = SFTPFile(
            remote_path="/docs/file.txt",
            filename="file.txt",
            size=100,
            mtime=datetime.utcnow(),
        )

        assert file_info.checksum is None
        assert file_info.local_path is None
        assert file_info.source_host == ""
        assert file_info.discovered_at is not None


class TestSFTPConnector:
    """Test suite for SFTPConnector."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sftp_config(self):
        """Create test SFTP configuration."""
        return SFTPConfig(
            host="sftp.test.com",
            port=22,
            username="testuser",
            password="testpassword",
            auth_method=SFTPAuthMethod.PASSWORD,
            remote_path="/incoming",
            processed_path="/processed",
            error_path="/errors",
            file_patterns=["*.pdf", "*.docx"],
            exclude_patterns=[".*", "*.tmp"],
            min_age_seconds=0,  # No wait for tests
        )

    @pytest.fixture
    def connector(self, sftp_config, temp_storage):
        """Create SFTPConnector instance."""
        return SFTPConnector(
            config=sftp_config,
            storage_path=temp_storage,
        )

    @pytest.fixture
    def mock_sftp_client(self):
        """Create mock SFTP client."""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def mock_transport(self):
        """Create mock paramiko transport."""
        mock = MagicMock()
        return mock

    # ===================
    # Pattern Matching Tests
    # ===================

    def test_matches_patterns_pdf(self, connector):
        """Test pattern matching for PDF files."""
        assert connector._matches_patterns("document.pdf", ["*.pdf"]) is True
        assert connector._matches_patterns("Document.PDF", ["*.pdf"]) is True

    def test_matches_patterns_docx(self, connector):
        """Test pattern matching for DOCX files."""
        assert connector._matches_patterns("contract.docx", ["*.docx"]) is True
        assert connector._matches_patterns("report.doc", ["*.doc*"]) is True

    def test_matches_patterns_no_match(self, connector):
        """Test pattern matching with no matches."""
        assert connector._matches_patterns("image.png", ["*.pdf", "*.docx"]) is False
        assert connector._matches_patterns("script.exe", ["*.pdf", "*.docx"]) is False

    def test_matches_patterns_hidden_files(self, connector):
        """Test pattern matching for hidden files."""
        assert connector._matches_patterns(".hidden", [".*"]) is True
        assert connector._matches_patterns(".gitignore", [".*"]) is True

    def test_matches_patterns_temp_files(self, connector):
        """Test pattern matching for temp files."""
        assert connector._matches_patterns("~document.docx", ["~*"]) is True
        assert connector._matches_patterns("file.tmp", ["*.tmp"]) is True

    # ===================
    # Directory Scanning Tests
    # ===================

    def test_scan_directory_basic(self, connector, mock_sftp_client):
        """Test basic directory scanning."""
        # Create mock directory entries
        mock_entry_pdf = MagicMock()
        mock_entry_pdf.filename = "document.pdf"
        mock_entry_pdf.st_mode = stat.S_IFREG
        mock_entry_pdf.st_size = 1024
        mock_entry_pdf.st_mtime = (datetime.utcnow() - timedelta(minutes=5)).timestamp()

        mock_entry_docx = MagicMock()
        mock_entry_docx.filename = "contract.docx"
        mock_entry_docx.st_mode = stat.S_IFREG
        mock_entry_docx.st_size = 2048
        mock_entry_docx.st_mtime = (datetime.utcnow() - timedelta(minutes=5)).timestamp()

        mock_sftp_client.listdir_attr.return_value = [mock_entry_pdf, mock_entry_docx]

        connector._sftp = mock_sftp_client
        files = connector._scan_directory("/incoming", 0)

        assert len(files) == 2
        assert any(f.filename == "document.pdf" for f in files)
        assert any(f.filename == "contract.docx" for f in files)

    def test_scan_directory_excludes_hidden(self, connector, mock_sftp_client):
        """Test that hidden files are excluded."""
        mock_entry_hidden = MagicMock()
        mock_entry_hidden.filename = ".hidden_doc.pdf"
        mock_entry_hidden.st_mode = stat.S_IFREG
        mock_entry_hidden.st_size = 1024
        mock_entry_hidden.st_mtime = datetime.utcnow().timestamp()

        mock_entry_normal = MagicMock()
        mock_entry_normal.filename = "normal.pdf"
        mock_entry_normal.st_mode = stat.S_IFREG
        mock_entry_normal.st_size = 1024
        mock_entry_normal.st_mtime = (datetime.utcnow() - timedelta(minutes=5)).timestamp()

        mock_sftp_client.listdir_attr.return_value = [mock_entry_hidden, mock_entry_normal]

        connector._sftp = mock_sftp_client
        files = connector._scan_directory("/incoming", 0)

        assert len(files) == 1
        assert files[0].filename == "normal.pdf"

    def test_scan_directory_excludes_too_large(self, connector, mock_sftp_client):
        """Test that oversized files are excluded."""
        # File larger than max_file_size (1GB default)
        mock_entry_large = MagicMock()
        mock_entry_large.filename = "huge.pdf"
        mock_entry_large.st_mode = stat.S_IFREG
        mock_entry_large.st_size = 2 * 1024 * 1024 * 1024  # 2GB
        mock_entry_large.st_mtime = (datetime.utcnow() - timedelta(minutes=5)).timestamp()

        mock_entry_normal = MagicMock()
        mock_entry_normal.filename = "normal.pdf"
        mock_entry_normal.st_mode = stat.S_IFREG
        mock_entry_normal.st_size = 1024
        mock_entry_normal.st_mtime = (datetime.utcnow() - timedelta(minutes=5)).timestamp()

        mock_sftp_client.listdir_attr.return_value = [mock_entry_large, mock_entry_normal]

        connector._sftp = mock_sftp_client
        files = connector._scan_directory("/incoming", 0)

        assert len(files) == 1
        assert files[0].filename == "normal.pdf"

    def test_scan_directory_excludes_too_new(self, temp_storage):
        """Test that files too new are excluded (stability check)."""
        config = SFTPConfig(
            host="sftp.test.com",
            username="user",
            min_age_seconds=300,  # 5 minutes
        )
        connector = SFTPConnector(config=config, storage_path=temp_storage)

        mock_sftp_client = MagicMock()

        # File created just now
        mock_entry_new = MagicMock()
        mock_entry_new.filename = "new_file.pdf"
        mock_entry_new.st_mode = stat.S_IFREG
        mock_entry_new.st_size = 1024
        mock_entry_new.st_mtime = datetime.utcnow().timestamp()

        # File created 10 minutes ago
        mock_entry_old = MagicMock()
        mock_entry_old.filename = "old_file.pdf"
        mock_entry_old.st_mode = stat.S_IFREG
        mock_entry_old.st_size = 1024
        mock_entry_old.st_mtime = (datetime.utcnow() - timedelta(minutes=10)).timestamp()

        mock_sftp_client.listdir_attr.return_value = [mock_entry_new, mock_entry_old]

        connector._sftp = mock_sftp_client
        files = connector._scan_directory("/incoming", 0)

        assert len(files) == 1
        assert files[0].filename == "old_file.pdf"

    def test_scan_directory_recursive(self, connector, mock_sftp_client):
        """Test recursive directory scanning."""
        # Root level entries
        mock_subdir = MagicMock()
        mock_subdir.filename = "subdir"
        mock_subdir.st_mode = stat.S_IFDIR

        mock_root_file = MagicMock()
        mock_root_file.filename = "root_doc.pdf"
        mock_root_file.st_mode = stat.S_IFREG
        mock_root_file.st_size = 1024
        mock_root_file.st_mtime = (datetime.utcnow() - timedelta(minutes=5)).timestamp()

        # Subdir entries
        mock_subdir_file = MagicMock()
        mock_subdir_file.filename = "subdir_doc.pdf"
        mock_subdir_file.st_mode = stat.S_IFREG
        mock_subdir_file.st_size = 2048
        mock_subdir_file.st_mtime = (datetime.utcnow() - timedelta(minutes=5)).timestamp()

        def listdir_attr_side_effect(path):
            if path == "/incoming":
                return [mock_subdir, mock_root_file]
            elif path == "/incoming/subdir":
                return [mock_subdir_file]
            return []

        mock_sftp_client.listdir_attr.side_effect = listdir_attr_side_effect

        connector._sftp = mock_sftp_client
        files = connector._scan_directory("/incoming", 0)

        assert len(files) == 2
        assert any(f.filename == "root_doc.pdf" for f in files)
        assert any(f.filename == "subdir_doc.pdf" for f in files)

    def test_scan_directory_max_depth(self, temp_storage, mock_sftp_client):
        """Test max depth limit in recursive scanning."""
        config = SFTPConfig(
            host="sftp.test.com",
            username="user",
            recursive=True,
            max_depth=1,
            min_age_seconds=0,
        )
        connector = SFTPConnector(config=config, storage_path=temp_storage)

        # Level 0 (root)
        mock_level1_dir = MagicMock()
        mock_level1_dir.filename = "level1"
        mock_level1_dir.st_mode = stat.S_IFDIR

        # Level 1
        mock_level2_dir = MagicMock()
        mock_level2_dir.filename = "level2"
        mock_level2_dir.st_mode = stat.S_IFDIR

        mock_level1_file = MagicMock()
        mock_level1_file.filename = "level1_doc.pdf"
        mock_level1_file.st_mode = stat.S_IFREG
        mock_level1_file.st_size = 1024
        mock_level1_file.st_mtime = (datetime.utcnow() - timedelta(minutes=5)).timestamp()

        # Level 2 - should not be reached
        mock_level2_file = MagicMock()
        mock_level2_file.filename = "level2_doc.pdf"
        mock_level2_file.st_mode = stat.S_IFREG
        mock_level2_file.st_size = 1024
        mock_level2_file.st_mtime = (datetime.utcnow() - timedelta(minutes=5)).timestamp()

        def listdir_attr_side_effect(path):
            if path == "/":
                return [mock_level1_dir]
            elif path == "/level1":
                return [mock_level2_dir, mock_level1_file]
            elif path == "/level1/level2":
                return [mock_level2_file]
            return []

        mock_sftp_client.listdir_attr.side_effect = listdir_attr_side_effect

        connector._sftp = mock_sftp_client
        files = connector._scan_directory("/", 0)

        # Should only find level1_doc.pdf, not level2_doc.pdf
        assert len(files) == 1
        assert files[0].filename == "level1_doc.pdf"

    def test_scan_directory_skips_processed(self, connector, mock_sftp_client):
        """Test that already processed files are skipped."""
        mock_entry = MagicMock()
        mock_entry.filename = "document.pdf"
        mock_entry.st_mode = stat.S_IFREG
        mock_entry.st_size = 1024
        mock_entry.st_mtime = (datetime.utcnow() - timedelta(minutes=5)).timestamp()

        mock_sftp_client.listdir_attr.return_value = [mock_entry]

        connector._sftp = mock_sftp_client

        # First scan
        files = connector._scan_directory("/incoming", 0)
        assert len(files) == 1

        # Mark as processed
        connector._processed_files.add("/incoming/document.pdf")

        # Second scan - should skip
        files = connector._scan_directory("/incoming", 0)
        assert len(files) == 0

    # ===================
    # Connection Tests (Mocked)
    # ===================

    def test_sync_connect_password(self, connector, mock_transport):
        """Test password authentication connection."""
        with patch("paramiko.Transport") as mock_transport_class:
            mock_transport_class.return_value = mock_transport

            with patch("paramiko.SFTPClient.from_transport") as mock_from_transport:
                mock_sftp = MagicMock()
                mock_from_transport.return_value = mock_sftp

                connector._sync_connect()

                mock_transport_class.assert_called_once_with(("sftp.test.com", 22))
                mock_transport.connect.assert_called_once_with(
                    username="testuser",
                    password="testpassword",
                )

    def test_sync_connect_key_file(self, temp_storage, mock_transport):
        """Test key file authentication."""
        config = SFTPConfig(
            host="sftp.test.com",
            username="user",
            auth_method=SFTPAuthMethod.KEY_FILE,
            key_file="/path/to/key",
        )
        connector = SFTPConnector(config=config, storage_path=temp_storage)

        with patch("paramiko.Transport") as mock_transport_class:
            mock_transport_class.return_value = mock_transport

            # Mock _load_private_key to avoid actual key loading
            with patch.object(connector, "_load_private_key") as mock_load_key:
                mock_key = MagicMock()
                mock_load_key.return_value = mock_key

                with patch("paramiko.SFTPClient.from_transport") as mock_from_transport:
                    mock_sftp = MagicMock()
                    mock_from_transport.return_value = mock_sftp

                    connector._sync_connect()

                    mock_transport.connect.assert_called_once_with(
                        username="user",
                        pkey=mock_key,
                    )

    def test_sync_disconnect(self, connector):
        """Test disconnection."""
        mock_sftp = MagicMock()
        mock_transport = MagicMock()
        connector._sftp = mock_sftp
        connector._transport = mock_transport

        connector._sync_disconnect()

        mock_sftp.close.assert_called_once()
        mock_transport.close.assert_called_once()
        assert connector._sftp is None
        assert connector._transport is None

    @pytest.mark.asyncio
    async def test_test_connection_success(self, connector):
        """Test successful connection test."""
        with patch.object(connector, "_connect") as mock_connect:
            with patch.object(connector, "_disconnect") as mock_disconnect:
                connector._sftp = MagicMock()
                connector._sftp.listdir.return_value = ["file1.pdf", "file2.docx", "dir1"]

                success, message = await connector.test_connection()

                assert success is True
                assert "3 items" in message

    @pytest.mark.asyncio
    async def test_test_connection_failure(self, connector):
        """Test failed connection test."""
        with patch.object(connector, "_connect") as mock_connect:
            mock_connect.side_effect = Exception("Connection refused")

            success, message = await connector.test_connection()

            assert success is False
            assert "Connection refused" in message

    # ===================
    # Checksum Tests
    # ===================

    def test_calculate_checksum(self, connector, temp_storage):
        """Test SHA-256 checksum calculation."""
        test_content = b"This is test content for checksum"
        test_file = temp_storage / "test_file.txt"
        test_file.write_bytes(test_content)

        checksum = connector._calculate_checksum(str(test_file))

        # Known SHA-256 hash
        import hashlib
        expected = hashlib.sha256(test_content).hexdigest()
        assert checksum == expected

    # ===================
    # Post-Processing Tests
    # ===================

    @pytest.mark.asyncio
    async def test_post_process_delete(self, temp_storage):
        """Test delete post-processing."""
        config = SFTPConfig(
            host="sftp.test.com",
            username="user",
            post_action=SFTPPostAction.DELETE,
        )
        connector = SFTPConnector(config=config, storage_path=temp_storage)

        mock_sftp = MagicMock()
        connector._sftp = mock_sftp

        file_info = SFTPFile(
            remote_path="/incoming/document.pdf",
            filename="document.pdf",
            size=1024,
            mtime=datetime.utcnow(),
        )

        await connector._post_process(file_info)

        mock_sftp.remove.assert_called_once_with("/incoming/document.pdf")

    @pytest.mark.asyncio
    async def test_post_process_move(self, connector):
        """Test move post-processing."""
        mock_sftp = MagicMock()
        connector._sftp = mock_sftp

        file_info = SFTPFile(
            remote_path="/incoming/document.pdf",
            filename="document.pdf",
            size=1024,
            mtime=datetime.utcnow(),
        )

        await connector._post_process(file_info)

        mock_sftp.rename.assert_called_once_with(
            "/incoming/document.pdf",
            "/processed/document.pdf",
        )

    @pytest.mark.asyncio
    async def test_post_process_rename(self, temp_storage):
        """Test rename post-processing."""
        config = SFTPConfig(
            host="sftp.test.com",
            username="user",
            post_action=SFTPPostAction.RENAME,
            processed_suffix=".done",
        )
        connector = SFTPConnector(config=config, storage_path=temp_storage)

        mock_sftp = MagicMock()
        connector._sftp = mock_sftp

        file_info = SFTPFile(
            remote_path="/incoming/document.pdf",
            filename="document.pdf",
            size=1024,
            mtime=datetime.utcnow(),
        )

        await connector._post_process(file_info)

        mock_sftp.rename.assert_called_once_with(
            "/incoming/document.pdf",
            "/incoming/document.pdf.done",
        )

    @pytest.mark.asyncio
    async def test_post_process_none(self, temp_storage):
        """Test no post-processing."""
        config = SFTPConfig(
            host="sftp.test.com",
            username="user",
            post_action=SFTPPostAction.NONE,
        )
        connector = SFTPConnector(config=config, storage_path=temp_storage)

        mock_sftp = MagicMock()
        connector._sftp = mock_sftp

        file_info = SFTPFile(
            remote_path="/incoming/document.pdf",
            filename="document.pdf",
            size=1024,
            mtime=datetime.utcnow(),
        )

        await connector._post_process(file_info)

        mock_sftp.remove.assert_not_called()
        mock_sftp.rename.assert_not_called()

    # ===================
    # Error Handling Tests
    # ===================

    @pytest.mark.asyncio
    async def test_handle_error(self, connector):
        """Test error handling moves file to error folder."""
        mock_sftp = MagicMock()
        connector._sftp = mock_sftp

        file_info = SFTPFile(
            remote_path="/incoming/bad_file.pdf",
            filename="bad_file.pdf",
            size=1024,
            mtime=datetime.utcnow(),
        )

        await connector._handle_error(file_info)

        # Should try to create error directory and move file
        mock_sftp.rename.assert_called_once_with(
            "/incoming/bad_file.pdf",
            "/errors/bad_file.pdf",
        )

        # File should be marked as processed to avoid retry loop
        assert "/incoming/bad_file.pdf" in connector._processed_files

    # ===================
    # Upload Tests
    # ===================

    @pytest.mark.asyncio
    async def test_upload_file(self, connector, temp_storage):
        """Test file upload to SFTP."""
        local_file = temp_storage / "upload_test.pdf"
        local_file.write_bytes(b"%PDF-1.4\nTest content")

        with patch.object(connector, "_connect"):
            mock_sftp = MagicMock()
            connector._sftp = mock_sftp

            result = await connector.upload_file(local_file, "/remote/upload_test.pdf")

            assert result is True
            mock_sftp.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_file_failure(self, connector, temp_storage):
        """Test file upload failure handling."""
        local_file = temp_storage / "upload_test.pdf"
        local_file.write_bytes(b"%PDF-1.4\nTest content")

        with patch.object(connector, "_connect"):
            mock_sftp = MagicMock()
            mock_sftp.put.side_effect = Exception("Upload failed")
            connector._sftp = mock_sftp

            result = await connector.upload_file(local_file, "/remote/upload_test.pdf")

            assert result is False

    # ===================
    # Lifecycle Tests
    # ===================

    @pytest.mark.asyncio
    async def test_stop(self, connector):
        """Test connector stop."""
        connector._running = True

        with patch.object(connector, "_disconnect") as mock_disconnect:
            await connector.stop()

        assert connector._running is False
        mock_disconnect.assert_called_once()

    # ===================
    # List Files Tests
    # ===================

    @pytest.mark.asyncio
    async def test_list_files(self, connector):
        """Test listing remote files."""
        with patch.object(connector, "_connect"):
            mock_sftp = MagicMock()

            mock_entry = MagicMock()
            mock_entry.filename = "document.pdf"
            mock_entry.st_mode = stat.S_IFREG
            mock_entry.st_size = 1024
            mock_entry.st_mtime = (datetime.utcnow() - timedelta(minutes=5)).timestamp()

            mock_sftp.listdir_attr.return_value = [mock_entry]
            connector._sftp = mock_sftp

            files = await connector.list_files("/custom_path")

            assert len(files) == 1
            assert files[0].filename == "document.pdf"
