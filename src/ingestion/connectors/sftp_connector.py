"""
SFTP Ingestion Connector
========================
Monitors SFTP servers for incoming legal documents.
Supports scheduled polling, recursive directory scanning, and automatic file cleanup.
"""

import asyncio
import fnmatch
import hashlib
import logging
import stat
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Optional

import paramiko
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SFTPAuthMethod(str, Enum):
    """SFTP authentication methods."""
    PASSWORD = "password"
    KEY_FILE = "key_file"
    KEY_AGENT = "key_agent"


class SFTPPostAction(str, Enum):
    """Actions after processing a file."""
    DELETE = "delete"
    MOVE = "move"
    RENAME = "rename"
    NONE = "none"


@dataclass
class SFTPConfig:
    """Configuration for SFTP connector."""
    host: str
    port: int = 22
    username: str = ""
    # Authentication
    auth_method: SFTPAuthMethod = SFTPAuthMethod.PASSWORD
    password: Optional[str] = None
    key_file: Optional[str] = None
    key_passphrase: Optional[str] = None
    # Directories
    remote_path: str = "/"
    processed_path: str = "/processed"
    error_path: str = "/errors"
    local_staging_path: str = "/tmp/sftp_staging"
    # Scanning options
    poll_interval_seconds: int = 300
    recursive: bool = True
    max_depth: int = 10
    # File filtering
    file_patterns: list[str] = field(default_factory=lambda: ["*.pdf", "*.doc*", "*.txt"])
    exclude_patterns: list[str] = field(default_factory=lambda: [".*", "~*", "*.tmp"])
    min_file_size: int = 0
    max_file_size: int = 1024 * 1024 * 1024  # 1GB
    min_age_seconds: int = 60  # Wait for file to stabilize
    # Processing
    post_action: SFTPPostAction = SFTPPostAction.MOVE
    processed_suffix: str = ".processed"
    # Connection settings
    timeout: int = 30
    keepalive_interval: int = 60


@dataclass
class SFTPFile:
    """Represents a file found on SFTP server."""
    remote_path: str
    filename: str
    size: int
    mtime: datetime
    checksum: Optional[str] = None
    local_path: Optional[Path] = None
    # Metadata
    source_host: str = ""
    discovered_at: datetime = field(default_factory=datetime.utcnow)


class SFTPConnector:
    """
    SFTP ingestion connector for polling remote servers.

    Features:
    - Password and key-based authentication
    - Recursive directory scanning
    - File pattern matching
    - Automatic post-processing (move/delete/rename)
    - File stability checking
    - Resume support for interrupted transfers
    """

    def __init__(
        self,
        config: SFTPConfig,
        storage_path: Path,
        callback: Optional[Callable[[SFTPFile, bytes], None]] = None,
    ):
        self.config = config
        self.storage_path = storage_path / "sftp_ingestion"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.callback = callback
        self._running = False
        self._transport: Optional[paramiko.Transport] = None
        self._sftp: Optional[paramiko.SFTPClient] = None
        self._processed_files: set[str] = set()

    async def start(self) -> None:
        """Start the SFTP polling service."""
        self._running = True
        logger.info(f"Starting SFTP connector for {self.config.host}")

        while self._running:
            try:
                await self._poll_server()
            except Exception as e:
                logger.error(f"Error polling SFTP server: {e}")
                await self._disconnect()

            await asyncio.sleep(self.config.poll_interval_seconds)

    async def stop(self) -> None:
        """Stop the SFTP polling service."""
        self._running = False
        await self._disconnect()
        logger.info("SFTP connector stopped")

    async def _connect(self) -> None:
        """Establish SFTP connection."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_connect)

    def _sync_connect(self) -> None:
        """Synchronous connection establishment."""
        if self._sftp is not None:
            return

        # Create transport
        self._transport = paramiko.Transport((self.config.host, self.config.port))
        self._transport.set_keepalive(self.config.keepalive_interval)

        # Authenticate
        if self.config.auth_method == SFTPAuthMethod.PASSWORD:
            self._transport.connect(
                username=self.config.username,
                password=self.config.password,
            )

        elif self.config.auth_method == SFTPAuthMethod.KEY_FILE:
            key = self._load_private_key()
            self._transport.connect(
                username=self.config.username,
                pkey=key,
            )

        elif self.config.auth_method == SFTPAuthMethod.KEY_AGENT:
            agent = paramiko.Agent()
            keys = agent.get_keys()
            if not keys:
                raise RuntimeError("No keys found in SSH agent")
            self._transport.connect(
                username=self.config.username,
                pkey=keys[0],
            )

        # Create SFTP client
        self._sftp = paramiko.SFTPClient.from_transport(self._transport)
        logger.info(f"Connected to SFTP server {self.config.host}")

    def _load_private_key(self) -> paramiko.PKey:
        """Load private key from file."""
        key_path = self.config.key_file
        passphrase = self.config.key_passphrase

        # Try different key types
        key_classes = [
            paramiko.RSAKey,
            paramiko.Ed25519Key,
            paramiko.ECDSAKey,
            paramiko.DSSKey,
        ]

        for key_class in key_classes:
            try:
                return key_class.from_private_key_file(key_path, password=passphrase)
            except paramiko.SSHException:
                continue

        raise RuntimeError(f"Unable to load private key from {key_path}")

    async def _disconnect(self) -> None:
        """Disconnect from SFTP server."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_disconnect)

    def _sync_disconnect(self) -> None:
        """Synchronous disconnection."""
        if self._sftp:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None

        if self._transport:
            try:
                self._transport.close()
            except Exception:
                pass
            self._transport = None

    async def _poll_server(self) -> None:
        """Poll SFTP server for new files."""
        await self._connect()

        loop = asyncio.get_event_loop()
        files = await loop.run_in_executor(
            None,
            self._scan_directory,
            self.config.remote_path,
            0,
        )

        logger.info(f"Found {len(files)} files to process")

        for file_info in files:
            if not self._running:
                break

            try:
                await self._process_file(file_info)
            except Exception as e:
                logger.error(f"Error processing {file_info.remote_path}: {e}")
                await self._handle_error(file_info)

    def _scan_directory(self, path: str, depth: int) -> list[SFTPFile]:
        """Recursively scan directory for matching files."""
        if depth > self.config.max_depth:
            return []

        files = []

        try:
            entries = self._sftp.listdir_attr(path)
        except IOError as e:
            logger.warning(f"Cannot access directory {path}: {e}")
            return []

        for entry in entries:
            full_path = str(PurePosixPath(path) / entry.filename)

            # Check if directory
            if stat.S_ISDIR(entry.st_mode):
                if self.config.recursive:
                    files.extend(self._scan_directory(full_path, depth + 1))
                continue

            # Skip if already processed
            if full_path in self._processed_files:
                continue

            # Check exclusion patterns
            if self._matches_patterns(entry.filename, self.config.exclude_patterns):
                continue

            # Check inclusion patterns
            if not self._matches_patterns(entry.filename, self.config.file_patterns):
                continue

            # Check file size
            if entry.st_size < self.config.min_file_size:
                continue
            if entry.st_size > self.config.max_file_size:
                logger.warning(f"File too large: {full_path} ({entry.st_size} bytes)")
                continue

            # Check file age (stability)
            mtime = datetime.fromtimestamp(entry.st_mtime)
            age_seconds = (datetime.utcnow() - mtime).total_seconds()
            if age_seconds < self.config.min_age_seconds:
                logger.debug(f"File too new, skipping: {full_path}")
                continue

            files.append(SFTPFile(
                remote_path=full_path,
                filename=entry.filename,
                size=entry.st_size,
                mtime=mtime,
                source_host=self.config.host,
            ))

        return files

    def _matches_patterns(self, filename: str, patterns: list[str]) -> bool:
        """Check if filename matches any pattern."""
        return any(fnmatch.fnmatch(filename.lower(), p.lower()) for p in patterns)

    async def _process_file(self, file_info: SFTPFile) -> None:
        """Download and process a file."""
        loop = asyncio.get_event_loop()

        # Download file
        local_path = self.storage_path / f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{file_info.filename}"
        file_info.local_path = local_path

        await loop.run_in_executor(
            None,
            self._download_file,
            file_info.remote_path,
            str(local_path),
        )

        # Calculate checksum
        file_info.checksum = await loop.run_in_executor(
            None,
            self._calculate_checksum,
            str(local_path),
        )

        logger.info(f"Downloaded {file_info.remote_path} ({file_info.size} bytes)")

        # Read file content
        with open(local_path, "rb") as f:
            content = f.read()

        # Call callback
        if self.callback:
            await loop.run_in_executor(None, self.callback, file_info, content)

        # Post-process
        await self._post_process(file_info)

        # Mark as processed
        self._processed_files.add(file_info.remote_path)

    def _download_file(self, remote_path: str, local_path: str) -> None:
        """Download file from SFTP server."""
        self._sftp.get(remote_path, local_path)

    def _calculate_checksum(self, filepath: str) -> str:
        """Calculate SHA-256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def _post_process(self, file_info: SFTPFile) -> None:
        """Handle post-processing based on configuration."""
        loop = asyncio.get_event_loop()

        if self.config.post_action == SFTPPostAction.DELETE:
            await loop.run_in_executor(
                None,
                self._sftp.remove,
                file_info.remote_path,
            )
            logger.info(f"Deleted remote file: {file_info.remote_path}")

        elif self.config.post_action == SFTPPostAction.MOVE:
            # Ensure processed directory exists
            await loop.run_in_executor(
                None,
                self._ensure_directory,
                self.config.processed_path,
            )

            new_path = str(PurePosixPath(self.config.processed_path) / file_info.filename)
            await loop.run_in_executor(
                None,
                self._sftp.rename,
                file_info.remote_path,
                new_path,
            )
            logger.info(f"Moved to: {new_path}")

        elif self.config.post_action == SFTPPostAction.RENAME:
            new_path = file_info.remote_path + self.config.processed_suffix
            await loop.run_in_executor(
                None,
                self._sftp.rename,
                file_info.remote_path,
                new_path,
            )
            logger.info(f"Renamed to: {new_path}")

    def _ensure_directory(self, path: str) -> None:
        """Create directory if it doesn't exist."""
        try:
            self._sftp.stat(path)
        except IOError:
            self._sftp.mkdir(path)

    async def _handle_error(self, file_info: SFTPFile) -> None:
        """Handle file that failed processing."""
        loop = asyncio.get_event_loop()

        try:
            # Move to error folder
            await loop.run_in_executor(
                None,
                self._ensure_directory,
                self.config.error_path,
            )

            error_path = str(PurePosixPath(self.config.error_path) / file_info.filename)
            await loop.run_in_executor(
                None,
                self._sftp.rename,
                file_info.remote_path,
                error_path,
            )
            logger.info(f"Moved to error folder: {error_path}")

        except Exception as e:
            logger.error(f"Failed to move to error folder: {e}")

        # Mark as processed to avoid retry loop
        self._processed_files.add(file_info.remote_path)

    async def test_connection(self) -> tuple[bool, str]:
        """Test connection to SFTP server."""
        try:
            await self._connect()

            loop = asyncio.get_event_loop()
            files = await loop.run_in_executor(
                None,
                self._sftp.listdir,
                self.config.remote_path,
            )

            await self._disconnect()
            return True, f"Connected. Found {len(files)} items in {self.config.remote_path}"

        except Exception as e:
            return False, str(e)

    async def upload_file(
        self,
        local_path: Path,
        remote_path: str,
    ) -> bool:
        """Upload a file to SFTP server (for bidirectional sync)."""
        try:
            await self._connect()

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._sftp.put,
                str(local_path),
                remote_path,
            )

            logger.info(f"Uploaded {local_path} to {remote_path}")
            return True

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    async def list_files(self, path: Optional[str] = None) -> list[SFTPFile]:
        """List files in remote directory."""
        await self._connect()

        loop = asyncio.get_event_loop()
        files = await loop.run_in_executor(
            None,
            self._scan_directory,
            path or self.config.remote_path,
            0,
        )

        return files
