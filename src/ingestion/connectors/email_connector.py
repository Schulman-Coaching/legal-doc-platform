"""
Email Ingestion Connector
=========================
Monitors email mailboxes for incoming legal documents via IMAP/POP3.
Supports attachment extraction, email threading, and automatic classification.
"""

import asyncio
import email
import email.policy
import imaplib
import logging
import poplib
import re
import ssl
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import aiofiles
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EmailProtocol(str, Enum):
    """Supported email protocols."""
    IMAP = "imap"
    IMAP_SSL = "imap_ssl"
    POP3 = "pop3"
    POP3_SSL = "pop3_ssl"


class EmailProcessingAction(str, Enum):
    """Actions to take after processing an email."""
    MARK_READ = "mark_read"
    MOVE_TO_FOLDER = "move_to_folder"
    DELETE = "delete"
    ARCHIVE = "archive"


@dataclass
class EmailConfig:
    """Configuration for email ingestion connector."""
    host: str
    port: int
    protocol: EmailProtocol
    username: str
    password: str
    mailbox: str = "INBOX"
    poll_interval_seconds: int = 60
    # Folders for processed emails
    processed_folder: str = "Processed"
    error_folder: str = "ProcessingErrors"
    # Processing options
    post_process_action: EmailProcessingAction = EmailProcessingAction.MOVE_TO_FOLDER
    extract_attachments: bool = True
    include_email_body: bool = True
    # Filtering
    subject_filters: list[str] = field(default_factory=list)
    sender_whitelist: list[str] = field(default_factory=list)
    sender_blacklist: list[str] = field(default_factory=list)
    # Legal-specific settings
    detect_matter_from_subject: bool = True
    matter_pattern: str = r"(?:Matter|Case|File)[:\s#]+(\w+-?\d+)"
    client_pattern: str = r"(?:Client|Account)[:\s#]+(\w+)"
    # SSL/TLS settings
    ssl_verify: bool = True
    ssl_cert_path: Optional[str] = None


@dataclass
class ExtractedEmail:
    """Represents an extracted email with metadata."""
    message_id: str
    subject: str
    sender: str
    recipients: list[str]
    cc: list[str]
    date: datetime
    body_text: str
    body_html: str
    thread_id: Optional[str]
    in_reply_to: Optional[str]
    references: list[str]
    headers: dict[str, str]
    attachments: list[dict[str, Any]]
    raw_size: int
    # Extracted legal metadata
    detected_matter_id: Optional[str] = None
    detected_client_id: Optional[str] = None


class EmailAttachment(BaseModel):
    """Model for email attachments."""
    filename: str
    content_type: str
    size: int
    content_id: Optional[str] = None
    is_inline: bool = False
    data: bytes = Field(exclude=True)


class EmailConnector:
    """
    Email ingestion connector supporting IMAP and POP3.

    Features:
    - Automatic mailbox polling
    - Attachment extraction
    - Email threading detection
    - Legal matter/client ID extraction from subject
    - Configurable post-processing actions
    """

    def __init__(
        self,
        config: EmailConfig,
        storage_path: Path,
        callback: Optional[callable] = None,
    ):
        self.config = config
        self.storage_path = storage_path / "email_ingestion"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.callback = callback
        self._running = False
        self._connection = None

        # Compile patterns for legal metadata extraction
        self._matter_pattern = re.compile(config.matter_pattern, re.IGNORECASE)
        self._client_pattern = re.compile(config.client_pattern, re.IGNORECASE)

    async def start(self) -> None:
        """Start the email polling service."""
        self._running = True
        logger.info(f"Starting email connector for {self.config.host}")

        while self._running:
            try:
                await self._poll_mailbox()
            except Exception as e:
                logger.error(f"Error polling mailbox: {e}")

            await asyncio.sleep(self.config.poll_interval_seconds)

    async def stop(self) -> None:
        """Stop the email polling service."""
        self._running = False
        if self._connection:
            try:
                if isinstance(self._connection, imaplib.IMAP4):
                    self._connection.logout()
                elif isinstance(self._connection, poplib.POP3):
                    self._connection.quit()
            except Exception:
                pass
        logger.info("Email connector stopped")

    def _connect(self) -> None:
        """Establish connection to mail server."""
        ssl_context = None
        if self.config.ssl_verify:
            ssl_context = ssl.create_default_context()
            if self.config.ssl_cert_path:
                ssl_context.load_verify_locations(self.config.ssl_cert_path)

        if self.config.protocol == EmailProtocol.IMAP_SSL:
            self._connection = imaplib.IMAP4_SSL(
                self.config.host,
                self.config.port,
                ssl_context=ssl_context,
            )
            self._connection.login(self.config.username, self.config.password)

        elif self.config.protocol == EmailProtocol.IMAP:
            self._connection = imaplib.IMAP4(self.config.host, self.config.port)
            self._connection.starttls(ssl_context=ssl_context)
            self._connection.login(self.config.username, self.config.password)

        elif self.config.protocol == EmailProtocol.POP3_SSL:
            self._connection = poplib.POP3_SSL(
                self.config.host,
                self.config.port,
                context=ssl_context,
            )
            self._connection.user(self.config.username)
            self._connection.pass_(self.config.password)

        elif self.config.protocol == EmailProtocol.POP3:
            self._connection = poplib.POP3(self.config.host, self.config.port)
            self._connection.stls(context=ssl_context)
            self._connection.user(self.config.username)
            self._connection.pass_(self.config.password)

        logger.info(f"Connected to {self.config.host}")

    async def _poll_mailbox(self) -> None:
        """Poll mailbox for new messages."""
        # Run synchronous email operations in executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_poll_mailbox)

    def _sync_poll_mailbox(self) -> None:
        """Synchronous mailbox polling."""
        try:
            self._connect()

            if isinstance(self._connection, imaplib.IMAP4):
                self._poll_imap()
            else:
                self._poll_pop3()

        finally:
            if self._connection:
                try:
                    if isinstance(self._connection, imaplib.IMAP4):
                        self._connection.logout()
                    else:
                        self._connection.quit()
                except Exception:
                    pass
                self._connection = None

    def _poll_imap(self) -> None:
        """Poll IMAP mailbox."""
        self._connection.select(self.config.mailbox)

        # Search for unread messages
        status, message_ids = self._connection.search(None, "UNSEEN")

        if status != "OK":
            logger.error("Failed to search mailbox")
            return

        ids = message_ids[0].split()
        logger.info(f"Found {len(ids)} unread messages")

        for msg_id in ids:
            try:
                self._process_imap_message(msg_id)
            except Exception as e:
                logger.error(f"Error processing message {msg_id}: {e}")
                self._move_to_error_folder_imap(msg_id)

    def _poll_pop3(self) -> None:
        """Poll POP3 mailbox."""
        num_messages = len(self._connection.list()[1])
        logger.info(f"Found {num_messages} messages")

        for i in range(1, num_messages + 1):
            try:
                self._process_pop3_message(i)
            except Exception as e:
                logger.error(f"Error processing message {i}: {e}")

    def _process_imap_message(self, msg_id: bytes) -> None:
        """Process a single IMAP message."""
        # Fetch message
        status, data = self._connection.fetch(msg_id, "(RFC822)")
        if status != "OK":
            return

        raw_email = data[0][1]
        msg = email.message_from_bytes(raw_email, policy=email.policy.default)

        # Check filters
        if not self._passes_filters(msg):
            logger.debug(f"Message {msg_id} filtered out")
            return

        # Extract email data
        extracted = self._extract_email(msg, len(raw_email))

        # Process attachments
        if self.config.extract_attachments:
            self._save_attachments(extracted)

        # Call callback if provided
        if self.callback:
            self.callback(extracted)

        # Post-processing
        self._post_process_imap(msg_id)

    def _process_pop3_message(self, msg_num: int) -> None:
        """Process a single POP3 message."""
        # Fetch message
        response = self._connection.retr(msg_num)
        raw_email = b"\n".join(response[1])
        msg = email.message_from_bytes(raw_email, policy=email.policy.default)

        # Check filters
        if not self._passes_filters(msg):
            return

        # Extract email data
        extracted = self._extract_email(msg, len(raw_email))

        # Process attachments
        if self.config.extract_attachments:
            self._save_attachments(extracted)

        # Call callback if provided
        if self.callback:
            self.callback(extracted)

        # For POP3, we can only delete messages
        if self.config.post_process_action == EmailProcessingAction.DELETE:
            self._connection.dele(msg_num)

    def _passes_filters(self, msg: EmailMessage) -> bool:
        """Check if message passes configured filters."""
        sender = msg.get("From", "")
        subject = msg.get("Subject", "")

        # Check sender whitelist
        if self.config.sender_whitelist:
            if not any(s.lower() in sender.lower() for s in self.config.sender_whitelist):
                return False

        # Check sender blacklist
        if self.config.sender_blacklist:
            if any(s.lower() in sender.lower() for s in self.config.sender_blacklist):
                return False

        # Check subject filters
        if self.config.subject_filters:
            if not any(f.lower() in subject.lower() for f in self.config.subject_filters):
                return False

        return True

    def _extract_email(self, msg: EmailMessage, raw_size: int) -> ExtractedEmail:
        """Extract structured data from email message."""
        # Parse recipients
        def parse_addresses(header: str) -> list[str]:
            addresses = msg.get_all(header, [])
            result = []
            for addr in addresses:
                if isinstance(addr, str):
                    result.append(addr)
            return result

        # Extract body
        body_text = ""
        body_html = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain" and not body_text:
                    body_text = part.get_content()
                elif content_type == "text/html" and not body_html:
                    body_html = part.get_content()
        else:
            content_type = msg.get_content_type()
            content = msg.get_content()
            if content_type == "text/plain":
                body_text = content
            elif content_type == "text/html":
                body_html = content

        # Extract attachments info
        attachments = []
        if msg.is_multipart():
            for part in msg.walk():
                content_disposition = part.get_content_disposition()
                if content_disposition in ["attachment", "inline"]:
                    filename = part.get_filename() or "unnamed"
                    attachments.append({
                        "filename": filename,
                        "content_type": part.get_content_type(),
                        "size": len(part.get_payload(decode=True) or b""),
                        "content_id": part.get("Content-ID"),
                        "is_inline": content_disposition == "inline",
                        "data": part.get_payload(decode=True),
                    })

        # Parse date
        date_str = msg.get("Date", "")
        try:
            parsed_date = email.utils.parsedate_to_datetime(date_str)
        except Exception:
            parsed_date = datetime.utcnow()

        # Build thread ID from references
        references = msg.get("References", "").split()
        in_reply_to = msg.get("In-Reply-To", "").strip()
        thread_id = references[0] if references else in_reply_to or msg.get("Message-ID")

        # Extract legal metadata from subject
        subject = msg.get("Subject", "")
        detected_matter = None
        detected_client = None

        if self.config.detect_matter_from_subject:
            matter_match = self._matter_pattern.search(subject)
            if matter_match:
                detected_matter = matter_match.group(1)

            client_match = self._client_pattern.search(subject)
            if client_match:
                detected_client = client_match.group(1)

        # Extract key headers
        headers = {
            "Message-ID": msg.get("Message-ID", ""),
            "From": msg.get("From", ""),
            "To": msg.get("To", ""),
            "Subject": subject,
            "Date": date_str,
            "Content-Type": msg.get_content_type(),
        }

        return ExtractedEmail(
            message_id=msg.get("Message-ID", str(uuid.uuid4())),
            subject=subject,
            sender=msg.get("From", ""),
            recipients=parse_addresses("To"),
            cc=parse_addresses("Cc"),
            date=parsed_date,
            body_text=body_text,
            body_html=body_html,
            thread_id=thread_id,
            in_reply_to=in_reply_to,
            references=references,
            headers=headers,
            attachments=attachments,
            raw_size=raw_size,
            detected_matter_id=detected_matter,
            detected_client_id=detected_client,
        )

    def _save_attachments(self, extracted: ExtractedEmail) -> list[Path]:
        """Save email attachments to storage."""
        saved_paths = []

        for attachment in extracted.attachments:
            if not attachment.get("data"):
                continue

            # Generate unique filename
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{unique_id}_{attachment['filename']}"
            filepath = self.storage_path / filename

            # Save attachment
            with open(filepath, "wb") as f:
                f.write(attachment["data"])

            saved_paths.append(filepath)
            logger.info(f"Saved attachment: {filename}")

        return saved_paths

    def _post_process_imap(self, msg_id: bytes) -> None:
        """Post-process IMAP message based on configuration."""
        if self.config.post_process_action == EmailProcessingAction.MARK_READ:
            self._connection.store(msg_id, "+FLAGS", "\\Seen")

        elif self.config.post_process_action == EmailProcessingAction.MOVE_TO_FOLDER:
            # Create folder if it doesn't exist
            self._connection.create(self.config.processed_folder)
            # Copy to processed folder
            self._connection.copy(msg_id, self.config.processed_folder)
            # Mark original for deletion
            self._connection.store(msg_id, "+FLAGS", "\\Deleted")
            self._connection.expunge()

        elif self.config.post_process_action == EmailProcessingAction.DELETE:
            self._connection.store(msg_id, "+FLAGS", "\\Deleted")
            self._connection.expunge()

        elif self.config.post_process_action == EmailProcessingAction.ARCHIVE:
            self._connection.create("Archive")
            self._connection.copy(msg_id, "Archive")
            self._connection.store(msg_id, "+FLAGS", "\\Deleted")
            self._connection.expunge()

    def _move_to_error_folder_imap(self, msg_id: bytes) -> None:
        """Move failed message to error folder."""
        try:
            self._connection.create(self.config.error_folder)
            self._connection.copy(msg_id, self.config.error_folder)
            self._connection.store(msg_id, "+FLAGS", "\\Deleted")
            self._connection.expunge()
        except Exception as e:
            logger.error(f"Failed to move message to error folder: {e}")

    async def test_connection(self) -> tuple[bool, str]:
        """Test connection to mail server."""
        try:
            loop = asyncio.get_event_loop()

            def _test():
                self._connect()
                if isinstance(self._connection, imaplib.IMAP4):
                    status, count = self._connection.select(self.config.mailbox)
                    self._connection.logout()
                    return True, f"Connected. Mailbox has {count[0].decode()} messages"
                else:
                    count = len(self._connection.list()[1])
                    self._connection.quit()
                    return True, f"Connected. Mailbox has {count} messages"

            return await loop.run_in_executor(None, _test)

        except Exception as e:
            return False, str(e)
