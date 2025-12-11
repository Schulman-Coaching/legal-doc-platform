"""
Tests for Email Connector
=========================
Comprehensive tests for email ingestion connector including
IMAP/POP3 support, filtering, and attachment extraction.
"""

import email
import pytest
import tempfile
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from ..connectors.email_connector import (
    EmailConnector,
    EmailConfig,
    EmailProtocol,
    EmailProcessingAction,
    ExtractedEmail,
    EmailAttachment,
)


class TestEmailConfig:
    """Test suite for EmailConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EmailConfig(
            host="mail.example.com",
            port=993,
            protocol=EmailProtocol.IMAP_SSL,
            username="user@example.com",
            password="secret",
        )

        assert config.mailbox == "INBOX"
        assert config.poll_interval_seconds == 60
        assert config.processed_folder == "Processed"
        assert config.error_folder == "ProcessingErrors"
        assert config.post_process_action == EmailProcessingAction.MOVE_TO_FOLDER
        assert config.extract_attachments is True
        assert config.include_email_body is True
        assert config.detect_matter_from_subject is True
        assert config.ssl_verify is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EmailConfig(
            host="imap.custom.com",
            port=143,
            protocol=EmailProtocol.IMAP,
            username="legal@firm.com",
            password="password123",
            mailbox="Legal/Incoming",
            poll_interval_seconds=120,
            post_process_action=EmailProcessingAction.DELETE,
            sender_whitelist=["trusted@client.com"],
            subject_filters=["Case:", "Matter:"],
        )

        assert config.mailbox == "Legal/Incoming"
        assert config.poll_interval_seconds == 120
        assert config.post_process_action == EmailProcessingAction.DELETE
        assert "trusted@client.com" in config.sender_whitelist
        assert "Case:" in config.subject_filters


class TestEmailConnector:
    """Test suite for EmailConnector."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def email_config(self):
        """Create test email configuration."""
        return EmailConfig(
            host="mail.test.com",
            port=993,
            protocol=EmailProtocol.IMAP_SSL,
            username="test@test.com",
            password="testpassword",
            subject_filters=["Legal:", "Case:"],
            detect_matter_from_subject=True,
            matter_pattern=r"(?:Matter|Case)[:\s#]+(\w+-?\d+)",
            client_pattern=r"(?:Client|Account)[:\s#]+(\w+)",
        )

    @pytest.fixture
    def connector(self, email_config, temp_storage):
        """Create EmailConnector instance."""
        return EmailConnector(
            config=email_config,
            storage_path=temp_storage,
        )

    @pytest.fixture
    def sample_email_message(self):
        """Create a sample email message."""
        msg = EmailMessage()
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"
        msg["Cc"] = "cc@example.com"
        msg["Subject"] = "Legal: Case#2024-001 - Contract Review"
        msg["Date"] = "Thu, 15 Feb 2024 10:30:00 +0000"
        msg["Message-ID"] = "<unique123@example.com>"
        msg.set_content("This is the email body text.")
        return msg

    @pytest.fixture
    def email_with_attachment(self):
        """Create email with attachment."""
        msg = EmailMessage()
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"
        msg["Subject"] = "Legal: Documents for Matter#M-2024"
        msg["Date"] = "Thu, 15 Feb 2024 10:30:00 +0000"
        msg["Message-ID"] = "<attach123@example.com>"

        msg.set_content("Please find attached documents.")

        # Add PDF attachment
        msg.add_attachment(
            b"%PDF-1.4\nSample PDF content",
            maintype="application",
            subtype="pdf",
            filename="contract.pdf",
        )

        return msg

    # ===================
    # Filter Tests
    # ===================

    def test_passes_filters_subject_match(self, connector, sample_email_message):
        """Test filter passes with matching subject."""
        result = connector._passes_filters(sample_email_message)
        assert result is True

    def test_passes_filters_subject_no_match(self, connector):
        """Test filter fails with non-matching subject."""
        msg = EmailMessage()
        msg["From"] = "sender@example.com"
        msg["Subject"] = "Random email without keywords"

        result = connector._passes_filters(msg)
        assert result is False

    def test_passes_filters_sender_whitelist(self, temp_storage):
        """Test sender whitelist filtering."""
        config = EmailConfig(
            host="mail.test.com",
            port=993,
            protocol=EmailProtocol.IMAP_SSL,
            username="test@test.com",
            password="test",
            sender_whitelist=["trusted@example.com", "allowed@firm.com"],
            subject_filters=[],  # No subject filters
        )
        connector = EmailConnector(config=config, storage_path=temp_storage)

        # Whitelisted sender
        msg_allowed = EmailMessage()
        msg_allowed["From"] = "trusted@example.com"
        msg_allowed["Subject"] = "Any subject"
        assert connector._passes_filters(msg_allowed) is True

        # Non-whitelisted sender
        msg_blocked = EmailMessage()
        msg_blocked["From"] = "random@unknown.com"
        msg_blocked["Subject"] = "Any subject"
        assert connector._passes_filters(msg_blocked) is False

    def test_passes_filters_sender_blacklist(self, temp_storage):
        """Test sender blacklist filtering."""
        config = EmailConfig(
            host="mail.test.com",
            port=993,
            protocol=EmailProtocol.IMAP_SSL,
            username="test@test.com",
            password="test",
            sender_blacklist=["spam@bad.com", "blocked@example.com"],
            subject_filters=[],  # No subject filters
        )
        connector = EmailConnector(config=config, storage_path=temp_storage)

        # Blacklisted sender
        msg_blocked = EmailMessage()
        msg_blocked["From"] = "spam@bad.com"
        msg_blocked["Subject"] = "Legal: Important"
        assert connector._passes_filters(msg_blocked) is False

        # Non-blacklisted sender
        msg_allowed = EmailMessage()
        msg_allowed["From"] = "good@example.com"
        msg_allowed["Subject"] = "Legal: Important"
        assert connector._passes_filters(msg_allowed) is True

    def test_passes_filters_case_insensitive(self, connector):
        """Test filters are case insensitive."""
        msg = EmailMessage()
        msg["From"] = "sender@example.com"
        msg["Subject"] = "LEGAL: Important Document"

        result = connector._passes_filters(msg)
        assert result is True

    # ===================
    # Email Extraction Tests
    # ===================

    def test_extract_email_basic(self, connector, sample_email_message):
        """Test basic email extraction."""
        extracted = connector._extract_email(sample_email_message, 1000)

        assert extracted.subject == "Legal: Case#2024-001 - Contract Review"
        assert extracted.sender == "sender@example.com"
        assert "recipient@example.com" in extracted.recipients
        assert "cc@example.com" in extracted.cc
        assert extracted.message_id == "<unique123@example.com>"
        assert extracted.body_text == "This is the email body text.\n"
        assert extracted.raw_size == 1000

    def test_extract_email_matter_detection(self, connector, sample_email_message):
        """Test legal matter ID detection from subject."""
        extracted = connector._extract_email(sample_email_message, 1000)

        # Should detect "2024-001" from "Case#2024-001"
        assert extracted.detected_matter_id == "2024-001"

    def test_extract_email_client_detection(self, connector, temp_storage):
        """Test client ID detection from subject."""
        config = EmailConfig(
            host="mail.test.com",
            port=993,
            protocol=EmailProtocol.IMAP_SSL,
            username="test@test.com",
            password="test",
            detect_matter_from_subject=True,
            matter_pattern=r"(?:Matter|Case)[:\s#]+(\w+[-\d]*)",
            client_pattern=r"(?:Client|Account)[:\s#]+(\w+)",
        )
        connector = EmailConnector(config=config, storage_path=temp_storage)

        msg = EmailMessage()
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"
        msg["Subject"] = "Re: Client: ACME123 - Matter: M-2024-500"
        msg["Date"] = "Thu, 15 Feb 2024 10:30:00 +0000"
        msg["Message-ID"] = "<test@example.com>"
        msg.set_content("Body text")

        extracted = connector._extract_email(msg, 500)

        assert extracted.detected_client_id == "ACME123"
        # The matter pattern captures alphanumeric with optional hyphens and digits
        assert extracted.detected_matter_id is not None
        assert "M-2024" in extracted.detected_matter_id

    def test_extract_email_with_attachments(self, connector, email_with_attachment):
        """Test email extraction with attachments."""
        extracted = connector._extract_email(email_with_attachment, 2000)

        assert len(extracted.attachments) == 1
        attachment = extracted.attachments[0]
        assert attachment["filename"] == "contract.pdf"
        assert attachment["content_type"] == "application/pdf"
        assert attachment["data"] is not None

    def test_extract_email_threading(self, connector):
        """Test email threading detection."""
        msg = EmailMessage()
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"
        msg["Subject"] = "Re: Legal: Original Thread"
        msg["Date"] = "Thu, 15 Feb 2024 10:30:00 +0000"
        msg["Message-ID"] = "<reply123@example.com>"
        msg["In-Reply-To"] = "<original123@example.com>"
        msg["References"] = "<thread001@example.com> <original123@example.com>"
        msg.set_content("Reply body")

        extracted = connector._extract_email(msg, 500)

        assert extracted.in_reply_to == "<original123@example.com>"
        assert "<thread001@example.com>" in extracted.references
        assert extracted.thread_id == "<thread001@example.com>"

    def test_extract_email_multipart(self, connector):
        """Test extraction of multipart email with HTML."""
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        msg = MIMEMultipart("alternative")
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"
        msg["Subject"] = "Legal: HTML Email"
        msg["Date"] = "Thu, 15 Feb 2024 10:30:00 +0000"
        msg["Message-ID"] = "<html123@example.com>"

        # Add plain text part
        text_part = MIMEText("Plain text version", "plain")
        msg.attach(text_part)

        # Add HTML part
        html_part = MIMEText("<html><body><p>HTML version</p></body></html>", "html")
        msg.attach(html_part)

        # Convert to EmailMessage for parsing
        import email
        raw = msg.as_bytes()
        parsed_msg = email.message_from_bytes(raw, policy=email.policy.default)

        extracted = connector._extract_email(parsed_msg, 1000)

        # Either plain text or HTML should be extracted
        assert extracted.body_text or extracted.body_html

    # ===================
    # Attachment Handling Tests
    # ===================

    def test_save_attachments(self, connector, email_with_attachment):
        """Test saving attachments to storage."""
        extracted = connector._extract_email(email_with_attachment, 2000)
        saved_paths = connector._save_attachments(extracted)

        assert len(saved_paths) == 1
        assert saved_paths[0].exists()
        assert "contract.pdf" in str(saved_paths[0])

        # Verify content
        content = saved_paths[0].read_bytes()
        assert b"%PDF-1.4" in content

    def test_save_attachments_empty(self, connector, sample_email_message):
        """Test saving when no attachments present."""
        extracted = connector._extract_email(sample_email_message, 1000)
        saved_paths = connector._save_attachments(extracted)

        assert len(saved_paths) == 0

    def test_save_multiple_attachments(self, connector):
        """Test saving multiple attachments."""
        msg = EmailMessage()
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"
        msg["Subject"] = "Legal: Multiple Documents"
        msg["Date"] = "Thu, 15 Feb 2024 10:30:00 +0000"
        msg["Message-ID"] = "<multi@example.com>"
        msg.set_content("Multiple attachments follow.")

        # Add multiple attachments
        msg.add_attachment(b"%PDF-1.4\nFirst PDF", maintype="application", subtype="pdf", filename="doc1.pdf")
        msg.add_attachment(b"PK\x03\x04DOCX content", maintype="application",
                          subtype="vnd.openxmlformats-officedocument.wordprocessingml.document",
                          filename="doc2.docx")
        msg.add_attachment(b"Plain text content", maintype="text", subtype="plain", filename="notes.txt")

        extracted = connector._extract_email(msg, 3000)
        saved_paths = connector._save_attachments(extracted)

        assert len(saved_paths) == 3

    # ===================
    # Connection Tests (Mocked)
    # ===================

    @pytest.mark.asyncio
    async def test_test_connection_imap_ssl(self, connector):
        """Test IMAP SSL connection test."""
        with patch("imaplib.IMAP4_SSL") as mock_imap:
            mock_connection = MagicMock()
            mock_connection.login.return_value = ("OK", [])
            mock_connection.select.return_value = ("OK", [b"10"])
            mock_imap.return_value = mock_connection

            success, message = await connector.test_connection()

            assert success is True
            # The message format is "Connected. Mailbox has X messages"
            assert "Connected" in message
            mock_connection.login.assert_called_once()

    @pytest.mark.asyncio
    async def test_test_connection_failure(self, connector):
        """Test connection failure handling."""
        with patch("imaplib.IMAP4_SSL") as mock_imap:
            mock_imap.side_effect = Exception("Connection refused")

            success, message = await connector.test_connection()

            assert success is False
            assert "Connection refused" in message

    # ===================
    # Protocol Tests
    # ===================

    def test_connect_imap_ssl(self, temp_storage):
        """Test IMAP SSL connection setup."""
        config = EmailConfig(
            host="mail.test.com",
            port=993,
            protocol=EmailProtocol.IMAP_SSL,
            username="test@test.com",
            password="test",
        )
        connector = EmailConnector(config=config, storage_path=temp_storage)

        with patch("imaplib.IMAP4_SSL") as mock_imap:
            mock_connection = MagicMock()
            mock_imap.return_value = mock_connection

            connector._connect()

            mock_imap.assert_called_once()
            mock_connection.login.assert_called_with("test@test.com", "test")

    def test_connect_imap_starttls(self, temp_storage):
        """Test IMAP with STARTTLS connection."""
        config = EmailConfig(
            host="mail.test.com",
            port=143,
            protocol=EmailProtocol.IMAP,
            username="test@test.com",
            password="test",
        )
        connector = EmailConnector(config=config, storage_path=temp_storage)

        with patch("imaplib.IMAP4") as mock_imap:
            mock_connection = MagicMock()
            mock_imap.return_value = mock_connection

            connector._connect()

            mock_imap.assert_called_once()
            mock_connection.starttls.assert_called_once()
            mock_connection.login.assert_called_with("test@test.com", "test")

    def test_connect_pop3_ssl(self, temp_storage):
        """Test POP3 SSL connection."""
        config = EmailConfig(
            host="mail.test.com",
            port=995,
            protocol=EmailProtocol.POP3_SSL,
            username="test@test.com",
            password="test",
        )
        connector = EmailConnector(config=config, storage_path=temp_storage)

        with patch("poplib.POP3_SSL") as mock_pop3:
            mock_connection = MagicMock()
            mock_pop3.return_value = mock_connection

            connector._connect()

            mock_pop3.assert_called_once()
            mock_connection.user.assert_called_with("test@test.com")
            mock_connection.pass_.assert_called_with("test")

    # ===================
    # Lifecycle Tests
    # ===================

    @pytest.mark.asyncio
    async def test_stop(self, connector):
        """Test connector stop."""
        connector._running = True
        connector._connection = MagicMock()

        await connector.stop()

        assert connector._running is False

    # ===================
    # Headers Extraction Tests
    # ===================

    def test_extract_headers(self, connector, sample_email_message):
        """Test header extraction."""
        extracted = connector._extract_email(sample_email_message, 1000)

        assert "Message-ID" in extracted.headers
        assert "From" in extracted.headers
        assert "To" in extracted.headers
        assert "Subject" in extracted.headers
        assert "Date" in extracted.headers


class TestExtractedEmail:
    """Test suite for ExtractedEmail dataclass."""

    def test_extracted_email_creation(self):
        """Test ExtractedEmail creation."""
        extracted = ExtractedEmail(
            message_id="<test@example.com>",
            subject="Test Subject",
            sender="sender@example.com",
            recipients=["recipient@example.com"],
            cc=["cc@example.com"],
            date=datetime.utcnow(),
            body_text="Body text",
            body_html="<p>Body HTML</p>",
            thread_id="<thread@example.com>",
            in_reply_to="<original@example.com>",
            references=["<ref1@example.com>"],
            headers={"X-Custom": "value"},
            attachments=[],
            raw_size=1000,
            detected_matter_id="M-2024-001",
            detected_client_id="CLIENT123",
        )

        assert extracted.message_id == "<test@example.com>"
        assert extracted.detected_matter_id == "M-2024-001"
        assert extracted.detected_client_id == "CLIENT123"


class TestEmailAttachment:
    """Test suite for EmailAttachment model."""

    def test_email_attachment_creation(self):
        """Test EmailAttachment creation."""
        attachment = EmailAttachment(
            filename="document.pdf",
            content_type="application/pdf",
            size=1024,
            content_id="<cid:123>",
            is_inline=False,
            data=b"%PDF-1.4\ncontent",
        )

        assert attachment.filename == "document.pdf"
        assert attachment.content_type == "application/pdf"
        assert attachment.size == 1024
        assert attachment.is_inline is False

    def test_email_attachment_inline(self):
        """Test inline attachment."""
        attachment = EmailAttachment(
            filename="logo.png",
            content_type="image/png",
            size=500,
            content_id="<logo@example.com>",
            is_inline=True,
            data=b"\x89PNG\r\n\x1a\n",
        )

        assert attachment.is_inline is True
        assert attachment.content_id == "<logo@example.com>"
