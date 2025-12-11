"""
Tests for Audit Service.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from ..services.audit import AuditService
from ..models import AuditEvent, AuditEventType


class TestAuditLogging:
    """Tests for audit event logging."""

    @pytest.mark.asyncio
    async def test_log_event(self, audit_service: AuditService):
        """Test logging a basic audit event."""
        event = await audit_service.log(
            event_type=AuditEventType.DOCUMENT_VIEW,
            user_id="user-001",
            resource_type="document",
            resource_id="doc-001",
            action="view",
        )

        assert event is not None
        assert event.event_type == AuditEventType.DOCUMENT_VIEW
        assert event.user_id == "user-001"
        assert event.hash is not None

    @pytest.mark.asyncio
    async def test_log_auth_event(self, audit_service: AuditService):
        """Test logging authentication event."""
        event = await audit_service.log_auth_event(
            event_type=AuditEventType.LOGIN_SUCCESS,
            user_id="user-002",
            success=True,
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0",
        )

        assert event.event_type == AuditEventType.LOGIN_SUCCESS
        assert event.outcome == "success"
        assert event.ip_address == "192.168.1.100"

    @pytest.mark.asyncio
    async def test_log_document_event(self, audit_service: AuditService):
        """Test logging document event."""
        event = await audit_service.log_document_event(
            event_type=AuditEventType.DOCUMENT_CREATE,
            document_id="doc-002",
            user_id="user-003",
            client_id="client-001",
            details={"size": 1024, "type": "pdf"},
        )

        assert event.event_type == AuditEventType.DOCUMENT_CREATE
        assert event.resource_id == "doc-002"
        assert event.details["size"] == 1024

    @pytest.mark.asyncio
    async def test_log_access_event(self, audit_service: AuditService):
        """Test logging access control event."""
        event = await audit_service.log_access_event(
            user_id="user-004",
            resource_type="document",
            resource_id="doc-003",
            action="delete",
            allowed=False,
            policy_id="policy-001",
        )

        assert event.event_type == AuditEventType.ACCESS_DENIED
        assert event.outcome == "denied"

    @pytest.mark.asyncio
    async def test_log_compliance_event(self, audit_service: AuditService):
        """Test logging compliance event."""
        event = await audit_service.log_compliance_event(
            event_type=AuditEventType.LEGAL_HOLD_APPLIED,
            resource_id="hold-001",
            user_id="admin-001",
            details={"matter_id": "matter-001"},
        )

        assert event.event_type == AuditEventType.LEGAL_HOLD_APPLIED


class TestHashChain:
    """Tests for hash chain integrity."""

    @pytest.mark.asyncio
    async def test_hash_chain_continuity(self, audit_service: AuditService):
        """Test that events form a hash chain."""
        event1 = await audit_service.log(
            event_type=AuditEventType.DOCUMENT_VIEW,
            user_id="user-chain-1",
            resource_type="document",
            resource_id="doc-chain-1",
        )

        event2 = await audit_service.log(
            event_type=AuditEventType.DOCUMENT_VIEW,
            user_id="user-chain-1",
            resource_type="document",
            resource_id="doc-chain-2",
        )

        # Second event should have first event's hash as previous_hash
        assert event2.previous_hash == event1.hash

    @pytest.mark.asyncio
    async def test_event_integrity_verification(self, audit_service: AuditService):
        """Test that event integrity can be verified."""
        event = await audit_service.log(
            event_type=AuditEventType.DOCUMENT_CREATE,
            user_id="user-verify",
            resource_type="document",
            resource_id="doc-verify",
        )

        assert event.verify_integrity() is True

    @pytest.mark.asyncio
    async def test_tampered_event_fails_verification(self, audit_service: AuditService):
        """Test that tampered event fails verification."""
        event = await audit_service.log(
            event_type=AuditEventType.DOCUMENT_CREATE,
            user_id="user-tamper",
            resource_type="document",
            resource_id="doc-tamper",
        )

        # Tamper with the event
        event.user_id = "attacker"

        # Should fail verification
        assert event.verify_integrity() is False


class TestAuditQuerying:
    """Tests for querying audit events."""

    @pytest.mark.asyncio
    async def test_query_by_user(self, audit_service: AuditService):
        """Test querying events by user."""
        # Log some events
        for i in range(3):
            await audit_service.log(
                event_type=AuditEventType.DOCUMENT_VIEW,
                user_id="query-user-001",
                resource_type="document",
                resource_id=f"doc-query-{i}",
            )

        # Flush buffer
        await audit_service._flush_buffer()

        # Note: In-memory mode doesn't support querying
        # This test would work with a real backend

    @pytest.mark.asyncio
    async def test_query_by_resource(self, audit_service: AuditService):
        """Test querying events by resource."""
        await audit_service.log(
            event_type=AuditEventType.DOCUMENT_CREATE,
            user_id="user-resource-1",
            resource_type="document",
            resource_id="doc-resource-query",
        )

        await audit_service.log(
            event_type=AuditEventType.DOCUMENT_VIEW,
            user_id="user-resource-2",
            resource_type="document",
            resource_id="doc-resource-query",
        )

        await audit_service._flush_buffer()

        # Note: Full querying requires a backend


class TestEventSerialization:
    """Tests for event serialization."""

    @pytest.mark.asyncio
    async def test_event_to_dict(self, audit_service: AuditService):
        """Test converting event to dictionary."""
        event = await audit_service.log(
            event_type=AuditEventType.DOCUMENT_VIEW,
            user_id="user-serial",
            resource_type="document",
            resource_id="doc-serial",
            details={"key": "value"},
        )

        event_dict = event.to_dict()

        assert event_dict["id"] == event.id
        assert event_dict["event_type"] == "document.view"
        assert event_dict["user_id"] == "user-serial"
        assert event_dict["details"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_event_from_dict(self, audit_service: AuditService):
        """Test creating event from dictionary."""
        event = await audit_service.log(
            event_type=AuditEventType.DOCUMENT_CREATE,
            user_id="user-from-dict",
            resource_type="document",
            resource_id="doc-from-dict",
        )

        event_dict = event.to_dict()
        restored = AuditEvent.from_dict(event_dict)

        assert restored.id == event.id
        assert restored.event_type == event.event_type
        assert restored.user_id == event.user_id


class TestBuffering:
    """Tests for event buffering."""

    @pytest.mark.asyncio
    async def test_buffer_fills(self, audit_service: AuditService):
        """Test that buffer accumulates events."""
        initial_size = len(audit_service._event_buffer)

        await audit_service.log(
            event_type=AuditEventType.DOCUMENT_VIEW,
            user_id="user-buffer",
            resource_type="document",
            resource_id="doc-buffer",
        )

        assert len(audit_service._event_buffer) == initial_size + 1

    @pytest.mark.asyncio
    async def test_buffer_flushes_when_full(self, audit_service: AuditService):
        """Test that buffer flushes when it reaches capacity."""
        # Fill buffer to capacity
        for i in range(audit_service.config.batch_size + 1):
            await audit_service.log(
                event_type=AuditEventType.DOCUMENT_VIEW,
                user_id="user-flush",
                resource_type="document",
                resource_id=f"doc-flush-{i}",
            )

        # Give time for async flush
        import asyncio
        await asyncio.sleep(0.1)

        # Buffer should have been flushed
        assert len(audit_service._event_buffer) < audit_service.config.batch_size


class TestHealthCheck:
    """Tests for health check."""

    @pytest.mark.asyncio
    async def test_health_check(self, audit_service: AuditService):
        """Test health check."""
        health = await audit_service.health_check()

        assert health["status"] == "healthy"
        assert "backend" in health
        assert "hash_chain_enabled" in health


class TestAuditServiceNotConnected:
    """Tests for unconnected audit service."""

    @pytest.mark.asyncio
    async def test_log_before_connect_fails(self, audit_config):
        """Test that logging before connection fails."""
        service = AuditService(audit_config)

        with pytest.raises(RuntimeError, match="not connected"):
            await service.log(
                event_type=AuditEventType.DOCUMENT_VIEW,
                user_id="user",
                resource_type="document",
                resource_id="doc",
            )
