"""
Tests for Unified Security Service.
"""

from __future__ import annotations

import pytest
from datetime import datetime

from ..security_service import SecurityService, AuthenticatedUser
from ..models import (
    AuditEventType,
    SecurityClassification,
)


class TestSecurityServiceInitialization:
    """Tests for service initialization."""

    @pytest.mark.asyncio
    async def test_connect(self, security_config):
        """Test connecting security service."""
        service = SecurityService(security_config)
        await service.connect()

        assert service._connected is True

        await service.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect(self, security_service: SecurityService):
        """Test disconnecting security service."""
        await security_service.disconnect()

        assert security_service._connected is False

    @pytest.mark.asyncio
    async def test_health_check(self, security_service: SecurityService):
        """Test health check."""
        health = await security_service.health_check()

        assert health["status"] in ("healthy", "degraded")
        assert "services" in health
        assert "timestamp" in health


class TestAuthentication:
    """Tests for authentication."""

    @pytest.mark.asyncio
    async def test_authenticate_mock(self, security_service: SecurityService):
        """Test authentication with mock Keycloak."""
        user = await security_service.authenticate(
            username="test-user",
            password="test-password",
            ip_address="192.168.1.100",
        )

        assert user is not None
        # Mock returns "mock-user" as username
        assert user.username is not None
        assert user.access_token is not None
        assert user.session_id is not None

    @pytest.mark.asyncio
    async def test_validate_session(self, security_service: SecurityService):
        """Test session validation."""
        user = await security_service.authenticate("user", "pass")

        validated = await security_service.validate_session(user.session_id)

        assert validated is not None
        assert validated.user_id == user.user_id

    @pytest.mark.asyncio
    async def test_invalid_session(self, security_service: SecurityService):
        """Test invalid session returns None."""
        validated = await security_service.validate_session("invalid-session")

        assert validated is None

    @pytest.mark.asyncio
    async def test_logout(self, security_service: SecurityService):
        """Test logout."""
        user = await security_service.authenticate("user", "pass")

        result = await security_service.logout(user)

        assert result is True

        # Session should be invalid
        validated = await security_service.validate_session(user.session_id)
        assert validated is None


class TestAuthorization:
    """Tests for authorization."""

    @pytest.mark.asyncio
    async def test_authorize_allowed(self, security_service: SecurityService, sample_user: AuthenticatedUser):
        """Test authorization when allowed."""
        # Assign user role
        await security_service.access_control.assign_role(sample_user.user_id, "user")

        decision = await security_service.authorize(
            user=sample_user,
            resource_type="documents",
            resource_id="doc-001",
            action="read",
        )

        assert decision.allowed is True

    @pytest.mark.asyncio
    async def test_authorize_denied(self, security_service: SecurityService, sample_user: AuthenticatedUser):
        """Test authorization when denied."""
        # Assign viewer role
        await security_service.access_control.assign_role(sample_user.user_id, "viewer")

        decision = await security_service.authorize(
            user=sample_user,
            resource_type="documents",
            resource_id="doc-001",
            action="delete",
        )

        assert decision.allowed is False

    @pytest.mark.asyncio
    async def test_require_permission_raises(self, security_service: SecurityService, sample_user: AuthenticatedUser):
        """Test require_permission raises on denied."""
        await security_service.access_control.assign_role(sample_user.user_id, "viewer")

        with pytest.raises(PermissionError):
            await security_service.require_permission(
                user=sample_user,
                resource_type="admin",
                resource_id="settings",
                action="admin",
            )


class TestEncryption:
    """Tests for encryption operations."""

    @pytest.mark.asyncio
    async def test_encrypt_document(self, security_service: SecurityService):
        """Test document encryption."""
        content = b"Sensitive document content"

        encrypted = await security_service.encrypt_document(content, "doc-001")

        assert encrypted is not None
        assert encrypted.ciphertext != content

    @pytest.mark.asyncio
    async def test_decrypt_document(self, security_service: SecurityService, sample_user: AuthenticatedUser):
        """Test document decryption."""
        content = b"Sensitive document content"
        encrypted = await security_service.encrypt_document(content, "doc-002")

        decrypted = await security_service.decrypt_document(
            encrypted,
            document_id="doc-002",
            user=sample_user,
        )

        assert decrypted == content


class TestDocumentSecurity:
    """Tests for document security operations."""

    @pytest.mark.asyncio
    async def test_secure_document(
        self,
        security_service: SecurityService,
        sample_user: AuthenticatedUser,
        sample_document_content: str,
        sample_document_metadata: dict,
    ):
        """Test securing a document."""
        secure_doc = await security_service.secure_document(
            document_id="doc-secure-001",
            content=sample_document_content,
            metadata=sample_document_metadata,
            user=sample_user,
        )

        assert secure_doc is not None
        assert secure_doc.classification is not None
        assert len(secure_doc.pii_detected) > 0

    @pytest.mark.asyncio
    async def test_check_document_compliance(
        self,
        security_service: SecurityService,
        sample_document_content: str,
        sample_document_metadata: dict,
    ):
        """Test checking document compliance."""
        result = await security_service.check_document_compliance(
            document_id="doc-compliance-001",
            content=sample_document_content,
            metadata=sample_document_metadata,
        )

        assert "compliant" in result
        assert "pii_detected" in result
        assert "classification" in result

    @pytest.mark.asyncio
    async def test_redact_document_pii(
        self,
        security_service: SecurityService,
        sample_user: AuthenticatedUser,
        sample_document_content: str,
    ):
        """Test PII redaction."""
        redacted, matches = await security_service.redact_document_pii(
            content=sample_document_content,
            user=sample_user,
            document_id="doc-redact-001",
        )

        assert "[REDACTED:" in redacted
        assert len(matches) > 0


class TestLegalHold:
    """Tests for legal hold operations."""

    @pytest.mark.asyncio
    async def test_apply_legal_hold(self, security_service: SecurityService, sample_user: AuthenticatedUser):
        """Test applying legal hold."""
        hold = await security_service.apply_legal_hold(
            matter_id="matter-service-001",
            matter_name="Test Matter",
            document_ids=["doc-hold-001", "doc-hold-002"],
            custodians=["user-001"],
            reason="Litigation",
            user=sample_user,
        )

        assert hold is not None
        assert hold.status == "active"

    @pytest.mark.asyncio
    async def test_is_under_legal_hold(self, security_service: SecurityService, sample_user: AuthenticatedUser):
        """Test checking legal hold status."""
        await security_service.apply_legal_hold(
            matter_id="matter-service-002",
            matter_name="Test Matter",
            document_ids=["doc-check-hold"],
            custodians=["user-001"],
            reason="Litigation",
            user=sample_user,
        )

        is_held = await security_service.is_under_legal_hold("doc-check-hold")

        assert is_held is True

    @pytest.mark.asyncio
    async def test_release_legal_hold(self, security_service: SecurityService, sample_user: AuthenticatedUser):
        """Test releasing legal hold."""
        hold = await security_service.apply_legal_hold(
            matter_id="matter-service-003",
            matter_name="Test Matter",
            document_ids=["doc-release-hold"],
            custodians=["user-001"],
            reason="Litigation",
            user=sample_user,
        )

        released = await security_service.release_legal_hold(hold.id, sample_user)

        assert released.status == "released"


class TestGDPR:
    """Tests for GDPR operations."""

    @pytest.mark.asyncio
    async def test_create_gdpr_request(self, security_service: SecurityService, sample_user: AuthenticatedUser):
        """Test creating GDPR request."""
        request = await security_service.create_gdpr_request(
            request_type="access",
            data_subject_id="subject-001",
            data_subject_email="subject@example.com",
            handler=sample_user,
        )

        assert request is not None
        assert request.status == "pending"

    @pytest.mark.asyncio
    async def test_process_gdpr_request(self, security_service: SecurityService, sample_user: AuthenticatedUser):
        """Test processing GDPR request."""
        request = await security_service.create_gdpr_request(
            request_type="access",
            data_subject_id="subject-002",
        )

        processed = await security_service.process_gdpr_request(
            request.id,
            response_data={"documents": ["doc1", "doc2"]},
            handler=sample_user,
        )

        assert processed.status == "completed"


class TestSecrets:
    """Tests for secrets management."""

    @pytest.mark.asyncio
    async def test_get_set_secret(self, security_service: SecurityService):
        """Test getting and setting secrets."""
        await security_service.set_secret("test/secret", {"api_key": "secret123"})

        secret = await security_service.get_secret("test/secret")

        assert secret is not None
        assert secret["api_key"] == "secret123"

    @pytest.mark.asyncio
    async def test_get_nonexistent_secret(self, security_service: SecurityService):
        """Test getting non-existent secret."""
        secret = await security_service.get_secret("nonexistent/path")

        assert secret is None


class TestAudit:
    """Tests for audit operations."""

    @pytest.mark.asyncio
    async def test_log_event(self, security_service: SecurityService):
        """Test logging audit event."""
        # Authenticate a user first
        user = await security_service.authenticate("test-user", "password")

        event = await security_service.log_event(
            event_type=AuditEventType.DOCUMENT_VIEW,
            user=user,
            resource_type="document",
            resource_id="doc-audit-001",
            action="view",
        )

        assert event is not None
        assert event.user_id == user.user_id

    @pytest.mark.asyncio
    async def test_verify_audit_integrity(self, security_service: SecurityService):
        """Test audit integrity verification."""
        result = await security_service.verify_audit_integrity()

        assert "status" in result


class TestRoleManagement:
    """Tests for role management."""

    @pytest.mark.asyncio
    async def test_assign_role(self, security_service: SecurityService, sample_admin_user: AuthenticatedUser):
        """Test assigning role."""
        result = await security_service.assign_role(
            user_id="new-user-001",
            role_name="user",
            assigned_by=sample_admin_user,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_revoke_role(self, security_service: SecurityService, sample_admin_user: AuthenticatedUser):
        """Test revoking role."""
        await security_service.assign_role("new-user-002", "user", sample_admin_user)

        result = await security_service.revoke_role(
            user_id="new-user-002",
            role_name="user",
            revoked_by=sample_admin_user,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_get_roles(self, security_service: SecurityService):
        """Test getting all roles."""
        roles = await security_service.get_roles()

        assert len(roles) > 0
        role_names = [r.name for r in roles]
        assert "admin" in role_names


class TestSecurityServiceNotConnected:
    """Tests for unconnected security service."""

    @pytest.mark.asyncio
    async def test_operations_fail_when_not_connected(self, security_config):
        """Test that operations fail when not connected."""
        service = SecurityService(security_config)

        with pytest.raises(RuntimeError, match="not connected"):
            await service.authenticate("user", "pass")
