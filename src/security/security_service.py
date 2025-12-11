"""
Unified Security Service
========================
Coordinator for all security subsystems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from .config import SecurityConfig
from .models import (
    AccessDecision,
    AuditEvent,
    AuditEventType,
    ComplianceViolation,
    EncryptedData,
    GDPRRequest,
    LegalHold,
    PIIMatch,
    Role,
    SecurityClassification,
)
from .services.access_control import AccessControlService
from .services.audit import AuditService
from .services.compliance import ComplianceService
from .services.encryption import EncryptionService
from .services.keycloak import KeycloakService
from .services.vault import VaultService

logger = logging.getLogger(__name__)


@dataclass
class AuthenticatedUser:
    """Authenticated user context."""
    user_id: str
    username: str
    email: Optional[str]
    roles: list[str]
    permissions: set[str]
    organization_id: Optional[str]
    attributes: dict[str, Any]
    access_token: str
    refresh_token: Optional[str]
    authenticated_at: datetime
    session_id: str


@dataclass
class SecureDocument:
    """Document with security metadata."""
    document_id: str
    classification: SecurityClassification
    encrypted: bool
    pii_detected: list[PIIMatch]
    legal_hold: Optional[LegalHold]
    compliance_status: dict[str, Any]


class SecurityService:
    """
    Unified security service coordinating all security subsystems.

    This is the main entry point for security operations, providing:
    - Authentication via Keycloak
    - Authorization via RBAC/ABAC
    - Encryption for data at rest
    - Audit logging with tamper detection
    - Compliance enforcement
    - Secrets management via Vault

    Usage:
        config = SecurityConfig()
        security = SecurityService(config)
        await security.connect()

        # Authenticate user
        user = await security.authenticate("username", "password")

        # Check access
        can_access = await security.authorize(user, "documents", doc_id, "read")

        # Encrypt data
        encrypted = await security.encrypt_document(content)

        # Check compliance
        compliance = await security.check_document_compliance(doc_id, content)
    """

    def __init__(self, config: SecurityConfig):
        self.config = config
        self._connected = False

        # Initialize sub-services
        self.vault = VaultService(config.vault)
        self.keycloak = KeycloakService(config.keycloak)
        self.encryption = EncryptionService(config.encryption, self.vault)
        self.audit = AuditService(config.audit)
        self.compliance = ComplianceService(config.compliance, self.audit)
        self.access_control = AccessControlService(config.access_control, self.audit)

        # Session cache
        self._sessions: dict[str, AuthenticatedUser] = {}

    async def connect(self) -> None:
        """Initialize all security services."""
        logger.info("Initializing security services...")

        try:
            # Connect to Vault first (needed for encryption keys)
            await self.vault.connect()

            # Connect to Keycloak
            await self.keycloak.connect()

            # Initialize encryption with keys from Vault
            await self.encryption.initialize()

            # Connect audit service
            await self.audit.connect()

            # Initialize compliance engine
            await self.compliance.initialize()

            # Initialize access control
            await self.access_control.initialize()

            self._connected = True
            logger.info("All security services initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize security services: %s", str(e))
            raise

    async def disconnect(self) -> None:
        """Shutdown all security services."""
        logger.info("Shutting down security services...")

        await self.audit.disconnect()
        await self.keycloak.disconnect()
        await self.vault.disconnect()

        self._sessions.clear()
        self._connected = False
        logger.info("Security services shut down")

    def _ensure_connected(self) -> None:
        """Ensure service is connected."""
        if not self._connected:
            raise RuntimeError("Security service not connected")

    # =========================================================================
    # Authentication
    # =========================================================================

    async def authenticate(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuthenticatedUser:
        """
        Authenticate a user.

        Args:
            username: Username or email
            password: Password
            ip_address: Client IP for audit
            user_agent: Client user agent for audit

        Returns:
            AuthenticatedUser context

        Raises:
            AuthenticationError: If credentials are invalid
        """
        self._ensure_connected()

        try:
            # Authenticate with Keycloak
            token_response = await self.keycloak.authenticate(username, password)
            access_token = token_response['access_token']
            refresh_token = token_response.get('refresh_token')

            # Get user info
            user_info = await self.keycloak.get_user_info(access_token)

            # Get roles and permissions
            user_id = user_info['sub']
            roles = await self.access_control.get_user_roles(user_id)
            if not roles:
                # Assign default role
                await self.access_control.assign_role(user_id, self.config.access_control.default_role)
                roles = [self.config.access_control.default_role]

            permissions = await self.access_control.get_user_permissions(user_id)

            # Create session
            session_id = str(uuid4())
            user = AuthenticatedUser(
                user_id=user_id,
                username=user_info.get('preferred_username', username),
                email=user_info.get('email'),
                roles=roles,
                permissions=permissions,
                organization_id=user_info.get('organization_id'),
                attributes=user_info,
                access_token=access_token,
                refresh_token=refresh_token,
                authenticated_at=datetime.utcnow(),
                session_id=session_id,
            )

            # Cache session
            self._sessions[session_id] = user

            # Log successful authentication
            await self.audit.log_auth_event(
                event_type=AuditEventType.LOGIN_SUCCESS,
                user_id=user_id,
                success=True,
                ip_address=ip_address,
                user_agent=user_agent,
                details={"username": username, "session_id": session_id},
            )

            logger.info("User authenticated: %s", username)
            return user

        except Exception as e:
            # Log failed authentication
            await self.audit.log_auth_event(
                event_type=AuditEventType.LOGIN_FAILURE,
                user_id=username,
                success=False,
                ip_address=ip_address,
                user_agent=user_agent,
                details={"error": str(e)},
            )
            raise

    async def validate_session(
        self,
        session_id: str,
        access_token: Optional[str] = None,
    ) -> Optional[AuthenticatedUser]:
        """
        Validate an existing session.

        Args:
            session_id: Session identifier
            access_token: Optional token for validation

        Returns:
            AuthenticatedUser if valid, None otherwise
        """
        self._ensure_connected()

        # Check session cache
        user = self._sessions.get(session_id)
        if not user:
            return None

        # Validate token if provided
        if access_token:
            token_info = await self.keycloak.validate_token(access_token)
            if not token_info.get('active'):
                del self._sessions[session_id]
                return None

        return user

    async def logout(
        self,
        user: AuthenticatedUser,
        ip_address: Optional[str] = None,
    ) -> bool:
        """
        Logout a user and invalidate session.

        Args:
            user: Authenticated user
            ip_address: Client IP for audit

        Returns:
            True if successful
        """
        self._ensure_connected()

        try:
            # Revoke tokens
            if user.refresh_token:
                await self.keycloak.logout(user.refresh_token, user.access_token)

            # Remove session
            if user.session_id in self._sessions:
                del self._sessions[user.session_id]

            # Log logout
            await self.audit.log_auth_event(
                event_type=AuditEventType.LOGOUT,
                user_id=user.user_id,
                success=True,
                ip_address=ip_address,
                details={"session_id": user.session_id},
            )

            logger.info("User logged out: %s", user.username)
            return True

        except Exception as e:
            logger.error("Logout failed: %s", str(e))
            return False

    async def refresh_session(self, user: AuthenticatedUser) -> AuthenticatedUser:
        """Refresh an authenticated session."""
        self._ensure_connected()

        if not user.refresh_token:
            raise ValueError("No refresh token available")

        token_response = await self.keycloak.refresh_token(user.refresh_token)

        # Update user with new tokens
        user.access_token = token_response['access_token']
        user.refresh_token = token_response.get('refresh_token', user.refresh_token)

        # Log token refresh
        await self.audit.log_auth_event(
            event_type=AuditEventType.TOKEN_REFRESH,
            user_id=user.user_id,
            success=True,
        )

        return user

    # =========================================================================
    # Authorization
    # =========================================================================

    async def authorize(
        self,
        user: AuthenticatedUser,
        resource_type: str,
        resource_id: str,
        action: str,
        resource_attributes: Optional[dict[str, Any]] = None,
    ) -> AccessDecision:
        """
        Check if user is authorized for an action.

        Args:
            user: Authenticated user
            resource_type: Type of resource
            resource_id: Resource identifier
            action: Action to perform
            resource_attributes: Additional resource attributes

        Returns:
            AccessDecision
        """
        self._ensure_connected()

        return await self.access_control.check_access(
            user_id=user.user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            resource_attributes=resource_attributes,
        )

    async def require_permission(
        self,
        user: AuthenticatedUser,
        resource_type: str,
        resource_id: str,
        action: str,
        **kwargs,
    ) -> None:
        """
        Require permission or raise exception.

        Args:
            user: Authenticated user
            resource_type: Type of resource
            resource_id: Resource identifier
            action: Action to perform

        Raises:
            PermissionError: If access is denied
        """
        decision = await self.authorize(user, resource_type, resource_id, action, **kwargs)

        if not decision.allowed:
            raise PermissionError(f"Access denied: {decision.reason}")

    # =========================================================================
    # Encryption
    # =========================================================================

    async def encrypt_document(
        self,
        content: bytes,
        document_id: Optional[str] = None,
    ) -> EncryptedData:
        """
        Encrypt document content.

        Args:
            content: Raw document content
            document_id: Optional document ID for AAD

        Returns:
            EncryptedData envelope
        """
        self._ensure_connected()

        aad = document_id.encode() if document_id else None
        return await self.encryption.encrypt(content, aad)

    async def decrypt_document(
        self,
        encrypted_data: EncryptedData,
        document_id: Optional[str] = None,
        user: Optional[AuthenticatedUser] = None,
    ) -> bytes:
        """
        Decrypt document content.

        Args:
            encrypted_data: Encrypted data envelope
            document_id: Optional document ID for AAD verification
            user: User requesting decryption (for audit)

        Returns:
            Decrypted content
        """
        self._ensure_connected()

        aad = document_id.encode() if document_id else None
        result = await self.encryption.decrypt(encrypted_data, aad)

        # Log decryption
        if user and document_id:
            await self.audit.log_document_event(
                event_type=AuditEventType.DOCUMENT_DECRYPT,
                document_id=document_id,
                user_id=user.user_id,
            )

        return result.plaintext

    # =========================================================================
    # Document Security
    # =========================================================================

    async def secure_document(
        self,
        document_id: str,
        content: str,
        metadata: dict[str, Any],
        user: AuthenticatedUser,
    ) -> SecureDocument:
        """
        Apply security controls to a document.

        Args:
            document_id: Document identifier
            content: Document text content
            metadata: Document metadata
            user: User performing the operation

        Returns:
            SecureDocument with security metadata
        """
        self._ensure_connected()

        # Classify document
        classification = await self.compliance.classify_document(content, metadata)

        # Detect PII
        pii_matches = await self.compliance.detect_pii(content)

        # Check for legal hold
        legal_hold = await self.compliance.check_legal_hold(document_id)

        # Evaluate compliance
        compliance_status = await self.compliance.evaluate_document_compliance(
            document_id, content, metadata
        )

        # Log classification
        await self.audit.log_document_event(
            event_type=AuditEventType.DOCUMENT_CLASSIFY,
            document_id=document_id,
            user_id=user.user_id,
            details={
                "classification": classification.value,
                "pii_count": len(pii_matches),
                "legal_hold": bool(legal_hold),
            },
        )

        return SecureDocument(
            document_id=document_id,
            classification=classification,
            encrypted=metadata.get('encrypted', False),
            pii_detected=pii_matches,
            legal_hold=legal_hold,
            compliance_status=compliance_status,
        )

    async def check_document_compliance(
        self,
        document_id: str,
        content: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Check document compliance against all frameworks.

        Args:
            document_id: Document identifier
            content: Document content
            metadata: Document metadata

        Returns:
            Compliance evaluation result
        """
        self._ensure_connected()

        return await self.compliance.evaluate_document_compliance(
            document_id, content, metadata
        )

    async def redact_document_pii(
        self,
        content: str,
        user: AuthenticatedUser,
        document_id: Optional[str] = None,
    ) -> tuple[str, list[PIIMatch]]:
        """
        Redact PII from document content.

        Args:
            content: Document content
            user: User performing redaction
            document_id: Optional document ID for audit

        Returns:
            Tuple of (redacted_content, pii_matches)
        """
        self._ensure_connected()

        redacted, matches = await self.compliance.redact_pii(content)

        # Log redaction
        if document_id:
            await self.audit.log_document_event(
                event_type=AuditEventType.DOCUMENT_UPDATE,
                document_id=document_id,
                user_id=user.user_id,
                details={"action": "pii_redaction", "pii_count": len(matches)},
            )

        return redacted, matches

    # =========================================================================
    # Legal Hold
    # =========================================================================

    async def apply_legal_hold(
        self,
        matter_id: str,
        matter_name: str,
        document_ids: list[str],
        custodians: list[str],
        reason: str,
        user: AuthenticatedUser,
    ) -> LegalHold:
        """Apply a legal hold."""
        self._ensure_connected()

        return await self.compliance.apply_legal_hold(
            matter_id=matter_id,
            matter_name=matter_name,
            document_ids=document_ids,
            custodians=custodians,
            reason=reason,
            applied_by=user.user_id,
        )

    async def release_legal_hold(
        self,
        hold_id: str,
        user: AuthenticatedUser,
    ) -> LegalHold:
        """Release a legal hold."""
        self._ensure_connected()

        return await self.compliance.release_legal_hold(hold_id, user.user_id)

    async def is_under_legal_hold(self, document_id: str) -> bool:
        """Check if document is under legal hold."""
        self._ensure_connected()

        hold = await self.compliance.check_legal_hold(document_id)
        return hold is not None

    # =========================================================================
    # GDPR
    # =========================================================================

    async def create_gdpr_request(
        self,
        request_type: str,
        data_subject_id: str,
        data_subject_email: Optional[str] = None,
        handler: Optional[AuthenticatedUser] = None,
    ) -> GDPRRequest:
        """Create a GDPR data subject request."""
        self._ensure_connected()

        return await self.compliance.create_gdpr_request(
            request_type=request_type,
            data_subject_id=data_subject_id,
            data_subject_email=data_subject_email,
            handled_by=handler.user_id if handler else None,
        )

    async def process_gdpr_request(
        self,
        request_id: str,
        response_data: Optional[dict[str, Any]] = None,
        handler: Optional[AuthenticatedUser] = None,
    ) -> GDPRRequest:
        """Process a GDPR request."""
        self._ensure_connected()

        return await self.compliance.process_gdpr_request(
            request_id=request_id,
            response_data=response_data,
            handled_by=handler.user_id if handler else None,
        )

    # =========================================================================
    # Secrets Management
    # =========================================================================

    async def get_secret(self, path: str) -> Optional[dict[str, Any]]:
        """Get a secret from Vault."""
        self._ensure_connected()
        return await self.vault.get_secret(path)

    async def set_secret(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """Store a secret in Vault."""
        self._ensure_connected()
        return await self.vault.set_secret(path, data)

    async def get_api_key(self, service: str) -> Optional[str]:
        """Get an API key for a service."""
        self._ensure_connected()
        return await self.vault.get_api_key(service)

    # =========================================================================
    # Audit
    # =========================================================================

    async def log_event(
        self,
        event_type: AuditEventType,
        user: Optional[AuthenticatedUser] = None,
        resource_type: str = "",
        resource_id: Optional[str] = None,
        action: str = "",
        outcome: str = "success",
        details: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> AuditEvent:
        """Log an audit event."""
        self._ensure_connected()

        return await self.audit.log(
            event_type=event_type,
            user_id=user.user_id if user else None,
            organization_id=user.organization_id if user else None,
            session_id=user.session_id if user else None,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome=outcome,
            details=details,
            **kwargs,
        )

    async def query_audit_events(
        self,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events."""
        self._ensure_connected()

        return await self.audit.query_events(
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

    async def verify_audit_integrity(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Verify audit log integrity."""
        self._ensure_connected()
        return await self.audit.verify_integrity(start_time, end_time)

    # =========================================================================
    # Role Management
    # =========================================================================

    async def assign_role(
        self,
        user_id: str,
        role_name: str,
        assigned_by: AuthenticatedUser,
    ) -> bool:
        """Assign a role to a user."""
        self._ensure_connected()

        result = await self.access_control.assign_role(user_id, role_name)

        # Clear session cache for affected user
        for session in list(self._sessions.values()):
            if session.user_id == user_id:
                session.roles = await self.access_control.get_user_roles(user_id)
                session.permissions = await self.access_control.get_user_permissions(user_id)

        return result

    async def revoke_role(
        self,
        user_id: str,
        role_name: str,
        revoked_by: AuthenticatedUser,
    ) -> bool:
        """Revoke a role from a user."""
        self._ensure_connected()

        result = await self.access_control.revoke_role(user_id, role_name)

        # Clear session cache for affected user
        for session in list(self._sessions.values()):
            if session.user_id == user_id:
                session.roles = await self.access_control.get_user_roles(user_id)
                session.permissions = await self.access_control.get_user_permissions(user_id)

        return result

    async def get_roles(self) -> list[Role]:
        """Get all available roles."""
        self._ensure_connected()
        return await self.access_control.list_roles()

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> dict[str, Any]:
        """Check health of all security services."""
        if not self._connected:
            return {"status": "disconnected"}

        results = {
            "status": "healthy",
            "services": {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Check each service
        services = [
            ("vault", self.vault),
            ("keycloak", self.keycloak),
            ("encryption", self.encryption),
            ("audit", self.audit),
            ("compliance", self.compliance),
            ("access_control", self.access_control),
        ]

        for name, service in services:
            try:
                health = await service.health_check()
                results["services"][name] = health
                if health.get("status") not in ("healthy", "mock"):
                    results["status"] = "degraded"
            except Exception as e:
                results["services"][name] = {"status": "error", "error": str(e)}
                results["status"] = "degraded"

        # Add session stats
        results["active_sessions"] = len(self._sessions)

        return results
