"""
Test fixtures for security layer tests.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Generator
from uuid import uuid4

import pytest
import pytest_asyncio

from ..config import (
    SecurityConfig,
    VaultConfig,
    KeycloakConfig,
    EncryptionConfig,
    AuditConfig,
    ComplianceConfig,
    AccessControlConfig,
)
from ..models import (
    AuditEvent,
    AuditEventType,
    SecurityClassification,
    ComplianceFramework,
    PIIType,
)
from ..services.vault import VaultService
from ..services.keycloak import KeycloakService
from ..services.encryption import EncryptionService
from ..services.audit import AuditService
from ..services.compliance import ComplianceService
from ..services.access_control import AccessControlService
from ..security_service import SecurityService, AuthenticatedUser


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Configuration fixtures

@pytest.fixture
def vault_config() -> VaultConfig:
    """Vault test configuration."""
    return VaultConfig(
        url="http://localhost:8200",
        token="test-token",
        mount_point="secret",
    )


@pytest.fixture
def keycloak_config() -> KeycloakConfig:
    """Keycloak test configuration."""
    return KeycloakConfig(
        server_url="http://localhost:8080",
        realm="test-realm",
        client_id="test-client",
        client_secret="test-secret",
    )


@pytest.fixture
def encryption_config() -> EncryptionConfig:
    """Encryption test configuration."""
    return EncryptionConfig(
        algorithm="aes-256-gcm",
        key_storage="local",
        local_key_file=None,  # Use ephemeral key
    )


@pytest.fixture
def audit_config() -> AuditConfig:
    """Audit test configuration."""
    return AuditConfig(
        backend="file",
        file_path=None,  # In-memory only
        enable_hash_chain=True,
        batch_size=10,
        flush_interval_seconds=1,
    )


@pytest.fixture
def compliance_config() -> ComplianceConfig:
    """Compliance test configuration."""
    return ComplianceConfig(
        enabled_frameworks=["soc2", "gdpr"],
        enable_pii_detection=True,
        pii_detection_confidence_threshold=0.7,
    )


@pytest.fixture
def access_control_config() -> AccessControlConfig:
    """Access control test configuration."""
    return AccessControlConfig(
        mode="hybrid",
        default_role="viewer",
        enable_role_inheritance=True,
        enable_attribute_policies=True,
    )


@pytest.fixture
def security_config(
    vault_config: VaultConfig,
    keycloak_config: KeycloakConfig,
    encryption_config: EncryptionConfig,
    audit_config: AuditConfig,
    compliance_config: ComplianceConfig,
    access_control_config: AccessControlConfig,
) -> SecurityConfig:
    """Combined security configuration."""
    return SecurityConfig(
        vault=vault_config,
        keycloak=keycloak_config,
        encryption=encryption_config,
        audit=audit_config,
        compliance=compliance_config,
        access_control=access_control_config,
        environment="test",
        debug=True,
    )


# Service fixtures

@pytest_asyncio.fixture
async def vault_service(vault_config: VaultConfig) -> VaultService:
    """Vault service instance."""
    service = VaultService(vault_config)
    await service.connect()
    yield service
    await service.disconnect()


@pytest_asyncio.fixture
async def keycloak_service(keycloak_config: KeycloakConfig) -> KeycloakService:
    """Keycloak service instance."""
    service = KeycloakService(keycloak_config)
    await service.connect()
    yield service
    await service.disconnect()


@pytest_asyncio.fixture
async def encryption_service(encryption_config: EncryptionConfig) -> EncryptionService:
    """Encryption service instance."""
    service = EncryptionService(encryption_config)
    await service.initialize()
    yield service


@pytest_asyncio.fixture
async def audit_service(audit_config: AuditConfig) -> AuditService:
    """Audit service instance."""
    service = AuditService(audit_config)
    await service.connect()
    yield service
    await service.disconnect()


@pytest_asyncio.fixture
async def compliance_service(
    compliance_config: ComplianceConfig,
    audit_service: AuditService,
) -> ComplianceService:
    """Compliance service instance."""
    service = ComplianceService(compliance_config, audit_service)
    await service.initialize()
    yield service


@pytest_asyncio.fixture
async def access_control_service(
    access_control_config: AccessControlConfig,
    audit_service: AuditService,
) -> AccessControlService:
    """Access control service instance."""
    service = AccessControlService(access_control_config, audit_service)
    await service.initialize()
    yield service


@pytest_asyncio.fixture
async def security_service(security_config: SecurityConfig) -> SecurityService:
    """Full security service instance."""
    service = SecurityService(security_config)
    await service.connect()
    yield service
    await service.disconnect()


# Sample data fixtures

@pytest.fixture
def sample_user() -> AuthenticatedUser:
    """Sample authenticated user."""
    return AuthenticatedUser(
        user_id=str(uuid4()),
        username="test-user",
        email="test@example.com",
        roles=["user"],
        permissions={"documents:read", "documents:write"},
        organization_id="org-001",
        attributes={"department": "legal"},
        access_token="mock-access-token",
        refresh_token="mock-refresh-token",
        authenticated_at=datetime.utcnow(),
        session_id=str(uuid4()),
    )


@pytest.fixture
def sample_admin_user() -> AuthenticatedUser:
    """Sample admin user."""
    return AuthenticatedUser(
        user_id=str(uuid4()),
        username="admin-user",
        email="admin@example.com",
        roles=["admin"],
        permissions={"*"},
        organization_id="org-001",
        attributes={"department": "it"},
        access_token="mock-admin-token",
        refresh_token="mock-admin-refresh",
        authenticated_at=datetime.utcnow(),
        session_id=str(uuid4()),
    )


@pytest.fixture
def sample_document_content() -> str:
    """Sample document content with PII."""
    return """
    CONFIDENTIAL AGREEMENT

    This Non-Disclosure Agreement is entered into by:

    John Smith
    SSN: 123-45-6789
    Email: john.smith@example.com
    Phone: (555) 123-4567
    Date of Birth: DOB: 01/15/1985

    Credit Card: 4111111111111111

    Bank Account: Account #12345678901234
    Routing: 021000021

    Address: 123 Main Street, New York, NY 10001

    The parties agree to maintain confidentiality of all proprietary information.

    This is privileged and confidential attorney-client communication.
    """


@pytest.fixture
def sample_clean_content() -> str:
    """Sample document content without PII."""
    return """
    INTERNAL MEMO

    Subject: Project Update

    This memo provides an update on the current project status.
    All deliverables are on track for completion.

    Please review and provide feedback by end of week.
    """


@pytest.fixture
def sample_audit_event() -> AuditEvent:
    """Sample audit event."""
    return AuditEvent(
        id=str(uuid4()),
        event_type=AuditEventType.DOCUMENT_VIEW,
        timestamp=datetime.utcnow(),
        user_id="user-001",
        organization_id="org-001",
        resource_type="document",
        resource_id="doc-001",
        action="view",
        outcome="success",
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0",
    )


@pytest.fixture
def sample_document_metadata() -> dict:
    """Sample document metadata."""
    return {
        "id": str(uuid4()),
        "title": "Test Agreement",
        "type": "contract",
        "client_id": "client-001",
        "matter_id": "matter-001",
        "created_by": "user-001",
        "created_at": datetime.utcnow().isoformat(),
        "encrypted": False,
    }
