"""
Legal Document Platform - Security & Compliance Layer
======================================================
Comprehensive security controls including:
- Secrets management (HashiCorp Vault integration)
- Authentication (Keycloak/OAuth2)
- Encryption (AES-256-GCM envelope encryption)
- Audit logging (tamper-evident, immutable logs)
- Access control (RBAC/ABAC)
- Compliance enforcement (SOC2, HIPAA, GDPR, CCPA)
"""

from .config import SecurityConfig
from .models import (
    SecurityClassification,
    ComplianceFramework,
    AuditEventType,
    DataResidency,
    AuditEvent,
    EncryptionKey,
    CompliancePolicy,
    DataRetentionPolicy,
    PIIType,
    PIIMatch,
    AccessPolicy,
    Role,
    Permission,
)
from .services.vault import VaultService, VaultConfig
from .services.keycloak import KeycloakService, KeycloakConfig
from .services.encryption import EncryptionService, EncryptionConfig
from .services.audit import AuditService, AuditConfig
from .services.compliance import ComplianceService, ComplianceConfig
from .services.access_control import AccessControlService, AccessControlConfig
from .security_service import SecurityService

__all__ = [
    # Config
    "SecurityConfig",
    # Models
    "SecurityClassification",
    "ComplianceFramework",
    "AuditEventType",
    "DataResidency",
    "AuditEvent",
    "EncryptionKey",
    "CompliancePolicy",
    "DataRetentionPolicy",
    "PIIType",
    "PIIMatch",
    "AccessPolicy",
    "Role",
    "Permission",
    # Services
    "VaultService",
    "VaultConfig",
    "KeycloakService",
    "KeycloakConfig",
    "EncryptionService",
    "EncryptionConfig",
    "AuditService",
    "AuditConfig",
    "ComplianceService",
    "ComplianceConfig",
    "AccessControlService",
    "AccessControlConfig",
    # Main Service
    "SecurityService",
]
