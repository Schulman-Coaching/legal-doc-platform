"""
Security Layer Models
=====================
Data models for security, compliance, and access control.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# =============================================================================
# Classification & Compliance Enums
# =============================================================================

class SecurityClassification(str, Enum):
    """Data security classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    ATTORNEY_CLIENT_PRIVILEGED = "attorney_client_privileged"


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""
    SOC2 = "soc2"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    CCPA = "ccpa"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    FINRA = "finra"
    GLBA = "glba"


class DataResidency(str, Enum):
    """Data residency regions."""
    US = "us"
    EU = "eu"
    UK = "uk"
    APAC = "apac"
    CANADA = "ca"
    GLOBAL = "global"


# =============================================================================
# Audit Event Types
# =============================================================================

class AuditEventType(str, Enum):
    """Types of audit events."""
    # Authentication
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    LOGOUT = "auth.logout"
    TOKEN_REFRESH = "auth.token.refresh"
    TOKEN_REVOKE = "auth.token.revoke"
    PASSWORD_CHANGE = "auth.password.change"
    PASSWORD_RESET = "auth.password.reset"
    MFA_ENABLED = "auth.mfa.enabled"
    MFA_DISABLED = "auth.mfa.disabled"
    MFA_CHALLENGE = "auth.mfa.challenge"
    SESSION_TIMEOUT = "auth.session.timeout"

    # Document operations
    DOCUMENT_VIEW = "document.view"
    DOCUMENT_CREATE = "document.create"
    DOCUMENT_UPDATE = "document.update"
    DOCUMENT_DELETE = "document.delete"
    DOCUMENT_DOWNLOAD = "document.download"
    DOCUMENT_SHARE = "document.share"
    DOCUMENT_EXPORT = "document.export"
    DOCUMENT_PRINT = "document.print"
    DOCUMENT_CLASSIFY = "document.classify"
    DOCUMENT_ENCRYPT = "document.encrypt"
    DOCUMENT_DECRYPT = "document.decrypt"

    # Access control
    PERMISSION_GRANT = "access.permission.grant"
    PERMISSION_REVOKE = "access.permission.revoke"
    PERMISSION_CHECK = "access.permission.check"
    ROLE_ASSIGN = "access.role.assign"
    ROLE_REVOKE = "access.role.revoke"
    ACCESS_DENIED = "access.denied"

    # Admin operations
    USER_CREATE = "admin.user.create"
    USER_UPDATE = "admin.user.update"
    USER_DELETE = "admin.user.delete"
    USER_DISABLE = "admin.user.disable"
    USER_ENABLE = "admin.user.enable"
    CONFIG_CHANGE = "admin.config.change"
    POLICY_CREATE = "admin.policy.create"
    POLICY_UPDATE = "admin.policy.update"
    POLICY_DELETE = "admin.policy.delete"

    # Security events
    ENCRYPTION_KEY_CREATE = "security.key.create"
    ENCRYPTION_KEY_ROTATION = "security.key.rotation"
    ENCRYPTION_KEY_DELETE = "security.key.delete"
    SECRET_ACCESS = "security.secret.access"
    SECRET_CREATE = "security.secret.create"
    SECRET_UPDATE = "security.secret.update"
    SECRET_DELETE = "security.secret.delete"
    SUSPICIOUS_ACTIVITY = "security.suspicious"
    BREACH_DETECTED = "security.breach"
    VULNERABILITY_DETECTED = "security.vulnerability"

    # Compliance
    LEGAL_HOLD_APPLIED = "compliance.hold.applied"
    LEGAL_HOLD_RELEASED = "compliance.hold.released"
    DATA_RETENTION_APPLIED = "compliance.retention.applied"
    DATA_RETENTION_EXPIRED = "compliance.retention.expired"
    DATA_DELETION = "compliance.deletion"
    DATA_EXPORT = "compliance.export"
    GDPR_REQUEST_ACCESS = "compliance.gdpr.access"
    GDPR_REQUEST_RECTIFY = "compliance.gdpr.rectify"
    GDPR_REQUEST_ERASE = "compliance.gdpr.erase"
    GDPR_REQUEST_PORTABILITY = "compliance.gdpr.portability"
    COMPLIANCE_VIOLATION = "compliance.violation"
    COMPLIANCE_AUDIT = "compliance.audit"

    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"
    BACKUP_STARTED = "system.backup.started"
    BACKUP_COMPLETED = "system.backup.completed"
    BACKUP_FAILED = "system.backup.failed"


# =============================================================================
# Audit Event Model
# =============================================================================

@dataclass
class AuditEvent:
    """Immutable audit log event."""
    id: str = field(default_factory=lambda: str(uuid4()))
    event_type: AuditEventType = AuditEventType.DOCUMENT_VIEW
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    client_id: Optional[str] = None
    session_id: Optional[str] = None
    resource_type: str = ""
    resource_id: Optional[str] = None
    action: str = ""
    outcome: str = "success"  # success, failure, error
    error_message: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # For tamper detection
    previous_hash: Optional[str] = None
    hash: Optional[str] = None

    def __post_init__(self):
        """Calculate hash after initialization."""
        if not self.hash:
            self.hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash for tamper detection."""
        data = (
            f"{self.id}:{self.event_type.value}:{self.timestamp.isoformat()}:"
            f"{self.user_id}:{self.resource_type}:{self.resource_id}:"
            f"{self.action}:{self.outcome}:{self.previous_hash}"
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify the event hasn't been tampered with."""
        return self.hash == self._calculate_hash()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "organization_id": self.organization_id,
            "client_id": self.client_id,
            "session_id": self.session_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "outcome": self.outcome,
            "error_message": self.error_message,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "details": self.details,
            "metadata": self.metadata,
            "previous_hash": self.previous_hash,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEvent:
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            event_type=AuditEventType(data.get("event_type", "document.view")),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
            user_id=data.get("user_id"),
            organization_id=data.get("organization_id"),
            client_id=data.get("client_id"),
            session_id=data.get("session_id"),
            resource_type=data.get("resource_type", ""),
            resource_id=data.get("resource_id"),
            action=data.get("action", ""),
            outcome=data.get("outcome", "success"),
            error_message=data.get("error_message"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            request_id=data.get("request_id"),
            correlation_id=data.get("correlation_id"),
            details=data.get("details", {}),
            metadata=data.get("metadata", {}),
            previous_hash=data.get("previous_hash"),
            hash=data.get("hash"),
        )


# =============================================================================
# Encryption Models
# =============================================================================

class EncryptionAlgorithm(str, Enum):
    """Supported encryption algorithms."""
    AES_256_GCM = "aes-256-gcm"
    AES_256_CBC = "aes-256-cbc"
    CHACHA20_POLY1305 = "chacha20-poly1305"


class EncryptionKey(BaseModel):
    """Encryption key metadata."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    version: int = 1
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    rotated_at: Optional[datetime] = None
    rotated_from: Optional[str] = None
    status: str = "active"  # active, rotated, revoked
    key_hash: Optional[str] = None  # For verification without storing key
    purpose: str = "data"  # data, master, signing

    class Config:
        use_enum_values = True


class EncryptedData(BaseModel):
    """Encrypted data envelope."""
    key_id: str
    algorithm: str = "aes-256-gcm"
    iv: str  # Base64 encoded
    ciphertext: str  # Base64 encoded
    auth_tag: Optional[str] = None  # Base64 encoded, for GCM mode
    encrypted_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# PII Detection Models
# =============================================================================

class PIIType(str, Enum):
    """Types of Personally Identifiable Information."""
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    EMAIL = "email"
    PHONE = "phone"
    DATE_OF_BIRTH = "date_of_birth"
    DRIVERS_LICENSE = "drivers_license"
    PASSPORT = "passport"
    BANK_ACCOUNT = "bank_account"
    IP_ADDRESS = "ip_address"
    ADDRESS = "address"
    NAME = "name"
    MEDICAL_RECORD = "medical_record"
    TAX_ID = "tax_id"


@dataclass
class PIIMatch:
    """PII detection match result."""
    pii_type: PIIType
    value: str
    masked_value: str
    start_position: int
    end_position: int
    confidence: float  # 0.0 to 1.0
    context: Optional[str] = None


# =============================================================================
# Access Control Models
# =============================================================================

class Permission(BaseModel):
    """Permission definition."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str  # e.g., "documents:read", "documents:write"
    resource_type: str  # e.g., "documents", "users", "settings"
    action: str  # e.g., "read", "write", "delete", "admin"
    description: Optional[str] = None
    conditions: dict[str, Any] = Field(default_factory=dict)


class Role(BaseModel):
    """Role with associated permissions."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    permissions: list[str] = Field(default_factory=list)  # Permission names
    inherits_from: list[str] = Field(default_factory=list)  # Parent role names
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_system: bool = False  # System roles cannot be deleted


class AccessPolicy(BaseModel):
    """Attribute-based access control policy."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    effect: str = "allow"  # allow, deny
    priority: int = 0  # Higher priority evaluated first
    conditions: dict[str, Any] = Field(default_factory=dict)
    # Conditions structure:
    # {
    #     "subject": {"user_id": "...", "roles": ["..."], "attributes": {...}},
    #     "resource": {"type": "...", "attributes": {...}},
    #     "action": ["read", "write"],
    #     "environment": {"time": {...}, "ip_range": [...]}
    # }
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AccessDecision(BaseModel):
    """Access control decision result."""
    allowed: bool
    policy_id: Optional[str] = None
    policy_name: Optional[str] = None
    reason: str = ""
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Compliance Models
# =============================================================================

class CompliancePolicy(BaseModel):
    """Compliance policy configuration."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    framework: ComplianceFramework
    name: str
    description: Optional[str] = None
    rules: list[dict[str, Any]] = Field(default_factory=list)
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DataRetentionPolicy(BaseModel):
    """Data retention policy."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    document_types: list[str] = Field(default_factory=list)  # Empty = all types
    classifications: list[SecurityClassification] = Field(default_factory=list)
    retention_days: int
    action: str = "archive"  # archive, delete, anonymize
    legal_hold_exempt: bool = False
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class LegalHold(BaseModel):
    """Legal hold record."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    matter_id: str
    matter_name: Optional[str] = None
    custodians: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    search_criteria: Optional[dict[str, Any]] = None
    reason: str
    applied_by: str
    applied_at: datetime = Field(default_factory=datetime.utcnow)
    released_by: Optional[str] = None
    released_at: Optional[datetime] = None
    status: str = "active"  # active, released


class ComplianceViolation(BaseModel):
    """Compliance violation record."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    policy_id: str
    policy_name: str
    framework: ComplianceFramework
    resource_type: str
    resource_id: str
    violation_type: str
    severity: str = "medium"  # low, medium, high, critical
    description: str
    remediation: Optional[str] = None
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    status: str = "open"  # open, acknowledged, resolved, false_positive

    class Config:
        use_enum_values = True


class GDPRRequest(BaseModel):
    """GDPR data subject request."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    request_type: str  # access, rectification, erasure, portability, restriction
    data_subject_id: str
    data_subject_email: Optional[str] = None
    requested_at: datetime = Field(default_factory=datetime.utcnow)
    deadline: datetime
    status: str = "pending"  # pending, in_progress, completed, rejected
    completed_at: Optional[datetime] = None
    handled_by: Optional[str] = None
    notes: Optional[str] = None
    response_data: Optional[dict[str, Any]] = None
