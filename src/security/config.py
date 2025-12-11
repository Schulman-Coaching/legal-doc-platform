"""
Security Layer Configuration
============================
Configuration classes for all security services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VaultConfig:
    """HashiCorp Vault configuration."""
    url: str = "http://localhost:8200"
    token: Optional[str] = None
    role_id: Optional[str] = None
    secret_id: Optional[str] = None
    mount_point: str = "secret"
    namespace: Optional[str] = None
    verify_ssl: bool = True
    timeout: int = 30
    # Key paths
    encryption_key_path: str = "legal-docs/encryption"
    api_keys_path: str = "legal-docs/api-keys"
    database_creds_path: str = "legal-docs/database"


@dataclass
class KeycloakConfig:
    """Keycloak authentication configuration."""
    server_url: str = "http://localhost:8080"
    realm: str = "legal-docs"
    client_id: str = "legal-doc-platform"
    client_secret: Optional[str] = None
    admin_username: Optional[str] = None
    admin_password: Optional[str] = None
    verify_ssl: bool = True
    # Token settings
    token_lifetime: int = 3600  # 1 hour
    refresh_token_lifetime: int = 86400  # 24 hours
    # OIDC endpoints
    authorization_endpoint: Optional[str] = None
    token_endpoint: Optional[str] = None
    userinfo_endpoint: Optional[str] = None
    jwks_uri: Optional[str] = None


@dataclass
class EncryptionConfig:
    """Encryption service configuration."""
    algorithm: str = "aes-256-gcm"
    key_rotation_days: int = 90
    envelope_encryption: bool = True
    # Key derivation
    kdf_algorithm: str = "pbkdf2"
    kdf_iterations: int = 100000
    salt_length: int = 32
    # Storage
    key_storage: str = "vault"  # vault, local, aws-kms
    local_key_file: Optional[str] = None
    # AWS KMS settings (if using AWS)
    aws_kms_key_id: Optional[str] = None
    aws_region: Optional[str] = None


@dataclass
class AuditConfig:
    """Audit logging configuration."""
    # Storage backend
    backend: str = "postgresql"  # postgresql, elasticsearch, file
    # PostgreSQL settings
    postgres_dsn: Optional[str] = None
    postgres_table: str = "audit_events"
    # Elasticsearch settings
    elasticsearch_hosts: list[str] = field(default_factory=lambda: ["http://localhost:9200"])
    elasticsearch_index: str = "audit-logs"
    # File settings
    file_path: Optional[str] = None
    file_rotation: str = "daily"  # daily, weekly, size
    file_max_size_mb: int = 100
    # Tamper detection
    enable_hash_chain: bool = True
    hash_algorithm: str = "sha256"
    # Retention
    retention_days: int = 2555  # 7 years for legal compliance
    archive_after_days: int = 365
    # Performance
    batch_size: int = 100
    flush_interval_seconds: int = 5


@dataclass
class ComplianceConfig:
    """Compliance engine configuration."""
    # Enabled frameworks
    enabled_frameworks: list[str] = field(default_factory=lambda: ["soc2", "gdpr"])
    # GDPR settings
    gdpr_request_deadline_days: int = 30
    gdpr_data_subject_verification: bool = True
    # Data retention
    default_retention_days: int = 2555  # 7 years
    enable_automatic_deletion: bool = False
    # Legal hold
    enable_legal_hold: bool = True
    legal_hold_notification_email: Optional[str] = None
    # PII detection
    enable_pii_detection: bool = True
    pii_detection_confidence_threshold: float = 0.8
    auto_redact_pii: bool = False
    # Reporting
    compliance_report_schedule: str = "weekly"
    compliance_alert_email: Optional[str] = None


@dataclass
class AccessControlConfig:
    """Access control configuration."""
    # Mode
    mode: str = "rbac"  # rbac, abac, hybrid
    # RBAC settings
    default_role: str = "viewer"
    enable_role_inheritance: bool = True
    # ABAC settings
    enable_attribute_policies: bool = True
    policy_evaluation_order: str = "deny-override"  # deny-override, allow-override
    # Session settings
    session_timeout_minutes: int = 60
    max_concurrent_sessions: int = 5
    # IP restrictions
    enable_ip_whitelist: bool = False
    ip_whitelist: list[str] = field(default_factory=list)
    # Time-based access
    enable_time_restrictions: bool = False
    business_hours_start: str = "09:00"
    business_hours_end: str = "18:00"
    business_days: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri


@dataclass
class SecurityConfig:
    """Combined security configuration."""
    vault: VaultConfig = field(default_factory=VaultConfig)
    keycloak: KeycloakConfig = field(default_factory=KeycloakConfig)
    encryption: EncryptionConfig = field(default_factory=EncryptionConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    access_control: AccessControlConfig = field(default_factory=AccessControlConfig)
    # Global settings
    environment: str = "development"  # development, staging, production
    debug: bool = False
