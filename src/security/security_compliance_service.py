"""
Legal Document Platform - Security & Compliance Layer
======================================================
Comprehensive security controls including secrets management,
encryption, audit logging, and compliance enforcement.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
# Security Configuration & Models
# ============================================================================

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


class AuditEventType(str, Enum):
    """Types of audit events."""
    # Authentication
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    LOGOUT = "auth.logout"
    TOKEN_REFRESH = "auth.token.refresh"
    MFA_ENABLED = "auth.mfa.enabled"
    MFA_DISABLED = "auth.mfa.disabled"

    # Document operations
    DOCUMENT_VIEW = "document.view"
    DOCUMENT_CREATE = "document.create"
    DOCUMENT_UPDATE = "document.update"
    DOCUMENT_DELETE = "document.delete"
    DOCUMENT_DOWNLOAD = "document.download"
    DOCUMENT_SHARE = "document.share"
    DOCUMENT_EXPORT = "document.export"

    # Access control
    PERMISSION_GRANT = "access.permission.grant"
    PERMISSION_REVOKE = "access.permission.revoke"
    ROLE_ASSIGN = "access.role.assign"
    ROLE_REVOKE = "access.role.revoke"

    # Admin operations
    USER_CREATE = "admin.user.create"
    USER_UPDATE = "admin.user.update"
    USER_DELETE = "admin.user.delete"
    CONFIG_CHANGE = "admin.config.change"
    POLICY_UPDATE = "admin.policy.update"

    # Security events
    ENCRYPTION_KEY_ROTATION = "security.key.rotation"
    SECRET_ACCESS = "security.secret.access"
    SUSPICIOUS_ACTIVITY = "security.suspicious"
    BREACH_DETECTED = "security.breach"

    # Compliance
    LEGAL_HOLD_APPLIED = "compliance.hold.applied"
    LEGAL_HOLD_RELEASED = "compliance.hold.released"
    DATA_RETENTION = "compliance.retention"
    DATA_DELETION = "compliance.deletion"
    GDPR_REQUEST = "compliance.gdpr.request"


class DataResidency(str, Enum):
    """Data residency regions."""
    US = "us"
    EU = "eu"
    UK = "uk"
    APAC = "apac"
    GLOBAL = "global"


@dataclass
class AuditEvent:
    """Audit log event."""
    id: str = field(default_factory=lambda: str(uuid4()))
    event_type: AuditEventType = AuditEventType.DOCUMENT_VIEW
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    resource_type: str = ""
    resource_id: Optional[str] = None
    action: str = ""
    outcome: str = "success"  # success, failure, error
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "organization_id": self.organization_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "outcome": self.outcome,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "request_id": self.request_id,
            "details": self.details,
            "metadata": self.metadata,
        }


class EncryptionKey(BaseModel):
    """Encryption key metadata."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    version: int = 1
    algorithm: str = "AES-256-GCM"
    purpose: str = "data_encryption"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    rotated_at: Optional[datetime] = None
    is_active: bool = True
    key_hash: str = ""  # Hash of the key for verification


class CompliancePolicy(BaseModel):
    """Compliance policy configuration."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    framework: ComplianceFramework
    description: str = ""
    rules: list[dict[str, Any]] = Field(default_factory=list)
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DataRetentionPolicy(BaseModel):
    """Data retention policy."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    document_types: list[str] = Field(default_factory=list)
    classification: SecurityClassification = SecurityClassification.INTERNAL
    retention_days: int = 2555  # 7 years default
    deletion_method: str = "secure_delete"  # secure_delete, archive, anonymize
    legal_hold_exempt: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Secrets Management (Vault Integration)
# ============================================================================

class SecretsManager:
    """
    Secrets management service.
    In production, integrate with HashiCorp Vault.
    """

    def __init__(
        self,
        vault_addr: str = "http://localhost:8200",
        vault_token: Optional[str] = None,
    ):
        self.vault_addr = vault_addr
        self.vault_token = vault_token or os.getenv("VAULT_TOKEN")
        self._secrets_cache: dict[str, tuple[str, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

    async def get_secret(
        self,
        path: str,
        key: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get a secret from Vault.

        Args:
            path: Secret path (e.g., "secret/data/database")
            key: Specific key within the secret

        Returns:
            Secret value or None
        """
        # Check cache
        cache_key = f"{path}:{key}" if key else path
        if cache_key in self._secrets_cache:
            value, cached_at = self._secrets_cache[cache_key]
            if datetime.utcnow() - cached_at < self._cache_ttl:
                return value

        # In production, call Vault API
        # import hvac
        # client = hvac.Client(url=self.vault_addr, token=self.vault_token)
        # secret = client.secrets.kv.v2.read_secret_version(path=path)
        # data = secret['data']['data']
        # value = data.get(key) if key else data

        # Placeholder
        value = f"secret-for-{path}"

        # Cache the result
        self._secrets_cache[cache_key] = (value, datetime.utcnow())

        return value

    async def set_secret(
        self,
        path: str,
        data: dict[str, str],
    ) -> bool:
        """Store a secret in Vault."""
        # In production:
        # client.secrets.kv.v2.create_or_update_secret(path=path, secret=data)

        logger.info(f"Secret stored at {path}")
        return True

    async def delete_secret(self, path: str) -> bool:
        """Delete a secret from Vault."""
        # In production:
        # client.secrets.kv.v2.delete_metadata_and_all_versions(path=path)

        # Invalidate cache
        keys_to_remove = [k for k in self._secrets_cache if k.startswith(path)]
        for k in keys_to_remove:
            del self._secrets_cache[k]

        logger.info(f"Secret deleted at {path}")
        return True

    async def rotate_secret(
        self,
        path: str,
        generator: Callable[[], str],
    ) -> str:
        """Rotate a secret with a new value."""
        new_value = generator()

        # Store new secret
        await self.set_secret(path, {"value": new_value})

        # Invalidate cache
        keys_to_remove = [k for k in self._secrets_cache if k.startswith(path)]
        for k in keys_to_remove:
            del self._secrets_cache[k]

        logger.info(f"Secret rotated at {path}")
        return new_value

    async def generate_dynamic_credentials(
        self,
        backend: str,
        role: str,
    ) -> dict[str, str]:
        """
        Generate dynamic credentials from Vault.

        Args:
            backend: Secrets backend (e.g., "database", "aws")
            role: Role name for credential generation

        Returns:
            Generated credentials
        """
        # In production:
        # if backend == "database":
        #     creds = client.secrets.database.generate_credentials(role)
        # elif backend == "aws":
        #     creds = client.secrets.aws.generate_credentials(role)

        return {
            "username": f"dynamic-{role}-{uuid4().hex[:8]}",
            "password": uuid4().hex,
            "lease_duration": "3600",
        }


# ============================================================================
# Encryption Service
# ============================================================================

class EncryptionService:
    """
    Encryption service for data at rest and in transit.
    Implements envelope encryption pattern.
    """

    def __init__(
        self,
        secrets_manager: SecretsManager,
        master_key_path: str = "secret/data/master-key",
    ):
        self.secrets_manager = secrets_manager
        self.master_key_path = master_key_path
        self._data_keys: dict[str, bytes] = {}
        self._key_metadata: dict[str, EncryptionKey] = {}

    async def _get_master_key(self) -> bytes:
        """Get the master encryption key from Vault."""
        key_b64 = await self.secrets_manager.get_secret(
            self.master_key_path,
            "key"
        )
        if not key_b64:
            # Generate new master key
            key = Fernet.generate_key()
            await self.secrets_manager.set_secret(
                self.master_key_path,
                {"key": key.decode()}
            )
            return key

        return key_b64.encode() if isinstance(key_b64, str) else key_b64

    async def generate_data_key(self) -> tuple[str, bytes]:
        """
        Generate a new data encryption key.

        Returns:
            Tuple of (key_id, encrypted_key)
        """
        # Generate random data key
        data_key = os.urandom(32)  # 256 bits

        # Encrypt data key with master key
        master_key = await self._get_master_key()
        fernet = Fernet(master_key)
        encrypted_key = fernet.encrypt(data_key)

        # Create key metadata
        key_id = str(uuid4())
        key_metadata = EncryptionKey(
            id=key_id,
            key_hash=hashlib.sha256(data_key).hexdigest(),
        )

        self._data_keys[key_id] = data_key
        self._key_metadata[key_id] = key_metadata

        logger.info(f"Generated data key {key_id}")
        return key_id, encrypted_key

    async def get_data_key(
        self,
        key_id: str,
        encrypted_key: bytes,
    ) -> bytes:
        """
        Get a data encryption key by decrypting it.

        Args:
            key_id: Key identifier
            encrypted_key: Encrypted key bytes

        Returns:
            Decrypted data key
        """
        # Check cache
        if key_id in self._data_keys:
            return self._data_keys[key_id]

        # Decrypt using master key
        master_key = await self._get_master_key()
        fernet = Fernet(master_key)
        data_key = fernet.decrypt(encrypted_key)

        # Cache the key
        self._data_keys[key_id] = data_key

        return data_key

    async def encrypt(
        self,
        data: bytes,
        key_id: Optional[str] = None,
    ) -> tuple[str, bytes, bytes]:
        """
        Encrypt data using envelope encryption.

        Args:
            data: Data to encrypt
            key_id: Optional existing key ID

        Returns:
            Tuple of (key_id, encrypted_key, ciphertext)
        """
        if key_id and key_id in self._data_keys:
            data_key = self._data_keys[key_id]
            encrypted_key = b""  # Key already stored
        else:
            key_id, encrypted_key = await self.generate_data_key()
            data_key = self._data_keys[key_id]

        # Encrypt data with AES-256-GCM
        iv = os.urandom(12)  # 96-bit IV for GCM
        cipher = Cipher(
            algorithms.AES(data_key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()

        # Combine IV, auth tag, and ciphertext
        result = iv + encryptor.tag + ciphertext

        return key_id, encrypted_key, result

    async def decrypt(
        self,
        key_id: str,
        encrypted_key: bytes,
        ciphertext: bytes,
    ) -> bytes:
        """
        Decrypt data.

        Args:
            key_id: Key identifier
            encrypted_key: Encrypted data key
            ciphertext: Encrypted data (IV + tag + ciphertext)

        Returns:
            Decrypted data
        """
        # Get data key
        data_key = await self.get_data_key(key_id, encrypted_key)

        # Extract IV, auth tag, and ciphertext
        iv = ciphertext[:12]
        tag = ciphertext[12:28]
        actual_ciphertext = ciphertext[28:]

        # Decrypt
        cipher = Cipher(
            algorithms.AES(data_key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(actual_ciphertext) + decryptor.finalize()

        return plaintext

    async def encrypt_field(
        self,
        value: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Encrypt a single field value.

        Args:
            value: Value to encrypt
            context: Optional context for key derivation

        Returns:
            Encrypted value as base64 string
        """
        import base64

        key_id, encrypted_key, ciphertext = await self.encrypt(value.encode())

        # Combine for storage
        result = {
            "k": key_id,
            "ek": base64.b64encode(encrypted_key).decode() if encrypted_key else "",
            "ct": base64.b64encode(ciphertext).decode(),
        }

        return base64.b64encode(json.dumps(result).encode()).decode()

    async def decrypt_field(self, encrypted_value: str) -> str:
        """Decrypt a field value."""
        import base64

        data = json.loads(base64.b64decode(encrypted_value))
        key_id = data["k"]
        encrypted_key = base64.b64decode(data["ek"]) if data["ek"] else b""
        ciphertext = base64.b64decode(data["ct"])

        plaintext = await self.decrypt(key_id, encrypted_key, ciphertext)
        return plaintext.decode()

    async def rotate_keys(self) -> dict[str, Any]:
        """Rotate encryption keys."""
        # Generate new master key
        new_master_key = Fernet.generate_key()

        # Re-encrypt all data keys with new master key
        old_master_key = await self._get_master_key()
        old_fernet = Fernet(old_master_key)
        new_fernet = Fernet(new_master_key)

        rotated_count = 0
        for key_id, data_key in self._data_keys.items():
            # Re-encrypt data key
            new_encrypted_key = new_fernet.encrypt(data_key)
            rotated_count += 1

        # Store new master key
        await self.secrets_manager.set_secret(
            self.master_key_path,
            {"key": new_master_key.decode()}
        )

        logger.info(f"Rotated {rotated_count} encryption keys")

        return {
            "rotated_keys": rotated_count,
            "timestamp": datetime.utcnow().isoformat(),
        }


# ============================================================================
# Audit Logging Service
# ============================================================================

class AuditLogger:
    """
    Immutable audit logging service.
    Logs all security-relevant events for compliance.
    """

    def __init__(
        self,
        storage_path: Path = Path("/var/log/audit"),
        elasticsearch_url: Optional[str] = None,
    ):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.elasticsearch_url = elasticsearch_url
        self._buffer: list[AuditEvent] = []
        self._buffer_size = 100
        self._lock = asyncio.Lock()

    async def log(self, event: AuditEvent) -> str:
        """
        Log an audit event.

        Args:
            event: Audit event to log

        Returns:
            Event ID
        """
        async with self._lock:
            # Add to buffer
            self._buffer.append(event)

            # Write to file immediately (append-only)
            await self._write_to_file(event)

            # Flush buffer if full
            if len(self._buffer) >= self._buffer_size:
                await self._flush_to_elasticsearch()

        logger.debug(f"Audit event logged: {event.event_type.value}")
        return event.id

    async def _write_to_file(self, event: AuditEvent) -> None:
        """Write event to audit log file."""
        filename = self.storage_path / f"audit-{datetime.utcnow().strftime('%Y-%m-%d')}.log"

        # Create tamper-evident log entry
        log_entry = {
            **event.to_dict(),
            "_hash": self._compute_hash(event),
        }

        with open(filename, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    async def _flush_to_elasticsearch(self) -> None:
        """Flush buffer to Elasticsearch."""
        if not self.elasticsearch_url or not self._buffer:
            return

        # In production, bulk index to Elasticsearch
        # from elasticsearch import AsyncElasticsearch
        # es = AsyncElasticsearch([self.elasticsearch_url])
        # actions = [
        #     {"_index": "audit-logs", "_source": event.to_dict()}
        #     for event in self._buffer
        # ]
        # await helpers.async_bulk(es, actions)

        self._buffer.clear()
        logger.debug("Flushed audit buffer to Elasticsearch")

    def _compute_hash(self, event: AuditEvent) -> str:
        """Compute hash for tamper detection."""
        data = json.dumps(event.to_dict(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    async def query(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit logs."""
        # In production, query Elasticsearch
        # Build query based on parameters

        return []

    async def export(
        self,
        start_time: datetime,
        end_time: datetime,
        format: str = "json",
    ) -> bytes:
        """Export audit logs for compliance reporting."""
        events = await self.query(start_time=start_time, end_time=end_time, limit=10000)

        if format == "json":
            return json.dumps([e.to_dict() for e in events]).encode()
        elif format == "csv":
            # Generate CSV
            import csv
            import io
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=events[0].to_dict().keys() if events else [])
            writer.writeheader()
            for event in events:
                writer.writerow(event.to_dict())
            return output.getvalue().encode()

        raise ValueError(f"Unsupported format: {format}")


# ============================================================================
# Compliance Engine
# ============================================================================

class ComplianceEngine:
    """
    Compliance enforcement engine.
    Ensures data handling meets regulatory requirements.
    """

    def __init__(
        self,
        audit_logger: AuditLogger,
        encryption_service: EncryptionService,
    ):
        self.audit_logger = audit_logger
        self.encryption = encryption_service
        self._policies: dict[str, CompliancePolicy] = {}
        self._retention_policies: dict[str, DataRetentionPolicy] = {}
        self._legal_holds: dict[str, set[str]] = {}  # document_id -> hold_ids

    def register_policy(self, policy: CompliancePolicy) -> None:
        """Register a compliance policy."""
        self._policies[policy.id] = policy
        logger.info(f"Registered compliance policy: {policy.name}")

    def register_retention_policy(self, policy: DataRetentionPolicy) -> None:
        """Register a data retention policy."""
        self._retention_policies[policy.id] = policy
        logger.info(f"Registered retention policy: {policy.name}")

    async def check_compliance(
        self,
        action: str,
        resource_type: str,
        resource_data: dict[str, Any],
        user_id: str,
        frameworks: Optional[list[ComplianceFramework]] = None,
    ) -> tuple[bool, list[str]]:
        """
        Check if an action is compliant with policies.

        Returns:
            Tuple of (is_compliant, list of violations)
        """
        violations = []

        # Filter relevant policies
        relevant_policies = [
            p for p in self._policies.values()
            if p.enabled and (not frameworks or p.framework in frameworks)
        ]

        for policy in relevant_policies:
            for rule in policy.rules:
                is_compliant, violation = await self._check_rule(
                    rule,
                    action,
                    resource_type,
                    resource_data,
                )
                if not is_compliant:
                    violations.append(f"{policy.name}: {violation}")

        return len(violations) == 0, violations

    async def _check_rule(
        self,
        rule: dict[str, Any],
        action: str,
        resource_type: str,
        resource_data: dict[str, Any],
    ) -> tuple[bool, str]:
        """Check a single compliance rule."""
        rule_type = rule.get("type")

        if rule_type == "data_classification":
            # Check data is properly classified
            classification = resource_data.get("classification")
            min_classification = rule.get("min_classification")
            if not classification:
                return False, "Data must have classification"

        elif rule_type == "encryption_required":
            # Check data is encrypted
            if not resource_data.get("encrypted", False):
                return False, "Data must be encrypted"

        elif rule_type == "retention_period":
            # Check retention period
            created_at = resource_data.get("created_at")
            max_days = rule.get("max_days", 2555)
            if created_at:
                age = (datetime.utcnow() - created_at).days
                if age > max_days:
                    return False, f"Data exceeds retention period ({age} > {max_days} days)"

        elif rule_type == "access_logging":
            # Ensure access is logged
            if action in ["read", "download"] and not rule.get("logging_enabled", True):
                return False, "Access logging required"

        elif rule_type == "pii_detection":
            # Check for PII in data
            text = str(resource_data.get("content", ""))
            if self._contains_pii(text):
                if not resource_data.get("pii_protected", False):
                    return False, "PII must be protected"

        return True, ""

    def _contains_pii(self, text: str) -> bool:
        """Check if text contains PII patterns."""
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Phone
        ]

        for pattern in pii_patterns:
            if re.search(pattern, text):
                return True

        return False

    async def apply_legal_hold(
        self,
        document_id: str,
        hold_id: str,
        reason: str,
        user_id: str,
    ) -> bool:
        """Apply legal hold to a document."""
        if document_id not in self._legal_holds:
            self._legal_holds[document_id] = set()

        self._legal_holds[document_id].add(hold_id)

        # Log the event
        await self.audit_logger.log(AuditEvent(
            event_type=AuditEventType.LEGAL_HOLD_APPLIED,
            user_id=user_id,
            resource_type="document",
            resource_id=document_id,
            action="apply_legal_hold",
            details={"hold_id": hold_id, "reason": reason},
        ))

        logger.info(f"Legal hold {hold_id} applied to document {document_id}")
        return True

    async def release_legal_hold(
        self,
        document_id: str,
        hold_id: str,
        user_id: str,
    ) -> bool:
        """Release legal hold from a document."""
        if document_id in self._legal_holds:
            self._legal_holds[document_id].discard(hold_id)

            if not self._legal_holds[document_id]:
                del self._legal_holds[document_id]

        # Log the event
        await self.audit_logger.log(AuditEvent(
            event_type=AuditEventType.LEGAL_HOLD_RELEASED,
            user_id=user_id,
            resource_type="document",
            resource_id=document_id,
            action="release_legal_hold",
            details={"hold_id": hold_id},
        ))

        logger.info(f"Legal hold {hold_id} released from document {document_id}")
        return True

    def is_under_legal_hold(self, document_id: str) -> bool:
        """Check if document is under legal hold."""
        return document_id in self._legal_holds and len(self._legal_holds[document_id]) > 0

    async def handle_gdpr_request(
        self,
        request_type: str,  # access, rectification, erasure, portability
        user_id: str,
        data_subject_id: str,
        request_id: str,
    ) -> dict[str, Any]:
        """Handle GDPR data subject request."""
        result = {
            "request_id": request_id,
            "request_type": request_type,
            "data_subject_id": data_subject_id,
            "status": "completed",
            "details": {},
        }

        if request_type == "access":
            # Return all data for the subject
            result["details"]["data_categories"] = [
                "documents", "metadata", "audit_logs"
            ]

        elif request_type == "erasure":
            # Right to be forgotten
            # Check for legal holds first
            # Then delete or anonymize data
            result["details"]["items_processed"] = 0

        elif request_type == "portability":
            # Export data in portable format
            result["details"]["export_format"] = "json"

        # Log the request
        await self.audit_logger.log(AuditEvent(
            event_type=AuditEventType.GDPR_REQUEST,
            user_id=user_id,
            resource_type="gdpr_request",
            resource_id=request_id,
            action=request_type,
            details=result,
        ))

        return result

    async def check_data_residency(
        self,
        document_id: str,
        required_region: DataResidency,
        current_region: DataResidency,
    ) -> bool:
        """Check if data residency requirements are met."""
        if required_region == DataResidency.GLOBAL:
            return True

        return required_region == current_region

    async def run_retention_check(self) -> dict[str, Any]:
        """Run retention policy check on all documents."""
        results = {
            "checked": 0,
            "flagged_for_deletion": 0,
            "under_legal_hold": 0,
            "errors": 0,
        }

        # In production, iterate through documents
        # For each document:
        # 1. Check if under legal hold
        # 2. Check retention policy
        # 3. Flag for deletion if past retention

        return results


# ============================================================================
# PII Detection Service
# ============================================================================

class PIIDetector:
    """
    Detects and masks Personally Identifiable Information.
    """

    # PII patterns with named groups
    PII_PATTERNS = {
        "ssn": re.compile(r'\b(\d{3})-(\d{2})-(\d{4})\b'),
        "credit_card": re.compile(r'\b(\d{4})[- ]?(\d{4})[- ]?(\d{4})[- ]?(\d{4})\b'),
        "email": re.compile(r'\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b'),
        "phone_us": re.compile(r'\b(\d{3})[-.\s]?(\d{3})[-.\s]?(\d{4})\b'),
        "date_of_birth": re.compile(r'\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/(\d{4})\b'),
        "drivers_license": re.compile(r'\b[A-Z]\d{7}\b'),
        "passport": re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'),
    }

    MASK_STRATEGIES = {
        "full": lambda m: "*" * len(m.group()),
        "partial": lambda m: m.group()[:2] + "*" * (len(m.group()) - 4) + m.group()[-2:],
        "hash": lambda m: hashlib.sha256(m.group().encode()).hexdigest()[:8],
    }

    async def detect(self, text: str) -> list[dict[str, Any]]:
        """
        Detect PII in text.

        Returns:
            List of detected PII with type and location
        """
        findings = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append({
                    "type": pii_type,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9,
                })

        return findings

    async def mask(
        self,
        text: str,
        strategy: str = "partial",
        pii_types: Optional[list[str]] = None,
    ) -> str:
        """
        Mask PII in text.

        Args:
            text: Text to mask
            strategy: Masking strategy (full, partial, hash)
            pii_types: Specific PII types to mask (None = all)

        Returns:
            Text with PII masked
        """
        mask_func = self.MASK_STRATEGIES.get(strategy, self.MASK_STRATEGIES["partial"])

        patterns_to_use = self.PII_PATTERNS
        if pii_types:
            patterns_to_use = {k: v for k, v in self.PII_PATTERNS.items() if k in pii_types}

        for pii_type, pattern in patterns_to_use.items():
            text = pattern.sub(mask_func, text)

        return text

    async def classify_sensitivity(self, text: str) -> str:
        """
        Classify document sensitivity based on PII content.

        Returns:
            Sensitivity level (public, internal, confidential, restricted)
        """
        findings = await self.detect(text)

        if not findings:
            return "internal"

        high_sensitivity_types = {"ssn", "credit_card", "passport"}
        medium_sensitivity_types = {"drivers_license", "date_of_birth"}

        found_types = {f["type"] for f in findings}

        if found_types & high_sensitivity_types:
            return "restricted"
        elif found_types & medium_sensitivity_types:
            return "confidential"
        else:
            return "internal"


# ============================================================================
# Access Control Service
# ============================================================================

class AccessControlService:
    """
    Role-Based Access Control (RBAC) and Attribute-Based Access Control (ABAC).
    """

    # Default role permissions
    ROLE_PERMISSIONS = {
        "viewer": [
            "documents:read",
            "search:read",
        ],
        "editor": [
            "documents:read",
            "documents:write",
            "documents:delete",
            "search:read",
            "search:write",
        ],
        "analyst": [
            "documents:read",
            "analysis:read",
            "analysis:write",
            "search:read",
        ],
        "admin": [
            "*",  # All permissions
        ],
    }

    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self._user_roles: dict[str, set[str]] = {}
        self._user_permissions: dict[str, set[str]] = {}
        self._resource_policies: dict[str, list[dict]] = {}

    def assign_role(
        self,
        user_id: str,
        role: str,
        assigned_by: str,
    ) -> bool:
        """Assign role to user."""
        if user_id not in self._user_roles:
            self._user_roles[user_id] = set()

        self._user_roles[user_id].add(role)

        # Log the assignment
        asyncio.create_task(self.audit_logger.log(AuditEvent(
            event_type=AuditEventType.ROLE_ASSIGN,
            user_id=assigned_by,
            resource_type="user",
            resource_id=user_id,
            action="assign_role",
            details={"role": role},
        )))

        return True

    def revoke_role(
        self,
        user_id: str,
        role: str,
        revoked_by: str,
    ) -> bool:
        """Revoke role from user."""
        if user_id in self._user_roles:
            self._user_roles[user_id].discard(role)

        asyncio.create_task(self.audit_logger.log(AuditEvent(
            event_type=AuditEventType.ROLE_REVOKE,
            user_id=revoked_by,
            resource_type="user",
            resource_id=user_id,
            action="revoke_role",
            details={"role": role},
        )))

        return True

    def check_permission(
        self,
        user_id: str,
        permission: str,
        resource: Optional[dict] = None,
    ) -> bool:
        """
        Check if user has permission.

        Args:
            user_id: User identifier
            permission: Required permission (e.g., "documents:read")
            resource: Optional resource for ABAC

        Returns:
            True if permitted
        """
        # Get user's roles
        roles = self._user_roles.get(user_id, set())

        # Check role-based permissions
        for role in roles:
            role_perms = self.ROLE_PERMISSIONS.get(role, [])

            if "*" in role_perms:
                return True

            if permission in role_perms:
                return True

            # Check wildcard permissions
            resource_type = permission.split(":")[0]
            if f"{resource_type}:*" in role_perms:
                return True

        # Check direct user permissions
        user_perms = self._user_permissions.get(user_id, set())
        if permission in user_perms:
            return True

        # ABAC - check resource-specific policies
        if resource:
            return self._check_abac_policies(user_id, permission, resource)

        return False

    def _check_abac_policies(
        self,
        user_id: str,
        permission: str,
        resource: dict,
    ) -> bool:
        """Check attribute-based access control policies."""
        resource_id = resource.get("id")
        if not resource_id:
            return False

        policies = self._resource_policies.get(resource_id, [])

        for policy in policies:
            if self._evaluate_policy(policy, user_id, permission, resource):
                return True

        return False

    def _evaluate_policy(
        self,
        policy: dict,
        user_id: str,
        permission: str,
        resource: dict,
    ) -> bool:
        """Evaluate a single ABAC policy."""
        # Check subject conditions
        subject_conditions = policy.get("subject", {})
        if subject_conditions.get("user_id") and subject_conditions["user_id"] != user_id:
            return False

        # Check action conditions
        action_conditions = policy.get("action", {})
        if action_conditions.get("permission") and action_conditions["permission"] != permission:
            return False

        # Check resource conditions
        resource_conditions = policy.get("resource", {})
        for attr, value in resource_conditions.items():
            if resource.get(attr) != value:
                return False

        # Check environment conditions (e.g., time, IP)
        env_conditions = policy.get("environment", {})
        # Implement time-based, location-based checks here

        return policy.get("effect", "deny") == "allow"


# ============================================================================
# Security Service Coordinator
# ============================================================================

class SecurityService:
    """
    Main security service coordinating all security components.
    """

    def __init__(
        self,
        vault_addr: str = "http://localhost:8200",
        audit_log_path: Path = Path("/var/log/audit"),
    ):
        self.secrets_manager = SecretsManager(vault_addr)
        self.encryption = EncryptionService(self.secrets_manager)
        self.audit_logger = AuditLogger(audit_log_path)
        self.compliance = ComplianceEngine(self.audit_logger, self.encryption)
        self.pii_detector = PIIDetector()
        self.access_control = AccessControlService(self.audit_logger)

    async def secure_document(
        self,
        document_id: str,
        content: bytes,
        metadata: dict[str, Any],
        user_id: str,
    ) -> dict[str, Any]:
        """
        Apply security controls to a document.

        Args:
            document_id: Document identifier
            content: Document content
            metadata: Document metadata
            user_id: User performing the operation

        Returns:
            Secured document with encryption info
        """
        # 1. Detect PII
        text_content = content.decode("utf-8", errors="ignore")
        pii_findings = await self.pii_detector.detect(text_content)

        # 2. Classify sensitivity
        sensitivity = await self.pii_detector.classify_sensitivity(text_content)
        metadata["classification"] = sensitivity
        metadata["pii_detected"] = len(pii_findings) > 0

        # 3. Encrypt content
        key_id, encrypted_key, ciphertext = await self.encryption.encrypt(content)

        # 4. Check compliance
        is_compliant, violations = await self.compliance.check_compliance(
            action="create",
            resource_type="document",
            resource_data=metadata,
            user_id=user_id,
        )

        if not is_compliant:
            logger.warning(f"Compliance violations for {document_id}: {violations}")

        # 5. Log the operation
        await self.audit_logger.log(AuditEvent(
            event_type=AuditEventType.DOCUMENT_CREATE,
            user_id=user_id,
            resource_type="document",
            resource_id=document_id,
            action="secure_document",
            details={
                "classification": sensitivity,
                "pii_found": len(pii_findings),
                "encrypted": True,
                "compliant": is_compliant,
            },
        ))

        return {
            "document_id": document_id,
            "key_id": key_id,
            "encrypted_key": encrypted_key,
            "ciphertext": ciphertext,
            "metadata": metadata,
            "pii_findings": pii_findings,
            "compliance": {
                "is_compliant": is_compliant,
                "violations": violations,
            },
        }

    async def access_document(
        self,
        document_id: str,
        user_id: str,
        permission: str = "documents:read",
        ip_address: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Check and log document access.

        Returns:
            Tuple of (is_allowed, denial_reason)
        """
        # Check permission
        if not self.access_control.check_permission(user_id, permission):
            await self.audit_logger.log(AuditEvent(
                event_type=AuditEventType.DOCUMENT_VIEW,
                user_id=user_id,
                resource_type="document",
                resource_id=document_id,
                action="access_denied",
                outcome="failure",
                ip_address=ip_address,
                details={"reason": "insufficient_permissions"},
            ))
            return False, "Insufficient permissions"

        # Check legal hold (if trying to delete)
        if permission == "documents:delete":
            if self.compliance.is_under_legal_hold(document_id):
                await self.audit_logger.log(AuditEvent(
                    event_type=AuditEventType.DOCUMENT_DELETE,
                    user_id=user_id,
                    resource_type="document",
                    resource_id=document_id,
                    action="access_denied",
                    outcome="failure",
                    ip_address=ip_address,
                    details={"reason": "legal_hold"},
                ))
                return False, "Document is under legal hold"

        # Log successful access
        await self.audit_logger.log(AuditEvent(
            event_type=AuditEventType.DOCUMENT_VIEW,
            user_id=user_id,
            resource_type="document",
            resource_id=document_id,
            action="access_granted",
            outcome="success",
            ip_address=ip_address,
        ))

        return True, None


# ============================================================================
# FastAPI Application
# ============================================================================

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer

app = FastAPI(
    title="Legal Document Security Service",
    description="Security and compliance controls for legal document platform",
    version="1.0.0",
)

security_service: Optional[SecurityService] = None
security = HTTPBearer()


@app.on_event("startup")
async def startup():
    global security_service
    security_service = SecurityService()
    logger.info("Security service initialized")


@app.post("/api/v1/security/encrypt")
async def encrypt_data(data: str, context: Optional[str] = None):
    """Encrypt sensitive data."""
    if not security_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    encrypted = await security_service.encryption.encrypt_field(data, context)
    return {"encrypted": encrypted}


@app.post("/api/v1/security/decrypt")
async def decrypt_data(encrypted_data: str):
    """Decrypt data."""
    if not security_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    decrypted = await security_service.encryption.decrypt_field(encrypted_data)
    return {"decrypted": decrypted}


@app.post("/api/v1/security/detect-pii")
async def detect_pii(text: str):
    """Detect PII in text."""
    if not security_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    findings = await security_service.pii_detector.detect(text)
    return {"findings": findings, "count": len(findings)}


@app.post("/api/v1/security/mask-pii")
async def mask_pii(text: str, strategy: str = "partial"):
    """Mask PII in text."""
    if not security_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    masked = await security_service.pii_detector.mask(text, strategy)
    return {"masked_text": masked}


@app.post("/api/v1/compliance/check")
async def check_compliance(
    action: str,
    resource_type: str,
    resource_data: dict,
    user_id: str,
):
    """Check compliance for an action."""
    if not security_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    is_compliant, violations = await security_service.compliance.check_compliance(
        action, resource_type, resource_data, user_id
    )
    return {"compliant": is_compliant, "violations": violations}


@app.post("/api/v1/compliance/legal-hold")
async def apply_legal_hold(
    document_id: str,
    hold_id: str,
    reason: str,
    user_id: str,
):
    """Apply legal hold to a document."""
    if not security_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    success = await security_service.compliance.apply_legal_hold(
        document_id, hold_id, reason, user_id
    )
    return {"success": success}


@app.get("/api/v1/audit/logs")
async def query_audit_logs(
    event_type: Optional[str] = None,
    user_id: Optional[str] = None,
    resource_id: Optional[str] = None,
    limit: int = 100,
):
    """Query audit logs."""
    if not security_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    event_type_enum = AuditEventType(event_type) if event_type else None
    events = await security_service.audit_logger.query(
        event_type=event_type_enum,
        user_id=user_id,
        resource_id=resource_id,
        limit=limit,
    )
    return {"events": [e.to_dict() for e in events]}


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "security"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
