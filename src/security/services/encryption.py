"""
Encryption Service
==================
Envelope encryption with AES-256-GCM.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import uuid4

from ..config import EncryptionConfig
from ..models import EncryptedData, EncryptionAlgorithm, EncryptionKey

logger = logging.getLogger(__name__)


@dataclass
class DecryptedData:
    """Result of decryption operation."""
    plaintext: bytes
    key_id: str
    algorithm: str
    decrypted_at: datetime


class EncryptionService:
    """
    Envelope encryption service using AES-256-GCM.

    Envelope encryption:
    1. Generate a unique Data Encryption Key (DEK) for each piece of data
    2. Encrypt data with DEK
    3. Encrypt DEK with Master Encryption Key (MEK) from Vault
    4. Store encrypted DEK with the encrypted data

    Features:
    - AES-256-GCM authenticated encryption
    - Envelope encryption pattern
    - Key rotation support
    - Multiple key versions
    - Vault integration for MEK storage
    """

    def __init__(self, config: EncryptionConfig, vault_service: Optional[Any] = None):
        self.config = config
        self.vault = vault_service
        self._master_keys: dict[str, bytes] = {}
        self._current_key_id: Optional[str] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize encryption service with master key from Vault."""
        try:
            if self.vault and self.config.key_storage == "vault":
                # Get master key from Vault
                key_data = await self.vault.get_secret("encryption/master")
                if key_data and 'key' in key_data:
                    key_bytes = base64.b64decode(key_data['key'])
                    key_id = key_data.get('key_id', 'master-v1')
                    self._master_keys[key_id] = key_bytes
                    self._current_key_id = key_id
                else:
                    # Generate new master key
                    await self._generate_master_key()
            elif self.config.local_key_file:
                # Load from local file (development only)
                await self._load_local_key()
            else:
                # Generate ephemeral key (testing only)
                logger.warning("Using ephemeral master key - data will not persist!")
                key_id = "ephemeral-v1"
                self._master_keys[key_id] = secrets.token_bytes(32)
                self._current_key_id = key_id

            self._initialized = True
            logger.info("Encryption service initialized with key: %s", self._current_key_id)

        except Exception as e:
            logger.error("Failed to initialize encryption service: %s", str(e))
            raise

    async def _generate_master_key(self) -> None:
        """Generate and store a new master key in Vault."""
        key_id = f"master-v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        key_bytes = secrets.token_bytes(32)

        if self.vault:
            await self.vault.store_encryption_key(
                key_id="master",
                key_bytes=key_bytes,
                metadata={'key_id': key_id, 'algorithm': self.config.algorithm},
            )

        self._master_keys[key_id] = key_bytes
        self._current_key_id = key_id
        logger.info("Generated new master key: %s", key_id)

    async def _load_local_key(self) -> None:
        """Load master key from local file."""
        if not self.config.local_key_file:
            raise ValueError("Local key file not configured")

        if os.path.exists(self.config.local_key_file):
            with open(self.config.local_key_file, 'rb') as f:
                key_data = f.read()
            key_id = "local-v1"
        else:
            # Generate and save
            key_data = secrets.token_bytes(32)
            os.makedirs(os.path.dirname(self.config.local_key_file), exist_ok=True)
            with open(self.config.local_key_file, 'wb') as f:
                f.write(key_data)
            key_id = "local-v1"
            logger.info("Generated local master key")

        self._master_keys[key_id] = key_data
        self._current_key_id = key_id

    def _ensure_initialized(self) -> None:
        """Ensure service is initialized."""
        if not self._initialized:
            raise RuntimeError("Encryption service not initialized")

    # =========================================================================
    # Encryption Operations
    # =========================================================================

    async def encrypt(
        self,
        plaintext: bytes,
        associated_data: Optional[bytes] = None,
    ) -> EncryptedData:
        """
        Encrypt data using envelope encryption.

        Args:
            plaintext: Data to encrypt
            associated_data: Additional authenticated data (not encrypted)

        Returns:
            EncryptedData envelope containing encrypted data and key info
        """
        self._ensure_initialized()

        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            # Fallback for environments without cryptography
            return await self._encrypt_fallback(plaintext)

        # Generate Data Encryption Key (DEK)
        dek = secrets.token_bytes(32)
        iv = secrets.token_bytes(12)  # 96-bit IV for GCM

        # Encrypt data with DEK
        aesgcm = AESGCM(dek)
        ciphertext = aesgcm.encrypt(iv, plaintext, associated_data)

        # Encrypt DEK with Master Key (envelope)
        master_key = self._master_keys[self._current_key_id]
        master_aesgcm = AESGCM(master_key)
        dek_iv = secrets.token_bytes(12)
        encrypted_dek = master_aesgcm.encrypt(dek_iv, dek, None)

        # Combine encrypted DEK info into key_id field
        key_info = f"{self._current_key_id}:{base64.b64encode(dek_iv).decode()}:{base64.b64encode(encrypted_dek).decode()}"

        return EncryptedData(
            key_id=key_info,
            algorithm=self.config.algorithm,
            iv=base64.b64encode(iv).decode(),
            ciphertext=base64.b64encode(ciphertext).decode(),
            encrypted_at=datetime.utcnow(),
        )

    async def decrypt(
        self,
        encrypted_data: EncryptedData,
        associated_data: Optional[bytes] = None,
    ) -> DecryptedData:
        """
        Decrypt envelope-encrypted data.

        Args:
            encrypted_data: EncryptedData envelope
            associated_data: Additional authenticated data (must match encryption)

        Returns:
            DecryptedData with plaintext
        """
        self._ensure_initialized()

        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            return await self._decrypt_fallback(encrypted_data)

        # Parse key info
        parts = encrypted_data.key_id.split(':')
        if len(parts) != 3:
            raise ValueError("Invalid key_id format")

        master_key_id, dek_iv_b64, encrypted_dek_b64 = parts
        dek_iv = base64.b64decode(dek_iv_b64)
        encrypted_dek = base64.b64decode(encrypted_dek_b64)

        # Get master key
        if master_key_id not in self._master_keys:
            # Try to load from Vault
            if self.vault:
                key_bytes = await self.vault.get_encryption_key("master")
                if key_bytes:
                    self._master_keys[master_key_id] = key_bytes
                else:
                    raise ValueError(f"Master key not found: {master_key_id}")
            else:
                raise ValueError(f"Master key not found: {master_key_id}")

        master_key = self._master_keys[master_key_id]

        # Decrypt DEK
        master_aesgcm = AESGCM(master_key)
        dek = master_aesgcm.decrypt(dek_iv, encrypted_dek, None)

        # Decrypt data
        iv = base64.b64decode(encrypted_data.iv)
        ciphertext = base64.b64decode(encrypted_data.ciphertext)
        aesgcm = AESGCM(dek)
        plaintext = aesgcm.decrypt(iv, ciphertext, associated_data)

        return DecryptedData(
            plaintext=plaintext,
            key_id=master_key_id,
            algorithm=encrypted_data.algorithm,
            decrypted_at=datetime.utcnow(),
        )

    async def _encrypt_fallback(self, plaintext: bytes) -> EncryptedData:
        """Fallback encryption using basic XOR (for testing only)."""
        logger.warning("Using fallback encryption - NOT SECURE!")
        key = self._master_keys[self._current_key_id]
        iv = secrets.token_bytes(16)

        # Simple XOR with key (NOT SECURE - testing only)
        extended_key = (key * ((len(plaintext) // len(key)) + 1))[:len(plaintext)]
        ciphertext = bytes(a ^ b for a, b in zip(plaintext, extended_key))

        return EncryptedData(
            key_id=f"{self._current_key_id}:fallback:fallback",
            algorithm="xor-fallback",
            iv=base64.b64encode(iv).decode(),
            ciphertext=base64.b64encode(ciphertext).decode(),
            encrypted_at=datetime.utcnow(),
        )

    async def _decrypt_fallback(self, encrypted_data: EncryptedData) -> DecryptedData:
        """Fallback decryption."""
        parts = encrypted_data.key_id.split(':')
        master_key_id = parts[0]
        key = self._master_keys[master_key_id]
        ciphertext = base64.b64decode(encrypted_data.ciphertext)

        extended_key = (key * ((len(ciphertext) // len(key)) + 1))[:len(ciphertext)]
        plaintext = bytes(a ^ b for a, b in zip(ciphertext, extended_key))

        return DecryptedData(
            plaintext=plaintext,
            key_id=master_key_id,
            algorithm="xor-fallback",
            decrypted_at=datetime.utcnow(),
        )

    # =========================================================================
    # Key Derivation
    # =========================================================================

    async def derive_key(
        self,
        password: str,
        salt: Optional[bytes] = None,
        iterations: Optional[int] = None,
    ) -> tuple[bytes, bytes]:
        """
        Derive a key from password using PBKDF2.

        Args:
            password: Password to derive from
            salt: Salt bytes (generated if not provided)
            iterations: Number of iterations

        Returns:
            Tuple of (derived_key, salt)
        """
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        except ImportError:
            # Fallback using hashlib
            return self._derive_key_fallback(password, salt, iterations)

        if salt is None:
            salt = secrets.token_bytes(self.config.salt_length)

        if iterations is None:
            iterations = self.config.kdf_iterations

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
        )
        derived_key = kdf.derive(password.encode())
        return derived_key, salt

    def _derive_key_fallback(
        self,
        password: str,
        salt: Optional[bytes] = None,
        iterations: Optional[int] = None,
    ) -> tuple[bytes, bytes]:
        """Fallback key derivation using hashlib."""
        if salt is None:
            salt = secrets.token_bytes(self.config.salt_length)
        if iterations is None:
            iterations = self.config.kdf_iterations

        derived_key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt,
            iterations,
            dklen=32,
        )
        return derived_key, salt

    # =========================================================================
    # Key Rotation
    # =========================================================================

    async def rotate_master_key(self) -> str:
        """
        Rotate the master encryption key.

        Returns:
            New key ID
        """
        self._ensure_initialized()

        # Generate new key
        new_key_id = f"master-v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        new_key = secrets.token_bytes(32)

        # Store in Vault
        if self.vault:
            await self.vault.rotate_encryption_key("master", new_key)

        # Keep old key for decryption
        old_key_id = self._current_key_id

        # Add new key
        self._master_keys[new_key_id] = new_key
        self._current_key_id = new_key_id

        logger.info("Rotated master key from %s to %s", old_key_id, new_key_id)
        return new_key_id

    async def re_encrypt(
        self,
        encrypted_data: EncryptedData,
        associated_data: Optional[bytes] = None,
    ) -> EncryptedData:
        """
        Re-encrypt data with current master key.

        Useful after key rotation.
        """
        # Decrypt with old key
        decrypted = await self.decrypt(encrypted_data, associated_data)

        # Encrypt with new key
        return await self.encrypt(decrypted.plaintext, associated_data)

    # =========================================================================
    # Hashing Operations
    # =========================================================================

    def hash_data(self, data: bytes) -> str:
        """
        Generate SHA-256 hash of data.

        Args:
            data: Data to hash

        Returns:
            Hex-encoded hash
        """
        return hashlib.sha256(data).hexdigest()

    def hash_password(self, password: str) -> str:
        """
        Hash a password for storage.

        Args:
            password: Password to hash

        Returns:
            Hash string with salt
        """
        try:
            import bcrypt
            return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        except ImportError:
            # Fallback using PBKDF2
            salt = secrets.token_bytes(16)
            hash_bytes = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
            return f"pbkdf2${base64.b64encode(salt).decode()}${base64.b64encode(hash_bytes).decode()}"

    def verify_password(self, password: str, hash_string: str) -> bool:
        """
        Verify a password against a hash.

        Args:
            password: Password to verify
            hash_string: Stored hash

        Returns:
            True if password matches
        """
        try:
            import bcrypt
            if hash_string.startswith('$2'):
                return bcrypt.checkpw(password.encode(), hash_string.encode())
        except ImportError:
            pass

        # Fallback PBKDF2 verification
        if hash_string.startswith('pbkdf2$'):
            parts = hash_string.split('$')
            if len(parts) == 3:
                salt = base64.b64decode(parts[1])
                stored_hash = base64.b64decode(parts[2])
                computed_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
                return secrets.compare_digest(computed_hash, stored_hash)

        return False

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def generate_random_key(self, length: int = 32) -> bytes:
        """Generate cryptographically secure random bytes."""
        return secrets.token_bytes(length)

    def generate_random_string(self, length: int = 32) -> str:
        """Generate cryptographically secure random hex string."""
        return secrets.token_hex(length // 2)

    async def get_key_metadata(self) -> dict[str, Any]:
        """Get metadata about current encryption keys."""
        return {
            'current_key_id': self._current_key_id,
            'algorithm': self.config.algorithm,
            'key_count': len(self._master_keys),
            'key_ids': list(self._master_keys.keys()),
            'envelope_encryption': self.config.envelope_encryption,
        }

    async def health_check(self) -> dict[str, Any]:
        """Check encryption service health."""
        if not self._initialized:
            return {"status": "not_initialized"}

        try:
            # Test encryption/decryption
            test_data = b"health-check-test"
            encrypted = await self.encrypt(test_data)
            decrypted = await self.decrypt(encrypted)

            if decrypted.plaintext == test_data:
                return {
                    "status": "healthy",
                    "current_key": self._current_key_id,
                    "algorithm": self.config.algorithm,
                }
            return {"status": "unhealthy", "error": "encryption test failed"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
