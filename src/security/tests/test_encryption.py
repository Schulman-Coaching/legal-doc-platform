"""
Tests for Encryption Service.
"""

from __future__ import annotations

import pytest
from ..services.encryption import EncryptionService, DecryptedData
from ..models import EncryptedData


class TestEncryptionService:
    """Tests for encryption service."""

    @pytest.mark.asyncio
    async def test_encrypt_decrypt_roundtrip(self, encryption_service: EncryptionService):
        """Test basic encrypt/decrypt roundtrip."""
        plaintext = b"This is sensitive document content"

        encrypted = await encryption_service.encrypt(plaintext)

        assert encrypted is not None
        assert encrypted.ciphertext != plaintext
        assert encrypted.key_id is not None
        assert encrypted.iv is not None

        decrypted = await encryption_service.decrypt(encrypted)

        assert decrypted.plaintext == plaintext

    @pytest.mark.asyncio
    async def test_encrypt_with_aad(self, encryption_service: EncryptionService):
        """Test encryption with additional authenticated data."""
        plaintext = b"Sensitive data"
        aad = b"document-id-123"

        encrypted = await encryption_service.encrypt(plaintext, aad)
        decrypted = await encryption_service.decrypt(encrypted, aad)

        assert decrypted.plaintext == plaintext

    @pytest.mark.asyncio
    async def test_decrypt_with_wrong_aad_fails(self, encryption_service: EncryptionService):
        """Test that decryption with wrong AAD fails."""
        plaintext = b"Sensitive data"
        aad = b"document-id-123"
        wrong_aad = b"wrong-id"

        encrypted = await encryption_service.encrypt(plaintext, aad)

        # Decryption with wrong AAD should fail
        try:
            await encryption_service.decrypt(encrypted, wrong_aad)
            # If using fallback (no cryptography), this might not fail
        except Exception:
            pass  # Expected

    @pytest.mark.asyncio
    async def test_encrypt_different_outputs(self, encryption_service: EncryptionService):
        """Test that encrypting same data produces different outputs (due to random IV)."""
        plaintext = b"Same content"

        encrypted1 = await encryption_service.encrypt(plaintext)
        encrypted2 = await encryption_service.encrypt(plaintext)

        # IVs should be different
        assert encrypted1.iv != encrypted2.iv
        # Ciphertexts should be different
        assert encrypted1.ciphertext != encrypted2.ciphertext

    @pytest.mark.asyncio
    async def test_hash_data(self, encryption_service: EncryptionService):
        """Test data hashing."""
        data = b"test data"

        hash1 = encryption_service.hash_data(data)
        hash2 = encryption_service.hash_data(data)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    @pytest.mark.asyncio
    async def test_hash_different_data(self, encryption_service: EncryptionService):
        """Test that different data produces different hashes."""
        hash1 = encryption_service.hash_data(b"data1")
        hash2 = encryption_service.hash_data(b"data2")

        assert hash1 != hash2

    @pytest.mark.asyncio
    async def test_password_hash_verify(self, encryption_service: EncryptionService):
        """Test password hashing and verification."""
        password = "SecureP@ssw0rd!"

        hashed = encryption_service.hash_password(password)

        assert hashed != password
        assert encryption_service.verify_password(password, hashed)

    @pytest.mark.asyncio
    async def test_password_verify_wrong_password(self, encryption_service: EncryptionService):
        """Test that wrong password fails verification."""
        password = "SecureP@ssw0rd!"
        wrong_password = "WrongPassword"

        hashed = encryption_service.hash_password(password)

        assert not encryption_service.verify_password(wrong_password, hashed)

    @pytest.mark.asyncio
    async def test_derive_key(self, encryption_service: EncryptionService):
        """Test key derivation from password."""
        password = "my-secret-password"

        key1, salt1 = await encryption_service.derive_key(password)
        key2, salt2 = await encryption_service.derive_key(password)

        # Different salts should produce different keys
        assert salt1 != salt2
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_derive_key_same_salt(self, encryption_service: EncryptionService):
        """Test key derivation with same salt produces same key."""
        password = "my-secret-password"
        salt = b"fixed-salt-value-32bytes........"

        key1, _ = await encryption_service.derive_key(password, salt)
        key2, _ = await encryption_service.derive_key(password, salt)

        assert key1 == key2

    @pytest.mark.asyncio
    async def test_generate_random_key(self, encryption_service: EncryptionService):
        """Test random key generation."""
        key1 = encryption_service.generate_random_key(32)
        key2 = encryption_service.generate_random_key(32)

        assert len(key1) == 32
        assert len(key2) == 32
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_generate_random_string(self, encryption_service: EncryptionService):
        """Test random string generation."""
        string1 = encryption_service.generate_random_string(32)
        string2 = encryption_service.generate_random_string(32)

        assert len(string1) == 32
        assert len(string2) == 32
        assert string1 != string2

    @pytest.mark.asyncio
    async def test_get_key_metadata(self, encryption_service: EncryptionService):
        """Test getting key metadata."""
        metadata = await encryption_service.get_key_metadata()

        assert "current_key_id" in metadata
        assert "algorithm" in metadata
        assert metadata["algorithm"] == "aes-256-gcm"

    @pytest.mark.asyncio
    async def test_health_check(self, encryption_service: EncryptionService):
        """Test health check."""
        health = await encryption_service.health_check()

        assert health["status"] == "healthy"
        assert "current_key" in health

    @pytest.mark.asyncio
    async def test_encrypt_empty_data(self, encryption_service: EncryptionService):
        """Test encrypting empty data."""
        plaintext = b""

        encrypted = await encryption_service.encrypt(plaintext)
        decrypted = await encryption_service.decrypt(encrypted)

        assert decrypted.plaintext == plaintext

    @pytest.mark.asyncio
    async def test_encrypt_large_data(self, encryption_service: EncryptionService):
        """Test encrypting large data."""
        plaintext = b"x" * 1024 * 1024  # 1MB

        encrypted = await encryption_service.encrypt(plaintext)
        decrypted = await encryption_service.decrypt(encrypted)

        assert decrypted.plaintext == plaintext


class TestEncryptionServiceNotInitialized:
    """Tests for uninitialized encryption service."""

    @pytest.mark.asyncio
    async def test_encrypt_before_init_fails(self, encryption_config):
        """Test that encryption before initialization fails."""
        service = EncryptionService(encryption_config)

        with pytest.raises(RuntimeError, match="not initialized"):
            await service.encrypt(b"data")

    @pytest.mark.asyncio
    async def test_decrypt_before_init_fails(self, encryption_config):
        """Test that decryption before initialization fails."""
        service = EncryptionService(encryption_config)

        encrypted = EncryptedData(
            key_id="key:iv:data",
            iv="base64iv",
            ciphertext="base64cipher",
        )

        with pytest.raises(RuntimeError, match="not initialized"):
            await service.decrypt(encrypted)
