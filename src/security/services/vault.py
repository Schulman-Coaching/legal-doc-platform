"""
HashiCorp Vault Integration Service
===================================
Secrets management using HashiCorp Vault.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Optional

from ..config import VaultConfig

logger = logging.getLogger(__name__)


class VaultService:
    """
    HashiCorp Vault integration for secrets management.

    Features:
    - Secret read/write with versioning
    - Dynamic database credentials
    - Encryption key management
    - AppRole authentication
    - Token renewal
    """

    def __init__(self, config: VaultConfig):
        self.config = config
        self._client = None
        self._connected = False
        self._token_expires_at: Optional[datetime] = None
        # In-memory cache for frequently accessed secrets
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

    async def connect(self) -> None:
        """Initialize connection to Vault."""
        try:
            # Import hvac here to avoid import errors if not installed
            import hvac

            self._client = hvac.Client(
                url=self.config.url,
                namespace=self.config.namespace,
                verify=self.config.verify_ssl,
                timeout=self.config.timeout,
            )

            # Authenticate
            if self.config.token:
                self._client.token = self.config.token
            elif self.config.role_id and self.config.secret_id:
                await self._approle_login()
            else:
                raise ValueError("Either token or role_id/secret_id required")

            # Verify connection
            if not self._client.is_authenticated():
                raise ConnectionError("Failed to authenticate with Vault")

            self._connected = True
            logger.info("Connected to Vault at %s", self.config.url)

        except ImportError:
            logger.warning("hvac not installed, using mock Vault client")
            self._client = MockVaultClient()
            self._connected = True
        except Exception as e:
            logger.error("Failed to connect to Vault: %s", str(e))
            raise

    async def disconnect(self) -> None:
        """Close connection to Vault."""
        if self._client and hasattr(self._client, 'adapter'):
            self._client.adapter.close()
        self._client = None
        self._connected = False
        self._cache.clear()
        logger.info("Disconnected from Vault")

    async def _approle_login(self) -> None:
        """Authenticate using AppRole."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.auth.approle.login(
                role_id=self.config.role_id,
                secret_id=self.config.secret_id,
            )
        )
        self._client.token = response['auth']['client_token']
        ttl = response['auth']['lease_duration']
        self._token_expires_at = datetime.utcnow() + timedelta(seconds=ttl)
        logger.debug("AppRole login successful, token expires in %d seconds", ttl)

    async def _ensure_authenticated(self) -> None:
        """Ensure we have a valid authentication token."""
        if self._token_expires_at:
            if datetime.utcnow() >= self._token_expires_at - timedelta(minutes=5):
                await self._approle_login()

    def _require_connection(func: Callable) -> Callable:
        """Decorator to ensure connection before operation."""
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not self._connected:
                raise RuntimeError("Vault service not connected")
            await self._ensure_authenticated()
            return await func(self, *args, **kwargs)
        return wrapper

    # =========================================================================
    # Secret Operations
    # =========================================================================

    @_require_connection
    async def get_secret(
        self,
        path: str,
        version: Optional[int] = None,
        use_cache: bool = True,
    ) -> Optional[dict[str, Any]]:
        """
        Retrieve a secret from Vault.

        Args:
            path: Secret path (relative to mount point)
            version: Specific version to retrieve (None for latest)
            use_cache: Whether to use cached value

        Returns:
            Secret data as dictionary or None if not found
        """
        cache_key = f"{path}:{version}"

        # Check cache
        if use_cache and cache_key in self._cache:
            data, cached_at = self._cache[cache_key]
            if datetime.utcnow() - cached_at < self._cache_ttl:
                return data

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.secrets.kv.v2.read_secret_version(
                    path=path,
                    mount_point=self.config.mount_point,
                    version=version,
                )
            )
            data = response['data']['data']

            # Update cache
            self._cache[cache_key] = (data, datetime.utcnow())

            return data
        except Exception as e:
            if "404" in str(e) or "InvalidPath" in str(e):
                return None
            logger.error("Failed to get secret %s: %s", path, str(e))
            raise

    @_require_connection
    async def set_secret(
        self,
        path: str,
        data: dict[str, Any],
        cas: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Store a secret in Vault.

        Args:
            path: Secret path
            data: Secret data to store
            cas: Check-and-set version (for optimistic locking)

        Returns:
            Metadata about the created secret version
        """
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.secrets.kv.v2.create_or_update_secret(
                    path=path,
                    secret=data,
                    mount_point=self.config.mount_point,
                    cas=cas,
                )
            )

            # Invalidate cache
            self._invalidate_cache(path)

            logger.info("Secret stored at %s", path)
            return response['data']
        except Exception as e:
            logger.error("Failed to set secret %s: %s", path, str(e))
            raise

    @_require_connection
    async def delete_secret(
        self,
        path: str,
        versions: Optional[list[int]] = None,
    ) -> bool:
        """
        Delete a secret or specific versions.

        Args:
            path: Secret path
            versions: Specific versions to delete (None for all)

        Returns:
            True if successful
        """
        try:
            loop = asyncio.get_event_loop()

            if versions:
                await loop.run_in_executor(
                    None,
                    lambda: self._client.secrets.kv.v2.delete_secret_versions(
                        path=path,
                        versions=versions,
                        mount_point=self.config.mount_point,
                    )
                )
            else:
                await loop.run_in_executor(
                    None,
                    lambda: self._client.secrets.kv.v2.delete_metadata_and_all_versions(
                        path=path,
                        mount_point=self.config.mount_point,
                    )
                )

            self._invalidate_cache(path)
            logger.info("Secret deleted at %s", path)
            return True
        except Exception as e:
            logger.error("Failed to delete secret %s: %s", path, str(e))
            raise

    @_require_connection
    async def list_secrets(self, path: str = "") -> list[str]:
        """List secrets at a path."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.secrets.kv.v2.list_secrets(
                    path=path,
                    mount_point=self.config.mount_point,
                )
            )
            return response['data']['keys']
        except Exception as e:
            if "404" in str(e):
                return []
            logger.error("Failed to list secrets at %s: %s", path, str(e))
            raise

    # =========================================================================
    # Encryption Key Management
    # =========================================================================

    @_require_connection
    async def get_encryption_key(self, key_id: str) -> Optional[bytes]:
        """
        Retrieve an encryption key from Vault.

        Args:
            key_id: Key identifier

        Returns:
            Key bytes or None if not found
        """
        path = f"{self.config.encryption_key_path}/{key_id}"
        secret = await self.get_secret(path)
        if secret and 'key' in secret:
            import base64
            return base64.b64decode(secret['key'])
        return None

    @_require_connection
    async def store_encryption_key(
        self,
        key_id: str,
        key_bytes: bytes,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Store an encryption key in Vault.

        Args:
            key_id: Key identifier
            key_bytes: Raw key bytes
            metadata: Additional metadata
        """
        import base64

        path = f"{self.config.encryption_key_path}/{key_id}"
        data = {
            'key': base64.b64encode(key_bytes).decode('utf-8'),
            'created_at': datetime.utcnow().isoformat(),
            **(metadata or {}),
        }
        await self.set_secret(path, data)
        logger.info("Encryption key stored: %s", key_id)

    @_require_connection
    async def rotate_encryption_key(
        self,
        key_id: str,
        new_key_bytes: bytes,
    ) -> str:
        """
        Rotate an encryption key (store new version).

        Args:
            key_id: Key identifier
            new_key_bytes: New key bytes

        Returns:
            New version identifier
        """
        import base64

        path = f"{self.config.encryption_key_path}/{key_id}"

        # Get current metadata
        current = await self.get_secret(path)
        version = (current.get('version', 0) + 1) if current else 1

        data = {
            'key': base64.b64encode(new_key_bytes).decode('utf-8'),
            'version': version,
            'rotated_at': datetime.utcnow().isoformat(),
            'previous_version': current.get('version') if current else None,
        }
        await self.set_secret(path, data)

        logger.info("Encryption key rotated: %s (version %d)", key_id, version)
        return f"{key_id}:v{version}"

    # =========================================================================
    # Database Credentials
    # =========================================================================

    @_require_connection
    async def get_database_credentials(
        self,
        database: str,
        role: str = "readonly",
    ) -> dict[str, str]:
        """
        Get dynamic database credentials from Vault.

        Args:
            database: Database name
            role: Database role (readonly, readwrite, admin)

        Returns:
            Credentials dict with username, password
        """
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.secrets.database.generate_credentials(
                    name=f"{database}-{role}",
                )
            )
            return {
                'username': response['data']['username'],
                'password': response['data']['password'],
                'lease_id': response['lease_id'],
                'lease_duration': response['lease_duration'],
            }
        except Exception as e:
            logger.error("Failed to get database credentials: %s", str(e))
            raise

    # =========================================================================
    # API Keys
    # =========================================================================

    @_require_connection
    async def get_api_key(self, service: str) -> Optional[str]:
        """Get an API key for an external service."""
        path = f"{self.config.api_keys_path}/{service}"
        secret = await self.get_secret(path)
        return secret.get('api_key') if secret else None

    @_require_connection
    async def set_api_key(
        self,
        service: str,
        api_key: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Store an API key for an external service."""
        path = f"{self.config.api_keys_path}/{service}"
        data = {
            'api_key': api_key,
            'service': service,
            'updated_at': datetime.utcnow().isoformat(),
            **(metadata or {}),
        }
        await self.set_secret(path, data)

    # =========================================================================
    # Transit Encryption (Vault-managed encryption)
    # =========================================================================

    @_require_connection
    async def transit_encrypt(
        self,
        plaintext: bytes,
        key_name: str = "legal-docs",
    ) -> str:
        """
        Encrypt data using Vault Transit engine.

        Args:
            plaintext: Data to encrypt
            key_name: Transit key name

        Returns:
            Ciphertext (vault:v1:... format)
        """
        import base64

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.secrets.transit.encrypt_data(
                name=key_name,
                plaintext=base64.b64encode(plaintext).decode('utf-8'),
            )
        )
        return response['data']['ciphertext']

    @_require_connection
    async def transit_decrypt(
        self,
        ciphertext: str,
        key_name: str = "legal-docs",
    ) -> bytes:
        """
        Decrypt data using Vault Transit engine.

        Args:
            ciphertext: Data to decrypt
            key_name: Transit key name

        Returns:
            Decrypted plaintext
        """
        import base64

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.secrets.transit.decrypt_data(
                name=key_name,
                ciphertext=ciphertext,
            )
        )
        return base64.b64decode(response['data']['plaintext'])

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _invalidate_cache(self, path: str) -> None:
        """Invalidate cached entries for a path."""
        keys_to_remove = [k for k in self._cache if k.startswith(path)]
        for key in keys_to_remove:
            del self._cache[key]

    async def health_check(self) -> dict[str, Any]:
        """Check Vault health status."""
        if not self._connected:
            return {"status": "disconnected"}

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.sys.read_health_status(method='GET')
            )
            return {
                "status": "healthy",
                "initialized": response.get('initialized', True),
                "sealed": response.get('sealed', False),
                "version": response.get('version', 'unknown'),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }


class MockVaultClient:
    """Mock Vault client for testing without Vault."""

    def __init__(self):
        self._secrets: dict[str, dict[str, Any]] = {}
        self.token = "mock-token"

    def is_authenticated(self) -> bool:
        return True

    @property
    def secrets(self):
        return self

    @property
    def kv(self):
        return self

    @property
    def v2(self):
        return self

    def read_secret_version(
        self,
        path: str,
        mount_point: str = "secret",
        version: Optional[int] = None,
    ) -> dict:
        key = f"{mount_point}/{path}"
        if key not in self._secrets:
            raise Exception("InvalidPath: secret not found (404)")
        return {
            'data': {
                'data': self._secrets[key],
                'metadata': {'version': 1},
            }
        }

    def create_or_update_secret(
        self,
        path: str,
        secret: dict,
        mount_point: str = "secret",
        cas: Optional[int] = None,
    ) -> dict:
        key = f"{mount_point}/{path}"
        self._secrets[key] = secret
        return {'data': {'version': 1}}

    def delete_metadata_and_all_versions(
        self,
        path: str,
        mount_point: str = "secret",
    ) -> None:
        key = f"{mount_point}/{path}"
        if key in self._secrets:
            del self._secrets[key]

    def delete_secret_versions(
        self,
        path: str,
        versions: list[int],
        mount_point: str = "secret",
    ) -> None:
        pass

    def list_secrets(
        self,
        path: str,
        mount_point: str = "secret",
    ) -> dict:
        prefix = f"{mount_point}/{path}"
        keys = [
            k.replace(prefix, '').lstrip('/')
            for k in self._secrets
            if k.startswith(prefix)
        ]
        return {'data': {'keys': keys}}

    @property
    def sys(self):
        return MockSys()

    @property
    def database(self):
        return MockDatabase()

    @property
    def transit(self):
        return MockTransit()

    @property
    def auth(self):
        return MockAuth()


class MockSys:
    def read_health_status(self, method: str = 'GET') -> dict:
        return {
            'initialized': True,
            'sealed': False,
            'version': 'mock-1.0.0',
        }


class MockDatabase:
    def generate_credentials(self, name: str) -> dict:
        return {
            'data': {
                'username': f'mock-{name}-user',
                'password': 'mock-password',
            },
            'lease_id': 'mock-lease-id',
            'lease_duration': 3600,
        }


class MockTransit:
    def encrypt_data(self, name: str, plaintext: str) -> dict:
        return {'data': {'ciphertext': f'vault:v1:{plaintext}'}}

    def decrypt_data(self, name: str, ciphertext: str) -> dict:
        # Remove vault:v1: prefix
        plaintext = ciphertext.replace('vault:v1:', '')
        return {'data': {'plaintext': plaintext}}


class MockAuth:
    @property
    def approle(self):
        return MockAppRole()


class MockAppRole:
    def login(self, role_id: str, secret_id: str) -> dict:
        return {
            'auth': {
                'client_token': 'mock-client-token',
                'lease_duration': 3600,
            }
        }
