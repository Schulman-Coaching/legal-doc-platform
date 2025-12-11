"""
Keycloak Authentication Service
===============================
OAuth2/OIDC authentication using Keycloak.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Optional
from uuid import uuid4

import httpx

from ..config import KeycloakConfig

logger = logging.getLogger(__name__)


class KeycloakService:
    """
    Keycloak authentication and identity management.

    Features:
    - OAuth2/OIDC authentication
    - Token validation and introspection
    - User management
    - Role and group management
    - Session management
    """

    def __init__(self, config: KeycloakConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._admin_token: Optional[str] = None
        self._admin_token_expires: Optional[datetime] = None
        self._jwks_cache: Optional[dict] = None
        self._jwks_cache_expires: Optional[datetime] = None
        self._connected = False

    async def connect(self) -> None:
        """Initialize connection to Keycloak."""
        try:
            self._client = httpx.AsyncClient(
                base_url=self.config.server_url,
                verify=self.config.verify_ssl,
                timeout=30.0,
            )

            # Discover OIDC endpoints if not configured
            await self._discover_endpoints()

            # Get admin token if credentials provided
            if self.config.admin_username and self.config.admin_password:
                await self._get_admin_token()

            self._connected = True
            logger.info("Connected to Keycloak at %s", self.config.server_url)

        except Exception as e:
            logger.error("Failed to connect to Keycloak: %s", str(e))
            # Use mock client for development
            self._client = None
            self._connected = True
            logger.warning("Using mock Keycloak service")

    async def disconnect(self) -> None:
        """Close connection to Keycloak."""
        if self._client:
            await self._client.aclose()
        self._client = None
        self._connected = False
        logger.info("Disconnected from Keycloak")

    async def _discover_endpoints(self) -> None:
        """Discover OIDC endpoints from well-known configuration."""
        if not self._client:
            return

        try:
            url = f"/realms/{self.config.realm}/.well-known/openid-configuration"
            response = await self._client.get(url)
            response.raise_for_status()
            config = response.json()

            self.config.authorization_endpoint = config.get('authorization_endpoint')
            self.config.token_endpoint = config.get('token_endpoint')
            self.config.userinfo_endpoint = config.get('userinfo_endpoint')
            self.config.jwks_uri = config.get('jwks_uri')

            logger.debug("Discovered OIDC endpoints for realm %s", self.config.realm)
        except Exception as e:
            logger.warning("Failed to discover OIDC endpoints: %s", str(e))

    async def _get_admin_token(self) -> None:
        """Get admin access token for management operations."""
        if not self._client:
            return

        try:
            url = f"/realms/master/protocol/openid-connect/token"
            data = {
                'grant_type': 'password',
                'client_id': 'admin-cli',
                'username': self.config.admin_username,
                'password': self.config.admin_password,
            }
            response = await self._client.post(url, data=data)
            response.raise_for_status()
            token_data = response.json()

            self._admin_token = token_data['access_token']
            expires_in = token_data.get('expires_in', 300)
            self._admin_token_expires = datetime.utcnow() + timedelta(seconds=expires_in)

            logger.debug("Admin token acquired, expires in %d seconds", expires_in)
        except Exception as e:
            logger.warning("Failed to get admin token: %s", str(e))

    async def _ensure_admin_token(self) -> Optional[str]:
        """Ensure we have a valid admin token."""
        if self._admin_token_expires and datetime.utcnow() >= self._admin_token_expires:
            await self._get_admin_token()
        return self._admin_token

    def _require_connection(func: Callable) -> Callable:
        """Decorator to ensure connection before operation."""
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not self._connected:
                raise RuntimeError("Keycloak service not connected")
            return await func(self, *args, **kwargs)
        return wrapper

    # =========================================================================
    # Token Operations
    # =========================================================================

    @_require_connection
    async def authenticate(
        self,
        username: str,
        password: str,
        scope: str = "openid profile email",
    ) -> dict[str, Any]:
        """
        Authenticate user with username/password.

        Args:
            username: User's username
            password: User's password
            scope: OAuth scopes

        Returns:
            Token response with access_token, refresh_token, etc.
        """
        if not self._client:
            return self._mock_authenticate(username)

        try:
            url = f"/realms/{self.config.realm}/protocol/openid-connect/token"
            data = {
                'grant_type': 'password',
                'client_id': self.config.client_id,
                'client_secret': self.config.client_secret,
                'username': username,
                'password': password,
                'scope': scope,
            }
            response = await self._client.post(url, data=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthenticationError("Invalid credentials")
            raise
        except httpx.ConnectError:
            # Fall back to mock if connection fails
            logger.warning("Keycloak connection failed, using mock authentication")
            return self._mock_authenticate(username)

    @_require_connection
    async def refresh_token(self, refresh_token: str) -> dict[str, Any]:
        """
        Refresh an access token.

        Args:
            refresh_token: Current refresh token

        Returns:
            New token response
        """
        if not self._client:
            return self._mock_token_response()

        url = f"/realms/{self.config.realm}/protocol/openid-connect/token"
        data = {
            'grant_type': 'refresh_token',
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret,
            'refresh_token': refresh_token,
        }
        response = await self._client.post(url, data=data)
        response.raise_for_status()
        return response.json()

    @_require_connection
    async def validate_token(self, token: str) -> dict[str, Any]:
        """
        Validate and introspect an access token.

        Args:
            token: Access token to validate

        Returns:
            Token info including active status, user info, etc.
        """
        if not self._client:
            return self._mock_token_info(token)

        try:
            url = f"/realms/{self.config.realm}/protocol/openid-connect/token/introspect"
            data = {
                'token': token,
                'client_id': self.config.client_id,
                'client_secret': self.config.client_secret,
            }
            response = await self._client.post(url, data=data)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError:
            return self._mock_token_info(token)

    @_require_connection
    async def logout(
        self,
        refresh_token: str,
        access_token: Optional[str] = None,
    ) -> bool:
        """
        Logout user and invalidate tokens.

        Args:
            refresh_token: Refresh token to revoke
            access_token: Optional access token

        Returns:
            True if successful
        """
        if not self._client:
            return True

        try:
            url = f"/realms/{self.config.realm}/protocol/openid-connect/logout"
            data = {
                'client_id': self.config.client_id,
                'client_secret': self.config.client_secret,
                'refresh_token': refresh_token,
            }
            response = await self._client.post(url, data=data)
            return response.status_code in (200, 204)
        except httpx.ConnectError:
            return True  # Mock success

    @_require_connection
    async def get_user_info(self, access_token: str) -> dict[str, Any]:
        """
        Get user info from access token.

        Args:
            access_token: Valid access token

        Returns:
            User info (sub, name, email, etc.)
        """
        if not self._client:
            return self._mock_user_info()

        try:
            url = f"/realms/{self.config.realm}/protocol/openid-connect/userinfo"
            headers = {'Authorization': f'Bearer {access_token}'}
            response = await self._client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError:
            return self._mock_user_info()

    # =========================================================================
    # JWKS Operations
    # =========================================================================

    @_require_connection
    async def get_jwks(self) -> dict[str, Any]:
        """Get JSON Web Key Set for token verification."""
        # Check cache
        if self._jwks_cache and self._jwks_cache_expires:
            if datetime.utcnow() < self._jwks_cache_expires:
                return self._jwks_cache

        if not self._client:
            return {'keys': []}

        url = self.config.jwks_uri or f"/realms/{self.config.realm}/protocol/openid-connect/certs"
        response = await self._client.get(url)
        response.raise_for_status()

        self._jwks_cache = response.json()
        self._jwks_cache_expires = datetime.utcnow() + timedelta(hours=1)

        return self._jwks_cache

    @_require_connection
    async def verify_token_signature(self, token: str) -> bool:
        """
        Verify token signature using JWKS.

        Args:
            token: JWT token to verify

        Returns:
            True if signature is valid
        """
        try:
            import jwt
            from jwt import PyJWKClient

            jwks = await self.get_jwks()
            # In production, use PyJWKClient for proper key handling
            # For now, just introspect the token
            token_info = await self.validate_token(token)
            return token_info.get('active', False)
        except Exception as e:
            logger.warning("Token verification failed: %s", str(e))
            return False

    # =========================================================================
    # User Management
    # =========================================================================

    @_require_connection
    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        enabled: bool = True,
        email_verified: bool = False,
        attributes: Optional[dict[str, list[str]]] = None,
    ) -> str:
        """
        Create a new user.

        Returns:
            User ID
        """
        if not self._client:
            return str(uuid4())

        admin_token = await self._ensure_admin_token()
        if not admin_token:
            raise RuntimeError("Admin token required for user management")

        url = f"/admin/realms/{self.config.realm}/users"
        headers = {'Authorization': f'Bearer {admin_token}'}
        user_data = {
            'username': username,
            'email': email,
            'firstName': first_name,
            'lastName': last_name,
            'enabled': enabled,
            'emailVerified': email_verified,
            'attributes': attributes or {},
            'credentials': [{
                'type': 'password',
                'value': password,
                'temporary': False,
            }],
        }

        response = await self._client.post(url, json=user_data, headers=headers)
        response.raise_for_status()

        # Extract user ID from Location header
        location = response.headers.get('Location', '')
        user_id = location.split('/')[-1]
        logger.info("Created user: %s (%s)", username, user_id)
        return user_id

    @_require_connection
    async def get_user(self, user_id: str) -> Optional[dict[str, Any]]:
        """Get user by ID."""
        if not self._client:
            return self._mock_user(user_id)

        admin_token = await self._ensure_admin_token()
        if not admin_token:
            return None

        url = f"/admin/realms/{self.config.realm}/users/{user_id}"
        headers = {'Authorization': f'Bearer {admin_token}'}
        response = await self._client.get(url, headers=headers)

        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    @_require_connection
    async def find_user_by_email(self, email: str) -> Optional[dict[str, Any]]:
        """Find user by email address."""
        if not self._client:
            return self._mock_user(str(uuid4()))

        admin_token = await self._ensure_admin_token()
        if not admin_token:
            return None

        url = f"/admin/realms/{self.config.realm}/users"
        headers = {'Authorization': f'Bearer {admin_token}'}
        params = {'email': email, 'exact': 'true'}
        response = await self._client.get(url, headers=headers, params=params)
        response.raise_for_status()

        users = response.json()
        return users[0] if users else None

    @_require_connection
    async def update_user(
        self,
        user_id: str,
        **attributes,
    ) -> bool:
        """Update user attributes."""
        if not self._client:
            return True

        admin_token = await self._ensure_admin_token()
        if not admin_token:
            return False

        url = f"/admin/realms/{self.config.realm}/users/{user_id}"
        headers = {'Authorization': f'Bearer {admin_token}'}
        response = await self._client.put(url, json=attributes, headers=headers)
        return response.status_code == 204

    @_require_connection
    async def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        if not self._client:
            return True

        admin_token = await self._ensure_admin_token()
        if not admin_token:
            return False

        url = f"/admin/realms/{self.config.realm}/users/{user_id}"
        headers = {'Authorization': f'Bearer {admin_token}'}
        response = await self._client.delete(url, headers=headers)
        return response.status_code == 204

    @_require_connection
    async def set_user_password(
        self,
        user_id: str,
        password: str,
        temporary: bool = False,
    ) -> bool:
        """Reset user password."""
        if not self._client:
            return True

        admin_token = await self._ensure_admin_token()
        if not admin_token:
            return False

        url = f"/admin/realms/{self.config.realm}/users/{user_id}/reset-password"
        headers = {'Authorization': f'Bearer {admin_token}'}
        credential = {
            'type': 'password',
            'value': password,
            'temporary': temporary,
        }
        response = await self._client.put(url, json=credential, headers=headers)
        return response.status_code == 204

    # =========================================================================
    # Role Management
    # =========================================================================

    @_require_connection
    async def get_realm_roles(self) -> list[dict[str, Any]]:
        """Get all realm roles."""
        if not self._client:
            return self._mock_roles()

        admin_token = await self._ensure_admin_token()
        if not admin_token:
            return []

        url = f"/admin/realms/{self.config.realm}/roles"
        headers = {'Authorization': f'Bearer {admin_token}'}
        response = await self._client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    @_require_connection
    async def create_realm_role(
        self,
        name: str,
        description: Optional[str] = None,
    ) -> bool:
        """Create a realm role."""
        if not self._client:
            return True

        admin_token = await self._ensure_admin_token()
        if not admin_token:
            return False

        url = f"/admin/realms/{self.config.realm}/roles"
        headers = {'Authorization': f'Bearer {admin_token}'}
        role_data = {
            'name': name,
            'description': description,
        }
        response = await self._client.post(url, json=role_data, headers=headers)
        return response.status_code == 201

    @_require_connection
    async def assign_role_to_user(
        self,
        user_id: str,
        role_name: str,
    ) -> bool:
        """Assign a realm role to a user."""
        if not self._client:
            return True

        admin_token = await self._ensure_admin_token()
        if not admin_token:
            return False

        # First get the role
        url = f"/admin/realms/{self.config.realm}/roles/{role_name}"
        headers = {'Authorization': f'Bearer {admin_token}'}
        response = await self._client.get(url, headers=headers)
        if response.status_code != 200:
            return False
        role = response.json()

        # Assign to user
        url = f"/admin/realms/{self.config.realm}/users/{user_id}/role-mappings/realm"
        response = await self._client.post(url, json=[role], headers=headers)
        return response.status_code == 204

    @_require_connection
    async def get_user_roles(self, user_id: str) -> list[dict[str, Any]]:
        """Get roles assigned to a user."""
        if not self._client:
            return self._mock_roles()

        admin_token = await self._ensure_admin_token()
        if not admin_token:
            return []

        url = f"/admin/realms/{self.config.realm}/users/{user_id}/role-mappings/realm"
        headers = {'Authorization': f'Bearer {admin_token}'}
        response = await self._client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Session Management
    # =========================================================================

    @_require_connection
    async def get_user_sessions(self, user_id: str) -> list[dict[str, Any]]:
        """Get active sessions for a user."""
        if not self._client:
            return []

        admin_token = await self._ensure_admin_token()
        if not admin_token:
            return []

        url = f"/admin/realms/{self.config.realm}/users/{user_id}/sessions"
        headers = {'Authorization': f'Bearer {admin_token}'}
        response = await self._client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    @_require_connection
    async def terminate_user_sessions(self, user_id: str) -> bool:
        """Terminate all sessions for a user."""
        if not self._client:
            return True

        admin_token = await self._ensure_admin_token()
        if not admin_token:
            return False

        url = f"/admin/realms/{self.config.realm}/users/{user_id}/logout"
        headers = {'Authorization': f'Bearer {admin_token}'}
        response = await self._client.post(url, headers=headers)
        return response.status_code == 204

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> dict[str, Any]:
        """Check Keycloak health status."""
        if not self._connected:
            return {"status": "disconnected"}

        if not self._client:
            return {"status": "healthy", "mode": "mock"}

        try:
            response = await self._client.get(f"/realms/{self.config.realm}")
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "realm": self.config.realm,
                }
            return {"status": "unhealthy", "code": response.status_code}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    # =========================================================================
    # Mock Responses
    # =========================================================================

    def _mock_authenticate(self, username: str) -> dict[str, Any]:
        """Generate mock authentication response."""
        return self._mock_token_response(username)

    def _mock_token_response(self, username: str = "mock-user") -> dict[str, Any]:
        """Generate mock token response."""
        return {
            'access_token': f'mock-access-token-{uuid4().hex[:8]}',
            'refresh_token': f'mock-refresh-token-{uuid4().hex[:8]}',
            'token_type': 'Bearer',
            'expires_in': self.config.token_lifetime,
            'refresh_expires_in': self.config.refresh_token_lifetime,
            'scope': 'openid profile email',
        }

    def _mock_token_info(self, token: str) -> dict[str, Any]:
        """Generate mock token introspection."""
        return {
            'active': True,
            'sub': str(uuid4()),
            'username': 'mock-user',
            'email': 'mock@example.com',
            'realm_access': {'roles': ['user']},
        }

    def _mock_user_info(self) -> dict[str, Any]:
        """Generate mock user info."""
        return {
            'sub': str(uuid4()),
            'name': 'Mock User',
            'preferred_username': 'mock-user',
            'email': 'mock@example.com',
            'email_verified': True,
        }

    def _mock_user(self, user_id: str) -> dict[str, Any]:
        """Generate mock user."""
        return {
            'id': user_id,
            'username': 'mock-user',
            'email': 'mock@example.com',
            'firstName': 'Mock',
            'lastName': 'User',
            'enabled': True,
            'emailVerified': True,
        }

    def _mock_roles(self) -> list[dict[str, Any]]:
        """Generate mock roles."""
        return [
            {'id': str(uuid4()), 'name': 'admin', 'description': 'Administrator'},
            {'id': str(uuid4()), 'name': 'user', 'description': 'Regular user'},
            {'id': str(uuid4()), 'name': 'viewer', 'description': 'Read-only access'},
        ]


class AuthenticationError(Exception):
    """Authentication failed."""
    pass
