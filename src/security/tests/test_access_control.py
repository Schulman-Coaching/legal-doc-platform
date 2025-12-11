"""
Tests for Access Control Service.
"""

from __future__ import annotations

import pytest
from ..services.access_control import AccessControlService
from ..models import AccessDecision, Role, Permission


class TestRoleManagement:
    """Tests for role management."""

    @pytest.mark.asyncio
    async def test_default_roles_created(self, access_control_service: AccessControlService):
        """Test that default roles are created."""
        roles = await access_control_service.list_roles()
        role_names = [r.name for r in roles]

        assert "admin" in role_names
        assert "manager" in role_names
        assert "user" in role_names
        assert "viewer" in role_names

    @pytest.mark.asyncio
    async def test_get_role(self, access_control_service: AccessControlService):
        """Test getting a specific role."""
        role = await access_control_service.get_role("admin")

        assert role is not None
        assert role.name == "admin"
        assert role.is_system is True
        assert len(role.permissions) > 0

    @pytest.mark.asyncio
    async def test_create_custom_role(self, access_control_service: AccessControlService):
        """Test creating a custom role."""
        role = await access_control_service.create_role(
            name="custom-role",
            description="A custom role",
            permissions=["documents:read", "documents:write"],
        )

        assert role.name == "custom-role"
        assert role.is_system is False
        assert "documents:read" in role.permissions

    @pytest.mark.asyncio
    async def test_create_duplicate_role_fails(self, access_control_service: AccessControlService):
        """Test that creating duplicate role fails."""
        with pytest.raises(ValueError, match="already exists"):
            await access_control_service.create_role(
                name="admin",  # Already exists
                permissions=["documents:read"],
            )

    @pytest.mark.asyncio
    async def test_update_custom_role(self, access_control_service: AccessControlService):
        """Test updating a custom role."""
        # First create the role
        await access_control_service.create_role(
            name="updatable-role",
            permissions=["documents:read"],
        )

        # Update it
        updated = await access_control_service.update_role(
            name="updatable-role",
            permissions=["documents:read", "documents:write"],
            description="Updated description",
        )

        assert "documents:write" in updated.permissions

    @pytest.mark.asyncio
    async def test_cannot_modify_system_role(self, access_control_service: AccessControlService):
        """Test that system roles cannot be modified."""
        with pytest.raises(ValueError, match="system role"):
            await access_control_service.update_role(
                name="admin",
                permissions=["documents:read"],
            )

    @pytest.mark.asyncio
    async def test_delete_custom_role(self, access_control_service: AccessControlService):
        """Test deleting a custom role."""
        await access_control_service.create_role(
            name="deletable-role",
            permissions=["documents:read"],
        )

        result = await access_control_service.delete_role("deletable-role")

        assert result is True
        assert await access_control_service.get_role("deletable-role") is None

    @pytest.mark.asyncio
    async def test_cannot_delete_system_role(self, access_control_service: AccessControlService):
        """Test that system roles cannot be deleted."""
        with pytest.raises(ValueError, match="system role"):
            await access_control_service.delete_role("admin")

    @pytest.mark.asyncio
    async def test_role_permissions_with_inheritance(self, access_control_service: AccessControlService):
        """Test that role inheritance works."""
        # User role inherits from viewer
        user_perms = await access_control_service.get_role_permissions("user")
        viewer_perms = await access_control_service.get_role_permissions("viewer")

        # User should have all viewer permissions plus its own
        assert viewer_perms.issubset(user_perms)


class TestUserRoleAssignment:
    """Tests for user role assignment."""

    @pytest.mark.asyncio
    async def test_assign_role_to_user(self, access_control_service: AccessControlService):
        """Test assigning a role to a user."""
        result = await access_control_service.assign_role("user-001", "user")

        assert result is True
        roles = await access_control_service.get_user_roles("user-001")
        assert "user" in roles

    @pytest.mark.asyncio
    async def test_assign_invalid_role_fails(self, access_control_service: AccessControlService):
        """Test assigning non-existent role fails."""
        with pytest.raises(ValueError, match="not found"):
            await access_control_service.assign_role("user-001", "nonexistent-role")

    @pytest.mark.asyncio
    async def test_revoke_role_from_user(self, access_control_service: AccessControlService):
        """Test revoking a role from a user."""
        await access_control_service.assign_role("user-002", "user")
        result = await access_control_service.revoke_role("user-002", "user")

        assert result is True
        roles = await access_control_service.get_user_roles("user-002")
        assert "user" not in roles

    @pytest.mark.asyncio
    async def test_get_user_permissions(self, access_control_service: AccessControlService):
        """Test getting all permissions for a user."""
        await access_control_service.assign_role("user-003", "user")

        permissions = await access_control_service.get_user_permissions("user-003")

        assert "documents:read" in permissions
        assert "documents:write" in permissions

    @pytest.mark.asyncio
    async def test_user_gets_default_role(self, access_control_service: AccessControlService):
        """Test that new user gets default role."""
        roles = await access_control_service.get_user_roles("new-user")

        assert "viewer" in roles  # Default role


class TestAccessDecisions:
    """Tests for access control decisions."""

    @pytest.mark.asyncio
    async def test_check_access_allowed(self, access_control_service: AccessControlService):
        """Test access check when allowed."""
        await access_control_service.assign_role("user-004", "user")

        decision = await access_control_service.check_access(
            user_id="user-004",
            resource_type="documents",
            resource_id="doc-001",
            action="read",
        )

        assert decision.allowed is True

    @pytest.mark.asyncio
    async def test_check_access_denied(self, access_control_service: AccessControlService):
        """Test access check when denied."""
        await access_control_service.assign_role("user-005", "viewer")

        decision = await access_control_service.check_access(
            user_id="user-005",
            resource_type="documents",
            resource_id="doc-001",
            action="delete",  # Viewers can't delete
        )

        assert decision.allowed is False

    @pytest.mark.asyncio
    async def test_admin_has_full_access(self, access_control_service: AccessControlService):
        """Test that admin has access to everything."""
        await access_control_service.assign_role("admin-user", "admin")

        # Check user permissions first
        perms = await access_control_service.get_user_permissions("admin-user")

        # Admin should have document permissions
        decision = await access_control_service.check_access(
            user_id="admin-user",
            resource_type="documents",
            resource_id="doc-001",
            action="delete",
        )

        assert decision.allowed is True

    @pytest.mark.asyncio
    async def test_convenience_methods(self, access_control_service: AccessControlService):
        """Test convenience access methods."""
        await access_control_service.assign_role("user-006", "user")

        assert await access_control_service.can_read("user-006", "documents", "doc-001")
        assert await access_control_service.can_write("user-006", "documents", "doc-001")
        assert not await access_control_service.can_delete("user-006", "documents", "doc-001")

    @pytest.mark.asyncio
    async def test_is_admin(self, access_control_service: AccessControlService):
        """Test is_admin check."""
        await access_control_service.assign_role("admin-test", "admin")
        await access_control_service.assign_role("user-test", "user")

        assert await access_control_service.is_admin("admin-test")
        assert not await access_control_service.is_admin("user-test")


class TestABACPolicies:
    """Tests for attribute-based access control."""

    @pytest.mark.asyncio
    async def test_create_policy(self, access_control_service: AccessControlService):
        """Test creating an ABAC policy."""
        policy = await access_control_service.create_policy(
            name="test-policy",
            effect="deny",
            conditions={"resource": {"classification": "restricted"}},
            priority=50,
        )

        assert policy.name == "test-policy"
        assert policy.effect == "deny"

    @pytest.mark.asyncio
    async def test_list_policies(self, access_control_service: AccessControlService):
        """Test listing policies."""
        policies = await access_control_service.list_policies()

        assert len(policies) > 0
        # Should be sorted by priority (descending)
        priorities = [p.priority for p in policies]
        assert priorities == sorted(priorities, reverse=True)

    @pytest.mark.asyncio
    async def test_policy_evaluation_with_attributes(self, access_control_service: AccessControlService):
        """Test policy evaluation with resource attributes."""
        await access_control_service.assign_role("user-007", "user")

        # User without clearance accessing restricted document
        decision = await access_control_service.check_access(
            user_id="user-007",
            resource_type="documents",
            resource_id="secret-doc",
            action="read",
            resource_attributes={"classification": "restricted"},
        )

        # Should be denied by restricted-document-clearance policy
        assert decision.allowed is False

    @pytest.mark.asyncio
    async def test_user_attributes(self, access_control_service: AccessControlService):
        """Test setting and getting user attributes."""
        await access_control_service.set_user_attributes(
            "user-008",
            {"department": "legal", "clearance_level": "secret"},
        )

        attributes = await access_control_service.get_user_attributes("user-008")

        assert attributes["department"] == "legal"
        assert attributes["clearance_level"] == "secret"


class TestHealthCheck:
    """Tests for health check."""

    @pytest.mark.asyncio
    async def test_health_check(self, access_control_service: AccessControlService):
        """Test health check."""
        health = await access_control_service.health_check()

        assert health["status"] == "healthy"
        assert health["mode"] == "hybrid"
        assert health["roles_count"] > 0
        assert health["permissions_count"] > 0


class TestAccessControlNotInitialized:
    """Tests for uninitialized access control service."""

    @pytest.mark.asyncio
    async def test_check_access_before_init_fails(self, access_control_config):
        """Test that access check before initialization fails."""
        service = AccessControlService(access_control_config)

        with pytest.raises(RuntimeError, match="not initialized"):
            await service.check_access("user", "resource", "id", "read")
