"""
Access Control Service
======================
Role-Based (RBAC) and Attribute-Based (ABAC) Access Control.
"""

from __future__ import annotations

import logging
from datetime import datetime, time
from typing import Any, Optional
from uuid import uuid4

from ..config import AccessControlConfig
from ..models import (
    AccessDecision,
    AccessPolicy,
    AuditEventType,
    Permission,
    Role,
)

logger = logging.getLogger(__name__)


class AccessControlService:
    """
    Hybrid RBAC/ABAC access control service.

    Features:
    - Role-based access control (RBAC)
    - Attribute-based access control (ABAC)
    - Policy evaluation with deny-override
    - Permission inheritance through roles
    - IP whitelist support
    - Time-based access restrictions
    """

    def __init__(self, config: AccessControlConfig, audit_service: Optional[Any] = None):
        self.config = config
        self.audit = audit_service
        self._roles: dict[str, Role] = {}
        self._permissions: dict[str, Permission] = {}
        self._policies: dict[str, AccessPolicy] = {}
        self._user_roles: dict[str, list[str]] = {}  # user_id -> role names
        self._user_attributes: dict[str, dict[str, Any]] = {}  # user_id -> attributes
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize access control with default roles and permissions."""
        # Create default permissions
        await self._create_default_permissions()

        # Create default roles
        await self._create_default_roles()

        # Create default policies
        if self.config.enable_attribute_policies:
            await self._create_default_policies()

        self._initialized = True
        logger.info("Access control service initialized (mode: %s)", self.config.mode)

    async def _create_default_permissions(self) -> None:
        """Create default system permissions."""
        default_permissions = [
            # Document permissions
            Permission(name="documents:read", resource_type="documents", action="read"),
            Permission(name="documents:write", resource_type="documents", action="write"),
            Permission(name="documents:delete", resource_type="documents", action="delete"),
            Permission(name="documents:share", resource_type="documents", action="share"),
            Permission(name="documents:export", resource_type="documents", action="export"),
            Permission(name="documents:classify", resource_type="documents", action="classify"),

            # User management
            Permission(name="users:read", resource_type="users", action="read"),
            Permission(name="users:write", resource_type="users", action="write"),
            Permission(name="users:delete", resource_type="users", action="delete"),
            Permission(name="users:admin", resource_type="users", action="admin"),

            # Client/Matter management
            Permission(name="clients:read", resource_type="clients", action="read"),
            Permission(name="clients:write", resource_type="clients", action="write"),
            Permission(name="matters:read", resource_type="matters", action="read"),
            Permission(name="matters:write", resource_type="matters", action="write"),

            # Admin permissions
            Permission(name="admin:settings", resource_type="settings", action="admin"),
            Permission(name="admin:audit", resource_type="audit", action="admin"),
            Permission(name="admin:compliance", resource_type="compliance", action="admin"),
            Permission(name="admin:security", resource_type="security", action="admin"),

            # Analytics
            Permission(name="analytics:read", resource_type="analytics", action="read"),
            Permission(name="analytics:export", resource_type="analytics", action="export"),
        ]

        for perm in default_permissions:
            self._permissions[perm.name] = perm

    async def _create_default_roles(self) -> None:
        """Create default system roles."""
        default_roles = [
            Role(
                name="admin",
                description="Full system administrator",
                permissions=[p for p in self._permissions.keys()],
                is_system=True,
            ),
            Role(
                name="manager",
                description="Team manager with elevated privileges",
                permissions=[
                    "documents:read", "documents:write", "documents:delete",
                    "documents:share", "documents:export",
                    "users:read", "users:write",
                    "clients:read", "clients:write",
                    "matters:read", "matters:write",
                    "analytics:read", "analytics:export",
                ],
                inherits_from=["user"],
                is_system=True,
            ),
            Role(
                name="user",
                description="Standard user",
                permissions=[
                    "documents:read", "documents:write",
                    "documents:share",
                    "clients:read",
                    "matters:read",
                ],
                inherits_from=["viewer"],
                is_system=True,
            ),
            Role(
                name="viewer",
                description="Read-only access",
                permissions=[
                    "documents:read",
                    "clients:read",
                    "matters:read",
                ],
                is_system=True,
            ),
            Role(
                name="compliance_officer",
                description="Compliance and audit access",
                permissions=[
                    "documents:read",
                    "admin:audit",
                    "admin:compliance",
                    "analytics:read",
                ],
                inherits_from=["viewer"],
                is_system=True,
            ),
            Role(
                name="external",
                description="External/guest user with limited access",
                permissions=[
                    "documents:read",
                ],
                is_system=True,
            ),
        ]

        for role in default_roles:
            self._roles[role.name] = role

    async def _create_default_policies(self) -> None:
        """Create default ABAC policies."""
        default_policies = [
            # Deny access to restricted documents without clearance
            AccessPolicy(
                name="restricted-document-clearance",
                description="Require clearance for restricted documents",
                effect="deny",
                priority=100,
                conditions={
                    "resource": {"classification": "restricted"},
                    "subject": {"attributes.clearance_level": {"not_in": ["top_secret", "secret"]}},
                },
            ),
            # Allow owners full access to their documents
            AccessPolicy(
                name="owner-full-access",
                description="Document owners have full access",
                effect="allow",
                priority=90,
                conditions={
                    "resource": {"owner_id": {"equals": "subject.user_id"}},
                },
            ),
            # Deny external users write access
            AccessPolicy(
                name="external-read-only",
                description="External users are read-only",
                effect="deny",
                priority=80,
                conditions={
                    "subject": {"roles": {"contains": "external"}},
                    "action": {"in": ["write", "delete", "admin"]},
                },
            ),
            # Time-based restrictions (if enabled)
            AccessPolicy(
                name="business-hours-only",
                description="Restrict access outside business hours",
                effect="deny",
                priority=70,
                conditions={
                    "environment": {"outside_business_hours": True},
                    "subject": {"attributes.remote_access": {"not_equals": True}},
                },
                enabled=False,  # Disabled by default
            ),
        ]

        for policy in default_policies:
            self._policies[policy.id] = policy

    # =========================================================================
    # Role Management
    # =========================================================================

    async def create_role(
        self,
        name: str,
        description: Optional[str] = None,
        permissions: Optional[list[str]] = None,
        inherits_from: Optional[list[str]] = None,
    ) -> Role:
        """Create a new role."""
        if name in self._roles:
            raise ValueError(f"Role already exists: {name}")

        role = Role(
            name=name,
            description=description,
            permissions=permissions or [],
            inherits_from=inherits_from or [],
            is_system=False,
        )
        self._roles[name] = role
        logger.info("Created role: %s", name)
        return role

    async def update_role(
        self,
        name: str,
        permissions: Optional[list[str]] = None,
        description: Optional[str] = None,
    ) -> Role:
        """Update an existing role."""
        role = self._roles.get(name)
        if not role:
            raise ValueError(f"Role not found: {name}")

        if role.is_system:
            raise ValueError(f"Cannot modify system role: {name}")

        if permissions is not None:
            role.permissions = permissions
        if description is not None:
            role.description = description
        role.updated_at = datetime.utcnow()

        logger.info("Updated role: %s", name)
        return role

    async def delete_role(self, name: str) -> bool:
        """Delete a role."""
        role = self._roles.get(name)
        if not role:
            raise ValueError(f"Role not found: {name}")

        if role.is_system:
            raise ValueError(f"Cannot delete system role: {name}")

        del self._roles[name]

        # Remove from users
        for user_id, roles in self._user_roles.items():
            if name in roles:
                roles.remove(name)

        logger.info("Deleted role: %s", name)
        return True

    async def get_role(self, name: str) -> Optional[Role]:
        """Get a role by name."""
        return self._roles.get(name)

    async def list_roles(self) -> list[Role]:
        """List all roles."""
        return list(self._roles.values())

    async def get_role_permissions(self, role_name: str) -> set[str]:
        """Get all permissions for a role (including inherited)."""
        role = self._roles.get(role_name)
        if not role:
            return set()

        permissions = set(role.permissions)

        # Add inherited permissions
        if self.config.enable_role_inheritance:
            for parent_name in role.inherits_from:
                parent_perms = await self.get_role_permissions(parent_name)
                permissions.update(parent_perms)

        return permissions

    # =========================================================================
    # User Role Assignment
    # =========================================================================

    async def assign_role(self, user_id: str, role_name: str) -> bool:
        """Assign a role to a user."""
        if role_name not in self._roles:
            raise ValueError(f"Role not found: {role_name}")

        if user_id not in self._user_roles:
            self._user_roles[user_id] = []

        if role_name not in self._user_roles[user_id]:
            self._user_roles[user_id].append(role_name)

            # Log audit event
            if self.audit:
                await self.audit.log(
                    event_type=AuditEventType.ROLE_ASSIGN,
                    resource_type="user",
                    resource_id=user_id,
                    action="role_assign",
                    details={"role": role_name},
                )

            logger.info("Assigned role %s to user %s", role_name, user_id)

        return True

    async def revoke_role(self, user_id: str, role_name: str) -> bool:
        """Revoke a role from a user."""
        if user_id in self._user_roles and role_name in self._user_roles[user_id]:
            self._user_roles[user_id].remove(role_name)

            # Log audit event
            if self.audit:
                await self.audit.log(
                    event_type=AuditEventType.ROLE_REVOKE,
                    resource_type="user",
                    resource_id=user_id,
                    action="role_revoke",
                    details={"role": role_name},
                )

            logger.info("Revoked role %s from user %s", role_name, user_id)
            return True

        return False

    async def get_user_roles(self, user_id: str) -> list[str]:
        """Get all roles assigned to a user."""
        roles = self._user_roles.get(user_id, [])
        if not roles and self.config.default_role:
            return [self.config.default_role]
        return roles

    async def get_user_permissions(self, user_id: str) -> set[str]:
        """Get all permissions for a user."""
        roles = await self.get_user_roles(user_id)
        permissions = set()

        for role_name in roles:
            role_perms = await self.get_role_permissions(role_name)
            permissions.update(role_perms)

        return permissions

    # =========================================================================
    # User Attributes (for ABAC)
    # =========================================================================

    async def set_user_attributes(self, user_id: str, attributes: dict[str, Any]) -> None:
        """Set attributes for a user (used in ABAC policies)."""
        if user_id not in self._user_attributes:
            self._user_attributes[user_id] = {}
        self._user_attributes[user_id].update(attributes)

    async def get_user_attributes(self, user_id: str) -> dict[str, Any]:
        """Get attributes for a user."""
        return self._user_attributes.get(user_id, {})

    # =========================================================================
    # Policy Management
    # =========================================================================

    async def create_policy(
        self,
        name: str,
        effect: str,
        conditions: dict[str, Any],
        description: Optional[str] = None,
        priority: int = 0,
    ) -> AccessPolicy:
        """Create an ABAC policy."""
        policy = AccessPolicy(
            name=name,
            description=description,
            effect=effect,
            priority=priority,
            conditions=conditions,
        )
        self._policies[policy.id] = policy
        logger.info("Created policy: %s", name)
        return policy

    async def update_policy(
        self,
        policy_id: str,
        conditions: Optional[dict[str, Any]] = None,
        enabled: Optional[bool] = None,
    ) -> AccessPolicy:
        """Update an ABAC policy."""
        policy = self._policies.get(policy_id)
        if not policy:
            raise ValueError(f"Policy not found: {policy_id}")

        if conditions is not None:
            policy.conditions = conditions
        if enabled is not None:
            policy.enabled = enabled

        logger.info("Updated policy: %s", policy_id)
        return policy

    async def delete_policy(self, policy_id: str) -> bool:
        """Delete an ABAC policy."""
        if policy_id in self._policies:
            del self._policies[policy_id]
            logger.info("Deleted policy: %s", policy_id)
            return True
        return False

    async def list_policies(self, enabled_only: bool = True) -> list[AccessPolicy]:
        """List all policies."""
        policies = list(self._policies.values())
        if enabled_only:
            policies = [p for p in policies if p.enabled]
        return sorted(policies, key=lambda p: -p.priority)

    # =========================================================================
    # Access Decision
    # =========================================================================

    async def check_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        resource_attributes: Optional[dict[str, Any]] = None,
        environment: Optional[dict[str, Any]] = None,
    ) -> AccessDecision:
        """
        Check if a user has access to perform an action on a resource.

        Args:
            user_id: User requesting access
            resource_type: Type of resource (documents, users, etc.)
            resource_id: Resource identifier
            action: Action to perform (read, write, delete, etc.)
            resource_attributes: Additional resource attributes for ABAC
            environment: Environment context (IP, time, etc.)

        Returns:
            AccessDecision with allowed status and reason
        """
        if not self._initialized:
            raise RuntimeError("Access control service not initialized")

        # Build permission name
        permission_name = f"{resource_type}:{action}"

        # Build context for policy evaluation
        context = {
            "subject": {
                "user_id": user_id,
                "roles": await self.get_user_roles(user_id),
                "attributes": await self.get_user_attributes(user_id),
            },
            "resource": {
                "type": resource_type,
                "id": resource_id,
                **(resource_attributes or {}),
            },
            "action": action,
            "environment": environment or self._build_environment_context(),
        }

        # Check RBAC first
        user_permissions = await self.get_user_permissions(user_id)
        has_permission = permission_name in user_permissions

        if not has_permission:
            decision = AccessDecision(
                allowed=False,
                reason=f"User lacks permission: {permission_name}",
            )
            await self._log_access_decision(user_id, resource_type, resource_id, action, decision)
            return decision

        # If RBAC-only mode, return decision
        if self.config.mode == "rbac":
            decision = AccessDecision(
                allowed=True,
                reason="RBAC permission granted",
            )
            await self._log_access_decision(user_id, resource_type, resource_id, action, decision)
            return decision

        # Evaluate ABAC policies
        decision = await self._evaluate_policies(context)

        # Log access decision
        await self._log_access_decision(user_id, resource_type, resource_id, action, decision)

        return decision

    async def _evaluate_policies(self, context: dict[str, Any]) -> AccessDecision:
        """Evaluate ABAC policies against context."""
        policies = await self.list_policies(enabled_only=True)

        deny_decision = None
        allow_decision = None

        for policy in policies:
            matches = self._evaluate_policy_conditions(policy.conditions, context)

            if matches:
                if policy.effect == "deny":
                    deny_decision = AccessDecision(
                        allowed=False,
                        policy_id=policy.id,
                        policy_name=policy.name,
                        reason=f"Denied by policy: {policy.name}",
                    )
                    if self.config.policy_evaluation_order == "deny-override":
                        return deny_decision
                else:
                    allow_decision = AccessDecision(
                        allowed=True,
                        policy_id=policy.id,
                        policy_name=policy.name,
                        reason=f"Allowed by policy: {policy.name}",
                    )
                    if self.config.policy_evaluation_order == "allow-override":
                        return allow_decision

        # Return based on evaluation order
        if self.config.policy_evaluation_order == "deny-override":
            return allow_decision or AccessDecision(
                allowed=True,
                reason="No deny policy matched",
            )
        else:
            return deny_decision or AccessDecision(
                allowed=True,
                reason="No policy matched, default allow",
            )

    def _evaluate_policy_conditions(
        self,
        conditions: dict[str, Any],
        context: dict[str, Any],
    ) -> bool:
        """Evaluate policy conditions against context."""
        for category, rules in conditions.items():
            if category not in context:
                continue

            category_context = context[category]

            for attr, constraint in rules.items():
                # Handle nested attribute paths
                value = self._get_nested_value(category_context, attr)

                if isinstance(constraint, dict):
                    # Complex constraint
                    if not self._evaluate_constraint(value, constraint, context):
                        return False
                else:
                    # Simple equality
                    if value != constraint:
                        return False

        return True

    def _get_nested_value(self, obj: Any, path: str) -> Any:
        """Get nested value from object using dot notation."""
        parts = path.split('.')
        current = obj

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return None

            if current is None:
                return None

        return current

    def _evaluate_constraint(
        self,
        value: Any,
        constraint: dict[str, Any],
        context: dict[str, Any],
    ) -> bool:
        """Evaluate a constraint against a value."""
        for op, expected in constraint.items():
            # Resolve expected value if it references context
            if isinstance(expected, str) and expected.startswith("subject."):
                expected = self._get_nested_value(context, expected)

            if op == "equals":
                if value != expected:
                    return False
            elif op == "not_equals":
                if value == expected:
                    return False
            elif op == "in":
                if value not in expected:
                    return False
            elif op == "not_in":
                if value in expected:
                    return False
            elif op == "contains":
                if not isinstance(value, (list, set)) or expected not in value:
                    return False
            elif op == "greater_than":
                if not (value and value > expected):
                    return False
            elif op == "less_than":
                if not (value and value < expected):
                    return False

        return True

    def _build_environment_context(self) -> dict[str, Any]:
        """Build environment context for policy evaluation."""
        now = datetime.utcnow()
        current_time = now.time()

        # Parse business hours
        start = time.fromisoformat(self.config.business_hours_start)
        end = time.fromisoformat(self.config.business_hours_end)

        is_business_hours = (
            start <= current_time <= end and
            now.weekday() in self.config.business_days
        )

        return {
            "timestamp": now.isoformat(),
            "day_of_week": now.weekday(),
            "hour": now.hour,
            "is_business_hours": is_business_hours,
            "outside_business_hours": not is_business_hours,
        }

    async def _log_access_decision(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        decision: AccessDecision,
    ) -> None:
        """Log access control decision to audit."""
        if self.audit:
            await self.audit.log_access_event(
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                allowed=decision.allowed,
                policy_id=decision.policy_id,
            )

    # =========================================================================
    # IP Whitelist
    # =========================================================================

    async def check_ip_whitelist(self, ip_address: str) -> bool:
        """Check if IP address is whitelisted."""
        if not self.config.enable_ip_whitelist:
            return True

        if not self.config.ip_whitelist:
            return True

        # Simple IP matching (could be enhanced with CIDR support)
        return ip_address in self.config.ip_whitelist

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def can_read(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        **kwargs,
    ) -> bool:
        """Check if user can read a resource."""
        decision = await self.check_access(user_id, resource_type, resource_id, "read", **kwargs)
        return decision.allowed

    async def can_write(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        **kwargs,
    ) -> bool:
        """Check if user can write to a resource."""
        decision = await self.check_access(user_id, resource_type, resource_id, "write", **kwargs)
        return decision.allowed

    async def can_delete(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        **kwargs,
    ) -> bool:
        """Check if user can delete a resource."""
        decision = await self.check_access(user_id, resource_type, resource_id, "delete", **kwargs)
        return decision.allowed

    async def is_admin(self, user_id: str) -> bool:
        """Check if user has admin role."""
        roles = await self.get_user_roles(user_id)
        return "admin" in roles

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> dict[str, Any]:
        """Check access control service health."""
        if not self._initialized:
            return {"status": "not_initialized"}

        return {
            "status": "healthy",
            "mode": self.config.mode,
            "roles_count": len(self._roles),
            "permissions_count": len(self._permissions),
            "policies_count": len(self._policies),
            "users_with_roles": len(self._user_roles),
            "ip_whitelist_enabled": self.config.enable_ip_whitelist,
            "time_restrictions_enabled": self.config.enable_time_restrictions,
        }
