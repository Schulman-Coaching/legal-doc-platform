"""
API Routes
==========
FastAPI router definitions for all API endpoints.
"""

from __future__ import annotations

from .documents import router as documents_router
from .auth import router as auth_router
from .users import router as users_router
from .clients import router as clients_router
from .webhooks import router as webhooks_router
from .admin import router as admin_router
from .analytics import router as analytics_router

__all__ = [
    "documents_router",
    "auth_router",
    "users_router",
    "clients_router",
    "webhooks_router",
    "admin_router",
    "analytics_router",
]
