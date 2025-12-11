"""
Legal Document Platform - API Gateway Layer
============================================
Unified REST API with authentication, rate limiting,
webhooks, and comprehensive endpoint coverage.
"""

from __future__ import annotations

from .config import APIConfig
from .models import (
    APIVersion,
    RateLimitTier,
    WebhookEventType,
    PaginatedResponse,
    ErrorResponse,
    HealthResponse,
)
from .app import create_app, APIGateway

__all__ = [
    # Config
    "APIConfig",
    # Models
    "APIVersion",
    "RateLimitTier",
    "WebhookEventType",
    "PaginatedResponse",
    "ErrorResponse",
    "HealthResponse",
    # App
    "create_app",
    "APIGateway",
]
