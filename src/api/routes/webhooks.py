"""
Webhook API Routes
==================
Webhook registration and event delivery management.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..config import WebhookConfig
from ..models import (
    ErrorResponse,
    PaginatedResponse,
    SuccessResponse,
    WebhookCreate,
    WebhookDelivery,
    WebhookEventType,
    WebhookResponse,
)
from ..middleware.auth import (
    AuthenticatedUser,
    get_authenticated_user,
    require_permission,
)


router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


# =============================================================================
# Storage (Demo)
# =============================================================================

_webhooks: dict[str, dict[str, Any]] = {}
_deliveries: dict[str, dict[str, Any]] = {}
_webhook_config = WebhookConfig()


def _generate_webhook_secret() -> str:
    """Generate secure webhook secret."""
    import secrets
    return secrets.token_urlsafe(32)


def _compute_signature(payload: bytes, secret: str, timestamp: str) -> str:
    """Compute HMAC signature for webhook payload."""
    message = f"{timestamp}.{payload.decode()}"
    signature = hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    return f"sha256={signature}"


# =============================================================================
# Webhook CRUD
# =============================================================================

@router.get(
    "",
    response_model=PaginatedResponse[WebhookResponse],
    summary="List webhooks",
    description="Get paginated list of registered webhooks.",
)
async def list_webhooks(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> PaginatedResponse[WebhookResponse]:
    """List registered webhooks."""
    webhooks = list(_webhooks.values())

    # Filter by organization
    if user.organization_id:
        webhooks = [w for w in webhooks if w.get("organization_id") == user.organization_id]

    # Apply filters
    if is_active is not None:
        webhooks = [w for w in webhooks if w.get("is_active") == is_active]

    # Sort by created_at descending
    webhooks.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)

    # Paginate
    total = len(webhooks)
    start = (page - 1) * size
    end = start + size
    page_webhooks = webhooks[start:end]

    items = [
        WebhookResponse(
            id=w["id"],
            url=w["url"],
            events=[WebhookEventType(e) for e in w["events"]],
            secret_preview=w["secret"][:8] + "***",
            description=w.get("description"),
            is_active=w.get("is_active", True),
            created_at=w["created_at"],
            last_triggered_at=w.get("last_triggered_at"),
            failure_count=w.get("failure_count", 0),
        )
        for w in page_webhooks
    ]

    return PaginatedResponse.create(items=items, total=total, page=page, size=size)


@router.get(
    "/{webhook_id}",
    response_model=WebhookResponse,
    summary="Get webhook",
    description="Get webhook by ID.",
    responses={404: {"model": ErrorResponse}},
)
async def get_webhook(
    webhook_id: str,
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> WebhookResponse:
    """Get webhook by ID."""
    webhook = _webhooks.get(webhook_id)
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    # Check organization access
    if user.organization_id and webhook.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    return WebhookResponse(
        id=webhook["id"],
        url=webhook["url"],
        events=[WebhookEventType(e) for e in webhook["events"]],
        secret_preview=webhook["secret"][:8] + "***",
        description=webhook.get("description"),
        is_active=webhook.get("is_active", True),
        created_at=webhook["created_at"],
        last_triggered_at=webhook.get("last_triggered_at"),
        failure_count=webhook.get("failure_count", 0),
    )


@router.post(
    "",
    response_model=WebhookResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create webhook",
    description="Register a new webhook endpoint.",
)
async def create_webhook(
    webhook_data: WebhookCreate,
    user: AuthenticatedUser = Depends(require_permission("webhooks:create")),
) -> WebhookResponse:
    """Create new webhook."""
    # Limit number of webhooks per organization
    org_webhooks = [
        w for w in _webhooks.values()
        if w.get("organization_id") == user.organization_id
    ]
    if len(org_webhooks) >= 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum number of webhooks reached (20)",
        )

    # Generate ID and secret
    webhook_id = str(uuid4())
    secret = webhook_data.secret or _generate_webhook_secret()
    now = datetime.utcnow()

    webhook = {
        "id": webhook_id,
        "url": webhook_data.url,
        "events": [e.value for e in webhook_data.events],
        "secret": secret,
        "description": webhook_data.description,
        "is_active": webhook_data.is_active,
        "organization_id": user.organization_id,
        "created_by": user.id,
        "created_at": now,
        "failure_count": 0,
    }

    _webhooks[webhook_id] = webhook

    return WebhookResponse(
        id=webhook["id"],
        url=webhook["url"],
        events=[WebhookEventType(e) for e in webhook["events"]],
        secret_preview=secret[:8] + "***",
        description=webhook.get("description"),
        is_active=webhook.get("is_active", True),
        created_at=webhook["created_at"],
        failure_count=0,
    )


@router.patch(
    "/{webhook_id}",
    response_model=WebhookResponse,
    summary="Update webhook",
    description="Update webhook settings.",
    responses={404: {"model": ErrorResponse}},
)
async def update_webhook(
    webhook_id: str,
    url: Optional[str] = None,
    events: Optional[list[WebhookEventType]] = None,
    description: Optional[str] = None,
    is_active: Optional[bool] = None,
    user: AuthenticatedUser = Depends(require_permission("webhooks:update")),
) -> WebhookResponse:
    """Update webhook."""
    webhook = _webhooks.get(webhook_id)
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    # Check organization access
    if user.organization_id and webhook.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    # Update fields
    if url is not None:
        webhook["url"] = url
    if events is not None:
        webhook["events"] = [e.value for e in events]
    if description is not None:
        webhook["description"] = description
    if is_active is not None:
        webhook["is_active"] = is_active
        # Reset failure count when re-enabling
        if is_active:
            webhook["failure_count"] = 0

    return WebhookResponse(
        id=webhook["id"],
        url=webhook["url"],
        events=[WebhookEventType(e) for e in webhook["events"]],
        secret_preview=webhook["secret"][:8] + "***",
        description=webhook.get("description"),
        is_active=webhook.get("is_active", True),
        created_at=webhook["created_at"],
        last_triggered_at=webhook.get("last_triggered_at"),
        failure_count=webhook.get("failure_count", 0),
    )


@router.delete(
    "/{webhook_id}",
    response_model=SuccessResponse,
    summary="Delete webhook",
    description="Delete a webhook.",
    responses={404: {"model": ErrorResponse}},
)
async def delete_webhook(
    webhook_id: str,
    user: AuthenticatedUser = Depends(require_permission("webhooks:delete")),
) -> SuccessResponse:
    """Delete webhook."""
    webhook = _webhooks.get(webhook_id)
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    # Check organization access
    if user.organization_id and webhook.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    del _webhooks[webhook_id]

    return SuccessResponse(
        success=True,
        message=f"Webhook {webhook_id} deleted",
    )


# =============================================================================
# Webhook Secret Management
# =============================================================================

@router.post(
    "/{webhook_id}/rotate-secret",
    response_model=dict,
    summary="Rotate webhook secret",
    description="Generate a new secret for the webhook.",
)
async def rotate_webhook_secret(
    webhook_id: str,
    user: AuthenticatedUser = Depends(require_permission("webhooks:update")),
) -> dict[str, Any]:
    """Rotate webhook secret."""
    webhook = _webhooks.get(webhook_id)
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    # Check organization access
    if user.organization_id and webhook.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    # Generate new secret
    new_secret = _generate_webhook_secret()
    webhook["secret"] = new_secret

    return {
        "webhook_id": webhook_id,
        "secret": new_secret,
        "message": "Secret rotated. Update your webhook handler with the new secret.",
    }


# =============================================================================
# Webhook Deliveries
# =============================================================================

@router.get(
    "/{webhook_id}/deliveries",
    response_model=PaginatedResponse[WebhookDelivery],
    summary="List webhook deliveries",
    description="Get delivery history for a webhook.",
)
async def list_webhook_deliveries(
    webhook_id: str,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    success_only: Optional[bool] = Query(None, description="Filter by success"),
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> PaginatedResponse[WebhookDelivery]:
    """List webhook deliveries."""
    webhook = _webhooks.get(webhook_id)
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    # Check organization access
    if user.organization_id and webhook.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    # Get deliveries for this webhook
    deliveries = [
        d for d in _deliveries.values()
        if d.get("webhook_id") == webhook_id
    ]

    # Apply filters
    if success_only is not None:
        if success_only:
            deliveries = [d for d in deliveries if d.get("response_status", 0) < 400]
        else:
            deliveries = [d for d in deliveries if d.get("response_status", 0) >= 400]

    # Sort by delivered_at descending
    deliveries.sort(key=lambda x: x.get("delivered_at") or datetime.min, reverse=True)

    # Paginate
    total = len(deliveries)
    start = (page - 1) * size
    end = start + size
    page_deliveries = deliveries[start:end]

    items = [
        WebhookDelivery(
            id=d["id"],
            webhook_id=d["webhook_id"],
            event_type=WebhookEventType(d["event_type"]),
            payload=d["payload"],
            response_status=d.get("response_status"),
            response_body=d.get("response_body"),
            delivered_at=d.get("delivered_at"),
            attempts=d.get("attempts", 0),
            next_retry_at=d.get("next_retry_at"),
        )
        for d in page_deliveries
    ]

    return PaginatedResponse.create(items=items, total=total, page=page, size=size)


@router.post(
    "/{webhook_id}/deliveries/{delivery_id}/retry",
    response_model=SuccessResponse,
    summary="Retry webhook delivery",
    description="Retry a failed webhook delivery.",
)
async def retry_webhook_delivery(
    webhook_id: str,
    delivery_id: str,
    user: AuthenticatedUser = Depends(require_permission("webhooks:update")),
) -> SuccessResponse:
    """Retry a failed delivery."""
    webhook = _webhooks.get(webhook_id)
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    delivery = _deliveries.get(delivery_id)
    if not delivery or delivery.get("webhook_id") != webhook_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Delivery {delivery_id} not found",
        )

    # Check organization access
    if user.organization_id and webhook.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    # Queue for retry (would be async in production)
    delivery["next_retry_at"] = datetime.utcnow()
    delivery["attempts"] = delivery.get("attempts", 0) + 1

    return SuccessResponse(
        success=True,
        message=f"Delivery {delivery_id} queued for retry",
    )


# =============================================================================
# Test Webhook
# =============================================================================

@router.post(
    "/{webhook_id}/test",
    response_model=dict,
    summary="Test webhook",
    description="Send a test event to the webhook.",
)
async def test_webhook(
    webhook_id: str,
    user: AuthenticatedUser = Depends(require_permission("webhooks:update")),
) -> dict[str, Any]:
    """Send test event to webhook."""
    webhook = _webhooks.get(webhook_id)
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    # Check organization access
    if user.organization_id and webhook.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    # Create test payload
    test_payload = {
        "event": "webhook.test",
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "message": "This is a test webhook delivery",
            "webhook_id": webhook_id,
        },
    }

    # Create delivery record
    delivery_id = str(uuid4())
    timestamp = str(int(time.time()))
    payload_bytes = json.dumps(test_payload).encode()
    signature = _compute_signature(payload_bytes, webhook["secret"], timestamp)

    delivery = {
        "id": delivery_id,
        "webhook_id": webhook_id,
        "event_type": "webhook.test",
        "payload": test_payload,
        "attempts": 1,
        "delivered_at": datetime.utcnow(),
        "response_status": 200,  # Mock success
        "response_body": "OK",
    }

    _deliveries[delivery_id] = delivery

    return {
        "delivery_id": delivery_id,
        "status": "delivered",
        "signature": signature,
        "timestamp": timestamp,
        "payload": test_payload,
    }


# =============================================================================
# Webhook Event Trigger (Internal)
# =============================================================================

async def trigger_webhook_event(
    event_type: WebhookEventType,
    payload: dict[str, Any],
    organization_id: Optional[str] = None,
) -> list[str]:
    """
    Trigger webhook event to all registered webhooks.

    Returns list of delivery IDs.
    """
    delivery_ids = []

    # Find matching webhooks
    for webhook in _webhooks.values():
        # Check organization
        if organization_id and webhook.get("organization_id") != organization_id:
            continue

        # Check if webhook is active
        if not webhook.get("is_active", True):
            continue

        # Check if webhook is subscribed to this event
        if event_type.value not in webhook.get("events", []):
            continue

        # Create delivery
        delivery_id = str(uuid4())
        timestamp = str(int(time.time()))

        full_payload = {
            "event": event_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "data": payload,
        }

        payload_bytes = json.dumps(full_payload).encode()
        signature = _compute_signature(payload_bytes, webhook["secret"], timestamp)

        delivery = {
            "id": delivery_id,
            "webhook_id": webhook["id"],
            "event_type": event_type.value,
            "payload": full_payload,
            "attempts": 0,
            "signature": signature,
            "timestamp": timestamp,
        }

        _deliveries[delivery_id] = delivery
        delivery_ids.append(delivery_id)

        # Update webhook
        webhook["last_triggered_at"] = datetime.utcnow()

        # In production, would queue for async delivery

    return delivery_ids
