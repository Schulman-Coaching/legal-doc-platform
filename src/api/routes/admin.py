"""
Admin & Compliance API Routes
=============================
Administrative endpoints for system management and compliance.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..models import (
    ErrorResponse,
    ExportRequest,
    ExportResponse,
    LegalHoldCreate,
    LegalHoldResponse,
    PaginatedResponse,
    SuccessResponse,
)
from ..middleware.auth import (
    AuthenticatedUser,
    get_authenticated_user,
    require_permission,
    require_role,
)


router = APIRouter(prefix="/admin", tags=["Administration"])


# =============================================================================
# Storage (Demo)
# =============================================================================

_legal_holds: dict[str, dict[str, Any]] = {}
_exports: dict[str, dict[str, Any]] = {}
_audit_logs: list[dict[str, Any]] = []


# =============================================================================
# Legal Hold Management
# =============================================================================

@router.get(
    "/legal-holds",
    response_model=PaginatedResponse[LegalHoldResponse],
    summary="List legal holds",
    description="Get all legal holds.",
)
async def list_legal_holds(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None, alias="status"),
    matter_id: Optional[str] = Query(None),
    user: AuthenticatedUser = Depends(require_role("admin")),
) -> PaginatedResponse[LegalHoldResponse]:
    """List legal holds with pagination."""
    holds = list(_legal_holds.values())

    # Filter by organization
    if user.organization_id:
        holds = [h for h in holds if h.get("organization_id") == user.organization_id]

    # Apply filters
    if status_filter:
        holds = [h for h in holds if h.get("status") == status_filter]
    if matter_id:
        holds = [h for h in holds if h.get("matter_id") == matter_id]

    # Sort by applied_at descending
    holds.sort(key=lambda x: x.get("applied_at", datetime.min), reverse=True)

    # Paginate
    total = len(holds)
    start = (page - 1) * size
    end = start + size
    page_holds = holds[start:end]

    items = [
        LegalHoldResponse(
            id=h["id"],
            matter_id=h["matter_id"],
            matter_name=h.get("matter_name"),
            document_ids=h["document_ids"],
            custodians=h.get("custodians", []),
            reason=h["reason"],
            status=h["status"],
            applied_by=h["applied_by"],
            applied_at=h["applied_at"],
            released_by=h.get("released_by"),
            released_at=h.get("released_at"),
        )
        for h in page_holds
    ]

    return PaginatedResponse.create(items=items, total=total, page=page, size=size)


@router.post(
    "/legal-holds",
    response_model=LegalHoldResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create legal hold",
    description="Apply legal hold to documents.",
)
async def create_legal_hold(
    hold_data: LegalHoldCreate,
    user: AuthenticatedUser = Depends(require_permission("legal_holds:create")),
) -> LegalHoldResponse:
    """Apply legal hold."""
    hold_id = str(uuid4())
    now = datetime.utcnow()

    hold = {
        "id": hold_id,
        "matter_id": hold_data.matter_id,
        "matter_name": hold_data.matter_name,
        "document_ids": hold_data.document_ids,
        "custodians": hold_data.custodians,
        "reason": hold_data.reason,
        "status": "active",
        "organization_id": user.organization_id,
        "applied_by": user.id,
        "applied_at": now,
    }

    _legal_holds[hold_id] = hold

    # Log audit event
    _audit_logs.append({
        "id": str(uuid4()),
        "event_type": "legal_hold.created",
        "user_id": user.id,
        "resource_type": "legal_hold",
        "resource_id": hold_id,
        "details": {"document_count": len(hold_data.document_ids)},
        "timestamp": now,
    })

    return LegalHoldResponse(
        id=hold["id"],
        matter_id=hold["matter_id"],
        matter_name=hold.get("matter_name"),
        document_ids=hold["document_ids"],
        custodians=hold.get("custodians", []),
        reason=hold["reason"],
        status=hold["status"],
        applied_by=hold["applied_by"],
        applied_at=hold["applied_at"],
    )


@router.post(
    "/legal-holds/{hold_id}/release",
    response_model=LegalHoldResponse,
    summary="Release legal hold",
    description="Release a legal hold.",
)
async def release_legal_hold(
    hold_id: str,
    reason: Optional[str] = None,
    user: AuthenticatedUser = Depends(require_permission("legal_holds:release")),
) -> LegalHoldResponse:
    """Release legal hold."""
    hold = _legal_holds.get(hold_id)
    if not hold:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Legal hold {hold_id} not found",
        )

    # Check organization access
    if user.organization_id and hold.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Legal hold {hold_id} not found",
        )

    if hold["status"] != "active":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Legal hold is not active",
        )

    now = datetime.utcnow()
    hold["status"] = "released"
    hold["released_by"] = user.id
    hold["released_at"] = now
    hold["release_reason"] = reason

    # Log audit event
    _audit_logs.append({
        "id": str(uuid4()),
        "event_type": "legal_hold.released",
        "user_id": user.id,
        "resource_type": "legal_hold",
        "resource_id": hold_id,
        "details": {"reason": reason},
        "timestamp": now,
    })

    return LegalHoldResponse(
        id=hold["id"],
        matter_id=hold["matter_id"],
        matter_name=hold.get("matter_name"),
        document_ids=hold["document_ids"],
        custodians=hold.get("custodians", []),
        reason=hold["reason"],
        status=hold["status"],
        applied_by=hold["applied_by"],
        applied_at=hold["applied_at"],
        released_by=hold.get("released_by"),
        released_at=hold.get("released_at"),
    )


# =============================================================================
# Export Management
# =============================================================================

@router.get(
    "/exports",
    response_model=PaginatedResponse[ExportResponse],
    summary="List exports",
    description="Get all document exports.",
)
async def list_exports(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None, alias="status"),
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> PaginatedResponse[ExportResponse]:
    """List exports with pagination."""
    exports = list(_exports.values())

    # Filter by user or organization
    if not any(r in user.roles for r in ["admin", "manager"]):
        exports = [e for e in exports if e.get("created_by") == user.id]
    elif user.organization_id:
        exports = [e for e in exports if e.get("organization_id") == user.organization_id]

    # Apply filters
    if status_filter:
        exports = [e for e in exports if e.get("status") == status_filter]

    # Sort by created_at descending
    exports.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)

    # Paginate
    total = len(exports)
    start = (page - 1) * size
    end = start + size
    page_exports = exports[start:end]

    items = [
        ExportResponse(
            id=e["id"],
            status=e["status"],
            document_count=e["document_count"],
            format=e["format"],
            download_url=e.get("download_url"),
            expires_at=e.get("expires_at"),
            created_at=e["created_at"],
        )
        for e in page_exports
    ]

    return PaginatedResponse.create(items=items, total=total, page=page, size=size)


@router.post(
    "/exports",
    response_model=ExportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create export",
    description="Request document export.",
)
async def create_export(
    export_request: ExportRequest,
    user: AuthenticatedUser = Depends(require_permission("export:create")),
) -> ExportResponse:
    """Create document export."""
    export_id = str(uuid4())
    now = datetime.utcnow()

    export = {
        "id": export_id,
        "document_ids": export_request.document_ids,
        "document_count": len(export_request.document_ids),
        "format": export_request.format,
        "include_metadata": export_request.include_metadata,
        "include_analysis": export_request.include_analysis,
        "status": "processing",
        "organization_id": user.organization_id,
        "created_by": user.id,
        "created_at": now,
    }

    _exports[export_id] = export

    # In production, would queue async processing
    # Simulate completion
    export["status"] = "completed"
    export["download_url"] = f"/api/v1/admin/exports/{export_id}/download"
    export["expires_at"] = now + timedelta(hours=24)

    return ExportResponse(
        id=export["id"],
        status=export["status"],
        document_count=export["document_count"],
        format=export["format"],
        download_url=export.get("download_url"),
        expires_at=export.get("expires_at"),
        created_at=export["created_at"],
    )


@router.get(
    "/exports/{export_id}",
    response_model=ExportResponse,
    summary="Get export status",
    description="Get export status and download URL.",
)
async def get_export(
    export_id: str,
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> ExportResponse:
    """Get export by ID."""
    export = _exports.get(export_id)
    if not export:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Export {export_id} not found",
        )

    # Check access
    if export.get("created_by") != user.id:
        if user.organization_id and export.get("organization_id") != user.organization_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Export {export_id} not found",
            )
        if "admin" not in user.roles:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Export {export_id} not found",
            )

    return ExportResponse(
        id=export["id"],
        status=export["status"],
        document_count=export["document_count"],
        format=export["format"],
        download_url=export.get("download_url"),
        expires_at=export.get("expires_at"),
        created_at=export["created_at"],
    )


# =============================================================================
# Audit Logs
# =============================================================================

@router.get(
    "/audit-logs",
    summary="List audit logs",
    description="Get audit logs (admin only).",
)
async def list_audit_logs(
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    user_id: Optional[str] = Query(None, description="Filter by user"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    date_from: Optional[datetime] = Query(None, description="Start date"),
    date_to: Optional[datetime] = Query(None, description="End date"),
    admin: AuthenticatedUser = Depends(require_role("admin")),
) -> dict[str, Any]:
    """List audit logs."""
    logs = _audit_logs.copy()

    # Filter by organization
    if admin.organization_id:
        logs = [l for l in logs if l.get("organization_id") == admin.organization_id]

    # Apply filters
    if event_type:
        logs = [l for l in logs if l.get("event_type") == event_type]
    if user_id:
        logs = [l for l in logs if l.get("user_id") == user_id]
    if resource_type:
        logs = [l for l in logs if l.get("resource_type") == resource_type]
    if date_from:
        logs = [l for l in logs if l.get("timestamp", datetime.min) >= date_from]
    if date_to:
        logs = [l for l in logs if l.get("timestamp", datetime.max) <= date_to]

    # Sort by timestamp descending
    logs.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)

    # Paginate
    total = len(logs)
    start = (page - 1) * size
    end = start + size
    page_logs = logs[start:end]

    return {
        "items": page_logs,
        "total": total,
        "page": page,
        "size": size,
        "pages": (total + size - 1) // size if size > 0 else 0,
    }


# =============================================================================
# System Statistics
# =============================================================================

@router.get(
    "/statistics",
    summary="Get system statistics",
    description="Get platform statistics (admin only).",
)
async def get_statistics(
    admin: AuthenticatedUser = Depends(require_role("admin")),
) -> dict[str, Any]:
    """Get system statistics."""
    # In production, would query actual data
    return {
        "documents": {
            "total": 0,
            "by_status": {
                "pending": 0,
                "processing": 0,
                "analyzed": 0,
                "failed": 0,
            },
            "by_type": {},
        },
        "users": {
            "total": 2,
            "active": 2,
            "by_role": {
                "admin": 1,
                "user": 2,
            },
        },
        "storage": {
            "used_bytes": 0,
            "total_bytes": 10737418240,  # 10 GB
            "percentage_used": 0,
        },
        "api_usage": {
            "requests_today": 0,
            "requests_this_month": 0,
            "average_response_time_ms": 0,
        },
        "legal_holds": {
            "active": len([h for h in _legal_holds.values() if h.get("status") == "active"]),
            "total": len(_legal_holds),
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


# =============================================================================
# Compliance Reports
# =============================================================================

@router.get(
    "/compliance/reports",
    summary="List compliance reports",
    description="Get available compliance reports.",
)
async def list_compliance_reports(
    admin: AuthenticatedUser = Depends(require_role("admin")),
) -> dict[str, Any]:
    """List compliance reports."""
    return {
        "reports": [
            {
                "id": "data-inventory",
                "name": "Data Inventory Report",
                "description": "Complete inventory of all stored documents and metadata",
                "formats": ["pdf", "csv", "json"],
            },
            {
                "id": "access-log",
                "name": "Access Log Report",
                "description": "Detailed access logs for compliance audit",
                "formats": ["pdf", "csv"],
            },
            {
                "id": "retention",
                "name": "Retention Policy Report",
                "description": "Documents approaching retention deadlines",
                "formats": ["pdf", "csv"],
            },
            {
                "id": "pii-inventory",
                "name": "PII Inventory Report",
                "description": "Documents containing personally identifiable information",
                "formats": ["pdf", "csv", "json"],
            },
            {
                "id": "legal-hold",
                "name": "Legal Hold Report",
                "description": "Active and historical legal holds",
                "formats": ["pdf", "csv"],
            },
        ]
    }


@router.post(
    "/compliance/reports/{report_id}/generate",
    summary="Generate compliance report",
    description="Generate a compliance report.",
)
async def generate_compliance_report(
    report_id: str,
    format: str = Query("pdf", description="Report format"),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    admin: AuthenticatedUser = Depends(require_role("admin")),
) -> dict[str, Any]:
    """Generate compliance report."""
    valid_reports = ["data-inventory", "access-log", "retention", "pii-inventory", "legal-hold"]
    if report_id not in valid_reports:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{report_id}' not found",
        )

    # In production, would queue report generation
    generation_id = str(uuid4())

    return {
        "generation_id": generation_id,
        "report_id": report_id,
        "format": format,
        "status": "processing",
        "estimated_completion": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
        "download_url": f"/api/v1/admin/compliance/reports/{generation_id}/download",
    }


# =============================================================================
# Retention Policy
# =============================================================================

@router.get(
    "/retention/policies",
    summary="List retention policies",
    description="Get document retention policies.",
)
async def list_retention_policies(
    admin: AuthenticatedUser = Depends(require_role("admin")),
) -> dict[str, Any]:
    """List retention policies."""
    return {
        "policies": [
            {
                "id": "default",
                "name": "Default Policy",
                "retention_days": 2555,  # 7 years
                "document_types": ["*"],
                "is_default": True,
            },
            {
                "id": "contracts",
                "name": "Contract Retention",
                "retention_days": 3650,  # 10 years
                "document_types": ["contract", "agreement"],
                "is_default": False,
            },
            {
                "id": "correspondence",
                "name": "Correspondence Retention",
                "retention_days": 1095,  # 3 years
                "document_types": ["email", "letter", "memo"],
                "is_default": False,
            },
        ]
    }


@router.get(
    "/retention/upcoming",
    summary="Get documents approaching retention deadline",
    description="List documents that will be affected by retention policies.",
)
async def get_upcoming_retention(
    days: int = Query(30, ge=1, le=365, description="Days until retention deadline"),
    admin: AuthenticatedUser = Depends(require_role("admin")),
) -> dict[str, Any]:
    """Get documents approaching retention deadline."""
    # In production, would query actual documents
    return {
        "period_days": days,
        "documents": [],
        "total_count": 0,
        "by_policy": {},
    }
