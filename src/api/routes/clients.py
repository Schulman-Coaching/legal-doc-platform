"""
Client & Matter API Routes
==========================
Client and matter management for document organization.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..models import (
    ClientCreate,
    ClientResponse,
    ErrorResponse,
    MatterCreate,
    MatterResponse,
    PaginatedResponse,
    SuccessResponse,
)
from ..middleware.auth import (
    AuthenticatedUser,
    get_authenticated_user,
    require_permission,
)


router = APIRouter(prefix="/clients", tags=["Clients & Matters"])


# =============================================================================
# Storage (Demo)
# =============================================================================

_clients: dict[str, dict[str, Any]] = {}
_matters: dict[str, dict[str, Any]] = {}


# =============================================================================
# Client CRUD
# =============================================================================

@router.get(
    "",
    response_model=PaginatedResponse[ClientResponse],
    summary="List clients",
    description="Get paginated list of clients.",
)
async def list_clients(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, description="Search by name or code"),
    industry: Optional[str] = Query(None, description="Filter by industry"),
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> PaginatedResponse[ClientResponse]:
    """List clients with pagination."""
    clients = list(_clients.values())

    # Filter by organization
    if user.organization_id:
        clients = [c for c in clients if c.get("organization_id") == user.organization_id]

    # Apply filters
    if search:
        search_lower = search.lower()
        clients = [
            c for c in clients
            if search_lower in c["name"].lower() or
               search_lower in (c.get("code") or "").lower()
        ]
    if industry:
        clients = [c for c in clients if c.get("industry") == industry]

    # Sort by name
    clients.sort(key=lambda x: x["name"])

    # Paginate
    total = len(clients)
    start = (page - 1) * size
    end = start + size
    page_clients = clients[start:end]

    items = [
        ClientResponse(
            id=c["id"],
            name=c["name"],
            code=c.get("code"),
            industry=c.get("industry"),
            contact_email=c.get("contact_email"),
            contact_phone=c.get("contact_phone"),
            address=c.get("address"),
            metadata=c.get("metadata", {}),
            document_count=c.get("document_count", 0),
            matter_count=c.get("matter_count", 0),
            created_at=c["created_at"],
            updated_at=c["updated_at"],
        )
        for c in page_clients
    ]

    return PaginatedResponse.create(items=items, total=total, page=page, size=size)


@router.get(
    "/{client_id}",
    response_model=ClientResponse,
    summary="Get client",
    description="Get client by ID.",
    responses={404: {"model": ErrorResponse}},
)
async def get_client(
    client_id: str,
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> ClientResponse:
    """Get client by ID."""
    client = _clients.get(client_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client {client_id} not found",
        )

    # Check organization access
    if user.organization_id and client.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client {client_id} not found",
        )

    return ClientResponse(
        id=client["id"],
        name=client["name"],
        code=client.get("code"),
        industry=client.get("industry"),
        contact_email=client.get("contact_email"),
        contact_phone=client.get("contact_phone"),
        address=client.get("address"),
        metadata=client.get("metadata", {}),
        document_count=client.get("document_count", 0),
        matter_count=client.get("matter_count", 0),
        created_at=client["created_at"],
        updated_at=client["updated_at"],
    )


@router.post(
    "",
    response_model=ClientResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create client",
    description="Create a new client.",
)
async def create_client(
    client_data: ClientCreate,
    user: AuthenticatedUser = Depends(require_permission("clients:create")),
) -> ClientResponse:
    """Create new client."""
    # Generate ID
    client_id = str(uuid4())
    now = datetime.utcnow()

    client = {
        "id": client_id,
        "name": client_data.name,
        "code": client_data.code,
        "industry": client_data.industry,
        "contact_email": client_data.contact_email,
        "contact_phone": client_data.contact_phone,
        "address": client_data.address,
        "metadata": client_data.metadata,
        "document_count": 0,
        "matter_count": 0,
        "organization_id": user.organization_id,
        "created_at": now,
        "updated_at": now,
        "created_by": user.id,
    }

    _clients[client_id] = client

    return ClientResponse(
        id=client["id"],
        name=client["name"],
        code=client.get("code"),
        industry=client.get("industry"),
        contact_email=client.get("contact_email"),
        contact_phone=client.get("contact_phone"),
        address=client.get("address"),
        metadata=client.get("metadata", {}),
        document_count=0,
        matter_count=0,
        created_at=client["created_at"],
        updated_at=client["updated_at"],
    )


@router.patch(
    "/{client_id}",
    response_model=ClientResponse,
    summary="Update client",
    description="Update client information.",
    responses={404: {"model": ErrorResponse}},
)
async def update_client(
    client_id: str,
    name: Optional[str] = None,
    code: Optional[str] = None,
    industry: Optional[str] = None,
    contact_email: Optional[str] = None,
    contact_phone: Optional[str] = None,
    address: Optional[str] = None,
    user: AuthenticatedUser = Depends(require_permission("clients:update")),
) -> ClientResponse:
    """Update client."""
    client = _clients.get(client_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client {client_id} not found",
        )

    # Check organization access
    if user.organization_id and client.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client {client_id} not found",
        )

    # Update fields
    if name is not None:
        client["name"] = name
    if code is not None:
        client["code"] = code
    if industry is not None:
        client["industry"] = industry
    if contact_email is not None:
        client["contact_email"] = contact_email
    if contact_phone is not None:
        client["contact_phone"] = contact_phone
    if address is not None:
        client["address"] = address

    client["updated_at"] = datetime.utcnow()

    return ClientResponse(
        id=client["id"],
        name=client["name"],
        code=client.get("code"),
        industry=client.get("industry"),
        contact_email=client.get("contact_email"),
        contact_phone=client.get("contact_phone"),
        address=client.get("address"),
        metadata=client.get("metadata", {}),
        document_count=client.get("document_count", 0),
        matter_count=client.get("matter_count", 0),
        created_at=client["created_at"],
        updated_at=client["updated_at"],
    )


@router.delete(
    "/{client_id}",
    response_model=SuccessResponse,
    summary="Delete client",
    description="Delete a client.",
    responses={404: {"model": ErrorResponse}},
)
async def delete_client(
    client_id: str,
    user: AuthenticatedUser = Depends(require_permission("clients:delete")),
) -> SuccessResponse:
    """Delete client."""
    client = _clients.get(client_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client {client_id} not found",
        )

    # Check organization access
    if user.organization_id and client.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client {client_id} not found",
        )

    # Check for associated matters
    client_matters = [m for m in _matters.values() if m.get("client_id") == client_id]
    if client_matters:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot delete client with {len(client_matters)} associated matters",
        )

    del _clients[client_id]

    return SuccessResponse(
        success=True,
        message=f"Client {client_id} deleted",
    )


# =============================================================================
# Matter CRUD
# =============================================================================

@router.get(
    "/{client_id}/matters",
    response_model=PaginatedResponse[MatterResponse],
    summary="List matters for client",
    description="Get paginated list of matters for a client.",
)
async def list_client_matters(
    client_id: str,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None, alias="status"),
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> PaginatedResponse[MatterResponse]:
    """List matters for a client."""
    # Verify client exists and access
    client = _clients.get(client_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client {client_id} not found",
        )

    if user.organization_id and client.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client {client_id} not found",
        )

    # Get matters for client
    matters = [m for m in _matters.values() if m.get("client_id") == client_id]

    # Apply filters
    if status_filter:
        matters = [m for m in matters if m.get("status") == status_filter]

    # Sort by created_at descending
    matters.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)

    # Paginate
    total = len(matters)
    start = (page - 1) * size
    end = start + size
    page_matters = matters[start:end]

    items = [
        MatterResponse(
            id=m["id"],
            client_id=m["client_id"],
            name=m["name"],
            code=m.get("code"),
            description=m.get("description"),
            practice_area=m.get("practice_area"),
            status=m["status"],
            metadata=m.get("metadata", {}),
            document_count=m.get("document_count", 0),
            created_at=m["created_at"],
            updated_at=m["updated_at"],
        )
        for m in page_matters
    ]

    return PaginatedResponse.create(items=items, total=total, page=page, size=size)


@router.post(
    "/{client_id}/matters",
    response_model=MatterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create matter",
    description="Create a new matter for a client.",
)
async def create_matter(
    client_id: str,
    matter_data: MatterCreate,
    user: AuthenticatedUser = Depends(require_permission("matters:create")),
) -> MatterResponse:
    """Create new matter."""
    # Verify client exists and access
    client = _clients.get(client_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client {client_id} not found",
        )

    if user.organization_id and client.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client {client_id} not found",
        )

    # Generate ID
    matter_id = str(uuid4())
    now = datetime.utcnow()

    matter = {
        "id": matter_id,
        "client_id": client_id,
        "name": matter_data.name,
        "code": matter_data.code,
        "description": matter_data.description,
        "practice_area": matter_data.practice_area,
        "status": matter_data.status,
        "metadata": matter_data.metadata,
        "document_count": 0,
        "organization_id": user.organization_id,
        "created_at": now,
        "updated_at": now,
        "created_by": user.id,
    }

    _matters[matter_id] = matter

    # Update client matter count
    client["matter_count"] = client.get("matter_count", 0) + 1

    return MatterResponse(
        id=matter["id"],
        client_id=matter["client_id"],
        name=matter["name"],
        code=matter.get("code"),
        description=matter.get("description"),
        practice_area=matter.get("practice_area"),
        status=matter["status"],
        metadata=matter.get("metadata", {}),
        document_count=0,
        created_at=matter["created_at"],
        updated_at=matter["updated_at"],
    )


# Top-level matters endpoint
@router.get(
    "/matters/{matter_id}",
    response_model=MatterResponse,
    summary="Get matter",
    description="Get matter by ID.",
    responses={404: {"model": ErrorResponse}},
    tags=["Matters"],
)
async def get_matter(
    matter_id: str,
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> MatterResponse:
    """Get matter by ID."""
    matter = _matters.get(matter_id)
    if not matter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Matter {matter_id} not found",
        )

    # Check organization access
    if user.organization_id and matter.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Matter {matter_id} not found",
        )

    return MatterResponse(
        id=matter["id"],
        client_id=matter["client_id"],
        name=matter["name"],
        code=matter.get("code"),
        description=matter.get("description"),
        practice_area=matter.get("practice_area"),
        status=matter["status"],
        metadata=matter.get("metadata", {}),
        document_count=matter.get("document_count", 0),
        created_at=matter["created_at"],
        updated_at=matter["updated_at"],
    )


@router.patch(
    "/matters/{matter_id}",
    response_model=MatterResponse,
    summary="Update matter",
    description="Update matter information.",
    responses={404: {"model": ErrorResponse}},
    tags=["Matters"],
)
async def update_matter(
    matter_id: str,
    name: Optional[str] = None,
    code: Optional[str] = None,
    description: Optional[str] = None,
    practice_area: Optional[str] = None,
    status_update: Optional[str] = None,
    user: AuthenticatedUser = Depends(require_permission("matters:update")),
) -> MatterResponse:
    """Update matter."""
    matter = _matters.get(matter_id)
    if not matter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Matter {matter_id} not found",
        )

    # Check organization access
    if user.organization_id and matter.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Matter {matter_id} not found",
        )

    # Update fields
    if name is not None:
        matter["name"] = name
    if code is not None:
        matter["code"] = code
    if description is not None:
        matter["description"] = description
    if practice_area is not None:
        matter["practice_area"] = practice_area
    if status_update is not None:
        matter["status"] = status_update

    matter["updated_at"] = datetime.utcnow()

    return MatterResponse(
        id=matter["id"],
        client_id=matter["client_id"],
        name=matter["name"],
        code=matter.get("code"),
        description=matter.get("description"),
        practice_area=matter.get("practice_area"),
        status=matter["status"],
        metadata=matter.get("metadata", {}),
        document_count=matter.get("document_count", 0),
        created_at=matter["created_at"],
        updated_at=matter["updated_at"],
    )


@router.delete(
    "/matters/{matter_id}",
    response_model=SuccessResponse,
    summary="Delete matter",
    description="Delete a matter.",
    responses={404: {"model": ErrorResponse}},
    tags=["Matters"],
)
async def delete_matter(
    matter_id: str,
    user: AuthenticatedUser = Depends(require_permission("matters:delete")),
) -> SuccessResponse:
    """Delete matter."""
    matter = _matters.get(matter_id)
    if not matter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Matter {matter_id} not found",
        )

    # Check organization access
    if user.organization_id and matter.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Matter {matter_id} not found",
        )

    # Check for documents
    if matter.get("document_count", 0) > 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot delete matter with {matter['document_count']} documents",
        )

    # Update client matter count
    client = _clients.get(matter["client_id"])
    if client:
        client["matter_count"] = max(0, client.get("matter_count", 1) - 1)

    del _matters[matter_id]

    return SuccessResponse(
        success=True,
        message=f"Matter {matter_id} deleted",
    )
