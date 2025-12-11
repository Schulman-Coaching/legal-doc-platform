"""
Document API Routes
===================
CRUD operations, search, and analysis endpoints for documents.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse

from ..models import (
    DocumentAnalysis,
    DocumentResponse,
    DocumentSearchRequest,
    DocumentSearchResult,
    DocumentStatus,
    DocumentUpdate,
    DocumentUploadRequest,
    ErrorResponse,
    PaginatedResponse,
    SortOrder,
    SuccessResponse,
)
from ..middleware.auth import (
    AuthenticatedUser,
    get_authenticated_user,
    require_permission,
)
from ..middleware.request_context import get_request_context


router = APIRouter(prefix="/documents", tags=["Documents"])


# =============================================================================
# In-Memory Storage (Demo)
# =============================================================================

# Document storage (would be replaced by actual storage layer)
_documents: dict[str, dict[str, Any]] = {}


def _generate_document_id() -> str:
    """Generate unique document ID."""
    return str(uuid4())


# =============================================================================
# Document CRUD
# =============================================================================

@router.get(
    "",
    response_model=PaginatedResponse[DocumentResponse],
    summary="List documents",
    description="Get paginated list of documents with optional filtering.",
)
async def list_documents(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    client_id: Optional[str] = Query(None, description="Filter by client"),
    matter_id: Optional[str] = Query(None, description="Filter by matter"),
    status_filter: Optional[DocumentStatus] = Query(None, alias="status", description="Filter by status"),
    content_type: Optional[str] = Query(None, description="Filter by content type"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: SortOrder = Query(SortOrder.DESC, description="Sort order"),
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> PaginatedResponse[DocumentResponse]:
    """List documents with pagination and filtering."""
    # Filter documents
    docs = list(_documents.values())

    # Apply filters
    if client_id:
        docs = [d for d in docs if d.get("client_id") == client_id]
    if matter_id:
        docs = [d for d in docs if d.get("matter_id") == matter_id]
    if status_filter:
        docs = [d for d in docs if d.get("status") == status_filter.value]
    if content_type:
        docs = [d for d in docs if d.get("content_type") == content_type]

    # Filter by organization (multi-tenancy)
    if user.organization_id:
        docs = [d for d in docs if d.get("organization_id") == user.organization_id]

    # Sort
    reverse = sort_order == SortOrder.DESC
    if sort_by in ["created_at", "updated_at", "filename", "file_size"]:
        docs.sort(key=lambda x: x.get(sort_by, ""), reverse=reverse)

    # Paginate
    total = len(docs)
    start = (page - 1) * size
    end = start + size
    page_docs = docs[start:end]

    # Convert to response models
    items = [
        DocumentResponse(
            id=d["id"],
            filename=d["filename"],
            original_filename=d["original_filename"],
            content_type=d["content_type"],
            file_size=d["file_size"],
            checksum=d["checksum"],
            status=DocumentStatus(d["status"]),
            client_id=d.get("client_id"),
            matter_id=d.get("matter_id"),
            classification=d.get("classification"),
            tags=d.get("tags", []),
            metadata=d.get("metadata", {}),
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            created_by=d["created_by"],
        )
        for d in page_docs
    ]

    return PaginatedResponse.create(items=items, total=total, page=page, size=size)


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document",
    description="Get document details by ID.",
    responses={404: {"model": ErrorResponse}},
)
async def get_document(
    document_id: str,
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> DocumentResponse:
    """Get document by ID."""
    doc = _documents.get(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Check access (multi-tenancy)
    if user.organization_id and doc.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    return DocumentResponse(
        id=doc["id"],
        filename=doc["filename"],
        original_filename=doc["original_filename"],
        content_type=doc["content_type"],
        file_size=doc["file_size"],
        checksum=doc["checksum"],
        status=DocumentStatus(doc["status"]),
        client_id=doc.get("client_id"),
        matter_id=doc.get("matter_id"),
        classification=doc.get("classification"),
        tags=doc.get("tags", []),
        metadata=doc.get("metadata", {}),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        created_by=doc["created_by"],
    )


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload document",
    description="Upload a new document with metadata.",
)
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="Document file"),
    client_id: Optional[str] = Form(None, description="Client ID"),
    matter_id: Optional[str] = Form(None, description="Matter ID"),
    tags: Optional[str] = Form(None, description="Comma-separated tags"),
    user: AuthenticatedUser = Depends(require_permission("documents:create")),
) -> DocumentResponse:
    """Upload a new document."""
    # Read file content
    content = await file.read()
    file_size = len(content)

    # Calculate checksum
    checksum = hashlib.sha256(content).hexdigest()

    # Generate unique filename
    doc_id = _generate_document_id()
    ext = os.path.splitext(file.filename or "")[1]
    filename = f"{doc_id}{ext}"

    # Parse tags
    tag_list = []
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    # Create document record
    now = datetime.utcnow()
    doc = {
        "id": doc_id,
        "filename": filename,
        "original_filename": file.filename or "unknown",
        "content_type": file.content_type or "application/octet-stream",
        "file_size": file_size,
        "checksum": checksum,
        "status": DocumentStatus.PENDING.value,
        "client_id": client_id,
        "matter_id": matter_id,
        "tags": tag_list,
        "metadata": {},
        "created_at": now,
        "updated_at": now,
        "created_by": user.id,
        "organization_id": user.organization_id,
        "content": content,  # In-memory storage of content
    }

    _documents[doc_id] = doc

    return DocumentResponse(
        id=doc["id"],
        filename=doc["filename"],
        original_filename=doc["original_filename"],
        content_type=doc["content_type"],
        file_size=doc["file_size"],
        checksum=doc["checksum"],
        status=DocumentStatus(doc["status"]),
        client_id=doc.get("client_id"),
        matter_id=doc.get("matter_id"),
        tags=doc.get("tags", []),
        metadata=doc.get("metadata", {}),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        created_by=doc["created_by"],
    )


@router.patch(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Update document",
    description="Update document metadata.",
    responses={404: {"model": ErrorResponse}},
)
async def update_document(
    document_id: str,
    update: DocumentUpdate,
    user: AuthenticatedUser = Depends(require_permission("documents:update")),
) -> DocumentResponse:
    """Update document metadata."""
    doc = _documents.get(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Check access
    if user.organization_id and doc.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Update fields
    if update.client_id is not None:
        doc["client_id"] = update.client_id
    if update.matter_id is not None:
        doc["matter_id"] = update.matter_id
    if update.tags is not None:
        doc["tags"] = update.tags
    if update.metadata is not None:
        doc["metadata"].update(update.metadata)
    if update.classification is not None:
        doc["classification"] = update.classification

    doc["updated_at"] = datetime.utcnow()

    return DocumentResponse(
        id=doc["id"],
        filename=doc["filename"],
        original_filename=doc["original_filename"],
        content_type=doc["content_type"],
        file_size=doc["file_size"],
        checksum=doc["checksum"],
        status=DocumentStatus(doc["status"]),
        client_id=doc.get("client_id"),
        matter_id=doc.get("matter_id"),
        classification=doc.get("classification"),
        tags=doc.get("tags", []),
        metadata=doc.get("metadata", {}),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        created_by=doc["created_by"],
    )


@router.delete(
    "/{document_id}",
    response_model=SuccessResponse,
    summary="Delete document",
    description="Delete a document by ID.",
    responses={404: {"model": ErrorResponse}},
)
async def delete_document(
    document_id: str,
    user: AuthenticatedUser = Depends(require_permission("documents:delete")),
) -> SuccessResponse:
    """Delete document."""
    doc = _documents.get(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Check access
    if user.organization_id and doc.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Check for legal hold
    if doc.get("legal_hold"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document is under legal hold and cannot be deleted",
        )

    del _documents[document_id]

    return SuccessResponse(
        success=True,
        message=f"Document {document_id} deleted successfully",
    )


# =============================================================================
# Document Download
# =============================================================================

@router.get(
    "/{document_id}/download",
    summary="Download document",
    description="Download document content.",
    responses={404: {"model": ErrorResponse}},
)
async def download_document(
    document_id: str,
    user: AuthenticatedUser = Depends(require_permission("documents:read")),
) -> StreamingResponse:
    """Download document content."""
    doc = _documents.get(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Check access
    if user.organization_id and doc.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    content = doc.get("content", b"")

    def content_iterator():
        yield content

    return StreamingResponse(
        content_iterator(),
        media_type=doc["content_type"],
        headers={
            "Content-Disposition": f'attachment; filename="{doc["original_filename"]}"',
            "Content-Length": str(len(content)),
        },
    )


# =============================================================================
# Document Search
# =============================================================================

@router.post(
    "/search",
    response_model=PaginatedResponse[DocumentSearchResult],
    summary="Search documents",
    description="Full-text search across documents.",
)
async def search_documents(
    request: Request,
    search: DocumentSearchRequest,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> PaginatedResponse[DocumentSearchResult]:
    """Search documents."""
    query = search.query.lower()

    # Simple in-memory search (would use Elasticsearch/etc in production)
    results = []
    for doc in _documents.values():
        # Check organization access
        if user.organization_id and doc.get("organization_id") != user.organization_id:
            continue

        # Apply filters
        if search.client_id and doc.get("client_id") != search.client_id:
            continue
        if search.matter_id and doc.get("matter_id") != search.matter_id:
            continue
        if search.document_types and doc.get("content_type") not in search.document_types:
            continue

        # Simple text matching
        score = 0
        highlights = []

        # Match filename
        if query in doc.get("original_filename", "").lower():
            score += 10
            highlights.append(f"Filename: {doc['original_filename']}")

        # Match tags
        for tag in doc.get("tags", []):
            if query in tag.lower():
                score += 5
                highlights.append(f"Tag: {tag}")

        # Match classification
        if query in doc.get("classification", "").lower():
            score += 5
            highlights.append(f"Classification: {doc.get('classification')}")

        if score > 0:
            results.append({
                "doc": doc,
                "score": score,
                "highlights": highlights,
            })

    # Sort by score (or specified field)
    if search.sort_by == "relevance":
        results.sort(key=lambda x: x["score"], reverse=True)
    else:
        reverse = search.sort_order == SortOrder.DESC
        results.sort(
            key=lambda x: x["doc"].get(search.sort_by, ""),
            reverse=reverse,
        )

    # Paginate
    total = len(results)
    start = (page - 1) * size
    end = start + size
    page_results = results[start:end]

    items = [
        DocumentSearchResult(
            id=r["doc"]["id"],
            filename=r["doc"]["original_filename"],
            content_type=r["doc"]["content_type"],
            score=r["score"],
            highlights=r["highlights"],
            classification=r["doc"].get("classification"),
            created_at=r["doc"]["created_at"],
        )
        for r in page_results
    ]

    return PaginatedResponse.create(items=items, total=total, page=page, size=size)


# =============================================================================
# Document Analysis
# =============================================================================

@router.get(
    "/{document_id}/analysis",
    response_model=DocumentAnalysis,
    summary="Get document analysis",
    description="Get AI-powered analysis of document.",
    responses={404: {"model": ErrorResponse}},
)
async def get_document_analysis(
    document_id: str,
    user: AuthenticatedUser = Depends(require_permission("documents:read")),
) -> DocumentAnalysis:
    """Get document analysis results."""
    doc = _documents.get(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Check access
    if user.organization_id and doc.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Return stored analysis or mock
    analysis = doc.get("analysis", {})

    return DocumentAnalysis(
        document_id=document_id,
        document_type=analysis.get("document_type", "unknown"),
        classification=analysis.get("classification", doc.get("classification", "unclassified")),
        summary=analysis.get("summary"),
        key_entities=analysis.get("key_entities", []),
        key_dates=analysis.get("key_dates", []),
        key_clauses=analysis.get("key_clauses", []),
        risk_factors=analysis.get("risk_factors", []),
        pii_detected=analysis.get("pii_detected", []),
        compliance_status=analysis.get("compliance_status", {}),
        analyzed_at=analysis.get("analyzed_at", datetime.utcnow()),
    )


@router.post(
    "/{document_id}/analyze",
    response_model=DocumentAnalysis,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger document analysis",
    description="Start AI analysis of document.",
    responses={404: {"model": ErrorResponse}},
)
async def analyze_document(
    document_id: str,
    user: AuthenticatedUser = Depends(require_permission("documents:analyze")),
) -> DocumentAnalysis:
    """Trigger document analysis."""
    doc = _documents.get(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Check access
    if user.organization_id and doc.get("organization_id") != user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Mock analysis (would call AI/ML service in production)
    now = datetime.utcnow()
    analysis = {
        "document_type": "contract",
        "classification": "confidential",
        "summary": "This document appears to be a standard legal agreement.",
        "key_entities": [
            {"type": "organization", "value": "Example Corp", "confidence": 0.95},
        ],
        "key_dates": [
            {"type": "effective_date", "value": "2024-01-01", "confidence": 0.90},
        ],
        "key_clauses": [
            {"type": "termination", "text": "Either party may terminate...", "section": "5.1"},
        ],
        "risk_factors": [],
        "pii_detected": [],
        "compliance_status": {"gdpr": "compliant", "hipaa": "not_applicable"},
        "analyzed_at": now,
    }

    doc["analysis"] = analysis
    doc["status"] = DocumentStatus.ANALYZED.value
    doc["classification"] = analysis["classification"]
    doc["updated_at"] = now

    return DocumentAnalysis(
        document_id=document_id,
        document_type=analysis["document_type"],
        classification=analysis["classification"],
        summary=analysis["summary"],
        key_entities=analysis["key_entities"],
        key_dates=analysis["key_dates"],
        key_clauses=analysis["key_clauses"],
        risk_factors=analysis["risk_factors"],
        pii_detected=analysis["pii_detected"],
        compliance_status=analysis["compliance_status"],
        analyzed_at=now,
    )


# =============================================================================
# Bulk Operations
# =============================================================================

@router.post(
    "/bulk/delete",
    response_model=SuccessResponse,
    summary="Bulk delete documents",
    description="Delete multiple documents at once.",
)
async def bulk_delete_documents(
    document_ids: list[str],
    user: AuthenticatedUser = Depends(require_permission("documents:delete")),
) -> SuccessResponse:
    """Delete multiple documents."""
    deleted = 0
    skipped = 0

    for doc_id in document_ids:
        doc = _documents.get(doc_id)
        if not doc:
            skipped += 1
            continue

        # Check access
        if user.organization_id and doc.get("organization_id") != user.organization_id:
            skipped += 1
            continue

        # Skip if under legal hold
        if doc.get("legal_hold"):
            skipped += 1
            continue

        del _documents[doc_id]
        deleted += 1

    return SuccessResponse(
        success=True,
        message=f"Deleted {deleted} documents, skipped {skipped}",
        data={"deleted": deleted, "skipped": skipped},
    )


@router.post(
    "/bulk/tag",
    response_model=SuccessResponse,
    summary="Bulk tag documents",
    description="Add tags to multiple documents.",
)
async def bulk_tag_documents(
    document_ids: list[str],
    tags: list[str],
    user: AuthenticatedUser = Depends(require_permission("documents:update")),
) -> SuccessResponse:
    """Add tags to multiple documents."""
    updated = 0

    for doc_id in document_ids:
        doc = _documents.get(doc_id)
        if not doc:
            continue

        # Check access
        if user.organization_id and doc.get("organization_id") != user.organization_id:
            continue

        # Add tags
        existing_tags = set(doc.get("tags", []))
        existing_tags.update(tags)
        doc["tags"] = list(existing_tags)
        doc["updated_at"] = datetime.utcnow()
        updated += 1

    return SuccessResponse(
        success=True,
        message=f"Updated {updated} documents",
        data={"updated": updated},
    )
