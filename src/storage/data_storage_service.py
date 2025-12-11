"""
Legal Document Data Storage Service
====================================
FastAPI service for the storage layer.

This module provides the REST API for the storage layer.
For direct programmatic access, use DataAccessLayer from data_access_layer.py.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Depends
from fastapi.responses import StreamingResponse

from .config import StorageConfig
from .data_access_layer import DataAccessLayer
from .models import DocumentRecord, SearchDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI Application
app = FastAPI(
    title="Legal Document Storage Service",
    description="Polyglot persistence layer for legal document platform",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Global DAL instance
dal: Optional[DataAccessLayer] = None


def get_dal() -> DataAccessLayer:
    """Dependency to get DAL instance."""
    if not dal:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return dal


@app.on_event("startup")
async def startup_event():
    """Initialize storage connections."""
    global dal

    config = StorageConfig.from_env()
    dal = DataAccessLayer(config)

    try:
        await dal.connect()
        logger.info("Storage service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize storage service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Close storage connections."""
    global dal
    if dal:
        await dal.disconnect()


# =============================================================================
# Health & Status Endpoints
# =============================================================================

@app.get("/api/v1/health")
async def health_check(dal: DataAccessLayer = Depends(get_dal)):
    """Health check endpoint."""
    return await dal.health_check()


@app.get("/api/v1/stats")
async def get_statistics(dal: DataAccessLayer = Depends(get_dal)):
    """Get storage statistics."""
    return await dal.get_statistics()


# =============================================================================
# Document Endpoints
# =============================================================================

@app.get("/api/v1/documents/{document_id}")
async def get_document(
    document_id: str,
    include_content: bool = Query(False),
    dal: DataAccessLayer = Depends(get_dal),
):
    """Get document metadata and optionally content."""
    result = await dal.get_document(document_id, include_content)
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")
    return result


@app.get("/api/v1/documents/{document_id}/context")
async def get_document_with_context(
    document_id: str,
    dal: DataAccessLayer = Depends(get_dal),
):
    """Get document with full knowledge graph context."""
    result = await dal.get_document_with_context(document_id)
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")
    return result


@app.delete("/api/v1/documents/{document_id}")
async def delete_document(
    document_id: str,
    deleted_by: str = Query(..., description="User ID performing deletion"),
    dal: DataAccessLayer = Depends(get_dal),
):
    """Delete a document."""
    try:
        success = await dal.delete_document(document_id, deleted_by)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"message": "Document deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/documents/{document_id}/download")
async def get_download_url(
    document_id: str,
    dal: DataAccessLayer = Depends(get_dal),
):
    """Get presigned URL for document download."""
    doc = await dal.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    url = await dal.minio.get_presigned_url(doc["storage_path"])
    return {"download_url": url, "expires_in_seconds": 3600}


@app.get("/api/v1/documents/{document_id}/similar")
async def get_similar_documents(
    document_id: str,
    limit: int = Query(10, le=50),
    dal: DataAccessLayer = Depends(get_dal),
):
    """Find documents similar to the given document."""
    results = await dal.find_similar_documents(document_id, limit)
    return {"similar_documents": results}


@app.get("/api/v1/documents/{document_id}/relationships")
async def get_document_relationships(
    document_id: str,
    relationship_type: Optional[str] = None,
    dal: DataAccessLayer = Depends(get_dal),
):
    """Get documents related to the given document."""
    results = await dal.get_document_relationships(document_id, relationship_type)
    return {"relationships": results}


# =============================================================================
# Search Endpoints
# =============================================================================

@app.get("/api/v1/search")
async def search_documents(
    q: str = Query(..., min_length=1, description="Search query"),
    document_type: Optional[str] = None,
    practice_area: Optional[str] = None,
    client_id: Optional[str] = None,
    matter_id: Optional[str] = None,
    classification: Optional[str] = None,
    from_: int = Query(0, alias="from", ge=0),
    size: int = Query(20, le=100),
    user_id: Optional[str] = None,
    dal: DataAccessLayer = Depends(get_dal),
):
    """Full-text search documents."""
    filters = {}
    if document_type:
        filters["document_type"] = document_type
    if practice_area:
        filters["practice_areas"] = practice_area
    if client_id:
        filters["client_id"] = client_id
    if matter_id:
        filters["matter_id"] = matter_id
    if classification:
        filters["classification"] = classification

    results = await dal.search(
        query=q,
        filters=filters if filters else None,
        from_=from_,
        size=size,
        user_id=user_id,
    )
    return results


@app.post("/api/v1/search/semantic")
async def semantic_search(
    query_vector: list[float],
    filters: Optional[dict[str, Any]] = None,
    size: int = Query(20, le=100),
    user_id: Optional[str] = None,
    dal: DataAccessLayer = Depends(get_dal),
):
    """Semantic search using embedding vector."""
    results = await dal.semantic_search(
        query_vector=query_vector,
        filters=filters,
        size=size,
        user_id=user_id,
    )
    return results


@app.get("/api/v1/search/entity")
async def search_by_entity(
    entity_value: str = Query(..., description="Entity value to search for"),
    entity_type: Optional[str] = Query(None, description="Entity type (person, organization, etc.)"),
    limit: int = Query(50, le=100),
    dal: DataAccessLayer = Depends(get_dal),
):
    """Find documents mentioning a specific entity."""
    results = await dal.find_documents_by_entity(
        entity_value=entity_value,
        entity_type=entity_type,
        limit=limit,
    )
    return {"documents": results}


# =============================================================================
# Analytics Endpoints
# =============================================================================

@app.get("/api/v1/analytics/documents")
async def get_document_analytics(
    client_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    dal: DataAccessLayer = Depends(get_dal),
):
    """Get document analytics."""
    return await dal.get_document_analytics(client_id, start_date, end_date)


@app.get("/api/v1/analytics/search")
async def get_search_analytics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    dal: DataAccessLayer = Depends(get_dal),
):
    """Get search analytics."""
    return await dal.get_search_analytics(start_date, end_date)


@app.get("/api/v1/analytics/api")
async def get_api_analytics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    dal: DataAccessLayer = Depends(get_dal),
):
    """Get API usage analytics."""
    return await dal.get_api_analytics(start_date, end_date)


# =============================================================================
# Knowledge Graph Endpoints
# =============================================================================

@app.post("/api/v1/relationships")
async def create_relationship(
    source_id: str,
    target_id: str,
    relationship_type: str,
    properties: Optional[dict[str, Any]] = None,
    dal: DataAccessLayer = Depends(get_dal),
):
    """Create a relationship between two documents."""
    success = await dal.create_document_relationship(
        source_id=source_id,
        target_id=target_id,
        relationship_type=relationship_type,
        properties=properties,
    )
    if not success:
        raise HTTPException(status_code=400, detail="Failed to create relationship")
    return {"message": "Relationship created successfully"}


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
