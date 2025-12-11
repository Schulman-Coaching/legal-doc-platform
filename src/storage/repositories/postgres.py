"""
PostgreSQL Repository
=====================
Async PostgreSQL repository using asyncpg for document metadata storage.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from uuid import UUID

import asyncpg
from asyncpg import Pool, Connection

from ..config import PostgresConfig
from ..models import DocumentRecord, DocumentStorageClass, RetentionPolicy, StorageMetrics, StorageBackend

logger = logging.getLogger(__name__)


# SQL Schema
SCHEMA_SQL = """
-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    original_filename VARCHAR(512) NOT NULL,
    mime_type VARCHAR(128) NOT NULL,
    file_size BIGINT NOT NULL,
    checksum_sha256 VARCHAR(64) NOT NULL,
    storage_path VARCHAR(1024) NOT NULL,
    storage_class VARCHAR(32) DEFAULT 'hot',
    encryption_key_id VARCHAR(128),

    -- Organization
    client_id VARCHAR(128),
    matter_id VARCHAR(128),
    user_id VARCHAR(128) NOT NULL,
    classification VARCHAR(64) DEFAULT 'internal',
    tags JSONB DEFAULT '[]',

    -- Processing
    processing_status VARCHAR(64) DEFAULT 'pending',
    document_type VARCHAR(128),
    practice_areas JSONB DEFAULT '[]',

    -- Extracted counts
    entity_count INTEGER DEFAULT 0,
    clause_count INTEGER DEFAULT 0,
    citation_count INTEGER DEFAULT 0,
    page_count INTEGER DEFAULT 1,
    word_count INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,
    archived_at TIMESTAMP WITH TIME ZONE,

    -- Retention
    retention_policy VARCHAR(64) DEFAULT 'standard',
    retention_until TIMESTAMP WITH TIME ZONE,
    legal_hold BOOLEAN DEFAULT FALSE,

    -- Custom metadata
    custom_metadata JSONB DEFAULT '{}',

    -- Soft delete
    is_deleted BOOLEAN DEFAULT FALSE,
    deleted_at TIMESTAMP WITH TIME ZONE,
    deleted_by VARCHAR(128)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_documents_client_id ON documents(client_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_documents_matter_id ON documents(matter_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_documents_checksum ON documents(checksum_sha256);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_documents_legal_hold ON documents(legal_hold) WHERE legal_hold = TRUE;
CREATE INDEX IF NOT EXISTS idx_documents_tags ON documents USING GIN(tags);

-- Document versions table
CREATE TABLE IF NOT EXISTS document_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    storage_path VARCHAR(1024) NOT NULL,
    file_size BIGINT NOT NULL,
    checksum_sha256 VARCHAR(64) NOT NULL,
    created_by VARCHAR(128) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    change_description TEXT,
    is_current BOOLEAN DEFAULT TRUE,

    UNIQUE(document_id, version_number)
);

CREATE INDEX IF NOT EXISTS idx_doc_versions_document ON document_versions(document_id);
CREATE INDEX IF NOT EXISTS idx_doc_versions_current ON document_versions(document_id) WHERE is_current = TRUE;

-- Audit log table
CREATE TABLE IF NOT EXISTS document_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    action VARCHAR(64) NOT NULL,
    action_detail TEXT,
    user_id VARCHAR(128) NOT NULL,
    user_ip VARCHAR(45),
    user_agent TEXT,
    old_values JSONB,
    new_values JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_document ON document_audit_log(document_id);
CREATE INDEX IF NOT EXISTS idx_audit_user ON document_audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON document_audit_log(timestamp DESC);

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS documents_updated_at ON documents;
CREATE TRIGGER documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
"""


class PostgresRepository:
    """
    PostgreSQL repository for document metadata.

    Uses asyncpg for high-performance async database access.
    """

    def __init__(self, config: PostgresConfig):
        self.config = config
        self._pool: Optional[Pool] = None

    async def connect(self) -> None:
        """Establish database connection pool."""
        try:
            self._pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=self.config.pool_min_size,
                max_size=self.config.pool_max_size,
                command_timeout=self.config.command_timeout,
                statement_cache_size=self.config.statement_cache_size,
            )
            logger.info(f"Connected to PostgreSQL at {self.config.host}:{self.config.port}/{self.config.database}")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Disconnected from PostgreSQL")

    async def initialize_schema(self) -> None:
        """Create database schema if not exists."""
        async with self.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
            logger.info("PostgreSQL schema initialized")

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Connection]:
        """Acquire a connection from the pool."""
        if not self._pool:
            raise RuntimeError("Database pool not initialized. Call connect() first.")
        async with self._pool.acquire() as conn:
            yield conn

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Connection]:
        """Start a transaction."""
        async with self.acquire() as conn:
            async with conn.transaction():
                yield conn

    # Document CRUD Operations

    async def create(self, document: DocumentRecord) -> DocumentRecord:
        """Create a new document record."""
        start_time = datetime.utcnow()

        async with self.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO documents (
                    id, original_filename, mime_type, file_size, checksum_sha256,
                    storage_path, storage_class, encryption_key_id,
                    client_id, matter_id, user_id, classification, tags,
                    processing_status, document_type, practice_areas,
                    entity_count, clause_count, citation_count, page_count, word_count,
                    retention_policy, retention_until, legal_hold, custom_metadata
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                    $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25
                )
                RETURNING *
                """,
                document.id,
                document.original_filename,
                document.mime_type,
                document.file_size,
                document.checksum_sha256,
                document.storage_path,
                document.storage_class,
                document.encryption_key_id,
                document.client_id,
                document.matter_id,
                document.user_id,
                document.classification,
                json.dumps(document.tags),
                document.processing_status,
                document.document_type,
                json.dumps(document.practice_areas),
                document.entity_count,
                document.clause_count,
                document.citation_count,
                document.page_count,
                document.word_count,
                document.retention_policy,
                document.retention_until,
                document.legal_hold,
                json.dumps(document.custom_metadata),
            )

        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(f"Created document {document.id} in {duration_ms:.2f}ms")

        return self._row_to_document(row)

    async def get(self, document_id: str) -> Optional[DocumentRecord]:
        """Get document by ID."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM documents
                WHERE id = $1 AND NOT is_deleted
                """,
                document_id,
            )

        if row:
            return self._row_to_document(row)
        return None

    async def get_by_checksum(self, checksum: str) -> Optional[DocumentRecord]:
        """Get document by SHA-256 checksum for deduplication."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM documents
                WHERE checksum_sha256 = $1 AND NOT is_deleted
                LIMIT 1
                """,
                checksum,
            )

        if row:
            return self._row_to_document(row)
        return None

    async def update(self, document_id: str, **kwargs) -> Optional[DocumentRecord]:
        """Update document fields."""
        if not kwargs:
            return await self.get(document_id)

        # Build dynamic update query
        set_clauses = []
        values = []
        idx = 1

        for key, value in kwargs.items():
            if key in ("tags", "practice_areas", "custom_metadata"):
                value = json.dumps(value)
            set_clauses.append(f"{key} = ${idx}")
            values.append(value)
            idx += 1

        values.append(document_id)

        async with self.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE documents
                SET {', '.join(set_clauses)}
                WHERE id = ${idx} AND NOT is_deleted
                RETURNING *
                """,
                *values,
            )

        if row:
            return self._row_to_document(row)
        return None

    async def delete(self, document_id: str, deleted_by: str) -> bool:
        """Soft delete a document."""
        async with self.acquire() as conn:
            # Check for legal hold
            doc = await conn.fetchrow(
                "SELECT legal_hold FROM documents WHERE id = $1",
                document_id
            )
            if doc and doc["legal_hold"]:
                raise ValueError("Cannot delete document under legal hold")

            result = await conn.execute(
                """
                UPDATE documents
                SET is_deleted = TRUE, deleted_at = NOW(), deleted_by = $2,
                    processing_status = 'deleted'
                WHERE id = $1 AND NOT is_deleted
                """,
                document_id,
                deleted_by,
            )

        return result == "UPDATE 1"

    async def list(
        self,
        client_id: Optional[str] = None,
        matter_id: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        document_type: Optional[str] = None,
        classification: Optional[str] = None,
        tags: Optional[list[str]] = None,
        legal_hold: Optional[bool] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list[DocumentRecord]:
        """List documents with filters."""
        conditions = ["NOT is_deleted"]
        values = []
        idx = 1

        if client_id:
            conditions.append(f"client_id = ${idx}")
            values.append(client_id)
            idx += 1
        if matter_id:
            conditions.append(f"matter_id = ${idx}")
            values.append(matter_id)
            idx += 1
        if user_id:
            conditions.append(f"user_id = ${idx}")
            values.append(user_id)
            idx += 1
        if status:
            conditions.append(f"processing_status = ${idx}")
            values.append(status)
            idx += 1
        if document_type:
            conditions.append(f"document_type = ${idx}")
            values.append(document_type)
            idx += 1
        if classification:
            conditions.append(f"classification = ${idx}")
            values.append(classification)
            idx += 1
        if tags:
            conditions.append(f"tags ?| ${idx}")
            values.append(tags)
            idx += 1
        if legal_hold is not None:
            conditions.append(f"legal_hold = ${idx}")
            values.append(legal_hold)
            idx += 1
        if created_after:
            conditions.append(f"created_at >= ${idx}")
            values.append(created_after)
            idx += 1
        if created_before:
            conditions.append(f"created_at <= ${idx}")
            values.append(created_before)
            idx += 1

        values.extend([limit, offset])

        query = f"""
            SELECT * FROM documents
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
            LIMIT ${idx} OFFSET ${idx + 1}
        """

        async with self.acquire() as conn:
            rows = await conn.fetch(query, *values)

        return [self._row_to_document(row) for row in rows]

    async def count(
        self,
        client_id: Optional[str] = None,
        matter_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> int:
        """Count documents matching criteria."""
        conditions = ["NOT is_deleted"]
        values = []
        idx = 1

        if client_id:
            conditions.append(f"client_id = ${idx}")
            values.append(client_id)
            idx += 1
        if matter_id:
            conditions.append(f"matter_id = ${idx}")
            values.append(matter_id)
            idx += 1
        if status:
            conditions.append(f"processing_status = ${idx}")
            values.append(status)
            idx += 1

        query = f"SELECT COUNT(*) FROM documents WHERE {' AND '.join(conditions)}"

        async with self.acquire() as conn:
            result = await conn.fetchval(query, *values)

        return result or 0

    # Legal Hold Operations

    async def set_legal_hold(
        self,
        document_id: str,
        hold: bool,
        reason: Optional[str] = None,
    ) -> bool:
        """Set or release legal hold on document."""
        async with self.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE documents
                SET legal_hold = $2,
                    custom_metadata = CASE
                        WHEN $2 THEN jsonb_set(COALESCE(custom_metadata, '{}'), '{legal_hold_reason}', to_jsonb($3::text))
                        ELSE custom_metadata - 'legal_hold_reason'
                    END
                WHERE id = $1 AND NOT is_deleted
                """,
                document_id,
                hold,
                reason,
            )
        return result == "UPDATE 1"

    # Status Updates

    async def update_processing_status(
        self,
        document_id: str,
        status: str,
        document_type: Optional[str] = None,
        practice_areas: Optional[list[str]] = None,
        entity_count: Optional[int] = None,
        clause_count: Optional[int] = None,
        citation_count: Optional[int] = None,
        page_count: Optional[int] = None,
        word_count: Optional[int] = None,
    ) -> bool:
        """Update document processing status and extracted data."""
        updates = ["processing_status = $2"]
        values = [document_id, status]
        idx = 3

        if document_type is not None:
            updates.append(f"document_type = ${idx}")
            values.append(document_type)
            idx += 1
        if practice_areas is not None:
            updates.append(f"practice_areas = ${idx}")
            values.append(json.dumps(practice_areas))
            idx += 1
        if entity_count is not None:
            updates.append(f"entity_count = ${idx}")
            values.append(entity_count)
            idx += 1
        if clause_count is not None:
            updates.append(f"clause_count = ${idx}")
            values.append(clause_count)
            idx += 1
        if citation_count is not None:
            updates.append(f"citation_count = ${idx}")
            values.append(citation_count)
            idx += 1
        if page_count is not None:
            updates.append(f"page_count = ${idx}")
            values.append(page_count)
            idx += 1
        if word_count is not None:
            updates.append(f"word_count = ${idx}")
            values.append(word_count)
            idx += 1

        if status == "completed":
            updates.append("processed_at = NOW()")

        async with self.acquire() as conn:
            result = await conn.execute(
                f"""
                UPDATE documents
                SET {', '.join(updates)}
                WHERE id = $1 AND NOT is_deleted
                """,
                *values,
            )
        return result == "UPDATE 1"

    # Version Management

    async def create_version(
        self,
        document_id: str,
        storage_path: str,
        file_size: int,
        checksum: str,
        created_by: str,
        change_description: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a new version of a document."""
        async with self.transaction() as conn:
            # Get next version number
            max_version = await conn.fetchval(
                "SELECT COALESCE(MAX(version_number), 0) FROM document_versions WHERE document_id = $1",
                document_id,
            )
            new_version = max_version + 1

            # Mark old versions as not current
            await conn.execute(
                "UPDATE document_versions SET is_current = FALSE WHERE document_id = $1",
                document_id,
            )

            # Create new version
            row = await conn.fetchrow(
                """
                INSERT INTO document_versions (
                    document_id, version_number, storage_path, file_size,
                    checksum_sha256, created_by, change_description, is_current
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, TRUE)
                RETURNING *
                """,
                document_id,
                new_version,
                storage_path,
                file_size,
                checksum,
                created_by,
                change_description,
            )

            # Update main document
            await conn.execute(
                """
                UPDATE documents
                SET storage_path = $2, file_size = $3, checksum_sha256 = $4
                WHERE id = $1
                """,
                document_id,
                storage_path,
                file_size,
                checksum,
            )

        return dict(row)

    async def get_versions(self, document_id: str) -> list[dict[str, Any]]:
        """Get all versions of a document."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM document_versions
                WHERE document_id = $1
                ORDER BY version_number DESC
                """,
                document_id,
            )
        return [dict(row) for row in rows]

    # Audit Logging

    async def log_audit(
        self,
        document_id: Optional[str],
        action: str,
        user_id: str,
        action_detail: Optional[str] = None,
        old_values: Optional[dict] = None,
        new_values: Optional[dict] = None,
        user_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> str:
        """Log an audit event."""
        async with self.acquire() as conn:
            audit_id = await conn.fetchval(
                """
                INSERT INTO document_audit_log (
                    document_id, action, action_detail, user_id,
                    user_ip, user_agent, old_values, new_values
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
                """,
                document_id,
                action,
                action_detail,
                user_id,
                user_ip,
                user_agent,
                json.dumps(old_values) if old_values else None,
                json.dumps(new_values) if new_values else None,
            )
        return str(audit_id)

    async def get_audit_logs(
        self,
        document_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get audit logs with filtering."""
        conditions = []
        values = []
        idx = 1

        if document_id:
            conditions.append(f"document_id = ${idx}")
            values.append(document_id)
            idx += 1
        if user_id:
            conditions.append(f"user_id = ${idx}")
            values.append(user_id)
            idx += 1
        if action:
            conditions.append(f"action = ${idx}")
            values.append(action)
            idx += 1
        if since:
            conditions.append(f"timestamp >= ${idx}")
            values.append(since)
            idx += 1

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        values.append(limit)

        query = f"""
            SELECT * FROM document_audit_log
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ${idx}
        """

        async with self.acquire() as conn:
            rows = await conn.fetch(query, *values)

        return [dict(row) for row in rows]

    # Statistics

    async def get_statistics(
        self,
        client_id: Optional[str] = None,
        matter_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get document statistics."""
        conditions = ["NOT is_deleted"]
        values = []
        idx = 1

        if client_id:
            conditions.append(f"client_id = ${idx}")
            values.append(client_id)
            idx += 1
        if matter_id:
            conditions.append(f"matter_id = ${idx}")
            values.append(matter_id)
            idx += 1

        where_clause = ' AND '.join(conditions)

        async with self.acquire() as conn:
            # Total count and size
            totals = await conn.fetchrow(
                f"""
                SELECT
                    COUNT(*) as total_count,
                    COALESCE(SUM(file_size), 0) as total_size
                FROM documents
                WHERE {where_clause}
                """,
                *values,
            )

            # By status
            status_rows = await conn.fetch(
                f"""
                SELECT processing_status, COUNT(*) as count
                FROM documents
                WHERE {where_clause}
                GROUP BY processing_status
                """,
                *values,
            )

            # By type
            type_rows = await conn.fetch(
                f"""
                SELECT document_type, COUNT(*) as count
                FROM documents
                WHERE {where_clause} AND document_type IS NOT NULL
                GROUP BY document_type
                ORDER BY count DESC
                LIMIT 10
                """,
                *values,
            )

            # Recent activity
            recent = await conn.fetchval(
                f"""
                SELECT COUNT(*) FROM documents
                WHERE {where_clause} AND created_at >= NOW() - INTERVAL '24 hours'
                """,
                *values,
            )

        return {
            "total_documents": totals["total_count"],
            "total_size_bytes": totals["total_size"],
            "by_status": {row["processing_status"]: row["count"] for row in status_rows},
            "by_type": {row["document_type"]: row["count"] for row in type_rows},
            "documents_last_24h": recent,
        }

    # Health Check

    async def health_check(self) -> dict[str, Any]:
        """Check database health."""
        try:
            async with self.acquire() as conn:
                result = await conn.fetchrow("SELECT 1 as ok, NOW() as server_time")
                pool_size = self._pool.get_size() if self._pool else 0
                pool_free = self._pool.get_idle_size() if self._pool else 0

            return {
                "status": "healthy",
                "server_time": result["server_time"].isoformat(),
                "pool_size": pool_size,
                "pool_free": pool_free,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    # Helper Methods

    def _row_to_document(self, row: asyncpg.Record) -> DocumentRecord:
        """Convert database row to DocumentRecord."""
        return DocumentRecord(
            id=str(row["id"]),
            original_filename=row["original_filename"],
            mime_type=row["mime_type"],
            file_size=row["file_size"],
            checksum_sha256=row["checksum_sha256"],
            storage_path=row["storage_path"],
            storage_class=row["storage_class"] or DocumentStorageClass.HOT,
            encryption_key_id=row["encryption_key_id"],
            client_id=row["client_id"],
            matter_id=row["matter_id"],
            user_id=row["user_id"],
            classification=row["classification"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            processing_status=row["processing_status"],
            document_type=row["document_type"],
            practice_areas=json.loads(row["practice_areas"]) if row["practice_areas"] else [],
            entity_count=row["entity_count"],
            clause_count=row["clause_count"],
            citation_count=row["citation_count"],
            page_count=row["page_count"],
            word_count=row["word_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            processed_at=row["processed_at"],
            archived_at=row["archived_at"],
            retention_policy=row["retention_policy"] or RetentionPolicy.STANDARD,
            retention_until=row["retention_until"],
            legal_hold=row["legal_hold"],
            custom_metadata=json.loads(row["custom_metadata"]) if row["custom_metadata"] else {},
        )
