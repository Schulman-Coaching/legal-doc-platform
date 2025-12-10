"""
Document Repository
===================
Repository pattern implementation for document data access.
"""

import logging
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import and_, or_, func, select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.document import (
    Document,
    DocumentVersion,
    DocumentMetadata,
    DocumentAuditLog,
    DocumentStatus,
    SecurityClassification,
    DocumentType,
    Tag,
)

logger = logging.getLogger(__name__)


class DocumentRepository:
    """
    Repository for document CRUD operations.

    Provides async database access with transaction support.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        original_filename: str,
        file_size: int,
        mime_type: str,
        storage_path: str,
        checksum_sha256: str,
        created_by: str,
        **kwargs,
    ) -> Document:
        """Create a new document record."""
        document = Document(
            original_filename=original_filename,
            file_size=file_size,
            mime_type=mime_type,
            storage_path=storage_path,
            checksum_sha256=checksum_sha256,
            created_by=created_by,
            **kwargs,
        )

        self.session.add(document)
        await self.session.flush()

        logger.info(f"Created document {document.id}: {original_filename}")
        return document

    async def get_by_id(
        self,
        document_id: UUID,
        include_versions: bool = False,
        include_metadata: bool = False,
    ) -> Optional[Document]:
        """Get document by ID with optional eager loading."""
        query = select(Document).where(
            and_(
                Document.id == document_id,
                Document.is_deleted == False,
            )
        )

        if include_versions:
            query = query.options(selectinload(Document.versions))
        if include_metadata:
            query = query.options(selectinload(Document.metadata_records))

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_by_checksum(self, checksum_sha256: str) -> Optional[Document]:
        """Get document by SHA-256 checksum for deduplication."""
        query = select(Document).where(
            and_(
                Document.checksum_sha256 == checksum_sha256,
                Document.is_deleted == False,
            )
        )

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def search(
        self,
        client_id: Optional[str] = None,
        matter_id: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
        classification: Optional[SecurityClassification] = None,
        document_type: Optional[DocumentType] = None,
        filename_contains: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        tags: Optional[list[str]] = None,
        legal_hold: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """Search documents with filters."""
        query = select(Document).where(Document.is_deleted == False)

        # Apply filters
        if client_id:
            query = query.where(Document.client_id == client_id)
        if matter_id:
            query = query.where(Document.matter_id == matter_id)
        if status:
            query = query.where(Document.status == status)
        if classification:
            query = query.where(Document.classification == classification)
        if document_type:
            query = query.where(Document.document_type == document_type)
        if filename_contains:
            query = query.where(Document.original_filename.ilike(f"%{filename_contains}%"))
        if created_after:
            query = query.where(Document.created_at >= created_after)
        if created_before:
            query = query.where(Document.created_at <= created_before)
        if legal_hold is not None:
            query = query.where(Document.legal_hold == legal_hold)

        # Tag filtering requires join
        if tags:
            query = query.join(Document.tags).where(Tag.name.in_(tags))

        # Ordering and pagination
        query = query.order_by(Document.created_at.desc())
        query = query.offset(offset).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def count(
        self,
        client_id: Optional[str] = None,
        matter_id: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
    ) -> int:
        """Count documents matching criteria."""
        query = select(func.count(Document.id)).where(Document.is_deleted == False)

        if client_id:
            query = query.where(Document.client_id == client_id)
        if matter_id:
            query = query.where(Document.matter_id == matter_id)
        if status:
            query = query.where(Document.status == status)

        result = await self.session.execute(query)
        return result.scalar_one()

    async def update(
        self,
        document_id: UUID,
        **kwargs,
    ) -> Optional[Document]:
        """Update document fields."""
        document = await self.get_by_id(document_id)
        if not document:
            return None

        for key, value in kwargs.items():
            if hasattr(document, key):
                setattr(document, key, value)

        document.updated_at = datetime.utcnow()
        await self.session.flush()

        return document

    async def update_status(
        self,
        document_id: UUID,
        status: DocumentStatus,
        error_message: Optional[str] = None,
    ) -> bool:
        """Update document status."""
        stmt = (
            update(Document)
            .where(Document.id == document_id)
            .values(
                status=status,
                processing_error=error_message,
                updated_at=datetime.utcnow(),
            )
        )

        result = await self.session.execute(stmt)
        return result.rowcount > 0

    async def soft_delete(
        self,
        document_id: UUID,
        deleted_by: str,
    ) -> bool:
        """Soft delete a document."""
        document = await self.get_by_id(document_id)
        if not document:
            return False

        # Check legal hold
        if document.legal_hold:
            raise ValueError("Cannot delete document under legal hold")

        document.is_deleted = True
        document.deleted_at = datetime.utcnow()
        document.deleted_by = deleted_by
        document.status = DocumentStatus.DELETED

        await self.session.flush()
        return True

    async def set_legal_hold(
        self,
        document_id: UUID,
        hold: bool,
        reason: Optional[str] = None,
    ) -> bool:
        """Set or release legal hold on document."""
        stmt = (
            update(Document)
            .where(Document.id == document_id)
            .values(
                legal_hold=hold,
                legal_hold_reason=reason if hold else None,
                updated_at=datetime.utcnow(),
            )
        )

        result = await self.session.execute(stmt)
        return result.rowcount > 0

    # Version management

    async def create_version(
        self,
        document_id: UUID,
        storage_path: str,
        file_size: int,
        checksum_sha256: str,
        created_by: str,
        change_description: Optional[str] = None,
    ) -> DocumentVersion:
        """Create a new version of a document."""
        # Get current max version
        query = select(func.max(DocumentVersion.version_number)).where(
            DocumentVersion.document_id == document_id
        )
        result = await self.session.execute(query)
        max_version = result.scalar_one() or 0

        # Mark old versions as not current
        stmt = (
            update(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .values(is_current=False)
        )
        await self.session.execute(stmt)

        # Create new version
        version = DocumentVersion(
            document_id=document_id,
            version_number=max_version + 1,
            storage_path=storage_path,
            file_size=file_size,
            checksum_sha256=checksum_sha256,
            created_by=created_by,
            change_description=change_description,
            is_current=True,
        )

        self.session.add(version)
        await self.session.flush()

        return version

    async def get_versions(self, document_id: UUID) -> list[DocumentVersion]:
        """Get all versions of a document."""
        query = (
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.version_number.desc())
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    # Metadata management

    async def add_metadata(
        self,
        document_id: UUID,
        category: str,
        key: str,
        value: Any,
        source: Optional[str] = None,
        confidence: Optional[int] = None,
    ) -> DocumentMetadata:
        """Add metadata to a document."""
        metadata = DocumentMetadata(
            document_id=document_id,
            category=category,
            key=key,
            value=value,
            source=source,
            confidence=confidence,
        )

        self.session.add(metadata)
        await self.session.flush()

        return metadata

    async def get_metadata(
        self,
        document_id: UUID,
        category: Optional[str] = None,
    ) -> list[DocumentMetadata]:
        """Get metadata for a document."""
        query = select(DocumentMetadata).where(
            DocumentMetadata.document_id == document_id
        )

        if category:
            query = query.where(DocumentMetadata.category == category)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    # Tag management

    async def add_tag(self, document_id: UUID, tag_name: str) -> bool:
        """Add a tag to a document."""
        # Get or create tag
        query = select(Tag).where(Tag.name == tag_name)
        result = await self.session.execute(query)
        tag = result.scalar_one_or_none()

        if not tag:
            tag = Tag(name=tag_name)
            self.session.add(tag)
            await self.session.flush()

        # Get document
        document = await self.get_by_id(document_id)
        if not document:
            return False

        if tag not in document.tags:
            document.tags.append(tag)
            await self.session.flush()

        return True

    async def remove_tag(self, document_id: UUID, tag_name: str) -> bool:
        """Remove a tag from a document."""
        document = await self.get_by_id(document_id)
        if not document:
            return False

        for tag in document.tags:
            if tag.name == tag_name:
                document.tags.remove(tag)
                await self.session.flush()
                return True

        return False

    # Audit logging

    async def log_action(
        self,
        document_id: UUID,
        action: str,
        user_id: str,
        action_detail: Optional[str] = None,
        old_values: Optional[dict] = None,
        new_values: Optional[dict] = None,
        user_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> DocumentAuditLog:
        """Log an action on a document."""
        audit = DocumentAuditLog(
            document_id=document_id,
            action=action,
            action_detail=action_detail,
            user_id=user_id,
            user_ip=user_ip,
            user_agent=user_agent,
            old_values=old_values,
            new_values=new_values,
        )

        self.session.add(audit)
        await self.session.flush()

        return audit

    async def get_audit_logs(
        self,
        document_id: Optional[UUID] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[DocumentAuditLog]:
        """Get audit logs with filtering."""
        query = select(DocumentAuditLog)

        if document_id:
            query = query.where(DocumentAuditLog.document_id == document_id)
        if user_id:
            query = query.where(DocumentAuditLog.user_id == user_id)
        if action:
            query = query.where(DocumentAuditLog.action == action)
        if since:
            query = query.where(DocumentAuditLog.timestamp >= since)

        query = query.order_by(DocumentAuditLog.timestamp.desc()).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    # Statistics

    async def get_statistics(
        self,
        client_id: Optional[str] = None,
    ) -> dict:
        """Get document statistics."""
        base_filter = Document.is_deleted == False
        if client_id:
            base_filter = and_(base_filter, Document.client_id == client_id)

        # Total count
        total_query = select(func.count(Document.id)).where(base_filter)
        total_result = await self.session.execute(total_query)
        total = total_result.scalar_one()

        # Count by status
        status_query = (
            select(Document.status, func.count(Document.id))
            .where(base_filter)
            .group_by(Document.status)
        )
        status_result = await self.session.execute(status_query)
        by_status = {row[0].value: row[1] for row in status_result.all()}

        # Total size
        size_query = select(func.sum(Document.file_size)).where(base_filter)
        size_result = await self.session.execute(size_query)
        total_size = size_result.scalar_one() or 0

        return {
            "total_documents": total,
            "by_status": by_status,
            "total_size_bytes": total_size,
        }
