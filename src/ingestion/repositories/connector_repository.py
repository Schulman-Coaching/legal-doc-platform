"""
Connector Configuration Repository
===================================
Repository for managing connector configurations.
"""

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.connector_config import (
    ConnectorConfig,
    ConnectorType,
    ConnectorStatus,
    EmailConnectorConfig,
    SFTPConnectorConfig,
    CloudConnectorConfig,
    ScannerConnectorConfig,
)

logger = logging.getLogger(__name__)


class ConnectorRepository:
    """Repository for connector configuration operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        name: str,
        connector_type: ConnectorType,
        created_by: str,
        **kwargs,
    ) -> ConnectorConfig:
        """Create a new connector configuration."""
        # Select appropriate model class
        model_class = self._get_model_class(connector_type)

        connector = model_class(
            name=name,
            connector_type=connector_type,
            created_by=created_by,
            **kwargs,
        )

        self.session.add(connector)
        await self.session.flush()

        logger.info(f"Created connector {connector.id}: {name}")
        return connector

    def _get_model_class(self, connector_type: ConnectorType):
        """Get the appropriate model class for connector type."""
        mapping = {
            ConnectorType.EMAIL_IMAP: EmailConnectorConfig,
            ConnectorType.EMAIL_POP3: EmailConnectorConfig,
            ConnectorType.SFTP: SFTPConnectorConfig,
            ConnectorType.CLOUD_S3: CloudConnectorConfig,
            ConnectorType.CLOUD_AZURE: CloudConnectorConfig,
            ConnectorType.CLOUD_GCS: CloudConnectorConfig,
            ConnectorType.SCANNER: ScannerConnectorConfig,
        }
        return mapping.get(connector_type, ConnectorConfig)

    async def get_by_id(self, connector_id: UUID) -> Optional[ConnectorConfig]:
        """Get connector by ID."""
        query = select(ConnectorConfig).where(ConnectorConfig.id == connector_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def list_connectors(
        self,
        connector_type: Optional[ConnectorType] = None,
        status: Optional[ConnectorStatus] = None,
        is_enabled: Optional[bool] = None,
        owned_by: Optional[str] = None,
        department: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ConnectorConfig]:
        """List connectors with filters."""
        query = select(ConnectorConfig)

        if connector_type:
            query = query.where(ConnectorConfig.connector_type == connector_type)
        if status:
            query = query.where(ConnectorConfig.status == status)
        if is_enabled is not None:
            query = query.where(ConnectorConfig.is_enabled == is_enabled)
        if owned_by:
            query = query.where(ConnectorConfig.owned_by == owned_by)
        if department:
            query = query.where(ConnectorConfig.department == department)

        query = query.order_by(ConnectorConfig.name)
        query = query.offset(offset).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_connectors_due_for_poll(
        self,
        limit: int = 10,
    ) -> list[ConnectorConfig]:
        """Get connectors that are due for polling."""
        now = datetime.utcnow()

        query = (
            select(ConnectorConfig)
            .where(
                and_(
                    ConnectorConfig.is_enabled == True,
                    ConnectorConfig.is_polling_enabled == True,
                    ConnectorConfig.status != ConnectorStatus.ERROR,
                    ConnectorConfig.next_poll_at <= now,
                )
            )
            .order_by(ConnectorConfig.next_poll_at)
            .limit(limit)
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def update(
        self,
        connector_id: UUID,
        **kwargs,
    ) -> Optional[ConnectorConfig]:
        """Update connector configuration."""
        connector = await self.get_by_id(connector_id)
        if not connector:
            return None

        for key, value in kwargs.items():
            if hasattr(connector, key):
                setattr(connector, key, value)

        connector.updated_at = datetime.utcnow()
        await self.session.flush()

        return connector

    async def update_status(
        self,
        connector_id: UUID,
        status: ConnectorStatus,
        error: Optional[str] = None,
    ) -> bool:
        """Update connector status."""
        values = {"status": status}

        if error:
            values["last_error"] = error
            values["last_error_at"] = datetime.utcnow()
            values["consecutive_errors"] = ConnectorConfig.consecutive_errors + 1
        else:
            values["consecutive_errors"] = 0

        stmt = (
            update(ConnectorConfig)
            .where(ConnectorConfig.id == connector_id)
            .values(**values)
        )

        result = await self.session.execute(stmt)
        return result.rowcount > 0

    async def record_poll(
        self,
        connector_id: UUID,
        success: bool,
        documents_ingested: int = 0,
        bytes_ingested: int = 0,
        error: Optional[str] = None,
    ) -> bool:
        """Record the result of a poll operation."""
        connector = await self.get_by_id(connector_id)
        if not connector:
            return False

        now = datetime.utcnow()
        connector.last_poll_at = now
        connector.next_poll_at = datetime.fromtimestamp(
            now.timestamp() + connector.poll_interval_seconds
        )

        if success:
            connector.status = ConnectorStatus.ACTIVE
            connector.consecutive_errors = 0
            connector.total_documents_ingested += documents_ingested
            connector.total_bytes_ingested += bytes_ingested
            if documents_ingested > 0:
                connector.last_successful_ingestion = now
        else:
            connector.consecutive_errors += 1
            connector.last_error = error
            connector.last_error_at = now

            if connector.consecutive_errors >= connector.max_consecutive_errors:
                connector.status = ConnectorStatus.ERROR

        await self.session.flush()
        return True

    async def enable(self, connector_id: UUID) -> bool:
        """Enable a connector."""
        stmt = (
            update(ConnectorConfig)
            .where(ConnectorConfig.id == connector_id)
            .values(
                is_enabled=True,
                status=ConnectorStatus.ACTIVE,
                updated_at=datetime.utcnow(),
            )
        )

        result = await self.session.execute(stmt)
        return result.rowcount > 0

    async def disable(self, connector_id: UUID) -> bool:
        """Disable a connector."""
        stmt = (
            update(ConnectorConfig)
            .where(ConnectorConfig.id == connector_id)
            .values(
                is_enabled=False,
                status=ConnectorStatus.INACTIVE,
                updated_at=datetime.utcnow(),
            )
        )

        result = await self.session.execute(stmt)
        return result.rowcount > 0

    async def delete(self, connector_id: UUID) -> bool:
        """Delete a connector configuration."""
        connector = await self.get_by_id(connector_id)
        if not connector:
            return False

        await self.session.delete(connector)
        await self.session.flush()

        logger.info(f"Deleted connector {connector_id}")
        return True

    async def get_statistics(self) -> dict:
        """Get connector statistics."""
        # Count by type
        type_query = (
            select(ConnectorConfig.connector_type, func.count(ConnectorConfig.id))
            .group_by(ConnectorConfig.connector_type)
        )
        type_result = await self.session.execute(type_query)
        by_type = {row[0].value: row[1] for row in type_result.all()}

        # Count by status
        status_query = (
            select(ConnectorConfig.status, func.count(ConnectorConfig.id))
            .group_by(ConnectorConfig.status)
        )
        status_result = await self.session.execute(status_query)
        by_status = {row[0].value: row[1] for row in status_result.all()}

        # Totals
        totals_query = select(
            func.count(ConnectorConfig.id),
            func.sum(ConnectorConfig.total_documents_ingested),
            func.sum(ConnectorConfig.total_bytes_ingested),
        )
        totals_result = await self.session.execute(totals_query)
        totals = totals_result.one()

        return {
            "total_connectors": totals[0],
            "by_type": by_type,
            "by_status": by_status,
            "total_documents_ingested": totals[1] or 0,
            "total_bytes_ingested": totals[2] or 0,
        }
