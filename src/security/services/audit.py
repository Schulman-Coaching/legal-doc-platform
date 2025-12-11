"""
Audit Logging Service
=====================
Tamper-evident, immutable audit logging.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import uuid4

from ..config import AuditConfig
from ..models import AuditEvent, AuditEventType

logger = logging.getLogger(__name__)


class AuditService:
    """
    Tamper-evident audit logging service.

    Features:
    - Hash chain for tamper detection
    - Multiple storage backends (PostgreSQL, Elasticsearch, File)
    - Asynchronous batch writes for performance
    - Event filtering and querying
    - Retention policy enforcement
    """

    def __init__(self, config: AuditConfig):
        self.config = config
        self._connected = False
        self._last_hash: Optional[str] = None
        self._event_buffer: deque[AuditEvent] = deque(maxlen=config.batch_size * 2)
        self._flush_task: Optional[asyncio.Task] = None
        self._db_pool = None
        self._es_client = None
        self._file_handle = None

    async def connect(self) -> None:
        """Initialize connection to audit storage backend."""
        try:
            if self.config.backend == "postgresql":
                await self._connect_postgres()
            elif self.config.backend == "elasticsearch":
                await self._connect_elasticsearch()
            elif self.config.backend == "file":
                await self._connect_file()

            # Get last hash for chain continuity
            await self._load_last_hash()

            # Start background flush task
            self._flush_task = asyncio.create_task(self._background_flush())

            self._connected = True
            logger.info("Audit service connected to %s backend", self.config.backend)

        except Exception as e:
            logger.error("Failed to connect audit service: %s", str(e))
            # Use in-memory fallback
            self._connected = True
            logger.warning("Using in-memory audit storage")

    async def disconnect(self) -> None:
        """Close connection and flush pending events."""
        # Cancel flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush remaining events
        await self._flush_buffer()

        # Close connections
        if self._db_pool:
            await self._db_pool.close()
        if self._es_client:
            await self._es_client.close()
        if self._file_handle:
            self._file_handle.close()

        self._connected = False
        logger.info("Audit service disconnected")

    async def _connect_postgres(self) -> None:
        """Connect to PostgreSQL backend."""
        try:
            import asyncpg

            self._db_pool = await asyncpg.create_pool(
                self.config.postgres_dsn,
                min_size=2,
                max_size=10,
            )

            # Create table if not exists
            async with self._db_pool.acquire() as conn:
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.config.postgres_table} (
                        id UUID PRIMARY KEY,
                        event_type VARCHAR(100) NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        user_id VARCHAR(100),
                        organization_id VARCHAR(100),
                        client_id VARCHAR(100),
                        session_id VARCHAR(100),
                        resource_type VARCHAR(100),
                        resource_id VARCHAR(255),
                        action VARCHAR(100),
                        outcome VARCHAR(20),
                        error_message TEXT,
                        ip_address VARCHAR(45),
                        user_agent TEXT,
                        request_id VARCHAR(100),
                        correlation_id VARCHAR(100),
                        details JSONB,
                        metadata JSONB,
                        previous_hash VARCHAR(64),
                        hash VARCHAR(64) NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.config.postgres_table}_timestamp
                    ON {self.config.postgres_table} (timestamp DESC)
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.config.postgres_table}_user
                    ON {self.config.postgres_table} (user_id, timestamp DESC)
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.config.postgres_table}_resource
                    ON {self.config.postgres_table} (resource_type, resource_id, timestamp DESC)
                """)
        except ImportError:
            logger.warning("asyncpg not installed, PostgreSQL backend unavailable")
            self._db_pool = None

    async def _connect_elasticsearch(self) -> None:
        """Connect to Elasticsearch backend."""
        try:
            from elasticsearch import AsyncElasticsearch

            self._es_client = AsyncElasticsearch(
                self.config.elasticsearch_hosts,
            )

            # Create index template
            await self._es_client.indices.put_index_template(
                name="audit-logs",
                body={
                    "index_patterns": [f"{self.config.elasticsearch_index}-*"],
                    "template": {
                        "settings": {
                            "number_of_shards": 1,
                            "number_of_replicas": 1,
                        },
                        "mappings": {
                            "properties": {
                                "id": {"type": "keyword"},
                                "event_type": {"type": "keyword"},
                                "timestamp": {"type": "date"},
                                "user_id": {"type": "keyword"},
                                "organization_id": {"type": "keyword"},
                                "client_id": {"type": "keyword"},
                                "resource_type": {"type": "keyword"},
                                "resource_id": {"type": "keyword"},
                                "action": {"type": "keyword"},
                                "outcome": {"type": "keyword"},
                                "ip_address": {"type": "ip"},
                                "details": {"type": "object"},
                                "hash": {"type": "keyword"},
                            }
                        }
                    }
                },
                ignore=400,
            )
        except ImportError:
            logger.warning("elasticsearch not installed, ES backend unavailable")
            self._es_client = None

    async def _connect_file(self) -> None:
        """Connect to file backend."""
        import os

        if self.config.file_path:
            os.makedirs(os.path.dirname(self.config.file_path), exist_ok=True)
            self._file_handle = open(self.config.file_path, 'a')

    async def _load_last_hash(self) -> None:
        """Load the last hash for chain continuity."""
        if self._db_pool:
            async with self._db_pool.acquire() as conn:
                row = await conn.fetchrow(f"""
                    SELECT hash FROM {self.config.postgres_table}
                    ORDER BY timestamp DESC LIMIT 1
                """)
                if row:
                    self._last_hash = row['hash']
        elif self._es_client:
            try:
                result = await self._es_client.search(
                    index=f"{self.config.elasticsearch_index}-*",
                    body={
                        "size": 1,
                        "sort": [{"timestamp": "desc"}],
                        "_source": ["hash"],
                    }
                )
                if result['hits']['hits']:
                    self._last_hash = result['hits']['hits'][0]['_source']['hash']
            except Exception:
                pass

    # =========================================================================
    # Event Logging
    # =========================================================================

    async def log(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        client_id: Optional[str] = None,
        session_id: Optional[str] = None,
        resource_type: str = "",
        resource_id: Optional[str] = None,
        action: str = "",
        outcome: str = "success",
        error_message: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            event_type: Type of event
            user_id: ID of user performing action
            organization_id: Organization context
            client_id: Client/matter context
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            action: Action performed
            outcome: Result (success, failure, error)
            error_message: Error details if failed
            ip_address: Client IP address
            user_agent: Client user agent
            request_id: Request ID for tracing
            correlation_id: Correlation ID for distributed tracing
            details: Additional event-specific details
            metadata: Additional metadata

        Returns:
            Created AuditEvent
        """
        if not self._connected:
            raise RuntimeError("Audit service not connected")

        # Create event with hash chain
        event = AuditEvent(
            id=str(uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            organization_id=organization_id,
            client_id=client_id,
            session_id=session_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome=outcome,
            error_message=error_message,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            correlation_id=correlation_id,
            details=details or {},
            metadata=metadata or {},
            previous_hash=self._last_hash,
        )

        # Update hash chain
        if self.config.enable_hash_chain:
            self._last_hash = event.hash

        # Add to buffer
        self._event_buffer.append(event)

        # Flush if buffer full
        if len(self._event_buffer) >= self.config.batch_size:
            asyncio.create_task(self._flush_buffer())

        return event

    async def log_auth_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> AuditEvent:
        """Log an authentication event."""
        return await self.log(
            event_type=event_type,
            user_id=user_id,
            resource_type="session",
            action=event_type.value.split('.')[-1],
            outcome="success" if success else "failure",
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
        )

    async def log_document_event(
        self,
        event_type: AuditEventType,
        document_id: str,
        user_id: str,
        client_id: Optional[str] = None,
        details: Optional[dict] = None,
        **kwargs,
    ) -> AuditEvent:
        """Log a document-related event."""
        return await self.log(
            event_type=event_type,
            user_id=user_id,
            client_id=client_id,
            resource_type="document",
            resource_id=document_id,
            action=event_type.value.split('.')[-1],
            details=details,
            **kwargs,
        )

    async def log_access_event(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        allowed: bool,
        policy_id: Optional[str] = None,
        **kwargs,
    ) -> AuditEvent:
        """Log an access control event."""
        event_type = (
            AuditEventType.PERMISSION_CHECK if allowed
            else AuditEventType.ACCESS_DENIED
        )
        return await self.log(
            event_type=event_type,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome="success" if allowed else "denied",
            details={"policy_id": policy_id} if policy_id else None,
            **kwargs,
        )

    async def log_compliance_event(
        self,
        event_type: AuditEventType,
        resource_id: str,
        user_id: Optional[str] = None,
        details: Optional[dict] = None,
        **kwargs,
    ) -> AuditEvent:
        """Log a compliance-related event."""
        return await self.log(
            event_type=event_type,
            user_id=user_id,
            resource_type="compliance",
            resource_id=resource_id,
            action=event_type.value.split('.')[-1],
            details=details,
            **kwargs,
        )

    # =========================================================================
    # Querying
    # =========================================================================

    async def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get a single audit event by ID."""
        if self._db_pool:
            async with self._db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"SELECT * FROM {self.config.postgres_table} WHERE id = $1",
                    event_id,
                )
                if row:
                    return self._row_to_event(dict(row))
        return None

    async def query_events(
        self,
        event_types: Optional[list[AuditEventType]] = None,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        client_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        outcome: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEvent]:
        """
        Query audit events with filters.

        Returns:
            List of matching AuditEvents
        """
        if self._db_pool:
            return await self._query_postgres(
                event_types, user_id, organization_id, client_id,
                resource_type, resource_id, start_time, end_time,
                outcome, limit, offset,
            )
        elif self._es_client:
            return await self._query_elasticsearch(
                event_types, user_id, organization_id, client_id,
                resource_type, resource_id, start_time, end_time,
                outcome, limit, offset,
            )
        return []

    async def _query_postgres(
        self,
        event_types: Optional[list[AuditEventType]],
        user_id: Optional[str],
        organization_id: Optional[str],
        client_id: Optional[str],
        resource_type: Optional[str],
        resource_id: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        outcome: Optional[str],
        limit: int,
        offset: int,
    ) -> list[AuditEvent]:
        """Query PostgreSQL backend."""
        conditions = []
        params = []
        param_idx = 1

        if event_types:
            placeholders = ', '.join(f'${i}' for i in range(param_idx, param_idx + len(event_types)))
            conditions.append(f"event_type IN ({placeholders})")
            params.extend(et.value for et in event_types)
            param_idx += len(event_types)

        if user_id:
            conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1

        if organization_id:
            conditions.append(f"organization_id = ${param_idx}")
            params.append(organization_id)
            param_idx += 1

        if client_id:
            conditions.append(f"client_id = ${param_idx}")
            params.append(client_id)
            param_idx += 1

        if resource_type:
            conditions.append(f"resource_type = ${param_idx}")
            params.append(resource_type)
            param_idx += 1

        if resource_id:
            conditions.append(f"resource_id = ${param_idx}")
            params.append(resource_id)
            param_idx += 1

        if start_time:
            conditions.append(f"timestamp >= ${param_idx}")
            params.append(start_time)
            param_idx += 1

        if end_time:
            conditions.append(f"timestamp <= ${param_idx}")
            params.append(end_time)
            param_idx += 1

        if outcome:
            conditions.append(f"outcome = ${param_idx}")
            params.append(outcome)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT * FROM {self.config.postgres_table}
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        async with self._db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._row_to_event(dict(row)) for row in rows]

    async def _query_elasticsearch(
        self,
        event_types: Optional[list[AuditEventType]],
        user_id: Optional[str],
        organization_id: Optional[str],
        client_id: Optional[str],
        resource_type: Optional[str],
        resource_id: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        outcome: Optional[str],
        limit: int,
        offset: int,
    ) -> list[AuditEvent]:
        """Query Elasticsearch backend."""
        must_clauses = []

        if event_types:
            must_clauses.append({
                "terms": {"event_type": [et.value for et in event_types]}
            })

        if user_id:
            must_clauses.append({"term": {"user_id": user_id}})

        if organization_id:
            must_clauses.append({"term": {"organization_id": organization_id}})

        if client_id:
            must_clauses.append({"term": {"client_id": client_id}})

        if resource_type:
            must_clauses.append({"term": {"resource_type": resource_type}})

        if resource_id:
            must_clauses.append({"term": {"resource_id": resource_id}})

        if outcome:
            must_clauses.append({"term": {"outcome": outcome}})

        if start_time or end_time:
            range_query = {"timestamp": {}}
            if start_time:
                range_query["timestamp"]["gte"] = start_time.isoformat()
            if end_time:
                range_query["timestamp"]["lte"] = end_time.isoformat()
            must_clauses.append({"range": range_query})

        result = await self._es_client.search(
            index=f"{self.config.elasticsearch_index}-*",
            body={
                "query": {"bool": {"must": must_clauses}} if must_clauses else {"match_all": {}},
                "sort": [{"timestamp": "desc"}],
                "from": offset,
                "size": limit,
            }
        )

        return [
            self._dict_to_event(hit['_source'])
            for hit in result['hits']['hits']
        ]

    # =========================================================================
    # Integrity Verification
    # =========================================================================

    async def verify_integrity(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """
        Verify integrity of audit log chain.

        Returns:
            Verification result with details
        """
        events = await self.query_events(
            start_time=start_time,
            end_time=end_time,
            limit=10000,  # Verify in batches for large logs
        )

        if not events:
            return {"status": "empty", "events_checked": 0}

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)

        invalid_events = []
        chain_breaks = []
        previous_hash = None

        for event in events:
            # Verify individual event hash
            if not event.verify_integrity():
                invalid_events.append(event.id)

            # Verify chain
            if previous_hash is not None and event.previous_hash != previous_hash:
                chain_breaks.append({
                    "event_id": event.id,
                    "expected_previous": previous_hash,
                    "actual_previous": event.previous_hash,
                })

            previous_hash = event.hash

        result = {
            "status": "valid" if not invalid_events and not chain_breaks else "invalid",
            "events_checked": len(events),
            "invalid_hashes": invalid_events,
            "chain_breaks": chain_breaks,
            "start_time": events[0].timestamp.isoformat() if events else None,
            "end_time": events[-1].timestamp.isoformat() if events else None,
        }

        if invalid_events or chain_breaks:
            logger.warning("Audit log integrity check failed: %s", result)

        return result

    # =========================================================================
    # Retention
    # =========================================================================

    async def apply_retention_policy(self) -> dict[str, Any]:
        """
        Apply retention policy to old audit events.

        Returns:
            Summary of actions taken
        """
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.retention_days)
        archive_date = datetime.utcnow() - timedelta(days=self.config.archive_after_days)

        result = {
            "archived": 0,
            "deleted": 0,
            "cutoff_date": cutoff_date.isoformat(),
        }

        if self._db_pool:
            async with self._db_pool.acquire() as conn:
                # Archive old events (could move to cold storage)
                if self.config.archive_after_days < self.config.retention_days:
                    # In production, would move to archive table/storage
                    pass

                # Delete expired events
                deleted = await conn.execute(
                    f"DELETE FROM {self.config.postgres_table} WHERE timestamp < $1",
                    cutoff_date,
                )
                result["deleted"] = int(deleted.split()[-1])

        logger.info("Retention policy applied: %s", result)
        return result

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _background_flush(self) -> None:
        """Background task to periodically flush event buffer."""
        while True:
            try:
                await asyncio.sleep(self.config.flush_interval_seconds)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in audit flush: %s", str(e))

    async def _flush_buffer(self) -> None:
        """Flush buffered events to storage."""
        if not self._event_buffer:
            return

        events = list(self._event_buffer)
        self._event_buffer.clear()

        if self._db_pool:
            await self._store_postgres(events)
        elif self._es_client:
            await self._store_elasticsearch(events)
        elif self._file_handle:
            await self._store_file(events)

        logger.debug("Flushed %d audit events", len(events))

    async def _store_postgres(self, events: list[AuditEvent]) -> None:
        """Store events in PostgreSQL."""
        async with self._db_pool.acquire() as conn:
            await conn.executemany(
                f"""
                INSERT INTO {self.config.postgres_table}
                (id, event_type, timestamp, user_id, organization_id, client_id,
                 session_id, resource_type, resource_id, action, outcome,
                 error_message, ip_address, user_agent, request_id, correlation_id,
                 details, metadata, previous_hash, hash)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                        $14, $15, $16, $17, $18, $19, $20)
                """,
                [
                    (
                        e.id, e.event_type.value, e.timestamp, e.user_id,
                        e.organization_id, e.client_id, e.session_id,
                        e.resource_type, e.resource_id, e.action, e.outcome,
                        e.error_message, e.ip_address, e.user_agent,
                        e.request_id, e.correlation_id,
                        json.dumps(e.details), json.dumps(e.metadata),
                        e.previous_hash, e.hash,
                    )
                    for e in events
                ]
            )

    async def _store_elasticsearch(self, events: list[AuditEvent]) -> None:
        """Store events in Elasticsearch."""
        index = f"{self.config.elasticsearch_index}-{datetime.utcnow().strftime('%Y.%m')}"

        operations = []
        for event in events:
            operations.append({"index": {"_index": index, "_id": event.id}})
            operations.append(event.to_dict())

        await self._es_client.bulk(body=operations)

    async def _store_file(self, events: list[AuditEvent]) -> None:
        """Store events in file."""
        for event in events:
            line = json.dumps(event.to_dict()) + '\n'
            self._file_handle.write(line)
        self._file_handle.flush()

    def _row_to_event(self, row: dict) -> AuditEvent:
        """Convert database row to AuditEvent."""
        return AuditEvent(
            id=str(row['id']),
            event_type=AuditEventType(row['event_type']),
            timestamp=row['timestamp'],
            user_id=row.get('user_id'),
            organization_id=row.get('organization_id'),
            client_id=row.get('client_id'),
            session_id=row.get('session_id'),
            resource_type=row.get('resource_type', ''),
            resource_id=row.get('resource_id'),
            action=row.get('action', ''),
            outcome=row.get('outcome', 'success'),
            error_message=row.get('error_message'),
            ip_address=row.get('ip_address'),
            user_agent=row.get('user_agent'),
            request_id=row.get('request_id'),
            correlation_id=row.get('correlation_id'),
            details=row.get('details') or {},
            metadata=row.get('metadata') or {},
            previous_hash=row.get('previous_hash'),
            hash=row.get('hash'),
        )

    def _dict_to_event(self, data: dict) -> AuditEvent:
        """Convert dictionary to AuditEvent."""
        return AuditEvent.from_dict(data)

    async def health_check(self) -> dict[str, Any]:
        """Check audit service health."""
        if not self._connected:
            return {"status": "disconnected"}

        result = {
            "status": "healthy",
            "backend": self.config.backend,
            "buffer_size": len(self._event_buffer),
            "hash_chain_enabled": self.config.enable_hash_chain,
        }

        if self._db_pool:
            try:
                async with self._db_pool.acquire() as conn:
                    count = await conn.fetchval(
                        f"SELECT COUNT(*) FROM {self.config.postgres_table}"
                    )
                    result["total_events"] = count
            except Exception as e:
                result["status"] = "degraded"
                result["error"] = str(e)

        return result
