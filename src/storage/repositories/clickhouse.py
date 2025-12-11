"""
ClickHouse Repository
=====================
ClickHouse repository for analytics and time-series data.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import uuid4

from clickhouse_driver import Client
from clickhouse_driver.errors import Error as ClickHouseError

from ..config import ClickHouseConfig
from ..models import AnalyticsEvent

logger = logging.getLogger(__name__)


class ClickHouseRepository:
    """
    ClickHouse repository for analytics and reporting.

    Handles document events, search analytics, API metrics, and time-series data.
    """

    def __init__(self, config: ClickHouseConfig):
        self.config = config
        self._client: Optional[Client] = None

    async def connect(self) -> None:
        """Connect to ClickHouse."""
        try:
            self._client = Client(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                secure=self.config.secure,
                verify=self.config.verify,
                connect_timeout=self.config.connect_timeout,
                send_receive_timeout=self.config.send_receive_timeout,
            )

            # Test connection
            self._client.execute("SELECT 1")

            logger.info(f"Connected to ClickHouse at {self.config.host}:{self.config.port}")

        except Exception as e:
            logger.error(f"Failed to connect to ClickHouse: {e}")
            raise

    async def disconnect(self) -> None:
        """Close ClickHouse connection."""
        if self._client:
            self._client.disconnect()
            self._client = None
            logger.info("Disconnected from ClickHouse")

    async def initialize_schema(self) -> None:
        """Create analytics tables."""
        if not self._client:
            raise RuntimeError("ClickHouse client not connected")

        schemas = [
            # Document events table
            """
            CREATE TABLE IF NOT EXISTS document_events (
                event_id UUID DEFAULT generateUUIDv4(),
                document_id String,
                event_type LowCardinality(String),
                event_time DateTime64(3) DEFAULT now64(3),
                user_id String,
                client_id Nullable(String),
                matter_id Nullable(String),
                document_type Nullable(String),
                practice_area Nullable(String),
                processing_time_ms UInt32 DEFAULT 0,
                file_size UInt64 DEFAULT 0,
                metadata String DEFAULT '{}'
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(event_time)
            ORDER BY (event_time, document_id, event_type)
            TTL event_time + INTERVAL 2 YEAR
            """,

            # Search analytics table
            """
            CREATE TABLE IF NOT EXISTS search_events (
                search_id UUID DEFAULT generateUUIDv4(),
                user_id String,
                query String,
                filters String DEFAULT '{}',
                results_count UInt32 DEFAULT 0,
                clicked_document_id Nullable(String),
                search_time DateTime64(3) DEFAULT now64(3),
                response_time_ms UInt32 DEFAULT 0,
                search_type LowCardinality(String) DEFAULT 'text'
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(search_time)
            ORDER BY (search_time, user_id)
            TTL search_time + INTERVAL 1 YEAR
            """,

            # API metrics table
            """
            CREATE TABLE IF NOT EXISTS api_metrics (
                request_id UUID DEFAULT generateUUIDv4(),
                endpoint String,
                method LowCardinality(String),
                user_id String,
                client_id Nullable(String),
                status_code UInt16,
                response_time_ms UInt32,
                request_size UInt32 DEFAULT 0,
                response_size UInt32 DEFAULT 0,
                timestamp DateTime64(3) DEFAULT now64(3),
                error_message Nullable(String)
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(timestamp)
            ORDER BY (timestamp, endpoint, method)
            TTL timestamp + INTERVAL 6 MONTH
            """,

            # User activity table
            """
            CREATE TABLE IF NOT EXISTS user_activity (
                activity_id UUID DEFAULT generateUUIDv4(),
                user_id String,
                activity_type LowCardinality(String),
                resource_type LowCardinality(String),
                resource_id Nullable(String),
                timestamp DateTime64(3) DEFAULT now64(3),
                session_id Nullable(String),
                ip_address Nullable(String),
                user_agent Nullable(String),
                metadata String DEFAULT '{}'
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(timestamp)
            ORDER BY (timestamp, user_id, activity_type)
            TTL timestamp + INTERVAL 1 YEAR
            """,

            # Processing metrics table
            """
            CREATE TABLE IF NOT EXISTS processing_metrics (
                metric_id UUID DEFAULT generateUUIDv4(),
                document_id String,
                stage LowCardinality(String),
                status LowCardinality(String),
                start_time DateTime64(3),
                end_time DateTime64(3),
                duration_ms UInt32,
                items_processed UInt32 DEFAULT 0,
                errors_count UInt32 DEFAULT 0,
                metadata String DEFAULT '{}'
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(start_time)
            ORDER BY (start_time, document_id, stage)
            TTL start_time + INTERVAL 1 YEAR
            """,

            # Daily aggregates (materialized view)
            """
            CREATE TABLE IF NOT EXISTS daily_document_stats (
                date Date,
                document_type LowCardinality(String),
                practice_area LowCardinality(String),
                total_documents UInt64,
                total_size UInt64,
                avg_processing_time Float64
            ) ENGINE = SummingMergeTree()
            PARTITION BY toYYYYMM(date)
            ORDER BY (date, document_type, practice_area)
            """,
        ]

        for schema in schemas:
            try:
                self._client.execute(schema)
            except ClickHouseError as e:
                logger.warning(f"Schema creation warning: {e}")

        logger.info("ClickHouse schema initialized")

    # Document Events

    async def record_document_event(
        self,
        document_id: str,
        event_type: str,
        user_id: str,
        client_id: Optional[str] = None,
        matter_id: Optional[str] = None,
        document_type: Optional[str] = None,
        practice_area: Optional[str] = None,
        processing_time_ms: int = 0,
        file_size: int = 0,
        metadata: Optional[dict] = None,
    ) -> str:
        """Record a document processing event."""
        if not self._client:
            raise RuntimeError("ClickHouse client not connected")

        event_id = str(uuid4())

        self._client.execute(
            """
            INSERT INTO document_events (
                event_id, document_id, event_type, user_id, client_id,
                matter_id, document_type, practice_area,
                processing_time_ms, file_size, metadata
            ) VALUES
            """,
            [(
                event_id,
                document_id,
                event_type,
                user_id,
                client_id,
                matter_id,
                document_type,
                practice_area,
                processing_time_ms,
                file_size,
                str(metadata or {}),
            )],
        )

        logger.debug(f"Recorded event {event_type} for document {document_id}")
        return event_id

    async def batch_record_events(self, events: list[dict[str, Any]]) -> int:
        """Batch insert multiple events."""
        if not self._client:
            raise RuntimeError("ClickHouse client not connected")

        if not events:
            return 0

        data = [
            (
                str(uuid4()),
                e.get("document_id", ""),
                e.get("event_type", ""),
                e.get("user_id", ""),
                e.get("client_id"),
                e.get("matter_id"),
                e.get("document_type"),
                e.get("practice_area"),
                e.get("processing_time_ms", 0),
                e.get("file_size", 0),
                str(e.get("metadata", {})),
            )
            for e in events
        ]

        self._client.execute(
            """
            INSERT INTO document_events (
                event_id, document_id, event_type, user_id, client_id,
                matter_id, document_type, practice_area,
                processing_time_ms, file_size, metadata
            ) VALUES
            """,
            data,
        )

        return len(data)

    # Search Analytics

    async def record_search_event(
        self,
        user_id: str,
        query: str,
        results_count: int,
        response_time_ms: int,
        filters: Optional[dict] = None,
        clicked_document_id: Optional[str] = None,
        search_type: str = "text",
    ) -> str:
        """Record a search event."""
        if not self._client:
            raise RuntimeError("ClickHouse client not connected")

        search_id = str(uuid4())

        self._client.execute(
            """
            INSERT INTO search_events (
                search_id, user_id, query, filters, results_count,
                clicked_document_id, response_time_ms, search_type
            ) VALUES
            """,
            [(
                search_id,
                user_id,
                query,
                str(filters or {}),
                results_count,
                clicked_document_id,
                response_time_ms,
                search_type,
            )],
        )

        return search_id

    async def record_search_click(
        self,
        search_id: str,
        document_id: str,
    ) -> None:
        """Update search event with clicked document."""
        if not self._client:
            raise RuntimeError("ClickHouse client not connected")

        # ClickHouse doesn't support UPDATE, so we record a new click event
        self._client.execute(
            """
            INSERT INTO search_events (
                search_id, user_id, query, clicked_document_id, search_type
            )
            SELECT search_id, user_id, query, %(doc_id)s, 'click'
            FROM search_events
            WHERE search_id = %(search_id)s
            LIMIT 1
            """,
            {"search_id": search_id, "doc_id": document_id},
        )

    # API Metrics

    async def record_api_metric(
        self,
        endpoint: str,
        method: str,
        user_id: str,
        status_code: int,
        response_time_ms: int,
        request_size: int = 0,
        response_size: int = 0,
        client_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> str:
        """Record an API metric."""
        if not self._client:
            raise RuntimeError("ClickHouse client not connected")

        request_id = str(uuid4())

        self._client.execute(
            """
            INSERT INTO api_metrics (
                request_id, endpoint, method, user_id, client_id,
                status_code, response_time_ms, request_size,
                response_size, error_message
            ) VALUES
            """,
            [(
                request_id,
                endpoint,
                method,
                user_id,
                client_id,
                status_code,
                response_time_ms,
                request_size,
                response_size,
                error_message,
            )],
        )

        return request_id

    # User Activity

    async def record_user_activity(
        self,
        user_id: str,
        activity_type: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Record user activity."""
        if not self._client:
            raise RuntimeError("ClickHouse client not connected")

        activity_id = str(uuid4())

        self._client.execute(
            """
            INSERT INTO user_activity (
                activity_id, user_id, activity_type, resource_type,
                resource_id, session_id, ip_address, user_agent, metadata
            ) VALUES
            """,
            [(
                activity_id,
                user_id,
                activity_type,
                resource_type,
                resource_id,
                session_id,
                ip_address,
                user_agent,
                str(metadata or {}),
            )],
        )

        return activity_id

    # Processing Metrics

    async def record_processing_metric(
        self,
        document_id: str,
        stage: str,
        status: str,
        start_time: datetime,
        end_time: datetime,
        items_processed: int = 0,
        errors_count: int = 0,
        metadata: Optional[dict] = None,
    ) -> str:
        """Record processing stage metrics."""
        if not self._client:
            raise RuntimeError("ClickHouse client not connected")

        metric_id = str(uuid4())
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        self._client.execute(
            """
            INSERT INTO processing_metrics (
                metric_id, document_id, stage, status,
                start_time, end_time, duration_ms,
                items_processed, errors_count, metadata
            ) VALUES
            """,
            [(
                metric_id,
                document_id,
                stage,
                status,
                start_time,
                end_time,
                duration_ms,
                items_processed,
                errors_count,
                str(metadata or {}),
            )],
        )

        return metric_id

    # Analytics Queries

    async def get_document_stats(
        self,
        client_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get document processing statistics."""
        if not self._client:
            raise RuntimeError("ClickHouse client not connected")

        start = start_date or (datetime.utcnow() - timedelta(days=30))
        end = end_date or datetime.utcnow()

        where_clauses = ["event_time >= %(start)s", "event_time <= %(end)s"]
        params = {"start": start, "end": end}

        if client_id:
            where_clauses.append("client_id = %(client_id)s")
            params["client_id"] = client_id

        where = " AND ".join(where_clauses)

        # Total counts
        result = self._client.execute(
            f"""
            SELECT
                count() as total_events,
                countDistinct(document_id) as unique_documents,
                sum(file_size) as total_size,
                avg(processing_time_ms) as avg_processing_time
            FROM document_events
            WHERE {where}
            """,
            params,
        )
        totals = result[0] if result else (0, 0, 0, 0)

        # By type
        by_type = self._client.execute(
            f"""
            SELECT document_type, count() as count
            FROM document_events
            WHERE {where} AND document_type IS NOT NULL
            GROUP BY document_type
            ORDER BY count DESC
            LIMIT 20
            """,
            params,
        )

        # By practice area
        by_practice = self._client.execute(
            f"""
            SELECT practice_area, count() as count
            FROM document_events
            WHERE {where} AND practice_area IS NOT NULL
            GROUP BY practice_area
            ORDER BY count DESC
            LIMIT 20
            """,
            params,
        )

        # Daily trend
        daily_trend = self._client.execute(
            f"""
            SELECT
                toDate(event_time) as date,
                count() as count,
                sum(file_size) as size
            FROM document_events
            WHERE {where}
            GROUP BY date
            ORDER BY date
            """,
            params,
        )

        return {
            "total_events": totals[0],
            "unique_documents": totals[1],
            "total_size_bytes": totals[2],
            "avg_processing_time_ms": round(totals[3] or 0, 2),
            "by_type": {row[0]: row[1] for row in by_type},
            "by_practice_area": {row[0]: row[1] for row in by_practice},
            "daily_trend": [
                {"date": str(row[0]), "count": row[1], "size": row[2]}
                for row in daily_trend
            ],
        }

    async def get_search_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get search analytics."""
        if not self._client:
            raise RuntimeError("ClickHouse client not connected")

        start = start_date or (datetime.utcnow() - timedelta(days=30))
        end = end_date or datetime.utcnow()

        params = {"start": start, "end": end}

        # Overview
        overview = self._client.execute(
            """
            SELECT
                count() as total_searches,
                avg(response_time_ms) as avg_response_time,
                avg(results_count) as avg_results,
                countIf(results_count = 0) as zero_results
            FROM search_events
            WHERE search_time >= %(start)s AND search_time <= %(end)s
                AND search_type = 'text'
            """,
            params,
        )[0]

        # Top queries
        top_queries = self._client.execute(
            """
            SELECT query, count() as count
            FROM search_events
            WHERE search_time >= %(start)s AND search_time <= %(end)s
                AND search_type = 'text'
            GROUP BY query
            ORDER BY count DESC
            LIMIT 20
            """,
            params,
        )

        # Zero result queries
        zero_result_queries = self._client.execute(
            """
            SELECT query, count() as count
            FROM search_events
            WHERE search_time >= %(start)s AND search_time <= %(end)s
                AND search_type = 'text' AND results_count = 0
            GROUP BY query
            ORDER BY count DESC
            LIMIT 20
            """,
            params,
        )

        # Click-through rate
        ctr = self._client.execute(
            """
            SELECT
                countIf(clicked_document_id IS NOT NULL) / count() as ctr
            FROM search_events
            WHERE search_time >= %(start)s AND search_time <= %(end)s
                AND search_type = 'text'
            """,
            params,
        )[0][0]

        return {
            "total_searches": overview[0],
            "avg_response_time_ms": round(overview[1] or 0, 2),
            "avg_results_count": round(overview[2] or 0, 2),
            "zero_result_rate": round(overview[3] / overview[0] if overview[0] else 0, 4),
            "click_through_rate": round(ctr or 0, 4),
            "top_queries": [{"query": q[0], "count": q[1]} for q in top_queries],
            "zero_result_queries": [{"query": q[0], "count": q[1]} for q in zero_result_queries],
        }

    async def get_api_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get API usage metrics."""
        if not self._client:
            raise RuntimeError("ClickHouse client not connected")

        start = start_date or (datetime.utcnow() - timedelta(days=7))
        end = end_date or datetime.utcnow()

        params = {"start": start, "end": end}

        # Overview
        overview = self._client.execute(
            """
            SELECT
                count() as total_requests,
                avg(response_time_ms) as avg_response_time,
                countIf(status_code >= 400) / count() as error_rate,
                countDistinct(user_id) as unique_users
            FROM api_metrics
            WHERE timestamp >= %(start)s AND timestamp <= %(end)s
            """,
            params,
        )[0]

        # By endpoint
        by_endpoint = self._client.execute(
            """
            SELECT
                endpoint,
                count() as count,
                avg(response_time_ms) as avg_time,
                countIf(status_code >= 400) as errors
            FROM api_metrics
            WHERE timestamp >= %(start)s AND timestamp <= %(end)s
            GROUP BY endpoint
            ORDER BY count DESC
            LIMIT 20
            """,
            params,
        )

        # Status code distribution
        status_codes = self._client.execute(
            """
            SELECT status_code, count() as count
            FROM api_metrics
            WHERE timestamp >= %(start)s AND timestamp <= %(end)s
            GROUP BY status_code
            ORDER BY count DESC
            """,
            params,
        )

        # Hourly trend
        hourly = self._client.execute(
            """
            SELECT
                toStartOfHour(timestamp) as hour,
                count() as count,
                avg(response_time_ms) as avg_time
            FROM api_metrics
            WHERE timestamp >= %(start)s AND timestamp <= %(end)s
            GROUP BY hour
            ORDER BY hour
            """,
            params,
        )

        return {
            "total_requests": overview[0],
            "avg_response_time_ms": round(overview[1] or 0, 2),
            "error_rate": round(overview[2] or 0, 4),
            "unique_users": overview[3],
            "by_endpoint": [
                {
                    "endpoint": e[0],
                    "count": e[1],
                    "avg_time_ms": round(e[2] or 0, 2),
                    "errors": e[3],
                }
                for e in by_endpoint
            ],
            "status_codes": {str(s[0]): s[1] for s in status_codes},
            "hourly_trend": [
                {"hour": str(h[0]), "count": h[1], "avg_time_ms": round(h[2] or 0, 2)}
                for h in hourly
            ],
        }

    async def get_user_activity_stats(
        self,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get user activity statistics."""
        if not self._client:
            raise RuntimeError("ClickHouse client not connected")

        start = start_date or (datetime.utcnow() - timedelta(days=30))
        end = end_date or datetime.utcnow()

        where_clauses = ["timestamp >= %(start)s", "timestamp <= %(end)s"]
        params = {"start": start, "end": end}

        if user_id:
            where_clauses.append("user_id = %(user_id)s")
            params["user_id"] = user_id

        where = " AND ".join(where_clauses)

        # Activity summary
        summary = self._client.execute(
            f"""
            SELECT
                count() as total_activities,
                countDistinct(user_id) as unique_users,
                countDistinct(session_id) as sessions
            FROM user_activity
            WHERE {where}
            """,
            params,
        )[0]

        # By activity type
        by_type = self._client.execute(
            f"""
            SELECT activity_type, count() as count
            FROM user_activity
            WHERE {where}
            GROUP BY activity_type
            ORDER BY count DESC
            """,
            params,
        )

        # By resource type
        by_resource = self._client.execute(
            f"""
            SELECT resource_type, count() as count
            FROM user_activity
            WHERE {where}
            GROUP BY resource_type
            ORDER BY count DESC
            """,
            params,
        )

        return {
            "total_activities": summary[0],
            "unique_users": summary[1],
            "total_sessions": summary[2],
            "by_activity_type": {t[0]: t[1] for t in by_type},
            "by_resource_type": {r[0]: r[1] for r in by_resource},
        }

    # Health Check

    async def health_check(self) -> dict[str, Any]:
        """Check ClickHouse health."""
        if not self._client:
            return {"status": "disconnected"}

        try:
            result = self._client.execute("SELECT version(), uptime()")
            version, uptime = result[0]

            # Get table sizes
            tables = self._client.execute(
                """
                SELECT
                    table,
                    formatReadableSize(sum(bytes)) as size,
                    sum(rows) as rows
                FROM system.parts
                WHERE database = %(db)s AND active
                GROUP BY table
                """,
                {"db": self.config.database},
            )

            return {
                "status": "healthy",
                "version": version,
                "uptime_seconds": uptime,
                "tables": [
                    {"name": t[0], "size": t[1], "rows": t[2]}
                    for t in tables
                ],
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
