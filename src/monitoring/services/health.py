"""
Health Check Service
====================
Health and readiness probes for Kubernetes-style deployments.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from ..config import HealthCheckConfig
from ..models import HealthCheckResult, HealthReport, ServiceHealth


# =============================================================================
# Health Check Interface
# =============================================================================

class HealthCheck(ABC):
    """
    Base class for health checks.

    Implement this for each dependency (database, cache, etc).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Health check name."""
        pass

    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Execute health check."""
        pass

    async def safe_check(self) -> HealthCheckResult:
        """Execute check with error handling."""
        start = time.perf_counter()
        try:
            result = await self.check()
            result.latency_ms = (time.perf_counter() - start) * 1000
            return result
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=ServiceHealth.UNHEALTHY,
                message="Health check failed",
                latency_ms=(time.perf_counter() - start) * 1000,
                error=str(e),
            )


# =============================================================================
# Common Health Checks
# =============================================================================

class DatabaseHealthCheck(HealthCheck):
    """Health check for database connection."""

    def __init__(
        self,
        name: str = "database",
        connection_func: Optional[Callable[[], Any]] = None,
        query: str = "SELECT 1",
    ):
        self._name = name
        self._connection_func = connection_func
        self._query = query

    @property
    def name(self) -> str:
        return self._name

    async def check(self) -> HealthCheckResult:
        """Check database connectivity."""
        if not self._connection_func:
            return HealthCheckResult(
                name=self.name,
                status=ServiceHealth.HEALTHY,
                message="No database configured (mock)",
            )

        try:
            conn = self._connection_func()
            if asyncio.iscoroutinefunction(conn):
                conn = await conn

            # Execute test query
            if hasattr(conn, "execute"):
                result = conn.execute(self._query)
                if asyncio.iscoroutine(result):
                    await result

            return HealthCheckResult(
                name=self.name,
                status=ServiceHealth.HEALTHY,
                message="Database connection successful",
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=ServiceHealth.UNHEALTHY,
                message="Database connection failed",
                error=str(e),
            )


class RedisHealthCheck(HealthCheck):
    """Health check for Redis connection."""

    def __init__(
        self,
        name: str = "redis",
        client: Optional[Any] = None,
    ):
        self._name = name
        self._client = client

    @property
    def name(self) -> str:
        return self._name

    async def check(self) -> HealthCheckResult:
        """Check Redis connectivity."""
        if not self._client:
            return HealthCheckResult(
                name=self.name,
                status=ServiceHealth.HEALTHY,
                message="No Redis configured (mock)",
            )

        try:
            # Ping Redis
            result = self._client.ping()
            if asyncio.iscoroutine(result):
                result = await result

            if result:
                return HealthCheckResult(
                    name=self.name,
                    status=ServiceHealth.HEALTHY,
                    message="Redis connection successful",
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=ServiceHealth.DEGRADED,
                    message="Redis ping returned false",
                )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=ServiceHealth.UNHEALTHY,
                message="Redis connection failed",
                error=str(e),
            )


class ElasticsearchHealthCheck(HealthCheck):
    """Health check for Elasticsearch."""

    def __init__(
        self,
        name: str = "elasticsearch",
        client: Optional[Any] = None,
    ):
        self._name = name
        self._client = client

    @property
    def name(self) -> str:
        return self._name

    async def check(self) -> HealthCheckResult:
        """Check Elasticsearch connectivity."""
        if not self._client:
            return HealthCheckResult(
                name=self.name,
                status=ServiceHealth.HEALTHY,
                message="No Elasticsearch configured (mock)",
            )

        try:
            # Cluster health
            result = self._client.cluster.health()
            if asyncio.iscoroutine(result):
                result = await result

            cluster_status = result.get("status", "unknown")

            if cluster_status == "green":
                status = ServiceHealth.HEALTHY
            elif cluster_status == "yellow":
                status = ServiceHealth.DEGRADED
            else:
                status = ServiceHealth.UNHEALTHY

            return HealthCheckResult(
                name=self.name,
                status=status,
                message=f"Elasticsearch cluster status: {cluster_status}",
                details=result,
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=ServiceHealth.UNHEALTHY,
                message="Elasticsearch connection failed",
                error=str(e),
            )


class StorageHealthCheck(HealthCheck):
    """Health check for object storage (S3, MinIO)."""

    def __init__(
        self,
        name: str = "storage",
        client: Optional[Any] = None,
        bucket: str = "",
    ):
        self._name = name
        self._client = client
        self._bucket = bucket

    @property
    def name(self) -> str:
        return self._name

    async def check(self) -> HealthCheckResult:
        """Check storage connectivity."""
        if not self._client:
            return HealthCheckResult(
                name=self.name,
                status=ServiceHealth.HEALTHY,
                message="No storage configured (mock)",
            )

        try:
            # List buckets or head bucket
            if hasattr(self._client, "list_buckets"):
                result = self._client.list_buckets()
                if asyncio.iscoroutine(result):
                    await result

            return HealthCheckResult(
                name=self.name,
                status=ServiceHealth.HEALTHY,
                message="Storage connection successful",
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=ServiceHealth.UNHEALTHY,
                message="Storage connection failed",
                error=str(e),
            )


class HTTPHealthCheck(HealthCheck):
    """Health check for HTTP endpoint."""

    def __init__(
        self,
        name: str,
        url: str,
        timeout: float = 5.0,
        expected_status: int = 200,
    ):
        self._name = name
        self._url = url
        self._timeout = timeout
        self._expected_status = expected_status

    @property
    def name(self) -> str:
        return self._name

    async def check(self) -> HealthCheckResult:
        """Check HTTP endpoint."""
        try:
            import urllib.request
            import urllib.error

            req = urllib.request.Request(self._url)
            response = urllib.request.urlopen(req, timeout=self._timeout)

            if response.getcode() == self._expected_status:
                return HealthCheckResult(
                    name=self.name,
                    status=ServiceHealth.HEALTHY,
                    message=f"HTTP {response.getcode()}",
                    details={"url": self._url},
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=ServiceHealth.DEGRADED,
                    message=f"Unexpected status: {response.getcode()}",
                    details={"url": self._url, "status": response.getcode()},
                )

        except urllib.error.HTTPError as e:
            return HealthCheckResult(
                name=self.name,
                status=ServiceHealth.UNHEALTHY,
                message=f"HTTP error: {e.code}",
                error=str(e),
                details={"url": self._url},
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=ServiceHealth.UNHEALTHY,
                message="Request failed",
                error=str(e),
                details={"url": self._url},
            )


class CustomHealthCheck(HealthCheck):
    """Custom health check using callable."""

    def __init__(
        self,
        name: str,
        check_func: Callable[[], HealthCheckResult],
    ):
        self._name = name
        self._check_func = check_func

    @property
    def name(self) -> str:
        return self._name

    async def check(self) -> HealthCheckResult:
        """Execute custom check."""
        result = self._check_func()
        if asyncio.iscoroutine(result):
            result = await result
        return result


# =============================================================================
# Health Checker
# =============================================================================

@dataclass
class HealthCheckerState:
    """Internal state for health checker."""
    start_time: datetime = field(default_factory=datetime.utcnow)
    last_check: Optional[datetime] = None
    last_report: Optional[HealthReport] = None
    consecutive_failures: int = 0


class HealthChecker:
    """
    Health checker that aggregates multiple health checks.

    Usage:
        checker = HealthChecker(config)
        checker.add_check(DatabaseHealthCheck("db", connection_func))
        checker.add_check(RedisHealthCheck("cache", redis_client))

        report = await checker.check_all()
    """

    def __init__(
        self,
        config: Optional[HealthCheckConfig] = None,
        version: str = "1.0.0",
    ):
        self.config = config or HealthCheckConfig()
        self.version = version
        self._checks: list[HealthCheck] = []
        self._state = HealthCheckerState()
        self._liveness_checks: list[HealthCheck] = []
        self._readiness_checks: list[HealthCheck] = []

    def add_check(
        self,
        check: HealthCheck,
        liveness: bool = True,
        readiness: bool = True,
    ) -> None:
        """Add a health check."""
        self._checks.append(check)
        if liveness:
            self._liveness_checks.append(check)
        if readiness:
            self._readiness_checks.append(check)

    def add_liveness_check(self, check: HealthCheck) -> None:
        """Add liveness-only check."""
        self.add_check(check, liveness=True, readiness=False)

    def add_readiness_check(self, check: HealthCheck) -> None:
        """Add readiness-only check."""
        self.add_check(check, liveness=False, readiness=True)

    @property
    def uptime_seconds(self) -> float:
        """Get service uptime in seconds."""
        return (datetime.utcnow() - self._state.start_time).total_seconds()

    async def check_all(self) -> HealthReport:
        """Run all health checks."""
        results = await asyncio.gather(
            *[check.safe_check() for check in self._checks],
            return_exceptions=True,
        )

        # Convert exceptions to results
        check_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                check_results.append(HealthCheckResult(
                    name=self._checks[i].name,
                    status=ServiceHealth.UNHEALTHY,
                    error=str(result),
                ))
            else:
                check_results.append(result)

        report = HealthReport.aggregate(check_results, self.version)
        report.uptime_seconds = self.uptime_seconds

        # Update state
        self._state.last_check = datetime.utcnow()
        self._state.last_report = report

        if report.status == ServiceHealth.UNHEALTHY:
            self._state.consecutive_failures += 1
        else:
            self._state.consecutive_failures = 0

        return report

    async def check_liveness(self) -> HealthReport:
        """Run liveness checks only."""
        if not self._liveness_checks:
            return HealthReport(
                status=ServiceHealth.HEALTHY,
                version=self.version,
                uptime_seconds=self.uptime_seconds,
            )

        results = await asyncio.gather(
            *[check.safe_check() for check in self._liveness_checks],
            return_exceptions=True,
        )

        check_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                check_results.append(HealthCheckResult(
                    name=self._liveness_checks[i].name,
                    status=ServiceHealth.UNHEALTHY,
                    error=str(result),
                ))
            else:
                check_results.append(result)

        report = HealthReport.aggregate(check_results, self.version)
        report.uptime_seconds = self.uptime_seconds
        return report

    async def check_readiness(self) -> HealthReport:
        """Run readiness checks only."""
        if not self._readiness_checks:
            return HealthReport(
                status=ServiceHealth.HEALTHY,
                version=self.version,
                uptime_seconds=self.uptime_seconds,
            )

        results = await asyncio.gather(
            *[check.safe_check() for check in self._readiness_checks],
            return_exceptions=True,
        )

        check_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                check_results.append(HealthCheckResult(
                    name=self._readiness_checks[i].name,
                    status=ServiceHealth.UNHEALTHY,
                    error=str(result),
                ))
            else:
                check_results.append(result)

        report = HealthReport.aggregate(check_results, self.version)
        report.uptime_seconds = self.uptime_seconds
        return report

    def get_cached_report(self) -> Optional[HealthReport]:
        """Get last cached health report."""
        return self._state.last_report

    @property
    def is_healthy(self) -> bool:
        """Quick check if last report was healthy."""
        if not self._state.last_report:
            return True  # Assume healthy if not checked
        return self._state.last_report.status == ServiceHealth.HEALTHY


# =============================================================================
# Health Check Endpoints
# =============================================================================

def create_health_endpoints(checker: HealthChecker):
    """
    Create FastAPI health check endpoints.

    Usage:
        from fastapi import FastAPI
        app = FastAPI()

        checker = HealthChecker(version="1.0.0")
        health_router = create_health_endpoints(checker)
        app.include_router(health_router)
    """
    from fastapi import APIRouter, Response

    router = APIRouter(tags=["Health"])

    @router.get("/health")
    async def health():
        """Full health check."""
        report = await checker.check_all()
        status_code = 200 if report.status != ServiceHealth.UNHEALTHY else 503
        return Response(
            content=json_dumps(report.to_dict()),
            media_type="application/json",
            status_code=status_code,
        )

    @router.get("/health/live")
    async def liveness():
        """Liveness probe."""
        report = await checker.check_liveness()
        status_code = 200 if report.status != ServiceHealth.UNHEALTHY else 503
        return Response(
            content=json_dumps(report.to_dict()),
            media_type="application/json",
            status_code=status_code,
        )

    @router.get("/health/ready")
    async def readiness():
        """Readiness probe."""
        report = await checker.check_readiness()
        status_code = 200 if report.status != ServiceHealth.UNHEALTHY else 503
        return Response(
            content=json_dumps(report.to_dict()),
            media_type="application/json",
            status_code=status_code,
        )

    return router


def json_dumps(data: dict) -> str:
    """JSON serialize with datetime support."""
    import json

    def default(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(data, default=default)


# =============================================================================
# Background Health Monitor
# =============================================================================

class HealthMonitor:
    """
    Background health monitor that periodically runs checks.

    Usage:
        monitor = HealthMonitor(checker, interval=30)
        await monitor.start()
        # ... later ...
        await monitor.stop()
    """

    def __init__(
        self,
        checker: HealthChecker,
        interval: float = 30.0,
        on_unhealthy: Optional[Callable[[HealthReport], None]] = None,
    ):
        self.checker = checker
        self.interval = interval
        self.on_unhealthy = on_unhealthy
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start background monitoring."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                report = await self.checker.check_all()

                if report.status == ServiceHealth.UNHEALTHY and self.on_unhealthy:
                    try:
                        callback_result = self.on_unhealthy(report)
                        if asyncio.iscoroutine(callback_result):
                            await callback_result
                    except Exception:
                        pass

            except Exception:
                pass

            await asyncio.sleep(self.interval)
