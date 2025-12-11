"""
Tests for health check service.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
from ..services.health import (
    HealthChecker,
    HealthCheck,
    DatabaseHealthCheck,
    RedisHealthCheck,
    ElasticsearchHealthCheck,
    StorageHealthCheck,
    HTTPHealthCheck,
    CustomHealthCheck,
    HealthMonitor,
    create_health_endpoints,
)
from ..models import HealthCheckResult, HealthReport, ServiceHealth
from ..config import HealthCheckConfig


class TestHealthCheckResult:
    """Tests for HealthCheckResult model."""

    def test_healthy_result(self):
        """Test healthy result creation."""
        result = HealthCheckResult(
            name="database",
            status=ServiceHealth.HEALTHY,
            message="Connection OK",
            latency_ms=15.5,
        )
        assert result.name == "database"
        assert result.status == ServiceHealth.HEALTHY
        assert result.latency_ms == 15.5

    def test_unhealthy_result(self):
        """Test unhealthy result creation."""
        result = HealthCheckResult(
            name="redis",
            status=ServiceHealth.UNHEALTHY,
            message="Connection refused",
            error="ConnectionError: refused",
        )
        assert result.status == ServiceHealth.UNHEALTHY
        assert result.error is not None

    def test_degraded_result(self):
        """Test degraded result creation."""
        result = HealthCheckResult(
            name="external_api",
            status=ServiceHealth.DEGRADED,
            message="Slow response time",
            latency_ms=2500.0,
        )
        assert result.status == ServiceHealth.DEGRADED


class TestHealthReport:
    """Tests for HealthReport model."""

    def test_aggregate_all_healthy(self):
        """Test aggregation with all healthy checks."""
        results = [
            HealthCheckResult(name="db", status=ServiceHealth.HEALTHY, message="OK"),
            HealthCheckResult(name="redis", status=ServiceHealth.HEALTHY, message="OK"),
            HealthCheckResult(name="es", status=ServiceHealth.HEALTHY, message="OK"),
        ]
        report = HealthReport.aggregate(results)

        assert report.status == ServiceHealth.HEALTHY
        assert len(report.checks) == 3

    def test_aggregate_with_degraded(self):
        """Test aggregation with degraded service."""
        results = [
            HealthCheckResult(name="db", status=ServiceHealth.HEALTHY, message="OK"),
            HealthCheckResult(name="redis", status=ServiceHealth.DEGRADED, message="Slow"),
        ]
        report = HealthReport.aggregate(results)

        assert report.status == ServiceHealth.DEGRADED

    def test_aggregate_with_unhealthy(self):
        """Test aggregation with unhealthy service."""
        results = [
            HealthCheckResult(name="db", status=ServiceHealth.HEALTHY, message="OK"),
            HealthCheckResult(name="redis", status=ServiceHealth.UNHEALTHY, message="Down"),
        ]
        report = HealthReport.aggregate(results)

        assert report.status == ServiceHealth.UNHEALTHY

    def test_aggregate_empty(self):
        """Test aggregation with no checks."""
        report = HealthReport.aggregate([])
        assert report.status == ServiceHealth.HEALTHY
        assert len(report.checks) == 0

    def test_report_to_dict(self):
        """Test report serialization."""
        results = [
            HealthCheckResult(name="db", status=ServiceHealth.HEALTHY, message="OK"),
        ]
        report = HealthReport.aggregate(results, version="1.0.0")
        report.uptime_seconds = 3600

        data = report.to_dict()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert data["uptime_seconds"] == 3600
        assert len(data["checks"]) == 1


class TestDatabaseHealthCheck:
    """Tests for DatabaseHealthCheck."""

    @pytest.mark.asyncio
    async def test_no_connection_func(self):
        """Test check without connection function returns healthy mock."""
        check = DatabaseHealthCheck(name="db")
        result = await check.check()

        assert result.status == ServiceHealth.HEALTHY
        assert "mock" in result.message.lower()

    @pytest.mark.asyncio
    async def test_successful_connection(self):
        """Test successful database connection."""
        mock_conn = MagicMock()
        mock_conn.execute = MagicMock(return_value=None)

        check = DatabaseHealthCheck(
            name="db",
            connection_func=lambda: mock_conn,
        )
        result = await check.check()

        assert result.status == ServiceHealth.HEALTHY
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_failure(self):
        """Test failed database connection."""
        def failing_connection():
            raise ConnectionError("Connection refused")

        check = DatabaseHealthCheck(
            name="db",
            connection_func=failing_connection,
        )
        result = await check.check()

        assert result.status == ServiceHealth.UNHEALTHY
        assert "failed" in result.message.lower()

    @pytest.mark.asyncio
    async def test_async_connection(self):
        """Test async database connection."""
        async def async_conn():
            mock = MagicMock()
            mock.execute = MagicMock(return_value=None)
            return mock

        check = DatabaseHealthCheck(
            name="db",
            connection_func=async_conn,
        )
        result = await check.check()

        assert result.status == ServiceHealth.HEALTHY


class TestRedisHealthCheck:
    """Tests for RedisHealthCheck."""

    @pytest.mark.asyncio
    async def test_no_client(self):
        """Test check without client returns healthy mock."""
        check = RedisHealthCheck(name="redis")
        result = await check.check()

        assert result.status == ServiceHealth.HEALTHY
        assert "mock" in result.message.lower()

    @pytest.mark.asyncio
    async def test_successful_ping(self):
        """Test successful Redis ping."""
        mock_client = MagicMock()
        mock_client.ping = MagicMock(return_value=True)

        check = RedisHealthCheck(name="redis", client=mock_client)
        result = await check.check()

        assert result.status == ServiceHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_failed_ping(self):
        """Test failed Redis ping."""
        mock_client = MagicMock()
        mock_client.ping = MagicMock(side_effect=ConnectionError("refused"))

        check = RedisHealthCheck(name="redis", client=mock_client)
        result = await check.check()

        assert result.status == ServiceHealth.UNHEALTHY

    @pytest.mark.asyncio
    async def test_ping_returns_false(self):
        """Test Redis ping returning false."""
        mock_client = MagicMock()
        mock_client.ping = MagicMock(return_value=False)

        check = RedisHealthCheck(name="redis", client=mock_client)
        result = await check.check()

        assert result.status == ServiceHealth.DEGRADED


class TestElasticsearchHealthCheck:
    """Tests for ElasticsearchHealthCheck."""

    @pytest.mark.asyncio
    async def test_no_client(self):
        """Test check without client returns healthy mock."""
        check = ElasticsearchHealthCheck(name="es")
        result = await check.check()

        assert result.status == ServiceHealth.HEALTHY
        assert "mock" in result.message.lower()

    @pytest.mark.asyncio
    async def test_green_cluster(self):
        """Test green cluster status."""
        mock_client = MagicMock()
        mock_client.cluster = MagicMock()
        mock_client.cluster.health = MagicMock(return_value={"status": "green"})

        check = ElasticsearchHealthCheck(name="es", client=mock_client)
        result = await check.check()

        assert result.status == ServiceHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_yellow_cluster(self):
        """Test yellow cluster status."""
        mock_client = MagicMock()
        mock_client.cluster = MagicMock()
        mock_client.cluster.health = MagicMock(return_value={"status": "yellow"})

        check = ElasticsearchHealthCheck(name="es", client=mock_client)
        result = await check.check()

        assert result.status == ServiceHealth.DEGRADED

    @pytest.mark.asyncio
    async def test_red_cluster(self):
        """Test red cluster status."""
        mock_client = MagicMock()
        mock_client.cluster = MagicMock()
        mock_client.cluster.health = MagicMock(return_value={"status": "red"})

        check = ElasticsearchHealthCheck(name="es", client=mock_client)
        result = await check.check()

        assert result.status == ServiceHealth.UNHEALTHY


class TestStorageHealthCheck:
    """Tests for StorageHealthCheck."""

    @pytest.mark.asyncio
    async def test_no_client(self):
        """Test check without client returns healthy mock."""
        check = StorageHealthCheck(name="s3")
        result = await check.check()

        assert result.status == ServiceHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_successful_list_buckets(self):
        """Test successful bucket listing."""
        mock_client = MagicMock()
        mock_client.list_buckets = MagicMock(return_value=[{"Name": "bucket1"}])

        check = StorageHealthCheck(name="s3", client=mock_client)
        result = await check.check()

        assert result.status == ServiceHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_failed_list_buckets(self):
        """Test failed bucket listing."""
        mock_client = MagicMock()
        mock_client.list_buckets = MagicMock(side_effect=Exception("Access denied"))

        check = StorageHealthCheck(name="s3", client=mock_client)
        result = await check.check()

        assert result.status == ServiceHealth.UNHEALTHY


class TestHTTPHealthCheck:
    """Tests for HTTPHealthCheck."""

    @pytest.mark.asyncio
    async def test_successful_http_check(self):
        """Test successful HTTP health check."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.getcode.return_value = 200
            mock_urlopen.return_value = mock_response

            check = HTTPHealthCheck(
                name="api",
                url="http://example.com/health",
            )
            result = await check.check()

            assert result.status == ServiceHealth.HEALTHY
            assert "200" in result.message

    @pytest.mark.asyncio
    async def test_unexpected_status_code(self):
        """Test unexpected HTTP status code."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.getcode.return_value = 503
            mock_urlopen.return_value = mock_response

            check = HTTPHealthCheck(
                name="api",
                url="http://example.com/health",
                expected_status=200,
            )
            result = await check.check()

            assert result.status == ServiceHealth.DEGRADED

    @pytest.mark.asyncio
    async def test_http_error(self):
        """Test HTTP error response."""
        import urllib.error

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                "http://example.com", 500, "Internal Server Error", {}, None
            )

            check = HTTPHealthCheck(
                name="api",
                url="http://example.com/health",
            )
            result = await check.check()

            assert result.status == ServiceHealth.UNHEALTHY


class TestCustomHealthCheck:
    """Tests for CustomHealthCheck."""

    @pytest.mark.asyncio
    async def test_custom_check_function(self):
        """Test custom health check function."""
        def custom_check():
            return HealthCheckResult(
                name="custom",
                status=ServiceHealth.HEALTHY,
                message="Custom check passed",
            )

        check = CustomHealthCheck(name="custom", check_func=custom_check)
        result = await check.check()

        assert result.status == ServiceHealth.HEALTHY
        assert result.message == "Custom check passed"

    @pytest.mark.asyncio
    async def test_async_custom_check(self):
        """Test async custom health check function."""
        async def async_custom_check():
            await asyncio.sleep(0.01)
            return HealthCheckResult(
                name="async_custom",
                status=ServiceHealth.HEALTHY,
                message="Async check passed",
            )

        check = CustomHealthCheck(name="async_custom", check_func=async_custom_check)
        result = await check.check()

        assert result.status == ServiceHealth.HEALTHY


class TestHealthChecker:
    """Tests for HealthChecker."""

    def test_checker_creation(self):
        """Test HealthChecker creation."""
        checker = HealthChecker()
        assert checker is not None

    def test_checker_with_config(self):
        """Test HealthChecker with configuration."""
        config = HealthCheckConfig(timeout=10.0)
        checker = HealthChecker(config=config, version="2.0.0")

        assert checker.version == "2.0.0"

    def test_add_check(self):
        """Test adding health checks."""
        checker = HealthChecker()
        check = DatabaseHealthCheck(name="db")
        checker.add_check(check)

        assert len(checker._checks) == 1

    def test_add_liveness_check(self):
        """Test adding liveness-only check."""
        checker = HealthChecker()
        check = DatabaseHealthCheck(name="db")
        checker.add_liveness_check(check)

        assert check in checker._liveness_checks
        assert check not in checker._readiness_checks

    def test_add_readiness_check(self):
        """Test adding readiness-only check."""
        checker = HealthChecker()
        check = DatabaseHealthCheck(name="db")
        checker.add_readiness_check(check)

        assert check in checker._readiness_checks
        assert check not in checker._liveness_checks

    @pytest.mark.asyncio
    async def test_check_all(self):
        """Test running all checks."""
        checker = HealthChecker(version="1.0.0")
        checker.add_check(DatabaseHealthCheck(name="db"))
        checker.add_check(RedisHealthCheck(name="redis"))

        report = await checker.check_all()

        assert report.version == "1.0.0"
        assert len(report.checks) == 2
        assert report.status == ServiceHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_check_liveness(self):
        """Test running liveness checks."""
        checker = HealthChecker()
        checker.add_liveness_check(DatabaseHealthCheck(name="db"))
        checker.add_readiness_check(RedisHealthCheck(name="redis"))

        report = await checker.check_liveness()

        assert len(report.checks) == 1
        assert report.checks[0].name == "db"

    @pytest.mark.asyncio
    async def test_check_readiness(self):
        """Test running readiness checks."""
        checker = HealthChecker()
        checker.add_liveness_check(DatabaseHealthCheck(name="db"))
        checker.add_readiness_check(RedisHealthCheck(name="redis"))

        report = await checker.check_readiness()

        assert len(report.checks) == 1
        assert report.checks[0].name == "redis"

    @pytest.mark.asyncio
    async def test_uptime_tracking(self):
        """Test uptime tracking."""
        import time

        checker = HealthChecker()
        time.sleep(0.1)

        uptime = checker.uptime_seconds
        assert uptime >= 0.1

    @pytest.mark.asyncio
    async def test_cached_report(self):
        """Test getting cached report."""
        checker = HealthChecker()
        checker.add_check(DatabaseHealthCheck(name="db"))

        await checker.check_all()
        cached = checker.get_cached_report()

        assert cached is not None
        assert len(cached.checks) == 1

    @pytest.mark.asyncio
    async def test_is_healthy(self):
        """Test is_healthy property."""
        checker = HealthChecker()
        checker.add_check(DatabaseHealthCheck(name="db"))

        # Before any check
        assert checker.is_healthy is True

        # After check
        await checker.check_all()
        assert checker.is_healthy is True


class TestHealthMonitor:
    """Tests for HealthMonitor."""

    @pytest.mark.asyncio
    async def test_monitor_start_stop(self):
        """Test starting and stopping monitor."""
        checker = HealthChecker()
        checker.add_check(DatabaseHealthCheck(name="db"))

        monitor = HealthMonitor(checker, interval=0.1)

        await monitor.start()
        assert monitor._running is True

        await asyncio.sleep(0.15)  # Allow one check cycle

        await monitor.stop()
        assert monitor._running is False

    @pytest.mark.asyncio
    async def test_monitor_unhealthy_callback(self):
        """Test unhealthy callback."""
        callback_called = False

        def on_unhealthy(report):
            nonlocal callback_called
            callback_called = True

        # Create checker with failing check
        checker = HealthChecker()

        def failing_check():
            return HealthCheckResult(
                name="failing",
                status=ServiceHealth.UNHEALTHY,
                message="Always fails",
            )

        checker.add_check(CustomHealthCheck(name="failing", check_func=failing_check))

        monitor = HealthMonitor(
            checker,
            interval=0.05,
            on_unhealthy=on_unhealthy,
        )

        await monitor.start()
        await asyncio.sleep(0.1)
        await monitor.stop()

        assert callback_called


class TestHealthEndpoints:
    """Tests for FastAPI health endpoints."""

    @pytest.fixture
    def checker(self):
        """Create a health checker for tests."""
        checker = HealthChecker(version="1.0.0")
        checker.add_check(DatabaseHealthCheck(name="db"))
        return checker

    def test_create_health_endpoints(self, checker):
        """Test creating health endpoints returns router."""
        router = create_health_endpoints(checker)
        assert router is not None

        # Check routes exist
        routes = [route.path for route in router.routes]
        assert "/health" in routes
        assert "/health/live" in routes
        assert "/health/ready" in routes


class TestSafeCheck:
    """Tests for safe_check error handling."""

    @pytest.mark.asyncio
    async def test_safe_check_catches_exception(self):
        """Test safe_check catches exceptions."""
        class FailingCheck(HealthCheck):
            @property
            def name(self) -> str:
                return "failing"

            async def check(self) -> HealthCheckResult:
                raise RuntimeError("Unexpected error")

        check = FailingCheck()
        result = await check.safe_check()

        assert result.status == ServiceHealth.UNHEALTHY
        assert result.error is not None
        assert "Unexpected error" in result.error

    @pytest.mark.asyncio
    async def test_safe_check_records_latency(self):
        """Test safe_check records latency."""
        import time

        class SlowCheck(HealthCheck):
            @property
            def name(self) -> str:
                return "slow"

            async def check(self) -> HealthCheckResult:
                time.sleep(0.01)
                return HealthCheckResult(
                    name=self.name,
                    status=ServiceHealth.HEALTHY,
                    message="OK",
                )

        check = SlowCheck()
        result = await check.safe_check()

        assert result.latency_ms >= 10  # At least 10ms
