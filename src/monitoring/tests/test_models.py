"""
Tests for monitoring models.
"""

import pytest
from datetime import datetime, timedelta
from ..models import (
    MetricType,
    MetricPoint,
    Span,
    SpanContext,
    SpanEvent,
    SpanStatus,
    LogLevel,
    LogRecord,
    Alert,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    HealthCheckResult,
    HealthReport,
    ServiceHealth,
    SLI,
    SLO,
)


class TestMetricModels:
    """Tests for metric models."""

    def test_metric_point_creation(self):
        """Test MetricPoint creation."""
        point = MetricPoint(
            name="requests_total",
            value=100.0,
            metric_type=MetricType.COUNTER,
            labels={"method": "GET"},
        )
        assert point.name == "requests_total"
        assert point.value == 100.0
        assert point.metric_type == MetricType.COUNTER
        assert point.labels == {"method": "GET"}
        assert point.timestamp is not None

    def test_metric_point_to_dict(self):
        """Test MetricPoint serialization."""
        point = MetricPoint(
            name="memory_bytes",
            value=1024.0,
            metric_type=MetricType.GAUGE,
        )
        data = point.to_dict()
        assert data["name"] == "memory_bytes"
        assert data["value"] == 1024.0
        assert data["type"] == "gauge"

    def test_metric_types(self):
        """Test all metric types."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"


class TestSpanModels:
    """Tests for span models."""

    def test_span_context_generation(self):
        """Test SpanContext generation."""
        ctx = SpanContext.generate()
        assert len(ctx.trace_id) == 32
        assert len(ctx.span_id) == 16
        assert ctx.trace_flags == 1

    def test_span_context_from_traceparent(self):
        """Test SpanContext from W3C traceparent header."""
        traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        ctx = SpanContext.from_traceparent(traceparent)
        assert ctx is not None
        assert ctx.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert ctx.span_id == "b7ad6b7169203331"
        assert ctx.trace_flags == 1

    def test_span_context_to_traceparent(self):
        """Test SpanContext to W3C traceparent header."""
        ctx = SpanContext(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            trace_flags=1,
        )
        traceparent = ctx.to_traceparent()
        assert traceparent == "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

    def test_span_creation(self):
        """Test Span creation."""
        ctx = SpanContext.generate()
        span = Span(
            name="test_operation",
            context=ctx,
            kind="internal",
        )
        assert span.name == "test_operation"
        assert span.context == ctx
        assert span.kind == "internal"
        assert span.status == SpanStatus.UNSET

    def test_span_attributes(self):
        """Test Span attribute setting."""
        ctx = SpanContext.generate()
        span = Span(name="test", context=ctx)
        span.set_attribute("key", "value")
        span.set_attribute("number", 42)

        assert span.attributes["key"] == "value"
        assert span.attributes["number"] == 42

    def test_span_events(self):
        """Test Span event adding."""
        ctx = SpanContext.generate()
        span = Span(name="test", context=ctx)
        span.add_event("event1", {"data": "value"})

        assert len(span.events) == 1
        assert span.events[0].name == "event1"
        assert span.events[0].attributes == {"data": "value"}

    def test_span_status(self):
        """Test Span status setting."""
        ctx = SpanContext.generate()
        span = Span(name="test", context=ctx)
        span.set_status(SpanStatus.OK)
        assert span.status == SpanStatus.OK

        span.set_status(SpanStatus.ERROR, "Something went wrong")
        assert span.status == SpanStatus.ERROR
        assert span.status_message == "Something went wrong"

    def test_span_duration(self):
        """Test Span duration calculation."""
        ctx = SpanContext.generate()
        span = Span(name="test", context=ctx)
        span.end()

        assert span.end_time is not None
        assert span.duration_ms >= 0

    def test_span_to_dict(self):
        """Test Span serialization."""
        ctx = SpanContext.generate()
        span = Span(
            name="test",
            context=ctx,
            service_name="test_service",
        )
        span.set_attribute("key", "value")
        span.end()

        data = span.to_dict()
        assert data["name"] == "test"
        assert data["traceId"] == ctx.trace_id
        assert data["spanId"] == ctx.span_id
        assert "key" in str(data["attributes"])


class TestLogModels:
    """Tests for log models."""

    def test_log_levels(self):
        """Test log levels."""
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.CRITICAL.value == "critical"

    def test_log_record_creation(self):
        """Test LogRecord creation."""
        record = LogRecord.create(
            message="Test message",
            level=LogLevel.INFO,
            logger_name="test",
        )
        assert record.message == "Test message"
        assert record.level == LogLevel.INFO
        assert record.logger_name == "test"
        assert record.timestamp is not None

    def test_log_record_with_context(self):
        """Test LogRecord with trace context."""
        record = LogRecord.create(
            message="Test",
            level=LogLevel.INFO,
            logger_name="test",
            trace_id="abc123",
            span_id="def456",
            attributes={"user_id": "user1"},
        )
        assert record.trace_id == "abc123"
        assert record.span_id == "def456"
        assert record.attributes == {"user_id": "user1"}

    def test_log_record_to_dict(self):
        """Test LogRecord serialization."""
        record = LogRecord.create(
            message="Test",
            level=LogLevel.ERROR,
            logger_name="test",
        )
        data = record.to_dict()
        assert data["message"] == "Test"
        assert data["level"] == "error"
        assert data["logger"] == "test"


class TestAlertModels:
    """Tests for alert models."""

    def test_alert_severities(self):
        """Test alert severities."""
        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.INFO.value == "info"

    def test_alert_states(self):
        """Test alert states."""
        assert AlertStatus.PENDING.value == "pending"
        assert AlertStatus.FIRING.value == "firing"
        assert AlertStatus.RESOLVED.value == "resolved"
        assert AlertStatus.ACKNOWLEDGED.value == "acknowledged"

    def test_alert_rule_creation(self):
        """Test AlertRule creation."""
        rule = AlertRule(
            name="high_latency",
            expr="latency_seconds",
            condition=">",
            threshold=1.0,
            severity=AlertSeverity.WARNING,
            description="Latency too high",
        )
        assert rule.name == "high_latency"
        assert rule.expr == "latency_seconds"
        assert rule.condition == ">"
        assert rule.threshold == 1.0
        assert rule.severity == AlertSeverity.WARNING

    def test_alert_creation(self):
        """Test Alert creation."""
        alert = Alert(
            name="test_alert",
            message="Test alert description",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.FIRING,
            value=95.0,
            labels={"host": "server1"},
        )
        assert alert.name == "test_alert"
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.status == AlertStatus.FIRING
        assert alert.value == 95.0

    def test_alert_methods(self):
        """Test Alert methods."""
        alert = Alert(
            name="test",
            message="Test alert",
        )

        # Test fire
        alert.fire()
        assert alert.status == AlertStatus.FIRING
        assert alert.fired_at is not None

        # Test resolve
        alert.resolve()
        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolved_at is not None

        # Test acknowledge
        alert2 = Alert(name="test2", message="Test")
        alert2.acknowledge("admin")
        assert alert2.status == AlertStatus.ACKNOWLEDGED
        assert alert2.acknowledged_by == "admin"


class TestHealthModels:
    """Tests for health check models."""

    def test_service_health_states(self):
        """Test service health states."""
        assert ServiceHealth.HEALTHY.value == "healthy"
        assert ServiceHealth.DEGRADED.value == "degraded"
        assert ServiceHealth.UNHEALTHY.value == "unhealthy"

    def test_health_check_result_creation(self):
        """Test HealthCheckResult creation."""
        result = HealthCheckResult(
            name="database",
            status=ServiceHealth.HEALTHY,
            message="Connection OK",
            latency_ms=15.0,
        )
        assert result.name == "database"
        assert result.status == ServiceHealth.HEALTHY
        assert result.latency_ms == 15.0

    def test_health_check_result_unhealthy(self):
        """Test unhealthy HealthCheckResult."""
        result = HealthCheckResult(
            name="redis",
            status=ServiceHealth.UNHEALTHY,
            message="Connection refused",
            error="ConnectionError",
        )
        assert result.status == ServiceHealth.UNHEALTHY
        assert result.error == "ConnectionError"

    def test_health_report_aggregation(self):
        """Test HealthReport aggregation."""
        results = [
            HealthCheckResult(name="db", status=ServiceHealth.HEALTHY, message="OK"),
            HealthCheckResult(name="redis", status=ServiceHealth.HEALTHY, message="OK"),
        ]
        report = HealthReport.aggregate(results)
        assert report.status == ServiceHealth.HEALTHY
        assert len(report.checks) == 2

    def test_health_report_degraded_aggregation(self):
        """Test HealthReport aggregation with degraded service."""
        results = [
            HealthCheckResult(name="db", status=ServiceHealth.HEALTHY, message="OK"),
            HealthCheckResult(name="redis", status=ServiceHealth.DEGRADED, message="Slow"),
        ]
        report = HealthReport.aggregate(results)
        assert report.status == ServiceHealth.DEGRADED

    def test_health_report_unhealthy_aggregation(self):
        """Test HealthReport aggregation with unhealthy service."""
        results = [
            HealthCheckResult(name="db", status=ServiceHealth.HEALTHY, message="OK"),
            HealthCheckResult(name="redis", status=ServiceHealth.UNHEALTHY, message="Down"),
        ]
        report = HealthReport.aggregate(results)
        assert report.status == ServiceHealth.UNHEALTHY

    def test_health_report_to_dict(self):
        """Test HealthReport serialization."""
        results = [
            HealthCheckResult(name="db", status=ServiceHealth.HEALTHY, message="OK"),
        ]
        report = HealthReport.aggregate(results, version="1.0.0")
        data = report.to_dict()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert len(data["checks"]) == 1


class TestSLIModels:
    """Tests for SLI/SLO models."""

    def test_sli_creation(self):
        """Test SLI creation."""
        sli = SLI(
            name="availability",
            description="Service availability",
            metric_name="service_up",
            good_query="sum(service_up == 1)",
            total_query="count(service_up)",
        )
        assert sli.name == "availability"
        assert sli.metric_name == "service_up"

    def test_slo_creation(self):
        """Test SLO creation."""
        sli = SLI(
            name="latency_p99",
            description="P99 latency",
            metric_name="latency",
            good_query="histogram_quantile(0.99, latency) < 0.5",
            total_query="count(latency)",
        )
        slo = SLO(
            name="latency_slo",
            description="99th percentile latency under 500ms",
            sli=sli,
            target=0.99,
        )
        assert slo.name == "latency_slo"
        assert slo.target == 0.99

    def test_slo_target(self):
        """Test SLO target."""
        sli = SLI(
            name="availability",
            description="Availability",
            metric_name="up",
            good_query="up == 1",
            total_query="count(up)",
        )
        slo = SLO(
            name="availability_slo",
            sli=sli,
            target=0.999,  # 99.9%
        )

        assert slo.target == 0.999
        assert slo.alert_burn_rate == 1.0
