"""
Legal Document Platform - Monitoring & Observability Layer
==========================================================
Comprehensive monitoring including metrics, logging, tracing,
alerting, and dashboards for platform health and performance.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Metrics Models
# ============================================================================

class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert statuses."""
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"


class ServiceHealth(str, Enum):
    """Service health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Span:
    """Distributed tracing span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    service_name: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: int = 0
    status: str = "ok"
    tags: dict[str, str] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)

    def finish(self, status: str = "ok") -> None:
        """Finish the span."""
        self.end_time = datetime.utcnow()
        self.duration_ms = int((self.end_time - self.start_time).total_seconds() * 1000)
        self.status = status


class Alert(BaseModel):
    """Alert definition."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    severity: AlertSeverity = AlertSeverity.WARNING
    status: AlertStatus = AlertStatus.FIRING
    message: str
    source: str
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    class Config:
        use_enum_values = True


class AlertRule(BaseModel):
    """Alert rule configuration."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    expr: str  # Prometheus-style expression
    for_duration: str = "5m"
    severity: AlertSeverity = AlertSeverity.WARNING
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class SLI(BaseModel):
    """Service Level Indicator."""
    name: str
    description: str
    metric_query: str
    unit: str = "ratio"


class SLO(BaseModel):
    """Service Level Objective."""
    name: str
    description: str
    sli: SLI
    target: float  # e.g., 0.999 for 99.9%
    window: str = "30d"


# ============================================================================
# Metrics Collector
# ============================================================================

class MetricsCollector:
    """
    Collects and exposes metrics in Prometheus format.
    """

    def __init__(self):
        self._counters: dict[str, dict[tuple, float]] = defaultdict(dict)
        self._gauges: dict[str, dict[tuple, float]] = defaultdict(dict)
        self._histograms: dict[str, dict[tuple, list]] = defaultdict(dict)
        self._histogram_buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def _labels_key(self, labels: dict[str, str]) -> tuple:
        """Convert labels dict to hashable tuple."""
        return tuple(sorted(labels.items()))

    def counter_inc(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        """Increment a counter."""
        labels = labels or {}
        key = self._labels_key(labels)
        self._counters[name][key] = self._counters[name].get(key, 0) + value

    def gauge_set(
        self,
        name: str,
        value: float,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        """Set a gauge value."""
        labels = labels or {}
        key = self._labels_key(labels)
        self._gauges[name][key] = value

    def gauge_inc(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        """Increment a gauge."""
        labels = labels or {}
        key = self._labels_key(labels)
        self._gauges[name][key] = self._gauges[name].get(key, 0) + value

    def gauge_dec(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        """Decrement a gauge."""
        labels = labels or {}
        key = self._labels_key(labels)
        self._gauges[name][key] = self._gauges[name].get(key, 0) - value

    def histogram_observe(
        self,
        name: str,
        value: float,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        """Record a histogram observation."""
        labels = labels or {}
        key = self._labels_key(labels)
        if key not in self._histograms[name]:
            self._histograms[name][key] = []
        self._histograms[name][key].append(value)

    def get_metrics(self) -> list[MetricPoint]:
        """Get all metrics as MetricPoint list."""
        metrics = []

        # Counters
        for name, values in self._counters.items():
            for labels_tuple, value in values.items():
                labels = dict(labels_tuple)
                metrics.append(MetricPoint(
                    name=name,
                    value=value,
                    labels=labels,
                    metric_type=MetricType.COUNTER,
                ))

        # Gauges
        for name, values in self._gauges.items():
            for labels_tuple, value in values.items():
                labels = dict(labels_tuple)
                metrics.append(MetricPoint(
                    name=name,
                    value=value,
                    labels=labels,
                    metric_type=MetricType.GAUGE,
                ))

        # Histograms (as multiple metrics)
        for name, values in self._histograms.items():
            for labels_tuple, observations in values.items():
                labels = dict(labels_tuple)
                if observations:
                    # Sum
                    metrics.append(MetricPoint(
                        name=f"{name}_sum",
                        value=sum(observations),
                        labels=labels,
                        metric_type=MetricType.HISTOGRAM,
                    ))
                    # Count
                    metrics.append(MetricPoint(
                        name=f"{name}_count",
                        value=len(observations),
                        labels=labels,
                        metric_type=MetricType.HISTOGRAM,
                    ))
                    # Buckets
                    for bucket in self._histogram_buckets:
                        count = sum(1 for o in observations if o <= bucket)
                        bucket_labels = {**labels, "le": str(bucket)}
                        metrics.append(MetricPoint(
                            name=f"{name}_bucket",
                            value=count,
                            labels=bucket_labels,
                            metric_type=MetricType.HISTOGRAM,
                        ))

        return metrics

    def to_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        metrics = self.get_metrics()

        for metric in metrics:
            labels_str = ""
            if metric.labels:
                labels_parts = [f'{k}="{v}"' for k, v in metric.labels.items()]
                labels_str = "{" + ",".join(labels_parts) + "}"

            lines.append(f"{metric.name}{labels_str} {metric.value}")

        return "\n".join(lines)


# ============================================================================
# Distributed Tracing
# ============================================================================

class Tracer:
    """
    Distributed tracing service.
    In production, integrate with Jaeger/Zipkin.
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._active_spans: dict[str, Span] = {}
        self._completed_spans: list[Span] = []

    def start_span(
        self,
        operation_name: str,
        parent_span: Optional[Span] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> Span:
        """Start a new span."""
        if parent_span:
            trace_id = parent_span.trace_id
            parent_span_id = parent_span.span_id
        else:
            trace_id = str(uuid4()).replace("-", "")[:32]
            parent_span_id = None

        span = Span(
            trace_id=trace_id,
            span_id=str(uuid4()).replace("-", "")[:16],
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            service_name=self.service_name,
            tags=tags or {},
        )

        self._active_spans[span.span_id] = span
        return span

    def finish_span(self, span: Span, status: str = "ok") -> None:
        """Finish a span."""
        span.finish(status)

        if span.span_id in self._active_spans:
            del self._active_spans[span.span_id]

        self._completed_spans.append(span)

        # In production, send to Jaeger
        # self._export_span(span)

    def trace(self, operation_name: str):
        """Decorator for tracing functions."""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                span = self.start_span(operation_name)
                try:
                    result = await func(*args, **kwargs)
                    self.finish_span(span, "ok")
                    return result
                except Exception as e:
                    span.tags["error"] = str(e)
                    self.finish_span(span, "error")
                    raise

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                span = self.start_span(operation_name)
                try:
                    result = func(*args, **kwargs)
                    self.finish_span(span, "ok")
                    return result
                except Exception as e:
                    span.tags["error"] = str(e)
                    self.finish_span(span, "error")
                    raise

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper

        return decorator

    def get_trace(self, trace_id: str) -> list[Span]:
        """Get all spans for a trace."""
        return [s for s in self._completed_spans if s.trace_id == trace_id]


# ============================================================================
# Structured Logging
# ============================================================================

class StructuredLogger:
    """
    Structured logging with JSON output for ELK stack.
    """

    def __init__(
        self,
        service_name: str,
        log_level: str = "INFO",
    ):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(self._json_formatter())
        self.logger.addHandler(handler)

    def _json_formatter(self):
        """Create JSON log formatter."""
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "service": record.name,
                    "logger": record.filename,
                    "line": record.lineno,
                }

                # Add extra fields
                if hasattr(record, "extra"):
                    log_data.update(record.extra)

                return json.dumps(log_data)

        return JsonFormatter()

    def _log(
        self,
        level: str,
        message: str,
        **kwargs,
    ) -> None:
        """Internal log method with structured data."""
        extra = {
            "extra": {
                "trace_id": kwargs.get("trace_id"),
                "span_id": kwargs.get("span_id"),
                "user_id": kwargs.get("user_id"),
                "request_id": kwargs.get("request_id"),
                **{k: v for k, v in kwargs.items()
                   if k not in ["trace_id", "span_id", "user_id", "request_id"]},
            }
        }

        getattr(self.logger, level.lower())(message, extra=extra)

    def debug(self, message: str, **kwargs) -> None:
        self._log("debug", message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        self._log("info", message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self._log("warning", message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self._log("error", message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        self._log("critical", message, **kwargs)


# ============================================================================
# Alert Manager
# ============================================================================

class AlertManager:
    """
    Manages alerts and notifications.
    """

    def __init__(
        self,
        metrics_collector: MetricsCollector,
    ):
        self.metrics = metrics_collector
        self._rules: dict[str, AlertRule] = {}
        self._active_alerts: dict[str, Alert] = {}
        self._alert_history: list[Alert] = []
        self._notification_channels: list[dict] = []

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self._rules[rule.id] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_id: str) -> None:
        """Remove an alert rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]

    def add_notification_channel(
        self,
        channel_type: str,  # pagerduty, slack, email
        config: dict[str, str],
    ) -> None:
        """Add a notification channel."""
        self._notification_channels.append({
            "type": channel_type,
            "config": config,
        })

    async def evaluate_rules(self) -> list[Alert]:
        """Evaluate all alert rules against current metrics."""
        new_alerts = []

        for rule in self._rules.values():
            if not rule.enabled:
                continue

            # Evaluate rule expression
            # In production, use Prometheus client to evaluate expr
            is_firing = await self._evaluate_expression(rule.expr)

            alert_key = f"{rule.name}"

            if is_firing:
                if alert_key not in self._active_alerts:
                    # New alert
                    alert = Alert(
                        name=rule.name,
                        severity=rule.severity,
                        message=rule.annotations.get("summary", rule.name),
                        source="alert_manager",
                        labels=rule.labels,
                        annotations=rule.annotations,
                    )
                    self._active_alerts[alert_key] = alert
                    self._alert_history.append(alert)
                    new_alerts.append(alert)

                    # Send notifications
                    await self._send_notifications(alert)
            else:
                if alert_key in self._active_alerts:
                    # Alert resolved
                    alert = self._active_alerts[alert_key]
                    alert.status = AlertStatus.RESOLVED
                    alert.resolved_at = datetime.utcnow()
                    del self._active_alerts[alert_key]

                    # Send resolution notification
                    await self._send_notifications(alert)

        return new_alerts

    async def _evaluate_expression(self, expr: str) -> bool:
        """
        Evaluate a Prometheus-style expression.
        In production, query Prometheus API.
        """
        # Placeholder - always return False
        return False

    async def _send_notifications(self, alert: Alert) -> None:
        """Send alert notifications to all channels."""
        for channel in self._notification_channels:
            try:
                if channel["type"] == "slack":
                    await self._send_slack_notification(alert, channel["config"])
                elif channel["type"] == "pagerduty":
                    await self._send_pagerduty_notification(alert, channel["config"])
                elif channel["type"] == "email":
                    await self._send_email_notification(alert, channel["config"])
            except Exception as e:
                logger.error(f"Failed to send notification to {channel['type']}: {e}")

    async def _send_slack_notification(
        self,
        alert: Alert,
        config: dict[str, str],
    ) -> None:
        """Send Slack notification."""
        # In production, use Slack webhook
        # import httpx
        # async with httpx.AsyncClient() as client:
        #     await client.post(
        #         config["webhook_url"],
        #         json={
        #             "text": f"[{alert.severity.upper()}] {alert.name}: {alert.message}",
        #             "attachments": [...]
        #         }
        #     )
        logger.info(f"Slack notification sent for alert: {alert.name}")

    async def _send_pagerduty_notification(
        self,
        alert: Alert,
        config: dict[str, str],
    ) -> None:
        """Send PagerDuty notification."""
        # In production, use PagerDuty Events API v2
        logger.info(f"PagerDuty notification sent for alert: {alert.name}")

    async def _send_email_notification(
        self,
        alert: Alert,
        config: dict[str, str],
    ) -> None:
        """Send email notification."""
        # In production, use SMTP or email service
        logger.info(f"Email notification sent for alert: {alert.name}")

    def acknowledge_alert(
        self,
        alert_id: str,
        user_id: str,
    ) -> bool:
        """Acknowledge an alert."""
        for key, alert in self._active_alerts.items():
            if alert.id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = user_id
                alert.acknowledged_at = datetime.utcnow()
                return True
        return False

    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
    ) -> list[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = list(self._active_alerts.values())
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts


# ============================================================================
# Health Checker
# ============================================================================

class HealthChecker:
    """
    Service health checking and readiness probes.
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._dependencies: dict[str, Callable] = {}
        self._health_cache: dict[str, tuple[ServiceHealth, datetime]] = {}
        self._cache_ttl = timedelta(seconds=5)

    def register_dependency(
        self,
        name: str,
        check_func: Callable[[], bool],
    ) -> None:
        """Register a dependency health check."""
        self._dependencies[name] = check_func

    async def check_health(self) -> dict[str, Any]:
        """Check overall service health."""
        results = {
            "service": self.service_name,
            "status": ServiceHealth.HEALTHY.value,
            "timestamp": datetime.utcnow().isoformat(),
            "dependencies": {},
        }

        all_healthy = True
        any_unhealthy = False

        for name, check_func in self._dependencies.items():
            try:
                # Check cache
                if name in self._health_cache:
                    cached_health, cached_at = self._health_cache[name]
                    if datetime.utcnow() - cached_at < self._cache_ttl:
                        results["dependencies"][name] = {
                            "status": cached_health.value,
                            "cached": True,
                        }
                        if cached_health == ServiceHealth.UNHEALTHY:
                            any_unhealthy = True
                        elif cached_health == ServiceHealth.DEGRADED:
                            all_healthy = False
                        continue

                # Run check
                if asyncio.iscoroutinefunction(check_func):
                    is_healthy = await check_func()
                else:
                    is_healthy = check_func()

                health = ServiceHealth.HEALTHY if is_healthy else ServiceHealth.UNHEALTHY
                self._health_cache[name] = (health, datetime.utcnow())

                results["dependencies"][name] = {
                    "status": health.value,
                    "cached": False,
                }

                if not is_healthy:
                    any_unhealthy = True

            except Exception as e:
                results["dependencies"][name] = {
                    "status": ServiceHealth.UNHEALTHY.value,
                    "error": str(e),
                }
                any_unhealthy = True

        # Determine overall status
        if any_unhealthy:
            results["status"] = ServiceHealth.UNHEALTHY.value
        elif not all_healthy:
            results["status"] = ServiceHealth.DEGRADED.value

        return results

    async def liveness_probe(self) -> bool:
        """Kubernetes liveness probe."""
        # Basic check that service is running
        return True

    async def readiness_probe(self) -> bool:
        """Kubernetes readiness probe."""
        health = await self.check_health()
        return health["status"] != ServiceHealth.UNHEALTHY.value


# ============================================================================
# SLO Monitor
# ============================================================================

class SLOMonitor:
    """
    Monitors Service Level Objectives.
    """

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._slos: dict[str, SLO] = {}
        self._error_budgets: dict[str, float] = {}

    def register_slo(self, slo: SLO) -> None:
        """Register an SLO."""
        self._slos[slo.name] = slo

        # Calculate initial error budget
        # Error budget = 1 - target (e.g., 0.001 for 99.9%)
        self._error_budgets[slo.name] = 1 - slo.target

        logger.info(f"Registered SLO: {slo.name} (target: {slo.target * 100}%)")

    async def calculate_sli(self, slo: SLO) -> float:
        """Calculate current SLI value."""
        # In production, query Prometheus
        # This is a placeholder
        return 0.999

    async def get_slo_status(self, slo_name: str) -> dict[str, Any]:
        """Get current status of an SLO."""
        if slo_name not in self._slos:
            return {"error": "SLO not found"}

        slo = self._slos[slo_name]
        current_sli = await self.calculate_sli(slo)

        error_budget = 1 - slo.target
        consumed_budget = max(0, 1 - current_sli)
        remaining_budget = max(0, error_budget - consumed_budget)
        budget_percentage = (remaining_budget / error_budget) * 100 if error_budget > 0 else 0

        return {
            "name": slo.name,
            "target": slo.target,
            "current_sli": current_sli,
            "meeting_slo": current_sli >= slo.target,
            "error_budget": {
                "total": error_budget,
                "consumed": consumed_budget,
                "remaining": remaining_budget,
                "percentage_remaining": budget_percentage,
            },
            "window": slo.window,
        }

    async def get_all_slo_status(self) -> list[dict[str, Any]]:
        """Get status of all SLOs."""
        statuses = []
        for slo_name in self._slos:
            status = await self.get_slo_status(slo_name)
            statuses.append(status)
        return statuses


# ============================================================================
# Dashboard Data Provider
# ============================================================================

class DashboardProvider:
    """
    Provides data for Grafana dashboards.
    """

    def __init__(
        self,
        metrics: MetricsCollector,
        health_checker: HealthChecker,
        alert_manager: AlertManager,
        slo_monitor: SLOMonitor,
    ):
        self.metrics = metrics
        self.health = health_checker
        self.alerts = alert_manager
        self.slo = slo_monitor

    async def get_overview_dashboard(self) -> dict[str, Any]:
        """Get data for overview dashboard."""
        health = await self.health.check_health()
        active_alerts = self.alerts.get_active_alerts()
        slo_status = await self.slo.get_all_slo_status()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "health": {
                "status": health["status"],
                "dependencies": health["dependencies"],
            },
            "alerts": {
                "total": len(active_alerts),
                "critical": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                "warning": len([a for a in active_alerts if a.severity == AlertSeverity.WARNING]),
                "active": [
                    {
                        "name": a.name,
                        "severity": a.severity.value,
                        "message": a.message,
                        "started_at": a.started_at.isoformat(),
                    }
                    for a in active_alerts[:5]  # Top 5
                ],
            },
            "slos": {
                "total": len(slo_status),
                "meeting": len([s for s in slo_status if s.get("meeting_slo", False)]),
                "breaching": len([s for s in slo_status if not s.get("meeting_slo", True)]),
                "details": slo_status,
            },
        }

    async def get_api_metrics(
        self,
        time_range: str = "1h",
    ) -> dict[str, Any]:
        """Get API-specific metrics."""
        return {
            "request_rate": {
                "value": 100,  # Placeholder
                "unit": "req/s",
            },
            "error_rate": {
                "value": 0.1,
                "unit": "%",
            },
            "latency": {
                "p50": 50,
                "p95": 150,
                "p99": 300,
                "unit": "ms",
            },
            "throughput": {
                "value": 1000,
                "unit": "req/min",
            },
        }

    async def get_processing_metrics(
        self,
        time_range: str = "1h",
    ) -> dict[str, Any]:
        """Get document processing metrics."""
        return {
            "documents_processed": {
                "value": 500,
                "rate": 50,
                "unit": "docs/hour",
            },
            "processing_time": {
                "avg": 5000,
                "p95": 15000,
                "unit": "ms",
            },
            "queue_depth": {
                "value": 10,
            },
            "success_rate": {
                "value": 99.5,
                "unit": "%",
            },
        }


# ============================================================================
# Monitoring Service Coordinator
# ============================================================================

class MonitoringService:
    """
    Main monitoring service coordinating all components.
    """

    def __init__(self, service_name: str):
        self.service_name = service_name

        # Initialize components
        self.metrics = MetricsCollector()
        self.tracer = Tracer(service_name)
        self.logger = StructuredLogger(service_name)
        self.health = HealthChecker(service_name)
        self.alerts = AlertManager(self.metrics)
        self.slo = SLOMonitor(self.metrics)
        self.dashboard = DashboardProvider(
            self.metrics,
            self.health,
            self.alerts,
            self.slo,
        )

        # Register default SLOs
        self._register_default_slos()

        # Register default alert rules
        self._register_default_alerts()

    def _register_default_slos(self) -> None:
        """Register default SLOs for legal document platform."""
        # API availability
        self.slo.register_slo(SLO(
            name="api_availability",
            description="API endpoint availability",
            sli=SLI(
                name="api_success_rate",
                description="Percentage of successful API requests",
                metric_query='sum(rate(http_requests_total{status!~"5.."}[5m])) / sum(rate(http_requests_total[5m]))',
            ),
            target=0.999,
            window="30d",
        ))

        # API latency
        self.slo.register_slo(SLO(
            name="api_latency_p99",
            description="API response time at p99",
            sli=SLI(
                name="api_latency",
                description="99th percentile latency under 500ms",
                metric_query='histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le)) < 0.5',
            ),
            target=0.99,
            window="30d",
        ))

        # Document processing
        self.slo.register_slo(SLO(
            name="document_processing_success",
            description="Document processing success rate",
            sli=SLI(
                name="processing_success_rate",
                description="Percentage of successfully processed documents",
                metric_query='sum(rate(documents_processed_total{status="success"}[1h])) / sum(rate(documents_processed_total[1h]))',
            ),
            target=0.995,
            window="30d",
        ))

        # Search latency
        self.slo.register_slo(SLO(
            name="search_latency_p95",
            description="Search response time at p95",
            sli=SLI(
                name="search_latency",
                description="95th percentile search latency under 200ms",
                metric_query='histogram_quantile(0.95, sum(rate(search_duration_seconds_bucket[5m])) by (le)) < 0.2',
            ),
            target=0.99,
            window="30d",
        ))

    def _register_default_alerts(self) -> None:
        """Register default alert rules."""
        # High error rate
        self.alerts.add_rule(AlertRule(
            name="HighErrorRate",
            expr='sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.01',
            for_duration="5m",
            severity=AlertSeverity.CRITICAL,
            annotations={
                "summary": "High error rate detected",
                "description": "Error rate is above 1% for the last 5 minutes",
            },
        ))

        # High latency
        self.alerts.add_rule(AlertRule(
            name="HighLatency",
            expr='histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le)) > 1',
            for_duration="5m",
            severity=AlertSeverity.WARNING,
            annotations={
                "summary": "High API latency",
                "description": "P99 latency is above 1 second",
            },
        ))

        # Processing queue backup
        self.alerts.add_rule(AlertRule(
            name="ProcessingQueueBackup",
            expr='processing_queue_depth > 100',
            for_duration="10m",
            severity=AlertSeverity.WARNING,
            annotations={
                "summary": "Processing queue backup",
                "description": "Document processing queue has more than 100 items",
            },
        ))

        # Storage low
        self.alerts.add_rule(AlertRule(
            name="StorageLow",
            expr='(1 - (storage_free_bytes / storage_total_bytes)) > 0.9',
            for_duration="30m",
            severity=AlertSeverity.WARNING,
            annotations={
                "summary": "Storage space low",
                "description": "Storage is more than 90% full",
            },
        ))

    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_ms: float,
    ) -> None:
        """Record an HTTP request."""
        labels = {
            "method": method,
            "endpoint": endpoint,
            "status_code": str(status_code),
        }

        self.metrics.counter_inc("http_requests_total", labels=labels)
        self.metrics.histogram_observe("http_request_duration_seconds", duration_ms / 1000, labels=labels)

    def record_document_processing(
        self,
        document_type: str,
        status: str,
        duration_ms: float,
    ) -> None:
        """Record document processing."""
        labels = {
            "document_type": document_type,
            "status": status,
        }

        self.metrics.counter_inc("documents_processed_total", labels=labels)
        self.metrics.histogram_observe("document_processing_duration_seconds", duration_ms / 1000, labels=labels)

    def set_queue_depth(self, queue_name: str, depth: int) -> None:
        """Set processing queue depth."""
        self.metrics.gauge_set("processing_queue_depth", depth, {"queue": queue_name})

    async def start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        asyncio.create_task(self._alert_evaluation_loop())

    async def _alert_evaluation_loop(self) -> None:
        """Periodically evaluate alert rules."""
        while True:
            try:
                await self.alerts.evaluate_rules()
            except Exception as e:
                self.logger.error(f"Alert evaluation failed: {e}")

            await asyncio.sleep(60)  # Every minute


# ============================================================================
# FastAPI Application
# ============================================================================

from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse

app = FastAPI(
    title="Legal Document Monitoring Service",
    description="Monitoring, metrics, and observability for legal document platform",
    version="1.0.0",
)

monitoring: Optional[MonitoringService] = None


@app.on_event("startup")
async def startup():
    global monitoring
    monitoring = MonitoringService("legal-doc-platform")
    await monitoring.start_background_tasks()

    # Register dependency health checks
    monitoring.health.register_dependency("database", lambda: True)
    monitoring.health.register_dependency("elasticsearch", lambda: True)
    monitoring.health.register_dependency("redis", lambda: True)
    monitoring.health.register_dependency("kafka", lambda: True)

    logger.info("Monitoring service initialized")


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    if not monitoring:
        return "# Service not initialized"

    return monitoring.metrics.to_prometheus_format()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not monitoring:
        return {"status": "unhealthy", "error": "Service not initialized"}

    return await monitoring.health.check_health()


@app.get("/health/live")
async def liveness_probe():
    """Kubernetes liveness probe."""
    if not monitoring:
        return Response(status_code=503)

    is_alive = await monitoring.health.liveness_probe()
    return Response(status_code=200 if is_alive else 503)


@app.get("/health/ready")
async def readiness_probe():
    """Kubernetes readiness probe."""
    if not monitoring:
        return Response(status_code=503)

    is_ready = await monitoring.health.readiness_probe()
    return Response(status_code=200 if is_ready else 503)


@app.get("/api/v1/alerts")
async def get_alerts(severity: Optional[str] = None):
    """Get active alerts."""
    if not monitoring:
        return {"error": "Service not initialized"}

    severity_enum = AlertSeverity(severity) if severity else None
    alerts = monitoring.alerts.get_active_alerts(severity_enum)

    return {
        "count": len(alerts),
        "alerts": [
            {
                "id": a.id,
                "name": a.name,
                "severity": a.severity.value,
                "status": a.status.value,
                "message": a.message,
                "started_at": a.started_at.isoformat(),
            }
            for a in alerts
        ],
    }


@app.post("/api/v1/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, user_id: str):
    """Acknowledge an alert."""
    if not monitoring:
        return {"error": "Service not initialized"}

    success = monitoring.alerts.acknowledge_alert(alert_id, user_id)
    return {"success": success}


@app.get("/api/v1/slos")
async def get_slos():
    """Get SLO status."""
    if not monitoring:
        return {"error": "Service not initialized"}

    return await monitoring.slo.get_all_slo_status()


@app.get("/api/v1/dashboard/overview")
async def dashboard_overview():
    """Get overview dashboard data."""
    if not monitoring:
        return {"error": "Service not initialized"}

    return await monitoring.dashboard.get_overview_dashboard()


@app.get("/api/v1/dashboard/api")
async def dashboard_api_metrics(time_range: str = "1h"):
    """Get API metrics for dashboard."""
    if not monitoring:
        return {"error": "Service not initialized"}

    return await monitoring.dashboard.get_api_metrics(time_range)


@app.get("/api/v1/dashboard/processing")
async def dashboard_processing_metrics(time_range: str = "1h"):
    """Get processing metrics for dashboard."""
    if not monitoring:
        return {"error": "Service not initialized"}

    return await monitoring.dashboard.get_processing_metrics(time_range)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
