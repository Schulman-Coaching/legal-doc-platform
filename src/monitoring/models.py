"""
Monitoring Models
=================
Data models for metrics, traces, logs, alerts, and health checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

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
    PENDING = "pending"
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


class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SpanStatus(str, Enum):
    """Span status codes."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class NotificationChannel(str, Enum):
    """Notification channels for alerts."""
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    SMS = "sms"
    OPSGENIE = "opsgenie"


# =============================================================================
# Metric Models
# =============================================================================

@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    description: str = ""
    unit: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "type": self.metric_type.value,
            "description": self.description,
            "unit": self.unit,
        }


@dataclass
class MetricMetadata:
    """Metric metadata for registration."""
    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    labels: list[str] = field(default_factory=list)


@dataclass
class HistogramBuckets:
    """Histogram bucket configuration."""
    boundaries: list[float] = field(default_factory=lambda: [
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ])

    @classmethod
    def linear(cls, start: float, width: float, count: int) -> "HistogramBuckets":
        """Create linear buckets."""
        return cls(boundaries=[start + i * width for i in range(count)])

    @classmethod
    def exponential(cls, start: float, factor: float, count: int) -> "HistogramBuckets":
        """Create exponential buckets."""
        return cls(boundaries=[start * (factor ** i) for i in range(count)])


# =============================================================================
# Tracing Models
# =============================================================================

@dataclass
class SpanContext:
    """Span context for propagation."""
    trace_id: str
    span_id: str
    trace_flags: int = 1
    trace_state: dict[str, str] = field(default_factory=dict)
    is_remote: bool = False

    @classmethod
    def generate(cls) -> "SpanContext":
        """Generate new span context."""
        import secrets
        return cls(
            trace_id=secrets.token_hex(16),
            span_id=secrets.token_hex(8),
        )

    def to_traceparent(self) -> str:
        """Convert to W3C traceparent header."""
        flags = f"{self.trace_flags:02x}"
        return f"00-{self.trace_id}-{self.span_id}-{flags}"

    @classmethod
    def from_traceparent(cls, traceparent: str) -> Optional["SpanContext"]:
        """Parse W3C traceparent header."""
        try:
            parts = traceparent.split("-")
            if len(parts) != 4:
                return None
            version, trace_id, span_id, flags = parts
            return cls(
                trace_id=trace_id,
                span_id=span_id,
                trace_flags=int(flags, 16),
                is_remote=True,
            )
        except Exception:
            return None


@dataclass
class SpanEvent:
    """Event within a span."""
    name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanLink:
    """Link to another span."""
    context: SpanContext
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """Distributed tracing span."""
    name: str
    context: SpanContext
    parent_context: Optional[SpanContext] = None
    kind: str = "internal"  # internal, server, client, producer, consumer
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: SpanStatus = SpanStatus.UNSET
    status_message: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    links: list[SpanLink] = field(default_factory=list)
    service_name: str = ""
    resource_attributes: dict[str, str] = field(default_factory=dict)

    @property
    def trace_id(self) -> str:
        """Get trace ID."""
        return self.context.trace_id

    @property
    def span_id(self) -> str:
        """Get span ID."""
        return self.context.span_id

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if not self.end_time:
            return 0
        return (self.end_time - self.start_time).total_seconds() * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[dict] = None) -> None:
        """Add event to span."""
        self.events.append(SpanEvent(
            name=name,
            attributes=attributes or {},
        ))

    def set_status(self, status: SpanStatus, message: str = "") -> None:
        """Set span status."""
        self.status = status
        self.status_message = message

    def end(self, status: Optional[SpanStatus] = None) -> None:
        """End the span."""
        self.end_time = datetime.utcnow()
        if status:
            self.status = status

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (OTLP-compatible)."""
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_context.span_id if self.parent_context else None,
            "name": self.name,
            "kind": self.kind,
            "startTimeUnixNano": int(self.start_time.timestamp() * 1e9),
            "endTimeUnixNano": int(self.end_time.timestamp() * 1e9) if self.end_time else None,
            "attributes": self.attributes,
            "events": [{"name": e.name, "attributes": e.attributes} for e in self.events],
            "status": {
                "code": self.status.value,
                "message": self.status_message,
            },
        }


# =============================================================================
# Logging Models
# =============================================================================

@dataclass
class LogRecord:
    """Structured log record."""
    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str = ""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    resource_attributes: dict[str, str] = field(default_factory=dict)
    exception: Optional[dict[str, Any]] = None

    @classmethod
    def create(
        cls,
        level: LogLevel,
        message: str,
        **kwargs,
    ) -> "LogRecord":
        """Create log record."""
        return cls(
            timestamp=datetime.utcnow(),
            level=level,
            message=message,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "logger": self.logger_name,
            "attributes": self.attributes,
        }
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.span_id:
            result["span_id"] = self.span_id
        if self.exception:
            result["exception"] = self.exception
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict())


# =============================================================================
# Alert Models
# =============================================================================

class Alert(BaseModel):
    """Alert instance."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    severity: AlertSeverity = AlertSeverity.WARNING
    status: AlertStatus = AlertStatus.PENDING
    message: str
    description: str = ""
    source: str = ""
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    value: Optional[float] = None
    threshold: Optional[float] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    fired_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    silenced_until: Optional[datetime] = None
    notification_sent: bool = False
    fingerprint: str = ""

    def fire(self) -> None:
        """Fire the alert."""
        self.status = AlertStatus.FIRING
        self.fired_at = datetime.utcnow()

    def resolve(self) -> None:
        """Resolve the alert."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.utcnow()

    def acknowledge(self, user: str) -> None:
        """Acknowledge the alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_by = user
        self.acknowledged_at = datetime.utcnow()

    def silence(self, duration: timedelta) -> None:
        """Silence the alert."""
        self.status = AlertStatus.SILENCED
        self.silenced_until = datetime.utcnow() + duration


class AlertRule(BaseModel):
    """Alert rule configuration."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    expr: str  # Prometheus-style expression or metric name
    condition: str = ">"  # >, <, >=, <=, ==, !=
    threshold: float = 0
    for_duration: timedelta = timedelta(minutes=5)
    severity: AlertSeverity = AlertSeverity.WARNING
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    notification_channels: list[NotificationChannel] = Field(default_factory=list)


class NotificationConfig(BaseModel):
    """Notification channel configuration."""
    channel: NotificationChannel
    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)
    # Email config: smtp_host, smtp_port, from_address, to_addresses
    # Slack config: webhook_url, channel
    # PagerDuty config: integration_key
    # Webhook config: url, headers


# =============================================================================
# Health Check Models
# =============================================================================

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: ServiceHealth
    message: str = ""
    latency_ms: float = 0
    details: dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None

    @property
    def is_healthy(self) -> bool:
        """Check if healthy."""
        return self.status == ServiceHealth.HEALTHY


@dataclass
class HealthReport:
    """Aggregated health report."""
    status: ServiceHealth
    version: str = ""
    uptime_seconds: float = 0
    checks: list[HealthCheckResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def aggregate(cls, checks: list[HealthCheckResult], version: str = "") -> "HealthReport":
        """Create report from check results."""
        if not checks:
            return cls(status=ServiceHealth.UNKNOWN, version=version)

        # Determine overall status
        statuses = [c.status for c in checks]
        if all(s == ServiceHealth.HEALTHY for s in statuses):
            status = ServiceHealth.HEALTHY
        elif any(s == ServiceHealth.UNHEALTHY for s in statuses):
            status = ServiceHealth.UNHEALTHY
        elif any(s == ServiceHealth.DEGRADED for s in statuses):
            status = ServiceHealth.DEGRADED
        else:
            status = ServiceHealth.UNKNOWN

        return cls(
            status=status,
            version=version,
            checks=checks,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "timestamp": self.timestamp.isoformat(),
            "checks": {
                c.name: {
                    "status": c.status.value,
                    "message": c.message,
                    "latency_ms": c.latency_ms,
                    "details": c.details,
                }
                for c in self.checks
            },
        }


# =============================================================================
# SLI/SLO Models
# =============================================================================

class SLI(BaseModel):
    """Service Level Indicator."""
    name: str
    description: str = ""
    metric_name: str
    good_query: str  # Query for good events
    total_query: str  # Query for total events
    unit: str = "ratio"


class SLO(BaseModel):
    """Service Level Objective."""
    name: str
    description: str = ""
    sli: SLI
    target: float  # e.g., 0.999 for 99.9%
    window: timedelta = timedelta(days=30)
    alert_burn_rate: float = 1.0  # Alert if burning error budget faster


@dataclass
class SLOStatus:
    """Current SLO status."""
    slo: SLO
    current_ratio: float
    target: float
    error_budget_remaining: float  # 0-1
    is_meeting_target: bool
    window_start: datetime
    window_end: datetime

    @property
    def error_budget_percentage(self) -> float:
        """Error budget as percentage."""
        return self.error_budget_remaining * 100


# =============================================================================
# Dashboard Models
# =============================================================================

@dataclass
class DashboardPanel:
    """Dashboard panel configuration."""
    id: str
    title: str
    panel_type: str  # graph, stat, table, gauge, heatmap
    metrics: list[str]
    query: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    position: dict[str, int] = field(default_factory=dict)  # x, y, w, h


@dataclass
class Dashboard:
    """Dashboard configuration."""
    id: str
    title: str
    description: str = ""
    panels: list[DashboardPanel] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    refresh_interval: int = 30  # seconds
    time_range: str = "1h"
    tags: list[str] = field(default_factory=list)
