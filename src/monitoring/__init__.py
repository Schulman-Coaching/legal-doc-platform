"""
Monitoring and Observability Layer
==================================
Comprehensive monitoring for the Legal Document Processing Platform.

This module provides:
- Prometheus-compatible metrics collection
- OpenTelemetry-compatible distributed tracing
- Structured JSON logging with trace correlation
- Health and readiness probes for Kubernetes
- Alerting with multiple notification channels
- Dashboard data aggregation and visualization

Quick Start:
------------
```python
from monitoring import (
    MetricsCollector,
    Tracer,
    get_logger,
    HealthChecker,
    AlertManager,
)

# Metrics
metrics = MetricsCollector()
counter = metrics.counter("requests_total", "Total requests")
counter.inc()

# Tracing
tracer = Tracer()
with tracer.start_span("operation") as span:
    span.set_attribute("user_id", "123")
    # ... do work ...

# Logging
logger = get_logger("myservice")
logger.info("Operation completed", user_id="123")

# Health checks
checker = HealthChecker()
checker.add_check(DatabaseHealthCheck("db", connection_func))
report = await checker.check_all()

# Alerting
alert_manager = AlertManager()
alert_manager.add_rule(AlertRule(
    name="high_latency",
    metric="latency_seconds",
    operator=">",
    threshold=1.0,
))
```
"""

from __future__ import annotations

# Configuration
from .config import (
    MonitoringConfig,
    MetricsConfig,
    TracingConfig,
    LoggingConfig,
    AlertingConfig,
    HealthCheckConfig,
    EmailNotificationConfig,
    SlackNotificationConfig,
    PagerDutyNotificationConfig,
)

# Models
from .models import (
    # Metrics
    MetricType,
    MetricPoint,
    # Tracing
    Span,
    SpanContext,
    SpanEvent,
    SpanStatus,
    # Logging
    LogLevel,
    LogRecord,
    # Alerting
    Alert,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    # Health
    HealthCheckResult,
    HealthReport,
    ServiceHealth,
    # SLI/SLO
    SLI,
    SLO,
)

# Services
from .services import (
    # Metrics
    MetricsCollector,
    Counter,
    Gauge,
    Histogram,
    Summary,
    # Tracing
    Tracer,
    trace,
    get_current_span,
    # Logging
    StructuredLogger,
    get_logger,
    # Health
    HealthChecker,
    HealthCheck,
    # Alerting
    AlertManager,
    AlertEvaluator,
    # Dashboard
    DashboardService,
    TimeSeriesStore,
    Dashboard,
    DashboardPanel,
    Aggregator,
)

# Health check implementations
from .services.health import (
    DatabaseHealthCheck,
    RedisHealthCheck,
    ElasticsearchHealthCheck,
    StorageHealthCheck,
    HTTPHealthCheck,
    CustomHealthCheck,
    HealthMonitor,
    create_health_endpoints,
)

# Notification channels
from .services.alerting import (
    NotificationChannel,
    EmailNotificationChannel,
    SlackNotificationChannel,
    PagerDutyNotificationChannel,
    WebhookNotificationChannel,
    ConsoleNotificationChannel,
    AlertMonitor,
    alert,
)

# Tracing exporters
from .services.tracing import (
    SpanExporter,
    ConsoleSpanExporter,
    InMemorySpanExporter,
    OTLPSpanExporter,
    TracingMiddleware,
    trace_method,
    get_default_tracer,
    set_default_tracer,
)

# Logging utilities
from .services.logging import (
    JSONFormatter,
    ConsoleFormatter,
    LoggingMiddleware,
    log_call,
    configure_logging,
)

# Dashboard utilities
from .services.dashboard import (
    AggregationType,
    TimeRange,
    TimeSeries,
    TimeSeriesPoint,
    PanelData,
    DashboardBuilder,
    dashboard,
    create_system_dashboard,
    create_application_dashboard,
    create_database_dashboard,
)

__version__ = "1.0.0"

__all__ = [
    # Version
    "__version__",
    # Config
    "MonitoringConfig",
    "MetricsConfig",
    "TracingConfig",
    "LoggingConfig",
    "AlertingConfig",
    "HealthCheckConfig",
    "EmailNotificationConfig",
    "SlackNotificationConfig",
    "PagerDutyNotificationConfig",
    # Models - Metrics
    "MetricType",
    "MetricPoint",
    # Models - Tracing
    "Span",
    "SpanContext",
    "SpanEvent",
    "SpanStatus",
    # Models - Logging
    "LogLevel",
    "LogRecord",
    # Models - Alerting
    "Alert",
    "AlertRule",
    "AlertSeverity",
    "AlertStatus",
    # Models - Health
    "HealthCheckResult",
    "HealthReport",
    "ServiceHealth",
    # Models - SLI/SLO
    "SLI",
    "SLO",
    # Services - Metrics
    "MetricsCollector",
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    # Services - Tracing
    "Tracer",
    "trace",
    "trace_method",
    "get_current_span",
    "get_default_tracer",
    "set_default_tracer",
    "SpanExporter",
    "ConsoleSpanExporter",
    "InMemorySpanExporter",
    "OTLPSpanExporter",
    "TracingMiddleware",
    # Services - Logging
    "StructuredLogger",
    "get_logger",
    "JSONFormatter",
    "ConsoleFormatter",
    "LoggingMiddleware",
    "log_call",
    "configure_logging",
    # Services - Health
    "HealthChecker",
    "HealthCheck",
    "DatabaseHealthCheck",
    "RedisHealthCheck",
    "ElasticsearchHealthCheck",
    "StorageHealthCheck",
    "HTTPHealthCheck",
    "CustomHealthCheck",
    "HealthMonitor",
    "create_health_endpoints",
    # Services - Alerting
    "AlertManager",
    "AlertEvaluator",
    "NotificationChannel",
    "EmailNotificationChannel",
    "SlackNotificationChannel",
    "PagerDutyNotificationChannel",
    "WebhookNotificationChannel",
    "ConsoleNotificationChannel",
    "AlertMonitor",
    "alert",
    # Services - Dashboard
    "DashboardService",
    "TimeSeriesStore",
    "Dashboard",
    "DashboardPanel",
    "Aggregator",
    "AggregationType",
    "TimeRange",
    "TimeSeries",
    "TimeSeriesPoint",
    "PanelData",
    "DashboardBuilder",
    "dashboard",
    "create_system_dashboard",
    "create_application_dashboard",
    "create_database_dashboard",
]
