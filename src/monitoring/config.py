"""
Monitoring Configuration
========================
Configuration classes for monitoring services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Optional

from .models import NotificationChannel


@dataclass
class MetricsConfig:
    """Metrics collection configuration."""
    enabled: bool = True
    prefix: str = "legal_doc_platform"
    default_labels: dict[str, str] = field(default_factory=lambda: {
        "service": "legal-doc-platform",
    })
    # Histogram buckets for different metric types
    http_latency_buckets: list[float] = field(default_factory=lambda: [
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ])
    db_latency_buckets: list[float] = field(default_factory=lambda: [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0
    ])
    # Prometheus push gateway (optional)
    push_gateway_url: Optional[str] = None
    push_interval_seconds: int = 15
    # Statsd (optional)
    statsd_host: Optional[str] = None
    statsd_port: int = 8125


@dataclass
class TracingConfig:
    """Distributed tracing configuration."""
    enabled: bool = True
    service_name: str = "legal-doc-platform"
    service_version: str = "1.0.0"
    environment: str = "development"
    # Sampling
    sample_rate: float = 1.0  # 1.0 = 100%
    # OTLP exporter
    otlp_endpoint: Optional[str] = None
    otlp_headers: dict[str, str] = field(default_factory=dict)
    # Jaeger exporter
    jaeger_agent_host: Optional[str] = None
    jaeger_agent_port: int = 6831
    # Zipkin exporter
    zipkin_endpoint: Optional[str] = None
    # Resource attributes
    resource_attributes: dict[str, str] = field(default_factory=dict)
    # Propagation format
    propagators: list[str] = field(default_factory=lambda: [
        "tracecontext",  # W3C Trace Context
        "baggage",       # W3C Baggage
    ])


@dataclass
class LoggingConfig:
    """Structured logging configuration."""
    enabled: bool = True
    level: str = "INFO"
    format: str = "json"  # json, text, console
    # Output destinations
    console_enabled: bool = True
    file_enabled: bool = False
    file_path: str = "/var/log/legal-doc-platform/app.log"
    file_max_size_mb: int = 100
    file_backup_count: int = 5
    # OTLP log exporter
    otlp_endpoint: Optional[str] = None
    # Loki exporter
    loki_url: Optional[str] = None
    # Include fields
    include_timestamp: bool = True
    include_level: bool = True
    include_logger: bool = True
    include_trace_id: bool = True
    include_span_id: bool = True
    include_caller: bool = False
    # Sensitive field masking
    mask_fields: list[str] = field(default_factory=lambda: [
        "password", "token", "secret", "api_key", "authorization",
    ])


@dataclass
class AlertingConfig:
    """Alerting configuration."""
    enabled: bool = True
    evaluation_interval: timedelta = timedelta(seconds=30)
    # Alert manager (Prometheus-compatible)
    alertmanager_url: Optional[str] = None
    # Notification defaults
    default_channels: list[NotificationChannel] = field(default_factory=list)
    # Throttling
    throttle_duration: timedelta = timedelta(minutes=5)
    # Grouping
    group_by: list[str] = field(default_factory=lambda: ["alertname", "severity"])
    group_wait: timedelta = timedelta(seconds=30)
    group_interval: timedelta = timedelta(minutes=5)
    # Repeat
    repeat_interval: timedelta = timedelta(hours=4)


@dataclass
class EmailNotificationConfig:
    """Email notification configuration."""
    enabled: bool = False
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True
    from_address: str = "alerts@legal-doc-platform.com"
    to_addresses: list[str] = field(default_factory=list)
    subject_prefix: str = "[Legal Doc Platform Alert]"


@dataclass
class SlackNotificationConfig:
    """Slack notification configuration."""
    enabled: bool = False
    webhook_url: Optional[str] = None
    channel: str = "#alerts"
    username: str = "Legal Doc Platform"
    icon_emoji: str = ":warning:"


@dataclass
class PagerDutyNotificationConfig:
    """PagerDuty notification configuration."""
    enabled: bool = False
    integration_key: Optional[str] = None
    service_url: str = "https://events.pagerduty.com/v2/enqueue"


@dataclass
class WebhookNotificationConfig:
    """Webhook notification configuration."""
    enabled: bool = False
    url: Optional[str] = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 30


@dataclass
class HealthCheckConfig:
    """Health check configuration."""
    enabled: bool = True
    interval: timedelta = timedelta(seconds=30)
    timeout: timedelta = timedelta(seconds=10)
    # Liveness vs Readiness
    liveness_path: str = "/health/live"
    readiness_path: str = "/health/ready"
    # Dependencies to check
    check_database: bool = True
    check_cache: bool = True
    check_storage: bool = True
    check_search: bool = True
    check_external_services: bool = True


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    enabled: bool = True
    default_refresh_interval: int = 30
    default_time_range: str = "1h"
    # Grafana integration
    grafana_url: Optional[str] = None
    grafana_api_key: Optional[str] = None


@dataclass
class MonitoringConfig:
    """Combined monitoring configuration."""
    # Sub-configs
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    health: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    # Notification channels
    email: EmailNotificationConfig = field(default_factory=EmailNotificationConfig)
    slack: SlackNotificationConfig = field(default_factory=SlackNotificationConfig)
    pagerduty: PagerDutyNotificationConfig = field(default_factory=PagerDutyNotificationConfig)
    webhook: WebhookNotificationConfig = field(default_factory=WebhookNotificationConfig)
    # General settings
    service_name: str = "legal-doc-platform"
    service_version: str = "1.0.0"
    environment: str = "development"
    instance_id: str = ""

    def __post_init__(self):
        """Initialize derived settings."""
        # Propagate service name
        self.tracing.service_name = self.service_name
        self.tracing.service_version = self.service_version
        self.tracing.environment = self.environment
