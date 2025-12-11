"""
Monitoring Services
===================
Core monitoring service implementations.
"""

from __future__ import annotations

from .metrics import MetricsCollector, Counter, Gauge, Histogram, Summary
from .tracing import Tracer, trace, get_current_span
from .logging import StructuredLogger, get_logger
from .health import HealthChecker, HealthCheck
from .alerting import AlertManager, AlertEvaluator
from .dashboard import (
    DashboardService,
    TimeSeriesStore,
    Dashboard,
    DashboardPanel,
    Aggregator,
)

__all__ = [
    # Metrics
    "MetricsCollector",
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    # Tracing
    "Tracer",
    "trace",
    "get_current_span",
    # Logging
    "StructuredLogger",
    "get_logger",
    # Health
    "HealthChecker",
    "HealthCheck",
    # Alerting
    "AlertManager",
    "AlertEvaluator",
    # Dashboard
    "DashboardService",
    "TimeSeriesStore",
    "Dashboard",
    "DashboardPanel",
    "Aggregator",
]
