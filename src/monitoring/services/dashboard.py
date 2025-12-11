"""
Dashboard Service
=================
Data aggregation and visualization support for monitoring dashboards.
"""

from __future__ import annotations

import asyncio
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum

from ..models import (
    MetricType,
    MetricPoint,
    ServiceHealth,
    AlertSeverity,
)


# =============================================================================
# Data Types
# =============================================================================

class AggregationType(Enum):
    """Aggregation functions for time series data."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    RATE = "rate"
    PERCENTILE_50 = "p50"
    PERCENTILE_90 = "p90"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"
    LAST = "last"
    FIRST = "first"


class TimeRange(Enum):
    """Common time ranges for dashboards."""
    LAST_5_MINUTES = "5m"
    LAST_15_MINUTES = "15m"
    LAST_30_MINUTES = "30m"
    LAST_1_HOUR = "1h"
    LAST_3_HOURS = "3h"
    LAST_6_HOURS = "6h"
    LAST_12_HOURS = "12h"
    LAST_24_HOURS = "24h"
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"


@dataclass
class TimeSeriesPoint:
    """Single point in time series."""
    timestamp: datetime
    value: float
    labels: Optional[dict[str, str]] = None


@dataclass
class TimeSeries:
    """Time series data."""
    name: str
    points: list[TimeSeriesPoint] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)

    def add_point(self, timestamp: datetime, value: float) -> None:
        """Add a point to the series."""
        self.points.append(TimeSeriesPoint(timestamp=timestamp, value=value))

    def get_values(self) -> list[float]:
        """Get all values."""
        return [p.value for p in self.points]

    def get_latest(self) -> Optional[TimeSeriesPoint]:
        """Get latest point."""
        if self.points:
            return max(self.points, key=lambda p: p.timestamp)
        return None


@dataclass
class DashboardPanel:
    """Dashboard panel configuration."""
    id: str
    title: str
    type: str  # "graph", "gauge", "stat", "table", "heatmap"
    metrics: list[str]
    aggregation: AggregationType = AggregationType.AVG
    time_range: TimeRange = TimeRange.LAST_1_HOUR
    refresh_interval: int = 30  # seconds
    thresholds: Optional[list[tuple[float, str]]] = None  # value, color
    labels: Optional[dict[str, str]] = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Dashboard:
    """Dashboard configuration."""
    id: str
    title: str
    description: str = ""
    panels: list[DashboardPanel] = field(default_factory=list)
    refresh_interval: int = 30
    time_range: TimeRange = TimeRange.LAST_1_HOUR
    variables: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def add_panel(self, panel: DashboardPanel) -> None:
        """Add panel to dashboard."""
        self.panels.append(panel)


@dataclass
class PanelData:
    """Data for a dashboard panel."""
    panel_id: str
    series: list[TimeSeries] = field(default_factory=list)
    current_value: Optional[float] = None
    aggregated_value: Optional[float] = None
    trend: Optional[str] = None  # "up", "down", "stable"
    change_percent: Optional[float] = None


# =============================================================================
# Time Series Store
# =============================================================================

class TimeSeriesStore:
    """
    In-memory time series store for dashboard data.

    Supports retention and downsampling.
    """

    def __init__(
        self,
        retention: timedelta = timedelta(hours=24),
        downsample_after: timedelta = timedelta(hours=1),
        downsample_interval: timedelta = timedelta(minutes=5),
    ):
        self.retention = retention
        self.downsample_after = downsample_after
        self.downsample_interval = downsample_interval
        self._series: dict[str, TimeSeries] = {}
        self._lock = threading.Lock()

    def record(
        self,
        name: str,
        value: float,
        timestamp: Optional[datetime] = None,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        """Record a metric value."""
        with self._lock:
            key = self._make_key(name, labels)

            if key not in self._series:
                self._series[key] = TimeSeries(name=name, labels=labels or {})

            series = self._series[key]
            series.add_point(
                timestamp=timestamp or datetime.utcnow(),
                value=value,
            )

    def _make_key(self, name: str, labels: Optional[dict[str, str]]) -> str:
        """Create key for series lookup."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def query(
        self,
        name: str,
        time_range: Optional[TimeRange] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        labels: Optional[dict[str, str]] = None,
    ) -> list[TimeSeries]:
        """Query time series data."""
        # Determine time bounds
        if time_range:
            end = datetime.utcnow()
            start = end - self._parse_time_range(time_range)
        elif not end:
            end = datetime.utcnow()
        if not start:
            start = end - timedelta(hours=1)

        with self._lock:
            results = []

            for key, series in self._series.items():
                # Check name match
                if series.name != name:
                    continue

                # Check label match
                if labels:
                    if not all(series.labels.get(k) == v for k, v in labels.items()):
                        continue

                # Filter points by time
                filtered_points = [
                    p for p in series.points
                    if start <= p.timestamp <= end
                ]

                if filtered_points:
                    result_series = TimeSeries(
                        name=series.name,
                        labels=series.labels.copy(),
                        points=filtered_points,
                    )
                    results.append(result_series)

            return results

    def _parse_time_range(self, time_range: TimeRange) -> timedelta:
        """Parse time range to timedelta."""
        mapping = {
            TimeRange.LAST_5_MINUTES: timedelta(minutes=5),
            TimeRange.LAST_15_MINUTES: timedelta(minutes=15),
            TimeRange.LAST_30_MINUTES: timedelta(minutes=30),
            TimeRange.LAST_1_HOUR: timedelta(hours=1),
            TimeRange.LAST_3_HOURS: timedelta(hours=3),
            TimeRange.LAST_6_HOURS: timedelta(hours=6),
            TimeRange.LAST_12_HOURS: timedelta(hours=12),
            TimeRange.LAST_24_HOURS: timedelta(hours=24),
            TimeRange.LAST_7_DAYS: timedelta(days=7),
            TimeRange.LAST_30_DAYS: timedelta(days=30),
        }
        return mapping.get(time_range, timedelta(hours=1))

    def cleanup(self) -> int:
        """Remove expired data. Returns number of points removed."""
        cutoff = datetime.utcnow() - self.retention
        removed = 0

        with self._lock:
            for series in self._series.values():
                original_count = len(series.points)
                series.points = [p for p in series.points if p.timestamp >= cutoff]
                removed += original_count - len(series.points)

            # Remove empty series
            empty_keys = [k for k, v in self._series.items() if not v.points]
            for key in empty_keys:
                del self._series[key]

        return removed

    def downsample(self) -> int:
        """Downsample old data. Returns number of points reduced."""
        cutoff = datetime.utcnow() - self.downsample_after
        reduced = 0

        with self._lock:
            for series in self._series.values():
                # Get old points
                old_points = [p for p in series.points if p.timestamp < cutoff]
                new_points = [p for p in series.points if p.timestamp >= cutoff]

                if not old_points:
                    continue

                # Group by downsample interval
                buckets: dict[datetime, list[float]] = defaultdict(list)
                for point in old_points:
                    bucket_time = self._bucket_time(point.timestamp)
                    buckets[bucket_time].append(point.value)

                # Average each bucket
                downsampled = [
                    TimeSeriesPoint(
                        timestamp=ts,
                        value=sum(values) / len(values),
                    )
                    for ts, values in sorted(buckets.items())
                ]

                original_old = len(old_points)
                series.points = downsampled + new_points
                reduced += original_old - len(downsampled)

        return reduced

    def _bucket_time(self, timestamp: datetime) -> datetime:
        """Round timestamp to bucket."""
        interval_seconds = self.downsample_interval.total_seconds()
        ts = timestamp.timestamp()
        bucket_ts = (ts // interval_seconds) * interval_seconds
        return datetime.fromtimestamp(bucket_ts)


# =============================================================================
# Aggregation Functions
# =============================================================================

class Aggregator:
    """Aggregation functions for time series data."""

    @staticmethod
    def aggregate(
        values: list[float],
        aggregation: AggregationType,
    ) -> Optional[float]:
        """Apply aggregation function to values."""
        if not values:
            return None

        if aggregation == AggregationType.SUM:
            return sum(values)
        elif aggregation == AggregationType.AVG:
            return sum(values) / len(values)
        elif aggregation == AggregationType.MIN:
            return min(values)
        elif aggregation == AggregationType.MAX:
            return max(values)
        elif aggregation == AggregationType.COUNT:
            return float(len(values))
        elif aggregation == AggregationType.LAST:
            return values[-1]
        elif aggregation == AggregationType.FIRST:
            return values[0]
        elif aggregation == AggregationType.PERCENTILE_50:
            return Aggregator.percentile(values, 50)
        elif aggregation == AggregationType.PERCENTILE_90:
            return Aggregator.percentile(values, 90)
        elif aggregation == AggregationType.PERCENTILE_95:
            return Aggregator.percentile(values, 95)
        elif aggregation == AggregationType.PERCENTILE_99:
            return Aggregator.percentile(values, 99)

        return None

    @staticmethod
    def percentile(values: list[float], p: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0
        sorted_values = sorted(values)
        idx = (len(sorted_values) - 1) * p / 100
        lower = int(idx)
        upper = lower + 1
        if upper >= len(sorted_values):
            return sorted_values[-1]
        weight = idx - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    @staticmethod
    def rate(
        points: list[TimeSeriesPoint],
        interval: timedelta = timedelta(seconds=60),
    ) -> list[TimeSeriesPoint]:
        """Calculate rate of change per interval."""
        if len(points) < 2:
            return []

        sorted_points = sorted(points, key=lambda p: p.timestamp)
        rate_points = []

        for i in range(1, len(sorted_points)):
            prev = sorted_points[i - 1]
            curr = sorted_points[i]

            time_diff = (curr.timestamp - prev.timestamp).total_seconds()
            if time_diff <= 0:
                continue

            value_diff = curr.value - prev.value
            rate = value_diff / time_diff * interval.total_seconds()

            rate_points.append(TimeSeriesPoint(
                timestamp=curr.timestamp,
                value=rate,
            ))

        return rate_points

    @staticmethod
    def moving_average(
        values: list[float],
        window: int = 5,
    ) -> list[float]:
        """Calculate moving average."""
        if len(values) < window:
            return values

        result = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            window_values = values[start:i + 1]
            result.append(sum(window_values) / len(window_values))

        return result


# =============================================================================
# Dashboard Service
# =============================================================================

class DashboardService:
    """
    Service for managing dashboards and fetching panel data.

    Usage:
        service = DashboardService(time_series_store)

        # Create dashboard
        dashboard = Dashboard(id="system", title="System Overview")
        dashboard.add_panel(DashboardPanel(
            id="cpu",
            title="CPU Usage",
            type="graph",
            metrics=["system_cpu_percent"],
        ))
        service.register_dashboard(dashboard)

        # Get panel data
        data = await service.get_panel_data("system", "cpu")
    """

    def __init__(
        self,
        store: Optional[TimeSeriesStore] = None,
        metrics_collector: Optional[Any] = None,
        health_checker: Optional[Any] = None,
        alert_manager: Optional[Any] = None,
    ):
        self.store = store or TimeSeriesStore()
        self.metrics_collector = metrics_collector
        self.health_checker = health_checker
        self.alert_manager = alert_manager
        self._dashboards: dict[str, Dashboard] = {}
        self._lock = threading.Lock()

    def register_dashboard(self, dashboard: Dashboard) -> None:
        """Register a dashboard."""
        with self._lock:
            self._dashboards[dashboard.id] = dashboard

    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard by ID."""
        return self._dashboards.get(dashboard_id)

    def list_dashboards(self) -> list[Dashboard]:
        """List all dashboards."""
        return list(self._dashboards.values())

    async def get_panel_data(
        self,
        dashboard_id: str,
        panel_id: str,
        time_range: Optional[TimeRange] = None,
    ) -> Optional[PanelData]:
        """Get data for a specific panel."""
        dashboard = self._dashboards.get(dashboard_id)
        if not dashboard:
            return None

        panel = next((p for p in dashboard.panels if p.id == panel_id), None)
        if not panel:
            return None

        # Use panel's time range if not specified
        effective_range = time_range or panel.time_range

        # Fetch data for each metric
        all_series: list[TimeSeries] = []
        for metric_name in panel.metrics:
            series_list = self.store.query(
                name=metric_name,
                time_range=effective_range,
                labels=panel.labels,
            )
            all_series.extend(series_list)

        # Calculate aggregated values
        all_values: list[float] = []
        for series in all_series:
            all_values.extend(series.get_values())

        current_value = None
        if all_series:
            latest = max(
                (s.get_latest() for s in all_series if s.get_latest()),
                key=lambda p: p.timestamp if p else datetime.min,
                default=None,
            )
            if latest:
                current_value = latest.value

        aggregated_value = Aggregator.aggregate(all_values, panel.aggregation)

        # Calculate trend
        trend = None
        change_percent = None
        if len(all_values) >= 2:
            first_half = all_values[:len(all_values)//2]
            second_half = all_values[len(all_values)//2:]

            first_avg = sum(first_half) / len(first_half) if first_half else 0
            second_avg = sum(second_half) / len(second_half) if second_half else 0

            if first_avg > 0:
                change_percent = ((second_avg - first_avg) / first_avg) * 100

                if change_percent > 5:
                    trend = "up"
                elif change_percent < -5:
                    trend = "down"
                else:
                    trend = "stable"

        return PanelData(
            panel_id=panel_id,
            series=all_series,
            current_value=current_value,
            aggregated_value=aggregated_value,
            trend=trend,
            change_percent=change_percent,
        )

    async def get_dashboard_data(
        self,
        dashboard_id: str,
        time_range: Optional[TimeRange] = None,
    ) -> dict[str, PanelData]:
        """Get data for all panels in a dashboard."""
        dashboard = self._dashboards.get(dashboard_id)
        if not dashboard:
            return {}

        result = {}
        for panel in dashboard.panels:
            data = await self.get_panel_data(dashboard_id, panel.id, time_range)
            if data:
                result[panel.id] = data

        return result

    async def get_overview(self) -> dict[str, Any]:
        """Get system overview data."""
        overview: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "health": None,
            "metrics_summary": {},
            "active_alerts": [],
        }

        # Health status
        if self.health_checker:
            try:
                report = await self.health_checker.check_all()
                overview["health"] = {
                    "status": report.status.value,
                    "checks": len(report.checks),
                    "uptime_seconds": report.uptime_seconds,
                }
            except Exception:
                pass

        # Active alerts
        if self.alert_manager:
            try:
                alerts = self.alert_manager.get_active_alerts()
                overview["active_alerts"] = [
                    {
                        "name": a.name,
                        "severity": a.severity.value,
                        "started_at": a.started_at.isoformat(),
                    }
                    for a in alerts
                ]
            except Exception:
                pass

        # Metrics summary
        if self.metrics_collector:
            try:
                # Get key metrics
                overview["metrics_summary"] = {
                    "total_metrics": len(self.metrics_collector.get_all_metrics()),
                }
            except Exception:
                pass

        return overview


# =============================================================================
# Dashboard Builder
# =============================================================================

class DashboardBuilder:
    """Fluent builder for dashboards."""

    def __init__(self, id: str, title: str):
        self._dashboard = Dashboard(id=id, title=title)
        self._current_panel: Optional[DashboardPanel] = None

    def description(self, text: str) -> "DashboardBuilder":
        """Set description."""
        self._dashboard.description = text
        return self

    def refresh(self, seconds: int) -> "DashboardBuilder":
        """Set refresh interval."""
        self._dashboard.refresh_interval = seconds
        return self

    def time_range(self, range: TimeRange) -> "DashboardBuilder":
        """Set default time range."""
        self._dashboard.time_range = range
        return self

    def tags(self, *tags: str) -> "DashboardBuilder":
        """Add tags."""
        self._dashboard.tags.extend(tags)
        return self

    def variable(self, name: str, value: Any) -> "DashboardBuilder":
        """Add variable."""
        self._dashboard.variables[name] = value
        return self

    def panel(
        self,
        id: str,
        title: str,
        type: str = "graph",
    ) -> "DashboardBuilder":
        """Start defining a panel."""
        self._current_panel = DashboardPanel(
            id=id,
            title=title,
            type=type,
            metrics=[],
        )
        return self

    def metric(self, name: str) -> "DashboardBuilder":
        """Add metric to current panel."""
        if self._current_panel:
            self._current_panel.metrics.append(name)
        return self

    def aggregate(self, agg: AggregationType) -> "DashboardBuilder":
        """Set aggregation for current panel."""
        if self._current_panel:
            self._current_panel.aggregation = agg
        return self

    def threshold(self, value: float, color: str) -> "DashboardBuilder":
        """Add threshold to current panel."""
        if self._current_panel:
            if self._current_panel.thresholds is None:
                self._current_panel.thresholds = []
            self._current_panel.thresholds.append((value, color))
        return self

    def option(self, key: str, value: Any) -> "DashboardBuilder":
        """Add option to current panel."""
        if self._current_panel:
            self._current_panel.options[key] = value
        return self

    def end_panel(self) -> "DashboardBuilder":
        """Finish current panel and add to dashboard."""
        if self._current_panel:
            self._dashboard.add_panel(self._current_panel)
            self._current_panel = None
        return self

    def build(self) -> Dashboard:
        """Build the dashboard."""
        # Add any pending panel
        if self._current_panel:
            self._dashboard.add_panel(self._current_panel)
        return self._dashboard


def dashboard(id: str, title: str) -> DashboardBuilder:
    """Create a dashboard builder."""
    return DashboardBuilder(id, title)


# =============================================================================
# Pre-built Dashboard Templates
# =============================================================================

def create_system_dashboard() -> Dashboard:
    """Create system metrics dashboard."""
    return (
        dashboard("system", "System Overview")
        .description("System-level metrics and health")
        .time_range(TimeRange.LAST_1_HOUR)
        .refresh(30)
        .tags("system", "infrastructure")

        .panel("cpu", "CPU Usage", "gauge")
        .metric("system_cpu_percent")
        .aggregate(AggregationType.AVG)
        .threshold(70, "yellow")
        .threshold(90, "red")
        .end_panel()

        .panel("memory", "Memory Usage", "gauge")
        .metric("system_memory_percent")
        .aggregate(AggregationType.AVG)
        .threshold(80, "yellow")
        .threshold(95, "red")
        .end_panel()

        .panel("disk", "Disk Usage", "gauge")
        .metric("system_disk_percent")
        .aggregate(AggregationType.AVG)
        .threshold(80, "yellow")
        .threshold(95, "red")
        .end_panel()

        .panel("network_in", "Network In", "graph")
        .metric("system_network_bytes_recv")
        .aggregate(AggregationType.RATE)
        .end_panel()

        .panel("network_out", "Network Out", "graph")
        .metric("system_network_bytes_sent")
        .aggregate(AggregationType.RATE)
        .end_panel()

        .build()
    )


def create_application_dashboard() -> Dashboard:
    """Create application metrics dashboard."""
    return (
        dashboard("application", "Application Metrics")
        .description("Application-level metrics and performance")
        .time_range(TimeRange.LAST_1_HOUR)
        .refresh(30)
        .tags("application", "performance")

        .panel("request_rate", "Request Rate", "graph")
        .metric("http_requests_total")
        .aggregate(AggregationType.RATE)
        .end_panel()

        .panel("latency_p50", "Latency (P50)", "graph")
        .metric("http_request_duration_seconds")
        .aggregate(AggregationType.PERCENTILE_50)
        .end_panel()

        .panel("latency_p99", "Latency (P99)", "graph")
        .metric("http_request_duration_seconds")
        .aggregate(AggregationType.PERCENTILE_99)
        .threshold(0.5, "yellow")
        .threshold(1.0, "red")
        .end_panel()

        .panel("errors", "Error Rate", "graph")
        .metric("http_requests_errors_total")
        .aggregate(AggregationType.RATE)
        .threshold(0.01, "yellow")
        .threshold(0.05, "red")
        .end_panel()

        .panel("active_requests", "Active Requests", "stat")
        .metric("http_requests_active")
        .aggregate(AggregationType.LAST)
        .end_panel()

        .build()
    )


def create_database_dashboard() -> Dashboard:
    """Create database metrics dashboard."""
    return (
        dashboard("database", "Database Metrics")
        .description("Database performance and health")
        .time_range(TimeRange.LAST_1_HOUR)
        .refresh(30)
        .tags("database", "performance")

        .panel("connections", "Active Connections", "gauge")
        .metric("db_connections_active")
        .aggregate(AggregationType.LAST)
        .threshold(80, "yellow")
        .threshold(95, "red")
        .end_panel()

        .panel("query_rate", "Query Rate", "graph")
        .metric("db_queries_total")
        .aggregate(AggregationType.RATE)
        .end_panel()

        .panel("query_latency", "Query Latency (P95)", "graph")
        .metric("db_query_duration_seconds")
        .aggregate(AggregationType.PERCENTILE_95)
        .threshold(0.1, "yellow")
        .threshold(0.5, "red")
        .end_panel()

        .panel("pool_usage", "Connection Pool Usage", "gauge")
        .metric("db_pool_usage_percent")
        .aggregate(AggregationType.AVG)
        .threshold(70, "yellow")
        .threshold(90, "red")
        .end_panel()

        .build()
    )
