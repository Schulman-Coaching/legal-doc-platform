"""
Metrics Collection Service
==========================
Prometheus-compatible metrics collection with counters, gauges, histograms.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Iterator, Optional, TypeVar

from ..config import MetricsConfig
from ..models import MetricPoint, MetricType, MetricMetadata, HistogramBuckets


T = TypeVar("T")


# =============================================================================
# Metric Instruments
# =============================================================================

@dataclass
class LabelSet:
    """Immutable label set for metric series."""
    _labels: tuple[tuple[str, str], ...]

    def __init__(self, labels: Optional[dict[str, str]] = None):
        self._labels = tuple(sorted((labels or {}).items()))

    def __hash__(self) -> int:
        return hash(self._labels)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LabelSet):
            return False
        return self._labels == other._labels

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        return dict(self._labels)

    def with_labels(self, **kwargs) -> "LabelSet":
        """Create new LabelSet with additional labels."""
        combined = dict(self._labels)
        combined.update(kwargs)
        return LabelSet(combined)


class Counter:
    """
    Counter metric - monotonically increasing value.

    Usage:
        counter = Counter("requests_total", "Total requests", ["method", "path"])
        counter.labels(method="GET", path="/api").inc()
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        label_names: Optional[list[str]] = None,
        registry: Optional["MetricsCollector"] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._values: dict[LabelSet, float] = defaultdict(float)
        self._lock = threading.Lock()

        if registry:
            registry.register(self)

    def labels(self, **kwargs) -> "_CounterChild":
        """Get counter with specific labels."""
        return _CounterChild(self, LabelSet(kwargs))

    def inc(self, value: float = 1.0) -> None:
        """Increment counter (no labels)."""
        if value < 0:
            raise ValueError("Counter can only be incremented")
        with self._lock:
            self._values[LabelSet()] += value

    def get(self, labels: Optional[dict[str, str]] = None) -> float:
        """Get current value."""
        return self._values.get(LabelSet(labels), 0)

    def collect(self) -> list[MetricPoint]:
        """Collect all metric points."""
        points = []
        with self._lock:
            for label_set, value in self._values.items():
                points.append(MetricPoint(
                    name=self.name,
                    value=value,
                    labels=label_set.to_dict(),
                    metric_type=MetricType.COUNTER,
                    description=self.description,
                ))
        return points


class _CounterChild:
    """Counter with bound labels."""

    def __init__(self, counter: Counter, label_set: LabelSet):
        self._counter = counter
        self._label_set = label_set

    def inc(self, value: float = 1.0) -> None:
        """Increment counter."""
        if value < 0:
            raise ValueError("Counter can only be incremented")
        with self._counter._lock:
            self._counter._values[self._label_set] += value


class Gauge:
    """
    Gauge metric - value that can go up and down.

    Usage:
        gauge = Gauge("active_connections", "Active connections")
        gauge.set(10)
        gauge.inc()
        gauge.dec()
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        label_names: Optional[list[str]] = None,
        registry: Optional["MetricsCollector"] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._values: dict[LabelSet, float] = defaultdict(float)
        self._lock = threading.Lock()

        if registry:
            registry.register(self)

    def labels(self, **kwargs) -> "_GaugeChild":
        """Get gauge with specific labels."""
        return _GaugeChild(self, LabelSet(kwargs))

    def set(self, value: float) -> None:
        """Set gauge value."""
        with self._lock:
            self._values[LabelSet()] = value

    def inc(self, value: float = 1.0) -> None:
        """Increment gauge."""
        with self._lock:
            self._values[LabelSet()] += value

    def dec(self, value: float = 1.0) -> None:
        """Decrement gauge."""
        with self._lock:
            self._values[LabelSet()] -= value

    def get(self, labels: Optional[dict[str, str]] = None) -> float:
        """Get current value."""
        return self._values.get(LabelSet(labels), 0)

    @contextmanager
    def track_inprogress(self) -> Iterator[None]:
        """Context manager to track in-progress operations."""
        self.inc()
        try:
            yield
        finally:
            self.dec()

    def set_to_current_time(self) -> None:
        """Set gauge to current Unix timestamp."""
        self.set(time.time())

    def collect(self) -> list[MetricPoint]:
        """Collect all metric points."""
        points = []
        with self._lock:
            for label_set, value in self._values.items():
                points.append(MetricPoint(
                    name=self.name,
                    value=value,
                    labels=label_set.to_dict(),
                    metric_type=MetricType.GAUGE,
                    description=self.description,
                ))
        return points


class _GaugeChild:
    """Gauge with bound labels."""

    def __init__(self, gauge: Gauge, label_set: LabelSet):
        self._gauge = gauge
        self._label_set = label_set

    def set(self, value: float) -> None:
        """Set gauge value."""
        with self._gauge._lock:
            self._gauge._values[self._label_set] = value

    def inc(self, value: float = 1.0) -> None:
        """Increment gauge."""
        with self._gauge._lock:
            self._gauge._values[self._label_set] += value

    def dec(self, value: float = 1.0) -> None:
        """Decrement gauge."""
        with self._gauge._lock:
            self._gauge._values[self._label_set] -= value


@dataclass
class _HistogramData:
    """Internal histogram data."""
    buckets: dict[float, int] = field(default_factory=dict)
    sum: float = 0
    count: int = 0


class Histogram:
    """
    Histogram metric - distribution of values.

    Usage:
        histogram = Histogram("request_duration_seconds", "Request duration")
        histogram.observe(0.5)

        with histogram.time():
            # ... code to measure ...
    """

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(
        self,
        name: str,
        description: str = "",
        label_names: Optional[list[str]] = None,
        buckets: Optional[tuple[float, ...]] = None,
        registry: Optional["MetricsCollector"] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._data: dict[LabelSet, _HistogramData] = {}
        self._lock = threading.Lock()

        if registry:
            registry.register(self)

    def labels(self, **kwargs) -> "_HistogramChild":
        """Get histogram with specific labels."""
        return _HistogramChild(self, LabelSet(kwargs))

    def observe(self, value: float) -> None:
        """Record a value."""
        self._observe(LabelSet(), value)

    def _observe(self, label_set: LabelSet, value: float) -> None:
        """Internal observe with label set."""
        with self._lock:
            if label_set not in self._data:
                self._data[label_set] = _HistogramData(
                    buckets={b: 0 for b in self.buckets}
                )

            data = self._data[label_set]
            data.sum += value
            data.count += 1

            for bucket in self.buckets:
                if value <= bucket:
                    data.buckets[bucket] += 1

    @contextmanager
    def time(self) -> Iterator[None]:
        """Context manager to time operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(time.perf_counter() - start)

    def collect(self) -> list[MetricPoint]:
        """Collect all metric points."""
        points = []
        with self._lock:
            for label_set, data in self._data.items():
                base_labels = label_set.to_dict()

                # _sum
                points.append(MetricPoint(
                    name=f"{self.name}_sum",
                    value=data.sum,
                    labels=base_labels,
                    metric_type=MetricType.HISTOGRAM,
                    description=self.description,
                ))

                # _count
                points.append(MetricPoint(
                    name=f"{self.name}_count",
                    value=data.count,
                    labels=base_labels,
                    metric_type=MetricType.HISTOGRAM,
                ))

                # _bucket
                cumulative = 0
                for bucket in sorted(self.buckets):
                    cumulative += data.buckets.get(bucket, 0)
                    bucket_labels = {**base_labels, "le": str(bucket)}
                    points.append(MetricPoint(
                        name=f"{self.name}_bucket",
                        value=cumulative,
                        labels=bucket_labels,
                        metric_type=MetricType.HISTOGRAM,
                    ))

                # +Inf bucket
                inf_labels = {**base_labels, "le": "+Inf"}
                points.append(MetricPoint(
                    name=f"{self.name}_bucket",
                    value=data.count,
                    labels=inf_labels,
                    metric_type=MetricType.HISTOGRAM,
                ))

        return points


class _HistogramChild:
    """Histogram with bound labels."""

    def __init__(self, histogram: Histogram, label_set: LabelSet):
        self._histogram = histogram
        self._label_set = label_set

    def observe(self, value: float) -> None:
        """Record a value."""
        self._histogram._observe(self._label_set, value)

    @contextmanager
    def time(self) -> Iterator[None]:
        """Context manager to time operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(time.perf_counter() - start)


@dataclass
class _SummaryData:
    """Internal summary data."""
    values: list[float] = field(default_factory=list)
    sum: float = 0
    count: int = 0


class Summary:
    """
    Summary metric - distribution with quantiles.

    Note: Quantiles are calculated client-side.
    """

    DEFAULT_QUANTILES = (0.5, 0.9, 0.95, 0.99)

    def __init__(
        self,
        name: str,
        description: str = "",
        label_names: Optional[list[str]] = None,
        quantiles: Optional[tuple[float, ...]] = None,
        max_age_seconds: int = 600,
        registry: Optional["MetricsCollector"] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self.quantiles = quantiles or self.DEFAULT_QUANTILES
        self.max_age_seconds = max_age_seconds
        self._data: dict[LabelSet, _SummaryData] = {}
        self._lock = threading.Lock()

        if registry:
            registry.register(self)

    def labels(self, **kwargs) -> "_SummaryChild":
        """Get summary with specific labels."""
        return _SummaryChild(self, LabelSet(kwargs))

    def observe(self, value: float) -> None:
        """Record a value."""
        self._observe(LabelSet(), value)

    def _observe(self, label_set: LabelSet, value: float) -> None:
        """Internal observe with label set."""
        with self._lock:
            if label_set not in self._data:
                self._data[label_set] = _SummaryData()

            data = self._data[label_set]
            data.values.append(value)
            data.sum += value
            data.count += 1

            # Limit stored values
            if len(data.values) > 10000:
                data.values = data.values[-5000:]

    @contextmanager
    def time(self) -> Iterator[None]:
        """Context manager to time operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(time.perf_counter() - start)

    def collect(self) -> list[MetricPoint]:
        """Collect all metric points."""
        points = []
        with self._lock:
            for label_set, data in self._data.items():
                base_labels = label_set.to_dict()

                # _sum
                points.append(MetricPoint(
                    name=f"{self.name}_sum",
                    value=data.sum,
                    labels=base_labels,
                    metric_type=MetricType.SUMMARY,
                    description=self.description,
                ))

                # _count
                points.append(MetricPoint(
                    name=f"{self.name}_count",
                    value=data.count,
                    labels=base_labels,
                    metric_type=MetricType.SUMMARY,
                ))

                # Quantiles
                if data.values:
                    sorted_values = sorted(data.values)
                    for q in self.quantiles:
                        idx = int(q * len(sorted_values))
                        idx = min(idx, len(sorted_values) - 1)
                        quantile_labels = {**base_labels, "quantile": str(q)}
                        points.append(MetricPoint(
                            name=self.name,
                            value=sorted_values[idx],
                            labels=quantile_labels,
                            metric_type=MetricType.SUMMARY,
                        ))

        return points


class _SummaryChild:
    """Summary with bound labels."""

    def __init__(self, summary: Summary, label_set: LabelSet):
        self._summary = summary
        self._label_set = label_set

    def observe(self, value: float) -> None:
        """Record a value."""
        self._summary._observe(self._label_set, value)

    @contextmanager
    def time(self) -> Iterator[None]:
        """Context manager to time operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(time.perf_counter() - start)


# =============================================================================
# Metrics Collector (Registry)
# =============================================================================

class MetricsCollector:
    """
    Metrics registry and collector.

    Central registry for all metrics with Prometheus-compatible export.
    """

    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()
        self._metrics: list[Counter | Gauge | Histogram | Summary] = []
        self._metadata: dict[str, MetricMetadata] = {}
        self._lock = threading.Lock()

        # Default metrics
        self._setup_default_metrics()

    def _setup_default_metrics(self) -> None:
        """Setup default process/runtime metrics."""
        # Process info
        self.info = Gauge(
            f"{self.config.prefix}_info",
            "Application information",
            registry=self,
        )

        # Build info with default labels
        self.info.labels(
            **self.config.default_labels
        ).set(1)

    def register(self, metric: Counter | Gauge | Histogram | Summary) -> None:
        """Register a metric."""
        with self._lock:
            self._metrics.append(metric)
            self._metadata[metric.name] = MetricMetadata(
                name=metric.name,
                metric_type=self._get_metric_type(metric),
                description=metric.description,
            )

    def _get_metric_type(self, metric: Any) -> MetricType:
        """Get metric type from instance."""
        if isinstance(metric, Counter):
            return MetricType.COUNTER
        elif isinstance(metric, Gauge):
            return MetricType.GAUGE
        elif isinstance(metric, Histogram):
            return MetricType.HISTOGRAM
        elif isinstance(metric, Summary):
            return MetricType.SUMMARY
        return MetricType.GAUGE

    def counter(
        self,
        name: str,
        description: str = "",
        label_names: Optional[list[str]] = None,
    ) -> Counter:
        """Create and register a counter."""
        return Counter(
            f"{self.config.prefix}_{name}",
            description,
            label_names,
            registry=self,
        )

    def gauge(
        self,
        name: str,
        description: str = "",
        label_names: Optional[list[str]] = None,
    ) -> Gauge:
        """Create and register a gauge."""
        return Gauge(
            f"{self.config.prefix}_{name}",
            description,
            label_names,
            registry=self,
        )

    def histogram(
        self,
        name: str,
        description: str = "",
        label_names: Optional[list[str]] = None,
        buckets: Optional[tuple[float, ...]] = None,
    ) -> Histogram:
        """Create and register a histogram."""
        return Histogram(
            f"{self.config.prefix}_{name}",
            description,
            label_names,
            buckets,
            registry=self,
        )

    def summary(
        self,
        name: str,
        description: str = "",
        label_names: Optional[list[str]] = None,
        quantiles: Optional[tuple[float, ...]] = None,
    ) -> Summary:
        """Create and register a summary."""
        return Summary(
            f"{self.config.prefix}_{name}",
            description,
            label_names,
            quantiles,
            registry=self,
        )

    def collect(self) -> list[MetricPoint]:
        """Collect all metrics."""
        points = []
        with self._lock:
            for metric in self._metrics:
                points.extend(metric.collect())
        return points

    def to_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        points = self.collect()

        # Group by metric name
        by_name: dict[str, list[MetricPoint]] = defaultdict(list)
        for point in points:
            base_name = point.name.rsplit("_", 1)[0] if point.name.endswith(("_sum", "_count", "_bucket")) else point.name
            by_name[base_name].append(point)

        for name, metric_points in sorted(by_name.items()):
            # Add HELP and TYPE
            if name in self._metadata:
                meta = self._metadata[name]
                if meta.description:
                    lines.append(f"# HELP {name} {meta.description}")
                lines.append(f"# TYPE {name} {meta.metric_type.value}")

            # Add metrics
            for point in metric_points:
                labels_str = ""
                if point.labels:
                    labels_parts = [f'{k}="{v}"' for k, v in sorted(point.labels.items())]
                    labels_str = "{" + ",".join(labels_parts) + "}"

                lines.append(f"{point.name}{labels_str} {point.value}")

        return "\n".join(lines)

    def to_json(self) -> list[dict[str, Any]]:
        """Export metrics as JSON."""
        return [p.to_dict() for p in self.collect()]


# =============================================================================
# Decorators
# =============================================================================

def count_calls(counter: Counter, labels: Optional[dict[str, str]] = None):
    """Decorator to count function calls."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if labels:
                counter.labels(**labels).inc()
            else:
                counter.inc()
            return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            if labels:
                counter.labels(**labels).inc()
            else:
                counter.inc()
            return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


def time_calls(histogram: Histogram, labels: Optional[dict[str, str]] = None):
    """Decorator to time function calls."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                if labels:
                    histogram.labels(**labels).observe(duration)
                else:
                    histogram.observe(duration)

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                if labels:
                    histogram.labels(**labels).observe(duration)
                else:
                    histogram.observe(duration)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


# =============================================================================
# Global Registry
# =============================================================================

_default_registry: Optional[MetricsCollector] = None


def get_default_registry() -> MetricsCollector:
    """Get or create default metrics registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = MetricsCollector()
    return _default_registry


def set_default_registry(registry: MetricsCollector) -> None:
    """Set default metrics registry."""
    global _default_registry
    _default_registry = registry
