"""
Tests for metrics collection service.
"""

import pytest
from ..services.metrics import (
    MetricsCollector,
    Counter,
    Gauge,
    Histogram,
    Summary,
    count_calls,
    time_calls,
)
from ..models import MetricType


class TestCounter:
    """Tests for Counter metric."""

    def test_counter_creation(self):
        """Test Counter creation."""
        counter = Counter(
            name="test_counter",
            description="A test counter",
            labels=["method", "status"],
        )
        assert counter.name == "test_counter"
        assert counter.description == "A test counter"

    def test_counter_increment(self):
        """Test Counter increment."""
        counter = Counter(name="test_counter", description="Test")
        counter.inc()
        assert counter.get() == 1

        counter.inc()
        assert counter.get() == 2

    def test_counter_increment_by(self):
        """Test Counter increment by value."""
        counter = Counter(name="test_counter", description="Test")
        counter.inc(5)
        assert counter.get() == 5

        counter.inc(3)
        assert counter.get() == 8

    def test_counter_with_labels(self):
        """Test Counter with labels."""
        counter = Counter(
            name="http_requests",
            description="HTTP requests",
            labels=["method", "status"],
        )
        counter.labels(method="GET", status="200").inc()
        counter.labels(method="GET", status="200").inc()
        counter.labels(method="POST", status="201").inc()

        assert counter.labels(method="GET", status="200").get() == 2
        assert counter.labels(method="POST", status="201").get() == 1

    def test_counter_negative_increment_raises(self):
        """Test Counter rejects negative increment."""
        counter = Counter(name="test", description="Test")
        with pytest.raises(ValueError):
            counter.inc(-1)


class TestGauge:
    """Tests for Gauge metric."""

    def test_gauge_creation(self):
        """Test Gauge creation."""
        gauge = Gauge(
            name="test_gauge",
            description="A test gauge",
        )
        assert gauge.name == "test_gauge"

    def test_gauge_set(self):
        """Test Gauge set."""
        gauge = Gauge(name="test_gauge", description="Test")
        gauge.set(42.0)
        assert gauge.get() == 42.0

        gauge.set(100.0)
        assert gauge.get() == 100.0

    def test_gauge_inc_dec(self):
        """Test Gauge increment and decrement."""
        gauge = Gauge(name="test_gauge", description="Test")
        gauge.set(10)

        gauge.inc()
        assert gauge.get() == 11

        gauge.dec()
        assert gauge.get() == 10

        gauge.inc(5)
        assert gauge.get() == 15

        gauge.dec(3)
        assert gauge.get() == 12

    def test_gauge_with_labels(self):
        """Test Gauge with labels."""
        gauge = Gauge(
            name="temperature",
            description="Temperature",
            labels=["location"],
        )
        gauge.labels(location="room1").set(22.5)
        gauge.labels(location="room2").set(20.0)

        assert gauge.labels(location="room1").get() == 22.5
        assert gauge.labels(location="room2").get() == 20.0


class TestHistogram:
    """Tests for Histogram metric."""

    def test_histogram_creation(self):
        """Test Histogram creation."""
        histogram = Histogram(
            name="request_duration",
            description="Request duration",
            buckets=[0.1, 0.5, 1.0, 5.0],
        )
        assert histogram.name == "request_duration"
        assert 0.1 in histogram.buckets
        assert float("inf") in histogram.buckets

    def test_histogram_observe(self):
        """Test Histogram observe."""
        histogram = Histogram(
            name="test",
            description="Test",
            buckets=[0.1, 0.5, 1.0],
        )
        histogram.observe(0.05)
        histogram.observe(0.3)
        histogram.observe(0.8)
        histogram.observe(1.5)

        assert histogram.sum == pytest.approx(0.05 + 0.3 + 0.8 + 1.5)
        assert histogram.count == 4

    def test_histogram_buckets(self):
        """Test Histogram bucket counts."""
        histogram = Histogram(
            name="test",
            description="Test",
            buckets=[0.1, 0.5, 1.0],
        )
        histogram.observe(0.05)  # goes in 0.1, 0.5, 1.0, inf
        histogram.observe(0.3)   # goes in 0.5, 1.0, inf
        histogram.observe(0.8)   # goes in 1.0, inf
        histogram.observe(1.5)   # goes in inf only

        assert histogram.bucket_counts[0.1] == 1
        assert histogram.bucket_counts[0.5] == 2
        assert histogram.bucket_counts[1.0] == 3
        assert histogram.bucket_counts[float("inf")] == 4

    def test_histogram_with_labels(self):
        """Test Histogram with labels."""
        histogram = Histogram(
            name="latency",
            description="Latency",
            labels=["endpoint"],
            buckets=[0.1, 0.5, 1.0],
        )
        histogram.labels(endpoint="/api/v1").observe(0.2)
        histogram.labels(endpoint="/api/v2").observe(0.8)

        h1 = histogram.labels(endpoint="/api/v1")
        h2 = histogram.labels(endpoint="/api/v2")

        assert h1.count == 1
        assert h2.count == 1

    def test_histogram_time_context_manager(self):
        """Test Histogram time context manager."""
        import time

        histogram = Histogram(
            name="test",
            description="Test",
            buckets=[0.01, 0.1, 1.0],
        )
        with histogram.time():
            time.sleep(0.01)

        assert histogram.count == 1
        assert histogram.sum >= 0.01


class TestSummary:
    """Tests for Summary metric."""

    def test_summary_creation(self):
        """Test Summary creation."""
        summary = Summary(
            name="test_summary",
            description="A test summary",
            quantiles=[0.5, 0.9, 0.99],
        )
        assert summary.name == "test_summary"

    def test_summary_observe(self):
        """Test Summary observe."""
        summary = Summary(
            name="test",
            description="Test",
            quantiles=[0.5, 0.9],
        )
        for i in range(100):
            summary.observe(i)

        assert summary.count == 100
        assert summary.sum == pytest.approx(sum(range(100)))

    def test_summary_quantiles(self):
        """Test Summary quantile calculation."""
        summary = Summary(
            name="test",
            description="Test",
            quantiles=[0.5, 0.9, 0.99],
        )
        for i in range(100):
            summary.observe(i)

        quantiles = summary.get_quantiles()
        # P50 should be around 49-50
        assert 45 <= quantiles[0.5] <= 55
        # P90 should be around 89-90
        assert 85 <= quantiles[0.9] <= 95
        # P99 should be around 98-99
        assert 95 <= quantiles[0.99] <= 100


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_collector_creation(self):
        """Test MetricsCollector creation."""
        collector = MetricsCollector()
        assert collector is not None

    def test_collector_counter(self):
        """Test creating counter via collector."""
        collector = MetricsCollector()
        counter = collector.counter(
            "requests_total",
            "Total requests",
            ["method"],
        )
        assert isinstance(counter, Counter)
        assert counter.name == "requests_total"

    def test_collector_gauge(self):
        """Test creating gauge via collector."""
        collector = MetricsCollector()
        gauge = collector.gauge(
            "temperature",
            "Temperature in celsius",
        )
        assert isinstance(gauge, Gauge)

    def test_collector_histogram(self):
        """Test creating histogram via collector."""
        collector = MetricsCollector()
        histogram = collector.histogram(
            "latency",
            "Request latency",
            buckets=[0.1, 0.5, 1.0],
        )
        assert isinstance(histogram, Histogram)

    def test_collector_summary(self):
        """Test creating summary via collector."""
        collector = MetricsCollector()
        summary = collector.summary(
            "response_size",
            "Response size",
        )
        assert isinstance(summary, Summary)

    def test_collector_get_metric(self):
        """Test getting metric by name."""
        collector = MetricsCollector()
        collector.counter("test_counter", "Test")

        metric = collector.get_metric("test_counter")
        assert metric is not None
        assert isinstance(metric, Counter)

    def test_collector_get_all_metrics(self):
        """Test getting all metrics."""
        collector = MetricsCollector()
        collector.counter("counter1", "Counter 1")
        collector.gauge("gauge1", "Gauge 1")

        metrics = collector.get_all_metrics()
        assert len(metrics) >= 2

    def test_collector_prometheus_format(self):
        """Test Prometheus format export."""
        collector = MetricsCollector()
        counter = collector.counter("http_requests", "HTTP requests", ["method"])
        counter.labels(method="GET").inc(10)
        counter.labels(method="POST").inc(5)

        output = collector.prometheus_format()
        assert "http_requests" in output
        assert "GET" in output
        assert "POST" in output

    def test_collector_duplicate_metric_reuse(self):
        """Test that duplicate metric names return same instance."""
        collector = MetricsCollector()
        c1 = collector.counter("test", "Test")
        c2 = collector.counter("test", "Test")
        assert c1 is c2


class TestDecorators:
    """Tests for metric decorators."""

    def test_count_calls_decorator(self):
        """Test @count_calls decorator."""
        collector = MetricsCollector()

        @count_calls(collector, "function_calls", "Function calls")
        def my_function():
            return "result"

        result = my_function()
        assert result == "result"

        counter = collector.get_metric("function_calls")
        assert counter.get() == 1

        my_function()
        my_function()
        assert counter.get() == 3

    def test_time_calls_decorator(self):
        """Test @time_calls decorator."""
        import time

        collector = MetricsCollector()

        @time_calls(collector, "function_duration", "Function duration")
        def slow_function():
            time.sleep(0.01)
            return "done"

        result = slow_function()
        assert result == "done"

        histogram = collector.get_metric("function_duration")
        assert histogram.count == 1
        assert histogram.sum >= 0.01

    @pytest.mark.asyncio
    async def test_count_calls_async(self):
        """Test @count_calls with async function."""
        collector = MetricsCollector()

        @count_calls(collector, "async_calls", "Async calls")
        async def async_function():
            return "async result"

        result = await async_function()
        assert result == "async result"

        counter = collector.get_metric("async_calls")
        assert counter.get() == 1

    @pytest.mark.asyncio
    async def test_time_calls_async(self):
        """Test @time_calls with async function."""
        import asyncio

        collector = MetricsCollector()

        @time_calls(collector, "async_duration", "Async duration")
        async def async_slow():
            await asyncio.sleep(0.01)
            return "done"

        result = await async_slow()
        assert result == "done"

        histogram = collector.get_metric("async_duration")
        assert histogram.count == 1
        assert histogram.sum >= 0.01


class TestMetricsIntegration:
    """Integration tests for metrics."""

    def test_full_metrics_workflow(self):
        """Test complete metrics workflow."""
        collector = MetricsCollector()

        # Create various metrics
        requests = collector.counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "status"],
        )
        latency = collector.histogram(
            "http_request_duration_seconds",
            "Request duration",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0],
        )
        active = collector.gauge(
            "http_active_requests",
            "Active requests",
        )

        # Simulate requests
        for _ in range(100):
            requests.labels(method="GET", status="200").inc()
            latency.observe(0.03)

        for _ in range(10):
            requests.labels(method="POST", status="201").inc()
            latency.observe(0.1)

        for _ in range(5):
            requests.labels(method="GET", status="500").inc()
            latency.observe(0.5)

        active.set(10)

        # Verify
        assert requests.labels(method="GET", status="200").get() == 100
        assert requests.labels(method="POST", status="201").get() == 10
        assert requests.labels(method="GET", status="500").get() == 5
        assert latency.count == 115
        assert active.get() == 10

        # Export to Prometheus format
        output = collector.prometheus_format()
        assert "http_requests_total" in output
        assert "http_request_duration_seconds" in output
        assert "http_active_requests" in output
