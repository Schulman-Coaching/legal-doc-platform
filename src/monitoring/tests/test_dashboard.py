"""
Tests for dashboard service.
"""

import pytest
from datetime import datetime, timedelta
from ..services.dashboard import (
    DashboardService,
    TimeSeriesStore,
    Dashboard,
    DashboardPanel,
    DashboardBuilder,
    TimeSeries,
    TimeSeriesPoint,
    PanelData,
    Aggregator,
    AggregationType,
    TimeRange,
    dashboard,
    create_system_dashboard,
    create_application_dashboard,
    create_database_dashboard,
)


class TestTimeSeriesPoint:
    """Tests for TimeSeriesPoint."""

    def test_point_creation(self):
        """Test TimeSeriesPoint creation."""
        now = datetime.utcnow()
        point = TimeSeriesPoint(timestamp=now, value=42.0)

        assert point.timestamp == now
        assert point.value == 42.0

    def test_point_with_labels(self):
        """Test TimeSeriesPoint with labels."""
        point = TimeSeriesPoint(
            timestamp=datetime.utcnow(),
            value=100.0,
            labels={"host": "server1"},
        )
        assert point.labels == {"host": "server1"}


class TestTimeSeries:
    """Tests for TimeSeries."""

    def test_series_creation(self):
        """Test TimeSeries creation."""
        series = TimeSeries(name="cpu_usage")
        assert series.name == "cpu_usage"
        assert len(series.points) == 0

    def test_series_add_point(self):
        """Test adding points to series."""
        series = TimeSeries(name="test")
        now = datetime.utcnow()

        series.add_point(now, 10.0)
        series.add_point(now + timedelta(seconds=1), 20.0)

        assert len(series.points) == 2

    def test_series_get_values(self):
        """Test getting all values."""
        series = TimeSeries(name="test")
        now = datetime.utcnow()

        series.add_point(now, 10.0)
        series.add_point(now + timedelta(seconds=1), 20.0)
        series.add_point(now + timedelta(seconds=2), 30.0)

        values = series.get_values()
        assert values == [10.0, 20.0, 30.0]

    def test_series_get_latest(self):
        """Test getting latest point."""
        series = TimeSeries(name="test")
        now = datetime.utcnow()

        series.add_point(now, 10.0)
        series.add_point(now + timedelta(seconds=1), 20.0)
        series.add_point(now + timedelta(seconds=2), 30.0)

        latest = series.get_latest()
        assert latest.value == 30.0

    def test_series_get_latest_empty(self):
        """Test getting latest from empty series."""
        series = TimeSeries(name="test")
        assert series.get_latest() is None


class TestTimeSeriesStore:
    """Tests for TimeSeriesStore."""

    def test_store_creation(self):
        """Test TimeSeriesStore creation."""
        store = TimeSeriesStore()
        assert store is not None

    def test_store_record(self):
        """Test recording values."""
        store = TimeSeriesStore()
        store.record("cpu_usage", 45.0)
        store.record("cpu_usage", 50.0)

        series = store.query("cpu_usage")
        assert len(series) == 1
        assert len(series[0].points) == 2

    def test_store_record_with_labels(self):
        """Test recording with labels."""
        store = TimeSeriesStore()
        store.record("cpu_usage", 45.0, labels={"host": "server1"})
        store.record("cpu_usage", 50.0, labels={"host": "server2"})

        all_series = store.query("cpu_usage")
        assert len(all_series) == 2

    def test_store_query_with_labels(self):
        """Test querying with label filter."""
        store = TimeSeriesStore()
        store.record("cpu_usage", 45.0, labels={"host": "server1"})
        store.record("cpu_usage", 50.0, labels={"host": "server2"})

        series = store.query("cpu_usage", labels={"host": "server1"})
        assert len(series) == 1
        assert series[0].labels["host"] == "server1"

    def test_store_query_time_range(self):
        """Test querying with time range."""
        store = TimeSeriesStore()
        now = datetime.utcnow()

        store.record("test", 10.0, timestamp=now - timedelta(hours=2))
        store.record("test", 20.0, timestamp=now - timedelta(minutes=30))
        store.record("test", 30.0, timestamp=now)

        # Query last hour
        series = store.query("test", time_range=TimeRange.LAST_1_HOUR)
        assert len(series) == 1
        assert len(series[0].points) == 2  # Only recent 2 points

    def test_store_cleanup(self):
        """Test data cleanup."""
        store = TimeSeriesStore(retention=timedelta(hours=1))
        now = datetime.utcnow()

        store.record("test", 10.0, timestamp=now - timedelta(hours=2))
        store.record("test", 20.0, timestamp=now)

        removed = store.cleanup()
        assert removed == 1

        series = store.query("test")
        assert len(series[0].points) == 1

    def test_store_downsample(self):
        """Test data downsampling."""
        store = TimeSeriesStore(
            downsample_after=timedelta(minutes=30),
            downsample_interval=timedelta(minutes=5),
        )
        now = datetime.utcnow()

        # Add many points an hour ago
        for i in range(60):
            store.record(
                "test",
                float(i),
                timestamp=now - timedelta(hours=1) + timedelta(seconds=i * 10),
            )

        # Add recent points
        store.record("test", 100.0, timestamp=now)

        reduced = store.downsample()
        assert reduced > 0


class TestAggregator:
    """Tests for Aggregator."""

    def test_aggregate_sum(self):
        """Test sum aggregation."""
        result = Aggregator.aggregate([1, 2, 3, 4, 5], AggregationType.SUM)
        assert result == 15

    def test_aggregate_avg(self):
        """Test average aggregation."""
        result = Aggregator.aggregate([10, 20, 30], AggregationType.AVG)
        assert result == 20

    def test_aggregate_min(self):
        """Test min aggregation."""
        result = Aggregator.aggregate([5, 2, 8, 1, 9], AggregationType.MIN)
        assert result == 1

    def test_aggregate_max(self):
        """Test max aggregation."""
        result = Aggregator.aggregate([5, 2, 8, 1, 9], AggregationType.MAX)
        assert result == 9

    def test_aggregate_count(self):
        """Test count aggregation."""
        result = Aggregator.aggregate([1, 2, 3, 4, 5], AggregationType.COUNT)
        assert result == 5

    def test_aggregate_last(self):
        """Test last aggregation."""
        result = Aggregator.aggregate([1, 2, 3, 4, 5], AggregationType.LAST)
        assert result == 5

    def test_aggregate_first(self):
        """Test first aggregation."""
        result = Aggregator.aggregate([1, 2, 3, 4, 5], AggregationType.FIRST)
        assert result == 1

    def test_aggregate_empty(self):
        """Test aggregation with empty list."""
        result = Aggregator.aggregate([], AggregationType.SUM)
        assert result is None

    def test_percentile_50(self):
        """Test 50th percentile."""
        result = Aggregator.aggregate(list(range(100)), AggregationType.PERCENTILE_50)
        assert 45 <= result <= 55

    def test_percentile_90(self):
        """Test 90th percentile."""
        result = Aggregator.aggregate(list(range(100)), AggregationType.PERCENTILE_90)
        assert 85 <= result <= 95

    def test_percentile_99(self):
        """Test 99th percentile."""
        result = Aggregator.aggregate(list(range(100)), AggregationType.PERCENTILE_99)
        assert 95 <= result <= 100

    def test_rate_calculation(self):
        """Test rate calculation."""
        now = datetime.utcnow()
        points = [
            TimeSeriesPoint(timestamp=now, value=100),
            TimeSeriesPoint(timestamp=now + timedelta(seconds=60), value=200),
        ]

        rate_points = Aggregator.rate(points)
        assert len(rate_points) == 1
        # Rate should be 100/60 * 60 = 100 per minute
        assert abs(rate_points[0].value - 100) < 1

    def test_moving_average(self):
        """Test moving average."""
        values = [10, 20, 30, 40, 50]
        result = Aggregator.moving_average(values, window=3)

        # First value: avg([10]) = 10
        # Second value: avg([10,20]) = 15
        # Third value: avg([10,20,30]) = 20
        # Fourth value: avg([20,30,40]) = 30
        # Fifth value: avg([30,40,50]) = 40
        assert result == [10, 15, 20, 30, 40]


class TestDashboardPanel:
    """Tests for DashboardPanel."""

    def test_panel_creation(self):
        """Test DashboardPanel creation."""
        panel = DashboardPanel(
            id="cpu_gauge",
            title="CPU Usage",
            type="gauge",
            metrics=["system_cpu_percent"],
        )
        assert panel.id == "cpu_gauge"
        assert panel.title == "CPU Usage"
        assert panel.type == "gauge"

    def test_panel_with_thresholds(self):
        """Test DashboardPanel with thresholds."""
        panel = DashboardPanel(
            id="cpu_gauge",
            title="CPU Usage",
            type="gauge",
            metrics=["system_cpu_percent"],
            thresholds=[(70, "yellow"), (90, "red")],
        )
        assert len(panel.thresholds) == 2

    def test_panel_with_aggregation(self):
        """Test DashboardPanel with aggregation."""
        panel = DashboardPanel(
            id="latency",
            title="P99 Latency",
            type="graph",
            metrics=["http_latency"],
            aggregation=AggregationType.PERCENTILE_99,
        )
        assert panel.aggregation == AggregationType.PERCENTILE_99


class TestDashboard:
    """Tests for Dashboard."""

    def test_dashboard_creation(self):
        """Test Dashboard creation."""
        dash = Dashboard(
            id="system",
            title="System Overview",
            description="System-level metrics",
        )
        assert dash.id == "system"
        assert dash.title == "System Overview"

    def test_dashboard_add_panel(self):
        """Test adding panels to dashboard."""
        dash = Dashboard(id="test", title="Test")
        panel = DashboardPanel(
            id="panel1",
            title="Panel 1",
            type="graph",
            metrics=["metric1"],
        )
        dash.add_panel(panel)

        assert len(dash.panels) == 1
        assert dash.panels[0].id == "panel1"


class TestDashboardBuilder:
    """Tests for DashboardBuilder."""

    def test_builder_basic(self):
        """Test basic dashboard building."""
        dash = (
            dashboard("test", "Test Dashboard")
            .description("A test dashboard")
            .time_range(TimeRange.LAST_1_HOUR)
            .refresh(30)
            .build()
        )

        assert dash.id == "test"
        assert dash.title == "Test Dashboard"
        assert dash.time_range == TimeRange.LAST_1_HOUR
        assert dash.refresh_interval == 30

    def test_builder_with_panels(self):
        """Test building dashboard with panels."""
        dash = (
            dashboard("test", "Test")
            .panel("cpu", "CPU", "gauge")
            .metric("system_cpu_percent")
            .aggregate(AggregationType.AVG)
            .threshold(70, "yellow")
            .threshold(90, "red")
            .end_panel()
            .panel("memory", "Memory", "gauge")
            .metric("system_memory_percent")
            .end_panel()
            .build()
        )

        assert len(dash.panels) == 2
        assert dash.panels[0].id == "cpu"
        assert dash.panels[1].id == "memory"
        assert len(dash.panels[0].thresholds) == 2

    def test_builder_with_tags(self):
        """Test building dashboard with tags."""
        dash = (
            dashboard("test", "Test")
            .tags("system", "infrastructure")
            .build()
        )

        assert "system" in dash.tags
        assert "infrastructure" in dash.tags

    def test_builder_with_variables(self):
        """Test building dashboard with variables."""
        dash = (
            dashboard("test", "Test")
            .variable("environment", "production")
            .variable("region", "us-east-1")
            .build()
        )

        assert dash.variables["environment"] == "production"
        assert dash.variables["region"] == "us-east-1"


class TestDashboardService:
    """Tests for DashboardService."""

    def test_service_creation(self):
        """Test DashboardService creation."""
        service = DashboardService()
        assert service is not None

    def test_register_dashboard(self):
        """Test registering dashboards."""
        service = DashboardService()
        dash = Dashboard(id="test", title="Test")
        service.register_dashboard(dash)

        retrieved = service.get_dashboard("test")
        assert retrieved is not None
        assert retrieved.id == "test"

    def test_list_dashboards(self):
        """Test listing dashboards."""
        service = DashboardService()
        service.register_dashboard(Dashboard(id="dash1", title="Dashboard 1"))
        service.register_dashboard(Dashboard(id="dash2", title="Dashboard 2"))

        dashboards = service.list_dashboards()
        assert len(dashboards) == 2

    @pytest.mark.asyncio
    async def test_get_panel_data(self):
        """Test getting panel data."""
        store = TimeSeriesStore()
        now = datetime.utcnow()

        # Add data
        for i in range(10):
            store.record(
                "cpu_percent",
                50.0 + i,
                timestamp=now - timedelta(minutes=10-i),
            )

        service = DashboardService(store=store)

        dash = (
            dashboard("system", "System")
            .panel("cpu", "CPU", "gauge")
            .metric("cpu_percent")
            .aggregate(AggregationType.AVG)
            .end_panel()
            .build()
        )
        service.register_dashboard(dash)

        data = await service.get_panel_data("system", "cpu")

        assert data is not None
        assert data.panel_id == "cpu"
        assert data.aggregated_value is not None

    @pytest.mark.asyncio
    async def test_get_panel_data_not_found(self):
        """Test getting data for non-existent panel."""
        service = DashboardService()

        data = await service.get_panel_data("nonexistent", "panel")
        assert data is None

    @pytest.mark.asyncio
    async def test_get_dashboard_data(self):
        """Test getting data for all panels."""
        store = TimeSeriesStore()
        now = datetime.utcnow()

        store.record("cpu", 50.0, timestamp=now)
        store.record("memory", 60.0, timestamp=now)

        service = DashboardService(store=store)

        dash = (
            dashboard("system", "System")
            .panel("cpu_panel", "CPU", "gauge")
            .metric("cpu")
            .end_panel()
            .panel("mem_panel", "Memory", "gauge")
            .metric("memory")
            .end_panel()
            .build()
        )
        service.register_dashboard(dash)

        data = await service.get_dashboard_data("system")

        assert "cpu_panel" in data
        assert "mem_panel" in data

    @pytest.mark.asyncio
    async def test_get_overview(self):
        """Test getting system overview."""
        service = DashboardService()
        overview = await service.get_overview()

        assert "timestamp" in overview
        assert "health" in overview
        assert "active_alerts" in overview


class TestPrebuiltDashboards:
    """Tests for pre-built dashboard templates."""

    def test_system_dashboard(self):
        """Test system dashboard template."""
        dash = create_system_dashboard()

        assert dash.id == "system"
        assert len(dash.panels) >= 3  # CPU, Memory, Disk at minimum
        assert any(p.id == "cpu" for p in dash.panels)

    def test_application_dashboard(self):
        """Test application dashboard template."""
        dash = create_application_dashboard()

        assert dash.id == "application"
        assert len(dash.panels) >= 3
        assert any(p.id == "request_rate" for p in dash.panels)

    def test_database_dashboard(self):
        """Test database dashboard template."""
        dash = create_database_dashboard()

        assert dash.id == "database"
        assert len(dash.panels) >= 2
        assert any(p.id == "connections" for p in dash.panels)


class TestTrendCalculation:
    """Tests for trend calculation."""

    @pytest.mark.asyncio
    async def test_trend_up(self):
        """Test upward trend detection."""
        store = TimeSeriesStore()
        now = datetime.utcnow()

        # Increasing values
        for i in range(10):
            store.record(
                "metric",
                10.0 + i * 5,  # 10, 15, 20, ...
                timestamp=now - timedelta(minutes=10-i),
            )

        service = DashboardService(store=store)
        dash = (
            dashboard("test", "Test")
            .panel("trend_panel", "Trend", "graph")
            .metric("metric")
            .end_panel()
            .build()
        )
        service.register_dashboard(dash)

        data = await service.get_panel_data("test", "trend_panel")

        assert data.trend == "up"
        assert data.change_percent > 0

    @pytest.mark.asyncio
    async def test_trend_down(self):
        """Test downward trend detection."""
        store = TimeSeriesStore()
        now = datetime.utcnow()

        # Decreasing values
        for i in range(10):
            store.record(
                "metric",
                100.0 - i * 5,  # 100, 95, 90, ...
                timestamp=now - timedelta(minutes=10-i),
            )

        service = DashboardService(store=store)
        dash = (
            dashboard("test", "Test")
            .panel("trend_panel", "Trend", "graph")
            .metric("metric")
            .end_panel()
            .build()
        )
        service.register_dashboard(dash)

        data = await service.get_panel_data("test", "trend_panel")

        assert data.trend == "down"
        assert data.change_percent < 0

    @pytest.mark.asyncio
    async def test_trend_stable(self):
        """Test stable trend detection."""
        store = TimeSeriesStore()
        now = datetime.utcnow()

        # Stable values with minor fluctuation
        for i in range(10):
            store.record(
                "metric",
                50.0 + (i % 3) - 1,  # Fluctuates around 50
                timestamp=now - timedelta(minutes=10-i),
            )

        service = DashboardService(store=store)
        dash = (
            dashboard("test", "Test")
            .panel("trend_panel", "Trend", "graph")
            .metric("metric")
            .end_panel()
            .build()
        )
        service.register_dashboard(dash)

        data = await service.get_panel_data("test", "trend_panel")

        assert data.trend == "stable"
