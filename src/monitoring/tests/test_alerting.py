"""
Tests for alerting service.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
from ..services.alerting import (
    AlertManager,
    AlertEvaluator,
    AlertMonitor,
    NotificationChannel,
    EmailNotificationChannel,
    SlackNotificationChannel,
    PagerDutyNotificationChannel,
    WebhookNotificationChannel,
    ConsoleNotificationChannel,
    AlertRuleBuilder,
    alert,
)
from ..models import Alert, AlertRule, AlertSeverity, AlertStatus
from ..config import (
    AlertingConfig,
    EmailNotificationConfig,
    SlackNotificationConfig,
    PagerDutyNotificationConfig,
)


class TestAlertRule:
    """Tests for AlertRule model."""

    def test_rule_creation(self):
        """Test AlertRule creation."""
        rule = AlertRule(
            name="high_latency",
            metric="http_request_duration_seconds",
            operator=">",
            threshold=1.0,
            severity=AlertSeverity.WARNING,
        )
        assert rule.name == "high_latency"
        assert rule.operator == ">"
        assert rule.threshold == 1.0

    def test_rule_with_labels(self):
        """Test AlertRule with label filters."""
        rule = AlertRule(
            name="api_errors",
            metric="http_errors_total",
            operator=">",
            threshold=10,
            severity=AlertSeverity.CRITICAL,
            labels={"endpoint": "/api/v1"},
        )
        assert rule.labels == {"endpoint": "/api/v1"}

    def test_rule_with_for_duration(self):
        """Test AlertRule with for duration."""
        rule = AlertRule(
            name="sustained_high_cpu",
            metric="cpu_percent",
            operator=">",
            threshold=90,
            severity=AlertSeverity.WARNING,
            for_duration=300,  # 5 minutes
        )
        assert rule.for_duration == 300


class TestAlertRuleBuilder:
    """Tests for AlertRuleBuilder."""

    def test_builder_basic(self):
        """Test basic rule building."""
        rule = (
            alert("high_latency")
            .metric("latency_seconds")
            .above(1.0)
            .warning()
            .description("Latency is too high")
            .build()
        )

        assert rule.name == "high_latency"
        assert rule.metric == "latency_seconds"
        assert rule.operator == ">"
        assert rule.threshold == 1.0
        assert rule.severity == AlertSeverity.WARNING

    def test_builder_critical(self):
        """Test critical severity."""
        rule = (
            alert("error_rate")
            .metric("error_rate")
            .above(0.05)
            .critical()
            .build()
        )
        assert rule.severity == AlertSeverity.CRITICAL

    def test_builder_below_threshold(self):
        """Test below threshold."""
        rule = (
            alert("low_disk")
            .metric("disk_free_percent")
            .below(10)
            .warning()
            .build()
        )
        assert rule.operator == "<"
        assert rule.threshold == 10

    def test_builder_equals(self):
        """Test equals operator."""
        rule = (
            alert("status_check")
            .metric("service_up")
            .equals(0)
            .critical()
            .build()
        )
        assert rule.operator == "=="
        assert rule.threshold == 0

    def test_builder_with_labels(self):
        """Test with label filters."""
        rule = (
            alert("endpoint_errors")
            .metric("http_errors")
            .above(10)
            .label("method", "POST")
            .label("endpoint", "/api/create")
            .warning()
            .build()
        )
        assert rule.labels == {"method": "POST", "endpoint": "/api/create"}

    def test_builder_with_annotations(self):
        """Test with annotations."""
        rule = (
            alert("high_latency")
            .metric("latency")
            .above(1.0)
            .annotate("runbook", "https://wiki/runbooks/latency")
            .annotate("team", "platform")
            .build()
        )
        assert rule.annotations["runbook"] == "https://wiki/runbooks/latency"

    def test_builder_for_duration(self):
        """Test for duration."""
        rule = (
            alert("sustained_error")
            .metric("error_rate")
            .above(0.01)
            .for_duration(300)
            .build()
        )
        assert rule.for_duration == 300


class TestAlertEvaluator:
    """Tests for AlertEvaluator."""

    def test_evaluator_greater_than(self):
        """Test greater than evaluation."""
        evaluator = AlertEvaluator()
        rule = AlertRule(
            name="test",
            metric="value",
            operator=">",
            threshold=10,
        )

        is_firing, reason = evaluator.evaluate(rule, 15)
        assert is_firing is True
        assert "15 > 10" in reason

        is_firing, _ = evaluator.evaluate(rule, 5)
        assert is_firing is False

    def test_evaluator_less_than(self):
        """Test less than evaluation."""
        evaluator = AlertEvaluator()
        rule = AlertRule(
            name="test",
            metric="value",
            operator="<",
            threshold=10,
        )

        is_firing, _ = evaluator.evaluate(rule, 5)
        assert is_firing is True

        is_firing, _ = evaluator.evaluate(rule, 15)
        assert is_firing is False

    def test_evaluator_greater_equal(self):
        """Test greater than or equal evaluation."""
        evaluator = AlertEvaluator()
        rule = AlertRule(
            name="test",
            metric="value",
            operator=">=",
            threshold=10,
        )

        is_firing, _ = evaluator.evaluate(rule, 10)
        assert is_firing is True

        is_firing, _ = evaluator.evaluate(rule, 9)
        assert is_firing is False

    def test_evaluator_less_equal(self):
        """Test less than or equal evaluation."""
        evaluator = AlertEvaluator()
        rule = AlertRule(
            name="test",
            metric="value",
            operator="<=",
            threshold=10,
        )

        is_firing, _ = evaluator.evaluate(rule, 10)
        assert is_firing is True

        is_firing, _ = evaluator.evaluate(rule, 11)
        assert is_firing is False

    def test_evaluator_equals(self):
        """Test equals evaluation."""
        evaluator = AlertEvaluator()
        rule = AlertRule(
            name="test",
            metric="value",
            operator="==",
            threshold=0,
        )

        is_firing, _ = evaluator.evaluate(rule, 0)
        assert is_firing is True

        is_firing, _ = evaluator.evaluate(rule, 1)
        assert is_firing is False

    def test_evaluator_not_equals(self):
        """Test not equals evaluation."""
        evaluator = AlertEvaluator()
        rule = AlertRule(
            name="test",
            metric="value",
            operator="!=",
            threshold=0,
        )

        is_firing, _ = evaluator.evaluate(rule, 1)
        assert is_firing is True

        is_firing, _ = evaluator.evaluate(rule, 0)
        assert is_firing is False

    def test_evaluator_unknown_operator(self):
        """Test unknown operator."""
        evaluator = AlertEvaluator()
        rule = AlertRule(
            name="test",
            metric="value",
            operator="~",  # Invalid
            threshold=10,
        )

        is_firing, reason = evaluator.evaluate(rule, 5)
        assert is_firing is False
        assert "Unknown operator" in reason

    def test_evaluator_average(self):
        """Test average evaluation."""
        evaluator = AlertEvaluator()
        rule = AlertRule(
            name="test",
            metric="value",
            operator=">",
            threshold=50,
        )

        # Average of [40, 50, 60, 70] = 55 > 50
        is_firing, _ = evaluator.evaluate_average(rule, [40, 50, 60, 70])
        assert is_firing is True

        # Average of [30, 40, 50] = 40 < 50
        is_firing, _ = evaluator.evaluate_average(rule, [30, 40, 50])
        assert is_firing is False


class TestAlertManager:
    """Tests for AlertManager."""

    def test_manager_creation(self):
        """Test AlertManager creation."""
        manager = AlertManager()
        assert manager is not None

    def test_add_rule(self):
        """Test adding alert rules."""
        manager = AlertManager()
        rule = AlertRule(
            name="test_rule",
            metric="test_metric",
            operator=">",
            threshold=10,
        )
        manager.add_rule(rule)

        rules = manager.get_rules()
        assert len(rules) == 1
        assert rules[0].name == "test_rule"

    def test_remove_rule(self):
        """Test removing alert rules."""
        manager = AlertManager()
        rule = AlertRule(
            name="test_rule",
            metric="test_metric",
            operator=">",
            threshold=10,
        )
        manager.add_rule(rule)
        manager.remove_rule("test_rule")

        rules = manager.get_rules()
        assert len(rules) == 0

    def test_add_channel(self):
        """Test adding notification channels."""
        manager = AlertManager()
        channel = ConsoleNotificationChannel()
        manager.add_channel(channel)

        assert len(manager._channels) == 1

    def test_add_channel_with_severities(self):
        """Test adding channel for specific severities."""
        manager = AlertManager()
        channel = ConsoleNotificationChannel()
        manager.add_channel(channel, severities=[AlertSeverity.CRITICAL])

        assert channel.name in manager._routes[AlertSeverity.CRITICAL]
        assert channel.name not in manager._routes[AlertSeverity.WARNING]

    @pytest.mark.asyncio
    async def test_evaluate_fires_alert(self):
        """Test alert evaluation fires alert."""
        manager = AlertManager()
        manager.add_rule(AlertRule(
            name="high_value",
            metric="test_metric",
            operator=">",
            threshold=10,
            severity=AlertSeverity.WARNING,
            for_duration=0,  # Fire immediately
        ))

        alerts = await manager.evaluate("test_metric", 15)

        assert len(alerts) == 1
        assert alerts[0].name == "high_value"

    @pytest.mark.asyncio
    async def test_evaluate_no_fire_below_threshold(self):
        """Test alert doesn't fire below threshold."""
        manager = AlertManager()
        manager.add_rule(AlertRule(
            name="high_value",
            metric="test_metric",
            operator=">",
            threshold=10,
            severity=AlertSeverity.WARNING,
        ))

        alerts = await manager.evaluate("test_metric", 5)

        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_evaluate_with_labels(self):
        """Test alert evaluation with label matching."""
        manager = AlertManager()
        manager.add_rule(AlertRule(
            name="api_errors",
            metric="errors",
            operator=">",
            threshold=5,
            severity=AlertSeverity.WARNING,
            labels={"endpoint": "/api"},
            for_duration=0,
        ))

        # Matching labels - should fire
        alerts = await manager.evaluate(
            "errors", 10,
            labels={"endpoint": "/api"}
        )
        assert len(alerts) == 1

        # Non-matching labels - should not fire
        alerts = await manager.evaluate(
            "errors", 10,
            labels={"endpoint": "/other"}
        )
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_evaluate_resolves_alert(self):
        """Test alert resolution when condition clears."""
        manager = AlertManager()
        manager.add_rule(AlertRule(
            name="test_alert",
            metric="test_metric",
            operator=">",
            threshold=10,
            severity=AlertSeverity.WARNING,
            for_duration=0,
        ))

        # Fire alert
        await manager.evaluate("test_metric", 15)
        assert len(manager.get_active_alerts()) == 1

        # Resolve alert
        alerts = await manager.evaluate("test_metric", 5)

        assert len(manager.get_active_alerts()) == 0
        assert len(alerts) == 1
        # Check that the alert in history shows resolved
        history = manager.get_alert_history()
        assert len(history) >= 1

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self):
        """Test acknowledging an alert."""
        manager = AlertManager()
        manager.add_rule(AlertRule(
            name="test",
            metric="test",
            operator=">",
            threshold=10,
            for_duration=0,
        ))

        await manager.evaluate("test", 15)
        active = manager.get_active_alerts()
        assert len(active) == 1

        alert_id = active[0].id
        result = await manager.acknowledge(alert_id, acknowledged_by="admin")

        assert result is True

    @pytest.mark.asyncio
    async def test_manual_resolve_alert(self):
        """Test manually resolving an alert."""
        manager = AlertManager()
        manager.add_rule(AlertRule(
            name="test",
            metric="test",
            operator=">",
            threshold=10,
            for_duration=0,
        ))

        await manager.evaluate("test", 15)
        active = manager.get_active_alerts()
        alert_id = active[0].id

        result = await manager.resolve(alert_id)

        assert result is True
        assert len(manager.get_active_alerts()) == 0


class TestNotificationChannels:
    """Tests for notification channels."""

    def test_console_channel(self):
        """Test ConsoleNotificationChannel."""
        channel = ConsoleNotificationChannel()
        assert channel.name == "console"

    @pytest.mark.asyncio
    async def test_console_channel_send(self, capsys):
        """Test ConsoleNotificationChannel send."""
        channel = ConsoleNotificationChannel()
        alert_obj = Alert(
            id="test123", message="Test alert",
            name="Test Alert",
            description="Test description",
            severity=AlertSeverity.WARNING,
        )

        result = await channel.send(alert_obj)

        assert result is True
        captured = capsys.readouterr()
        assert "Test Alert" in captured.out

    def test_webhook_channel_creation(self):
        """Test WebhookNotificationChannel creation."""
        channel = WebhookNotificationChannel(
            name="custom_webhook",
            url="http://example.com/webhook",
            headers={"Authorization": "Bearer token"},
        )
        assert channel.name == "custom_webhook"
        assert channel.url == "http://example.com/webhook"

    @pytest.mark.asyncio
    async def test_webhook_channel_disabled(self):
        """Test disabled webhook channel."""
        channel = WebhookNotificationChannel(
            name="webhook",
            url="http://example.com",
            enabled=False,
        )
        alert_obj = Alert(
            id="test", message="Test",
            name="Test",
            description="Test",
            severity=AlertSeverity.INFO,
        )

        result = await channel.send(alert_obj)
        assert result is False

    def test_email_channel_creation(self):
        """Test EmailNotificationChannel creation."""
        config = EmailNotificationConfig(
            smtp_host="smtp.example.com",
            smtp_port=587,
            from_address="alerts@example.com",
            recipients=["team@example.com"],
        )
        channel = EmailNotificationChannel(config)
        assert channel.name == "email"

    @pytest.mark.asyncio
    async def test_email_channel_disabled(self):
        """Test disabled email channel."""
        config = EmailNotificationConfig(
            enabled=False,
            smtp_host="smtp.example.com",
            from_address="alerts@example.com",
            recipients=["team@example.com"],
        )
        channel = EmailNotificationChannel(config)
        alert_obj = Alert(
            id="test", message="Test",
            name="Test",
            description="Test",
            severity=AlertSeverity.INFO,
        )

        result = await channel.send(alert_obj)
        assert result is False

    def test_slack_channel_creation(self):
        """Test SlackNotificationChannel creation."""
        config = SlackNotificationConfig(
            webhook_url="https://hooks.slack.com/services/xxx",
            channel="#alerts",
        )
        channel = SlackNotificationChannel(config)
        assert channel.name == "slack"

    @pytest.mark.asyncio
    async def test_slack_channel_disabled(self):
        """Test disabled Slack channel."""
        config = SlackNotificationConfig(
            enabled=False,
            webhook_url="https://hooks.slack.com/services/xxx",
            channel="#alerts",
        )
        channel = SlackNotificationChannel(config)
        alert_obj = Alert(
            id="test", message="Test",
            name="Test",
            description="Test",
            severity=AlertSeverity.INFO,
        )

        result = await channel.send(alert_obj)
        assert result is False

    def test_pagerduty_channel_creation(self):
        """Test PagerDutyNotificationChannel creation."""
        config = PagerDutyNotificationConfig(
            routing_key="xxxx",
        )
        channel = PagerDutyNotificationChannel(config)
        assert channel.name == "pagerduty"

    @pytest.mark.asyncio
    async def test_pagerduty_channel_disabled(self):
        """Test disabled PagerDuty channel."""
        config = PagerDutyNotificationConfig(
            enabled=False,
            routing_key="xxxx",
        )
        channel = PagerDutyNotificationChannel(config)
        alert_obj = Alert(
            id="test", message="Test",
            name="Test",
            description="Test",
            severity=AlertSeverity.INFO,
        )

        result = await channel.send(alert_obj)
        assert result is False


class TestAlertMonitor:
    """Tests for AlertMonitor."""

    @pytest.mark.asyncio
    async def test_monitor_start_stop(self):
        """Test starting and stopping monitor."""
        manager = AlertManager()
        manager.add_rule(AlertRule(
            name="test",
            metric="test",
            operator=">",
            threshold=10,
        ))

        def get_metrics():
            return {"test": 5}

        monitor = AlertMonitor(
            manager,
            metrics_func=get_metrics,
            interval=0.1,
        )

        await monitor.start()
        assert monitor._running is True

        await asyncio.sleep(0.15)

        await monitor.stop()
        assert monitor._running is False

    @pytest.mark.asyncio
    async def test_monitor_evaluates_metrics(self):
        """Test monitor evaluates metrics."""
        manager = AlertManager()
        manager.add_rule(AlertRule(
            name="high_value",
            metric="test_metric",
            operator=">",
            threshold=10,
            for_duration=0,
        ))

        evaluation_count = 0

        def get_metrics():
            nonlocal evaluation_count
            evaluation_count += 1
            return {"test_metric": 15}

        monitor = AlertMonitor(
            manager,
            metrics_func=get_metrics,
            interval=0.05,
        )

        await monitor.start()
        await asyncio.sleep(0.12)
        await monitor.stop()

        # Should have evaluated at least once
        assert evaluation_count >= 1
        # Should have fired alert
        assert len(manager.get_active_alerts()) >= 1


class TestAlertIntegration:
    """Integration tests for alerting."""

    @pytest.mark.asyncio
    async def test_full_alert_workflow(self):
        """Test complete alert workflow."""
        # Setup
        manager = AlertManager()
        notifications = []

        class TestChannel(NotificationChannel):
            @property
            def name(self):
                return "test"

            async def send(self, alert):
                notifications.append(alert)
                return True

        manager.add_channel(TestChannel())
        manager.add_rule(AlertRule(
            name="cpu_high",
            metric="cpu_percent",
            operator=">",
            threshold=90,
            severity=AlertSeverity.CRITICAL,
            description="CPU usage is too high",
            for_duration=0,
        ))

        # Fire alert
        await manager.evaluate("cpu_percent", 95)

        assert len(manager.get_active_alerts()) == 1
        assert len(notifications) == 1
        assert notifications[0].name == "cpu_high"

        # Value goes back to normal - resolve
        await manager.evaluate("cpu_percent", 50)

        assert len(manager.get_active_alerts()) == 0
        # Should have sent resolution notification
        assert len(notifications) >= 2

    @pytest.mark.asyncio
    async def test_multiple_rules_same_metric(self):
        """Test multiple rules for same metric."""
        manager = AlertManager()

        manager.add_rule(AlertRule(
            name="cpu_warning",
            metric="cpu_percent",
            operator=">",
            threshold=70,
            severity=AlertSeverity.WARNING,
            for_duration=0,
        ))
        manager.add_rule(AlertRule(
            name="cpu_critical",
            metric="cpu_percent",
            operator=">",
            threshold=90,
            severity=AlertSeverity.CRITICAL,
            for_duration=0,
        ))

        # 75% - should fire warning only
        await manager.evaluate("cpu_percent", 75)
        active = manager.get_active_alerts()
        assert len(active) == 1
        assert active[0].severity == AlertSeverity.WARNING

        # 95% - should fire critical too
        await manager.evaluate("cpu_percent", 95)
        active = manager.get_active_alerts()
        assert len(active) == 2
