"""
Alerting Service
================
Alert management with evaluation, notification channels, and escalation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
from enum import Enum

from ..config import (
    AlertingConfig,
    EmailNotificationConfig,
    SlackNotificationConfig,
    PagerDutyNotificationConfig,
)
from ..models import (
    Alert,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    MetricType,
)


# =============================================================================
# Notification Channels
# =============================================================================

class NotificationChannel(ABC):
    """Base class for notification channels."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Channel name."""
        pass

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Send notification. Returns True if successful."""
        pass


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""

    def __init__(self, config: EmailNotificationConfig):
        self.config = config
        self._name = "email"

    @property
    def name(self) -> str:
        return self._name

    async def send(self, alert: Alert) -> bool:
        """Send email notification."""
        if not self.config.enabled:
            return False

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = self._format_subject(alert)
            msg["From"] = self.config.from_address
            msg["To"] = ", ".join(self.config.recipients)

            # Plain text body
            text_body = self._format_text_body(alert)
            msg.attach(MIMEText(text_body, "plain"))

            # HTML body
            html_body = self._format_html_body(alert)
            msg.attach(MIMEText(html_body, "html"))

            # Send
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                if self.config.use_tls:
                    server.starttls()
                if self.config.username and self.config.password:
                    server.login(self.config.username, self.config.password)
                server.sendmail(
                    self.config.from_address,
                    self.config.recipients,
                    msg.as_string(),
                )

            return True

        except Exception:
            return False

    def _format_subject(self, alert: Alert) -> str:
        """Format email subject."""
        severity_emoji = {
            AlertSeverity.CRITICAL: "🔴",
            AlertSeverity.WARNING: "🟠",
            AlertSeverity.INFO: "🔵",
        }
        emoji = severity_emoji.get(alert.severity, "⚪")
        return f"{emoji} [{alert.severity.value.upper()}] {alert.name}"

    def _format_text_body(self, alert: Alert) -> str:
        """Format plain text email body."""
        lines = [
            f"Alert: {alert.name}",
            f"Severity: {alert.severity.value.upper()}",
            f"State: {alert.status.value}",
            f"Description: {alert.message}",
            f"",
            f"Started: {alert.started_at.isoformat()}",
        ]

        if alert.value is not None:
            lines.append(f"Current Value: {alert.value}")

        if alert.labels:
            lines.append("")
            lines.append("Labels:")
            for k, v in alert.labels.items():
                lines.append(f"  {k}: {v}")

        if alert.annotations:
            lines.append("")
            lines.append("Annotations:")
            for k, v in alert.annotations.items():
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)

    def _format_html_body(self, alert: Alert) -> str:
        """Format HTML email body."""
        severity_color = {
            AlertSeverity.CRITICAL: "#dc3545",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.INFO: "#17a2b8",
        }
        color = severity_color.get(alert.severity, "#6c757d")

        labels_html = ""
        if alert.labels:
            labels_html = "<br>".join(f"<b>{k}:</b> {v}" for k, v in alert.labels.items())

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <div style="border-left: 4px solid {color}; padding-left: 15px;">
                <h2 style="color: {color}; margin: 0;">{alert.name}</h2>
                <p style="color: #666; margin: 5px 0;">
                    <strong>Severity:</strong> {alert.severity.value.upper()} |
                    <strong>State:</strong> {alert.status.value}
                </p>
            </div>
            <div style="margin-top: 20px;">
                <p><strong>Description:</strong> {alert.message}</p>
                <p><strong>Started:</strong> {alert.started_at.isoformat()}</p>
                {f'<p><strong>Current Value:</strong> {alert.value}</p>' if alert.value else ''}
            </div>
            {f'<div style="margin-top: 20px;"><strong>Labels:</strong><br>{labels_html}</div>' if labels_html else ''}
        </body>
        </html>
        """


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel."""

    def __init__(self, config: SlackNotificationConfig):
        self.config = config
        self._name = "slack"

    @property
    def name(self) -> str:
        return self._name

    async def send(self, alert: Alert) -> bool:
        """Send Slack notification."""
        if not self.config.enabled:
            return False

        try:
            import urllib.request

            payload = self._build_payload(alert)
            data = json.dumps(payload).encode("utf-8")

            req = urllib.request.Request(
                self.config.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
            return True

        except Exception:
            return False

    def _build_payload(self, alert: Alert) -> dict:
        """Build Slack message payload."""
        severity_color = {
            AlertSeverity.CRITICAL: "#dc3545",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.INFO: "#17a2b8",
        }
        color = severity_color.get(alert.severity, "#6c757d")

        severity_emoji = {
            AlertSeverity.CRITICAL: ":red_circle:",
            AlertSeverity.WARNING: ":large_orange_circle:",
            AlertSeverity.INFO: ":large_blue_circle:",
        }
        emoji = severity_emoji.get(alert.severity, ":white_circle:")

        fields = [
            {
                "title": "Severity",
                "value": alert.severity.value.upper(),
                "short": True,
            },
            {
                "title": "State",
                "value": alert.status.value,
                "short": True,
            },
        ]

        if alert.value is not None:
            fields.append({
                "title": "Value",
                "value": str(alert.value),
                "short": True,
            })

        # Add labels as fields
        for k, v in (alert.labels or {}).items():
            fields.append({
                "title": k,
                "value": str(v),
                "short": True,
            })

        payload = {
            "channel": self.config.channel,
            "username": self.config.username,
            "icon_emoji": self.config.icon_emoji,
            "attachments": [
                {
                    "color": color,
                    "title": f"{emoji} {alert.name}",
                    "text": alert.message,
                    "fields": fields,
                    "footer": f"Alert ID: {alert.id}",
                    "ts": int(alert.started_at.timestamp()),
                }
            ],
        }

        return payload


class PagerDutyNotificationChannel(NotificationChannel):
    """PagerDuty notification channel."""

    def __init__(self, config: PagerDutyNotificationConfig):
        self.config = config
        self._name = "pagerduty"

    @property
    def name(self) -> str:
        return self._name

    async def send(self, alert: Alert) -> bool:
        """Send PagerDuty notification."""
        if not self.config.enabled:
            return False

        try:
            import urllib.request

            # Map alert status to PagerDuty event action
            action = "trigger"
            if alert.status == AlertStatus.RESOLVED:
                action = "resolve"
            elif alert.status == AlertStatus.ACKNOWLEDGED:
                action = "acknowledge"

            # Map severity
            pd_severity = {
                AlertSeverity.CRITICAL: "critical",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.INFO: "info",
            }

            payload = {
                "routing_key": self.config.routing_key,
                "event_action": action,
                "dedup_key": alert.id,
                "payload": {
                    "summary": f"[{alert.severity.value.upper()}] {alert.name}: {alert.message}",
                    "source": self.config.source,
                    "severity": pd_severity.get(alert.severity, "warning"),
                    "timestamp": alert.started_at.isoformat(),
                    "custom_details": {
                        "alert_id": alert.id,
                        "value": alert.value,
                        "labels": alert.labels,
                        "annotations": alert.annotations,
                    },
                },
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                "https://events.pagerduty.com/v2/enqueue",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
            return True

        except Exception:
            return False


class WebhookNotificationChannel(NotificationChannel):
    """Generic webhook notification channel."""

    def __init__(
        self,
        name: str,
        url: str,
        headers: Optional[dict[str, str]] = None,
        enabled: bool = True,
    ):
        self._name = name
        self.url = url
        self.headers = headers or {}
        self.enabled = enabled

    @property
    def name(self) -> str:
        return self._name

    async def send(self, alert: Alert) -> bool:
        """Send webhook notification."""
        if not self.enabled:
            return False

        try:
            import urllib.request

            payload = {
                "alert_id": alert.id,
                "name": alert.name,
                "description": alert.message,
                "severity": alert.severity.value,
                "state": alert.status.value,
                "value": alert.value,
                "labels": alert.labels,
                "annotations": alert.annotations,
                "started_at": alert.started_at.isoformat(),
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    **self.headers,
                },
            )
            urllib.request.urlopen(req, timeout=10)
            return True

        except Exception:
            return False


class ConsoleNotificationChannel(NotificationChannel):
    """Console notification channel (for testing/debugging)."""

    def __init__(self, name: str = "console"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def send(self, alert: Alert) -> bool:
        """Print alert to console."""
        severity_symbol = {
            AlertSeverity.CRITICAL: "🔴",
            AlertSeverity.WARNING: "🟠",
            AlertSeverity.INFO: "🔵",
        }
        symbol = severity_symbol.get(alert.severity, "⚪")

        print(f"\n{symbol} [{alert.severity.value.upper()}] {alert.name}")
        print(f"   State: {alert.status.value}")
        print(f"   Description: {alert.message}")
        if alert.value is not None:
            print(f"   Value: {alert.value}")
        if alert.labels:
            print(f"   Labels: {alert.labels}")
        print()

        return True


# =============================================================================
# Alert Evaluator
# =============================================================================

class AlertEvaluator:
    """
    Evaluates alert rules against metric values.

    Supports:
    - Threshold alerts (>, <, >=, <=, ==, !=)
    - Rate of change alerts
    - Anomaly detection (simple)
    """

    def __init__(self):
        self._operators = {
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }

    def evaluate(
        self,
        rule: AlertRule,
        current_value: float,
        history: Optional[list[float]] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Evaluate rule against current value.

        Returns (is_firing, reason).
        """
        operator = rule.operator
        threshold = rule.threshold

        if operator not in self._operators:
            return False, f"Unknown operator: {operator}"

        is_firing = self._operators[operator](current_value, threshold)

        if is_firing:
            reason = f"Value {current_value} {operator} {threshold}"
            return True, reason

        return False, None

    def evaluate_rate(
        self,
        rule: AlertRule,
        values: list[tuple[datetime, float]],
        window: timedelta = timedelta(minutes=5),
    ) -> tuple[bool, Optional[str]]:
        """
        Evaluate rate of change.

        Returns (is_firing, reason).
        """
        if len(values) < 2:
            return False, None

        # Filter to window
        cutoff = datetime.utcnow() - window
        recent = [(t, v) for t, v in values if t >= cutoff]

        if len(recent) < 2:
            return False, None

        # Calculate rate
        first_time, first_value = recent[0]
        last_time, last_value = recent[-1]

        time_diff = (last_time - first_time).total_seconds()
        if time_diff <= 0:
            return False, None

        rate = (last_value - first_value) / time_diff

        # Evaluate
        return self.evaluate(rule, rate)

    def evaluate_average(
        self,
        rule: AlertRule,
        values: list[float],
    ) -> tuple[bool, Optional[str]]:
        """
        Evaluate average over values.

        Returns (is_firing, reason).
        """
        if not values:
            return False, None

        avg = sum(values) / len(values)
        return self.evaluate(rule, avg)


# =============================================================================
# Alert Manager
# =============================================================================

@dataclass
class AlertRuleState:
    """Internal state for an alert rule."""
    rule: AlertRule
    firing_since: Optional[datetime] = None
    pending_since: Optional[datetime] = None
    last_value: Optional[float] = None
    last_evaluated: Optional[datetime] = None
    notification_count: int = 0
    last_notified: Optional[datetime] = None


class AlertManager:
    """
    Manages alert rules, evaluation, and notification.

    Usage:
        manager = AlertManager(config)
        manager.add_channel(SlackNotificationChannel(slack_config))

        manager.add_rule(AlertRule(
            name="high_latency",
            metric="http_request_duration_seconds",
            operator=">",
            threshold=1.0,
            severity=AlertSeverity.WARNING,
        ))

        # In your metrics collection loop:
        await manager.evaluate("http_request_duration_seconds", current_value)
    """

    def __init__(self, config: Optional[AlertingConfig] = None):
        self.config = config or AlertingConfig()
        self._rules: dict[str, AlertRule] = {}
        self._states: dict[str, AlertRuleState] = {}
        self._channels: list[NotificationChannel] = []
        self._active_alerts: dict[str, Alert] = {}
        self._alert_history: list[Alert] = []
        self._lock = threading.Lock()
        self._evaluator = AlertEvaluator()

        # Route map: severity -> channel names
        self._routes: dict[AlertSeverity, list[str]] = {
            AlertSeverity.CRITICAL: [],
            AlertSeverity.WARNING: [],
            AlertSeverity.INFO: [],
        }

    def add_channel(
        self,
        channel: NotificationChannel,
        severities: Optional[list[AlertSeverity]] = None,
    ) -> None:
        """Add notification channel."""
        with self._lock:
            self._channels.append(channel)

            # Route channel to severities
            if severities is None:
                # Route to all severities
                for severity in AlertSeverity:
                    if channel.name not in self._routes[severity]:
                        self._routes[severity].append(channel.name)
            else:
                for severity in severities:
                    if channel.name not in self._routes[severity]:
                        self._routes[severity].append(channel.name)

    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        with self._lock:
            self._rules[rule.name] = rule
            self._states[rule.name] = AlertRuleState(rule=rule)

    def remove_rule(self, name: str) -> None:
        """Remove alert rule."""
        with self._lock:
            self._rules.pop(name, None)
            self._states.pop(name, None)

    def get_rules(self) -> list[AlertRule]:
        """Get all rules."""
        with self._lock:
            return list(self._rules.values())

    def get_active_alerts(self) -> list[Alert]:
        """Get currently active alerts."""
        with self._lock:
            return list(self._active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> list[Alert]:
        """Get alert history."""
        with self._lock:
            return self._alert_history[-limit:]

    async def evaluate(
        self,
        metric_name: str,
        value: float,
        labels: Optional[dict[str, str]] = None,
    ) -> list[Alert]:
        """
        Evaluate all rules against a metric value.

        Returns list of newly fired or resolved alerts.
        """
        triggered_alerts: list[Alert] = []

        with self._lock:
            # Find rules for this metric
            rules_to_evaluate = [
                r for r in self._rules.values()
                if r.metric == metric_name
            ]

        for rule in rules_to_evaluate:
            # Check label filters
            if rule.labels:
                if labels is None:
                    continue
                if not all(labels.get(k) == v for k, v in rule.labels.items()):
                    continue

            # Evaluate rule
            is_firing, reason = self._evaluator.evaluate(rule, value)

            with self._lock:
                state = self._states.get(rule.name)
                if not state:
                    continue

                state.last_value = value
                state.last_evaluated = datetime.utcnow()

                alert = await self._process_evaluation(
                    rule, state, is_firing, value, reason, labels
                )

                if alert:
                    triggered_alerts.append(alert)

        return triggered_alerts

    async def _process_evaluation(
        self,
        rule: AlertRule,
        state: AlertRuleState,
        is_firing: bool,
        value: float,
        reason: Optional[str],
        labels: Optional[dict[str, str]],
    ) -> Optional[Alert]:
        """Process evaluation result and manage alert state."""
        now = datetime.utcnow()
        alert_id = self._generate_alert_id(rule, labels)

        if is_firing:
            # Check if already active
            if alert_id in self._active_alerts:
                # Update existing alert
                alert = self._active_alerts[alert_id]
                alert.value = value
                alert.updated_at = now
                return None

            # Check pending duration
            if state.pending_since is None:
                state.pending_since = now

            pending_duration = (now - state.pending_since).total_seconds()
            for_duration = rule.for_duration or self.config.default_for_duration

            if pending_duration < for_duration:
                # Still pending
                return None

            # Fire alert
            alert = Alert(
                id=alert_id,
                name=rule.name,
                message=rule.description or f"Alert: {rule.metric} {rule.operator} {rule.threshold}",
                severity=rule.severity,
                status=AlertStatus.FIRING,
                value=value,
                labels=labels or {},
                annotations=rule.annotations or {},
                started_at=state.pending_since,
            )

            self._active_alerts[alert_id] = alert
            state.firing_since = state.pending_since
            state.pending_since = None

            # Send notifications
            await self._notify(alert)

            return alert

        else:
            # Not firing - reset pending
            state.pending_since = None

            # Check if we should resolve
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = now
                alert.value = value

                # Move to history
                del self._active_alerts[alert_id]
                self._alert_history.append(alert)

                # Limit history
                if len(self._alert_history) > 1000:
                    self._alert_history = self._alert_history[-500:]

                # Notify resolution
                await self._notify(alert)

                state.firing_since = None

                return alert

        return None

    def _generate_alert_id(
        self,
        rule: AlertRule,
        labels: Optional[dict[str, str]],
    ) -> str:
        """Generate unique alert ID."""
        parts = [rule.name]
        if labels:
            for k, v in sorted(labels.items()):
                parts.append(f"{k}={v}")
        id_string = "|".join(parts)
        return hashlib.sha256(id_string.encode()).hexdigest()[:16]

    async def _notify(self, alert: Alert) -> None:
        """Send notifications for alert."""
        # Get channels for this severity
        channel_names = self._routes.get(alert.severity, [])

        for channel in self._channels:
            if channel.name in channel_names:
                try:
                    await channel.send(alert)
                except Exception:
                    pass

    async def acknowledge(
        self,
        alert_id: str,
        acknowledged_by: Optional[str] = None,
    ) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            alert = self._active_alerts.get(alert_id)
            if not alert:
                return False

            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by

            return True

    async def resolve(self, alert_id: str) -> bool:
        """Manually resolve an alert."""
        with self._lock:
            alert = self._active_alerts.get(alert_id)
            if not alert:
                return False

            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()

            del self._active_alerts[alert_id]
            self._alert_history.append(alert)

            return True

    async def silence(
        self,
        rule_name: str,
        duration: timedelta,
        reason: Optional[str] = None,
    ) -> bool:
        """Silence a rule for a duration."""
        with self._lock:
            if rule_name not in self._rules:
                return False

            rule = self._rules[rule_name]
            rule.silenced_until = datetime.utcnow() + duration
            return True


# =============================================================================
# Alert Monitor (Background Evaluation)
# =============================================================================

class AlertMonitor:
    """
    Background alert monitor that periodically evaluates rules.

    Usage:
        monitor = AlertMonitor(
            alert_manager,
            metrics_func=lambda: get_current_metrics(),
            interval=60,
        )
        await monitor.start()
    """

    def __init__(
        self,
        manager: AlertManager,
        metrics_func: Callable[[], dict[str, float]],
        interval: float = 60.0,
    ):
        self.manager = manager
        self.metrics_func = metrics_func
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start background monitoring."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                # Get current metrics
                metrics = self.metrics_func()
                if asyncio.iscoroutine(metrics):
                    metrics = await metrics

                # Evaluate each metric
                for metric_name, value in metrics.items():
                    try:
                        await self.manager.evaluate(metric_name, value)
                    except Exception:
                        pass

            except Exception:
                pass

            await asyncio.sleep(self.interval)


# =============================================================================
# Alert Rule Builder
# =============================================================================

class AlertRuleBuilder:
    """Fluent builder for alert rules."""

    def __init__(self, name: str):
        self._name = name
        self._metric: str = ""
        self._operator: str = ">"
        self._threshold: float = 0
        self._severity: AlertSeverity = AlertSeverity.WARNING
        self._description: Optional[str] = None
        self._labels: dict[str, str] = {}
        self._annotations: dict[str, str] = {}
        self._for_duration: float = 0

    def metric(self, name: str) -> "AlertRuleBuilder":
        """Set metric to monitor."""
        self._metric = name
        return self

    def above(self, threshold: float) -> "AlertRuleBuilder":
        """Alert when above threshold."""
        self._operator = ">"
        self._threshold = threshold
        return self

    def below(self, threshold: float) -> "AlertRuleBuilder":
        """Alert when below threshold."""
        self._operator = "<"
        self._threshold = threshold
        return self

    def equals(self, value: float) -> "AlertRuleBuilder":
        """Alert when equals value."""
        self._operator = "=="
        self._threshold = value
        return self

    def severity(self, level: AlertSeverity) -> "AlertRuleBuilder":
        """Set alert severity."""
        self._severity = level
        return self

    def critical(self) -> "AlertRuleBuilder":
        """Set severity to critical."""
        self._severity = AlertSeverity.CRITICAL
        return self

    def warning(self) -> "AlertRuleBuilder":
        """Set severity to warning."""
        self._severity = AlertSeverity.WARNING
        return self

    def info(self) -> "AlertRuleBuilder":
        """Set severity to info."""
        self._severity = AlertSeverity.INFO
        return self

    def description(self, text: str) -> "AlertRuleBuilder":
        """Set description."""
        self._description = text
        return self

    def label(self, key: str, value: str) -> "AlertRuleBuilder":
        """Add label filter."""
        self._labels[key] = value
        return self

    def annotate(self, key: str, value: str) -> "AlertRuleBuilder":
        """Add annotation."""
        self._annotations[key] = value
        return self

    def for_duration(self, seconds: float) -> "AlertRuleBuilder":
        """Set pending duration before firing."""
        self._for_duration = seconds
        return self

    def build(self) -> AlertRule:
        """Build the alert rule."""
        return AlertRule(
            name=self._name,
            metric=self._metric,
            operator=self._operator,
            threshold=self._threshold,
            severity=self._severity,
            description=self._description,
            labels=self._labels if self._labels else None,
            annotations=self._annotations if self._annotations else None,
            for_duration=self._for_duration,
        )


def alert(name: str) -> AlertRuleBuilder:
    """Create an alert rule builder."""
    return AlertRuleBuilder(name)
