"""
Tests for structured logging service.
"""

import pytest
import json
import io
import logging
from datetime import datetime
from unittest.mock import patch, MagicMock
from ..services.logging import (
    StructuredLogger,
    BoundLogger,
    get_logger,
    configure_logging,
    JSONFormatter,
    ConsoleFormatter,
    LoggingMiddleware,
    log_call,
)
from ..models import LogLevel, LogRecord
from ..config import LoggingConfig


class TestLogRecord:
    """Tests for LogRecord model."""

    def test_log_record_creation(self):
        """Test LogRecord creation."""
        record = LogRecord(
            message="Test message",
            level=LogLevel.INFO,
            logger_name="test",
        )
        assert record.message == "Test message"
        assert record.level == LogLevel.INFO
        assert record.logger_name == "test"
        assert record.timestamp is not None

    def test_log_record_with_extra(self):
        """Test LogRecord with extra fields."""
        record = LogRecord(
            message="User action",
            level=LogLevel.INFO,
            logger_name="test",
            extra={"user_id": "123", "action": "login"},
        )
        assert record.extra["user_id"] == "123"
        assert record.extra["action"] == "login"

    def test_log_record_with_trace_context(self):
        """Test LogRecord with trace context."""
        record = LogRecord(
            message="Traced operation",
            level=LogLevel.INFO,
            logger_name="test",
            trace_id="abc123",
            span_id="def456",
        )
        assert record.trace_id == "abc123"
        assert record.span_id == "def456"

    def test_log_record_to_dict(self):
        """Test LogRecord serialization."""
        record = LogRecord(
            message="Test",
            level=LogLevel.WARNING,
            logger_name="mylogger",
        )
        data = record.to_dict()

        assert data["message"] == "Test"
        assert data["level"] == "WARNING"
        assert data["logger"] == "mylogger"
        assert "timestamp" in data


class TestStructuredLogger:
    """Tests for StructuredLogger."""

    def test_logger_creation(self):
        """Test StructuredLogger creation."""
        logger = StructuredLogger("test")
        assert logger.name == "test"

    def test_logger_with_config(self):
        """Test StructuredLogger with configuration."""
        config = LoggingConfig(
            level="DEBUG",
            format="json",
        )
        logger = StructuredLogger("test", config=config)
        assert logger.config.level == "DEBUG"

    def test_logger_info(self):
        """Test info level logging."""
        logger = StructuredLogger("test")
        records = []
        logger._emit = lambda r: records.append(r)

        logger.info("Test message", key="value")

        assert len(records) == 1
        assert records[0].message == "Test message"
        assert records[0].level == LogLevel.INFO
        assert records[0].extra.get("key") == "value"

    def test_logger_debug(self):
        """Test debug level logging."""
        logger = StructuredLogger("test")
        records = []
        logger._emit = lambda r: records.append(r)

        logger.debug("Debug message")

        assert len(records) == 1
        assert records[0].level == LogLevel.DEBUG

    def test_logger_warning(self):
        """Test warning level logging."""
        logger = StructuredLogger("test")
        records = []
        logger._emit = lambda r: records.append(r)

        logger.warning("Warning message")

        assert len(records) == 1
        assert records[0].level == LogLevel.WARNING

    def test_logger_error(self):
        """Test error level logging."""
        logger = StructuredLogger("test")
        records = []
        logger._emit = lambda r: records.append(r)

        logger.error("Error message", error_code="E001")

        assert len(records) == 1
        assert records[0].level == LogLevel.ERROR
        assert records[0].extra.get("error_code") == "E001"

    def test_logger_critical(self):
        """Test critical level logging."""
        logger = StructuredLogger("test")
        records = []
        logger._emit = lambda r: records.append(r)

        logger.critical("Critical error")

        assert len(records) == 1
        assert records[0].level == LogLevel.CRITICAL

    def test_logger_exception(self):
        """Test exception logging."""
        logger = StructuredLogger("test")
        records = []
        logger._emit = lambda r: records.append(r)

        try:
            raise ValueError("Test error")
        except ValueError:
            logger.exception("An error occurred")

        assert len(records) == 1
        assert records[0].level == LogLevel.ERROR
        assert "exception" in records[0].extra or "exc_info" in records[0].extra


class TestBoundLogger:
    """Tests for BoundLogger."""

    def test_bound_logger_creation(self):
        """Test BoundLogger creation."""
        base_logger = StructuredLogger("test")
        bound = BoundLogger(base_logger, user_id="123")

        assert bound.context["user_id"] == "123"

    def test_bound_logger_logging(self):
        """Test BoundLogger includes context."""
        base_logger = StructuredLogger("test")
        records = []
        base_logger._emit = lambda r: records.append(r)

        bound = base_logger.bind(user_id="123", request_id="req-456")
        bound.info("User action", action="click")

        assert len(records) == 1
        assert records[0].extra.get("user_id") == "123"
        assert records[0].extra.get("request_id") == "req-456"
        assert records[0].extra.get("action") == "click"

    def test_bound_logger_nested_binding(self):
        """Test nested binding."""
        base_logger = StructuredLogger("test")
        records = []
        base_logger._emit = lambda r: records.append(r)

        bound1 = base_logger.bind(service="api")
        bound2 = bound1.bind(endpoint="/users")

        bound2.info("Request processed")

        assert len(records) == 1
        assert records[0].extra.get("service") == "api"
        assert records[0].extra.get("endpoint") == "/users"


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger(self):
        """Test get_logger returns logger."""
        logger = get_logger("myapp")
        assert logger is not None
        assert logger.name == "myapp"

    def test_get_logger_caching(self):
        """Test get_logger returns same instance."""
        logger1 = get_logger("cached")
        logger2 = get_logger("cached")
        assert logger1 is logger2


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_json_formatter_basic(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        record = LogRecord(
            message="Test message",
            level=LogLevel.INFO,
            logger_name="test",
        )
        output = formatter.format(record)

        data = json.loads(output)
        assert data["message"] == "Test message"
        assert data["level"] == "INFO"
        assert data["logger"] == "test"

    def test_json_formatter_with_extra(self):
        """Test JSON formatting with extra fields."""
        formatter = JSONFormatter()
        record = LogRecord(
            message="User login",
            level=LogLevel.INFO,
            logger_name="auth",
            extra={"user_id": "123", "ip": "192.168.1.1"},
        )
        output = formatter.format(record)

        data = json.loads(output)
        assert data["user_id"] == "123"
        assert data["ip"] == "192.168.1.1"

    def test_json_formatter_with_trace_context(self):
        """Test JSON formatting with trace context."""
        formatter = JSONFormatter()
        record = LogRecord(
            message="Traced",
            level=LogLevel.INFO,
            logger_name="test",
            trace_id="trace123",
            span_id="span456",
        )
        output = formatter.format(record)

        data = json.loads(output)
        assert data["trace_id"] == "trace123"
        assert data["span_id"] == "span456"


class TestConsoleFormatter:
    """Tests for ConsoleFormatter."""

    def test_console_formatter_basic(self):
        """Test basic console formatting."""
        formatter = ConsoleFormatter()
        record = LogRecord(
            message="Test message",
            level=LogLevel.INFO,
            logger_name="test",
        )
        output = formatter.format(record)

        assert "INFO" in output
        assert "test" in output
        assert "Test message" in output

    def test_console_formatter_with_color(self):
        """Test console formatting with color."""
        formatter = ConsoleFormatter(use_color=True)
        record = LogRecord(
            message="Warning",
            level=LogLevel.WARNING,
            logger_name="test",
        )
        output = formatter.format(record)

        # Output should contain ANSI color codes
        assert "Warning" in output

    def test_console_formatter_error_level(self):
        """Test console formatting for error level."""
        formatter = ConsoleFormatter()
        record = LogRecord(
            message="Error occurred",
            level=LogLevel.ERROR,
            logger_name="test",
        )
        output = formatter.format(record)

        assert "ERROR" in output


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_basic(self):
        """Test basic logging configuration."""
        config = LoggingConfig(level="INFO", format="json")
        configure_logging(config)

        logger = get_logger("configured")
        assert logger is not None

    def test_configure_logging_debug(self):
        """Test debug level configuration."""
        config = LoggingConfig(level="DEBUG")
        configure_logging(config)

        # Debug logging should be enabled
        logger = get_logger("debug_test")
        assert logger is not None


class TestLogCallDecorator:
    """Tests for log_call decorator."""

    def test_log_call_decorator(self):
        """Test @log_call decorator."""
        logger = StructuredLogger("test")
        records = []
        logger._emit = lambda r: records.append(r)

        @log_call(logger)
        def my_function(x, y):
            return x + y

        result = my_function(1, 2)
        assert result == 3

        # Should have at least one log record
        assert len(records) >= 1

    def test_log_call_with_result(self):
        """Test @log_call logs result."""
        logger = StructuredLogger("test")
        records = []
        logger._emit = lambda r: records.append(r)

        @log_call(logger, log_result=True)
        def get_value():
            return "secret_value"

        result = get_value()
        assert result == "secret_value"

        # Check for result in logs
        assert any("secret_value" in str(r.extra) for r in records)

    def test_log_call_on_exception(self):
        """Test @log_call logs exceptions."""
        logger = StructuredLogger("test")
        records = []
        logger._emit = lambda r: records.append(r)

        @log_call(logger)
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        # Should have error log
        assert any(r.level == LogLevel.ERROR for r in records)

    @pytest.mark.asyncio
    async def test_log_call_async(self):
        """Test @log_call with async function."""
        import asyncio

        logger = StructuredLogger("test")
        records = []
        logger._emit = lambda r: records.append(r)

        @log_call(logger)
        async def async_function():
            await asyncio.sleep(0.01)
            return "async result"

        result = await async_function()
        assert result == "async result"
        assert len(records) >= 1


class TestLoggingMiddleware:
    """Tests for LoggingMiddleware."""

    @pytest.mark.asyncio
    async def test_middleware_logs_request(self):
        """Test middleware logs HTTP request."""
        logger = StructuredLogger("test")
        records = []
        logger._emit = lambda r: records.append(r)

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"OK"})

        middleware = LoggingMiddleware(app, logger)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/test",
            "headers": [],
        }

        await middleware(scope, lambda: {"type": "http.request"}, lambda m: None)

        # Should have logged the request
        assert len(records) >= 1

    @pytest.mark.asyncio
    async def test_middleware_logs_status_code(self):
        """Test middleware logs response status."""
        logger = StructuredLogger("test")
        records = []
        logger._emit = lambda r: records.append(r)

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 404})
            await send({"type": "http.response.body", "body": b"Not Found"})

        middleware = LoggingMiddleware(app, logger)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/missing",
            "headers": [],
        }

        await middleware(scope, lambda: {"type": "http.request"}, lambda m: None)

        # Check for status code in logs
        assert any(r.extra.get("status_code") == 404 for r in records)

    @pytest.mark.asyncio
    async def test_middleware_logs_duration(self):
        """Test middleware logs request duration."""
        import asyncio

        logger = StructuredLogger("test")
        records = []
        logger._emit = lambda r: records.append(r)

        async def slow_app(scope, receive, send):
            await asyncio.sleep(0.01)
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b""})

        middleware = LoggingMiddleware(app=slow_app, logger=logger)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/slow",
            "headers": [],
        }

        await middleware(scope, lambda: {"type": "http.request"}, lambda m: None)

        # Check for duration in logs
        assert any("duration" in str(r.extra) for r in records)

    @pytest.mark.asyncio
    async def test_middleware_ignores_non_http(self):
        """Test middleware ignores non-HTTP requests."""
        logger = StructuredLogger("test")
        records = []
        logger._emit = lambda r: records.append(r)

        called = False

        async def app(scope, receive, send):
            nonlocal called
            called = True

        middleware = LoggingMiddleware(app, logger)

        scope = {"type": "websocket"}
        await middleware(scope, None, None)

        assert called
        # Should not log for non-HTTP
        request_logs = [r for r in records if "http" in str(r.extra).lower()]
        assert len(request_logs) == 0


class TestLoggingIntegration:
    """Integration tests for logging."""

    def test_full_logging_workflow(self):
        """Test complete logging workflow."""
        config = LoggingConfig(
            level="DEBUG",
            format="json",
        )
        logger = StructuredLogger("integration_test", config=config)
        records = []
        logger._emit = lambda r: records.append(r)

        # Create bound logger
        request_logger = logger.bind(request_id="req-123")

        # Log at various levels
        request_logger.debug("Starting request")
        request_logger.info("Processing", step=1)
        request_logger.info("Processing", step=2)
        request_logger.warning("Slow operation", duration_ms=500)

        try:
            raise ValueError("Simulated error")
        except ValueError:
            request_logger.error("Error occurred", error_type="validation")

        request_logger.info("Request completed", status="success")

        # Verify logs
        assert len(records) == 6

        # All logs should have request_id
        for record in records:
            assert record.extra.get("request_id") == "req-123"

        # Check levels
        assert records[0].level == LogLevel.DEBUG
        assert records[1].level == LogLevel.INFO
        assert records[3].level == LogLevel.WARNING
        assert records[4].level == LogLevel.ERROR

    def test_trace_correlation(self):
        """Test log correlation with traces."""
        from ..services.tracing import Tracer, get_current_span

        tracer = Tracer()
        logger = StructuredLogger("traced")
        records = []
        logger._emit = lambda r: records.append(r)

        with tracer.start_span("operation") as span:
            # Manually set trace context (normally done automatically)
            current_span = get_current_span()
            logger.info(
                "Operation in progress",
                trace_id=current_span.context.trace_id if current_span else None,
                span_id=current_span.context.span_id if current_span else None,
            )

        assert len(records) == 1
        # Should have trace context
        assert records[0].extra.get("trace_id") is not None
