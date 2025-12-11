"""
Structured Logging Service
==========================
JSON-structured logging with trace correlation and multiple outputs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
import traceback
from datetime import datetime
from functools import wraps
from io import StringIO
from typing import Any, Callable, Optional, TextIO, TypeVar

from ..config import LoggingConfig
from ..models import LogLevel, LogRecord
from .tracing import get_current_span


T = TypeVar("T")


# =============================================================================
# Log Formatters
# =============================================================================

class JSONFormatter(logging.Formatter):
    """JSON log formatter."""

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_trace_id: bool = True,
        include_span_id: bool = True,
        include_caller: bool = False,
        mask_fields: Optional[list[str]] = None,
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_trace_id = include_trace_id
        self.include_span_id = include_span_id
        self.include_caller = include_caller
        self.mask_fields = set(f.lower() for f in (mask_fields or []))

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_dict: dict[str, Any] = {}

        if self.include_timestamp:
            log_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"

        if self.include_level:
            log_dict["level"] = record.levelname.lower()

        if self.include_logger:
            log_dict["logger"] = record.name

        log_dict["message"] = record.getMessage()

        # Add trace context
        if self.include_trace_id or self.include_span_id:
            span = get_current_span()
            if span:
                if self.include_trace_id:
                    log_dict["trace_id"] = span.trace_id
                if self.include_span_id:
                    log_dict["span_id"] = span.span_id

        # Add caller info
        if self.include_caller:
            log_dict["caller"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "asctime",
            ):
                # Mask sensitive fields
                if key.lower() in self.mask_fields:
                    value = "***MASKED***"
                log_dict[key] = value

        # Add exception info
        if record.exc_info:
            log_dict["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self._format_traceback(record.exc_info),
            }

        return json.dumps(log_dict, default=str)

    def _format_traceback(self, exc_info) -> Optional[str]:
        """Format exception traceback."""
        if not exc_info or not exc_info[2]:
            return None
        sio = StringIO()
        traceback.print_exception(exc_info[0], exc_info[1], exc_info[2], file=sio)
        return sio.getvalue()


class ConsoleFormatter(logging.Formatter):
    """Console-friendly log formatter with colors."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console."""
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET

        # Get trace context
        trace_info = ""
        span = get_current_span()
        if span:
            trace_info = f" [{span.trace_id[:8]}:{span.span_id}]"

        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname.ljust(8)

        formatted = f"{timestamp} {color}{level}{reset}{trace_info} {record.getMessage()}"

        if record.exc_info:
            formatted += "\n" + self._format_traceback(record.exc_info)

        return formatted

    def _format_traceback(self, exc_info) -> str:
        """Format exception traceback."""
        if not exc_info:
            return ""
        sio = StringIO()
        traceback.print_exception(exc_info[0], exc_info[1], exc_info[2], file=sio)
        return sio.getvalue()


class TextFormatter(logging.Formatter):
    """Plain text log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as text."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname.ljust(8)

        # Get trace context
        trace_info = ""
        span = get_current_span()
        if span:
            trace_info = f" trace_id={span.trace_id[:8]} span_id={span.span_id}"

        formatted = f"{timestamp} {level} {record.name}{trace_info} - {record.getMessage()}"

        if record.exc_info:
            formatted += "\n" + self._format_traceback(record.exc_info)

        return formatted

    def _format_traceback(self, exc_info) -> str:
        """Format exception traceback."""
        if not exc_info:
            return ""
        sio = StringIO()
        traceback.print_exception(exc_info[0], exc_info[1], exc_info[2], file=sio)
        return sio.getvalue()


# =============================================================================
# Log Handlers
# =============================================================================

class RotatingFileHandler(logging.FileHandler):
    """File handler with rotation support."""

    def __init__(
        self,
        filename: str,
        max_bytes: int = 100 * 1024 * 1024,
        backup_count: int = 5,
    ):
        super().__init__(filename)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record with rotation check."""
        with self._lock:
            self._rotate_if_needed()
            super().emit(record)

    def _rotate_if_needed(self) -> None:
        """Check and rotate log file if needed."""
        import os

        if not os.path.exists(self.baseFilename):
            return

        if os.path.getsize(self.baseFilename) < self.max_bytes:
            return

        # Close current file
        self.close()

        # Rotate files
        for i in range(self.backup_count - 1, 0, -1):
            src = f"{self.baseFilename}.{i}"
            dst = f"{self.baseFilename}.{i + 1}"
            if os.path.exists(src):
                os.rename(src, dst)

        # Rename current to .1
        os.rename(self.baseFilename, f"{self.baseFilename}.1")

        # Reopen
        self.stream = self._open()


# =============================================================================
# Structured Logger
# =============================================================================

class StructuredLogger:
    """
    Structured logger with JSON output and trace correlation.

    Usage:
        logger = StructuredLogger("my-service")
        logger.info("Request received", user_id="123", path="/api")
        logger.error("Request failed", exc_info=True)
    """

    def __init__(
        self,
        name: str,
        config: Optional[LoggingConfig] = None,
    ):
        self.name = name
        self.config = config or LoggingConfig()
        self._logger = logging.getLogger(name)
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup log handlers."""
        self._logger.setLevel(getattr(logging, self.config.level.upper()))
        self._logger.handlers.clear()

        # Create formatter
        if self.config.format == "json":
            formatter = JSONFormatter(
                include_timestamp=self.config.include_timestamp,
                include_level=self.config.include_level,
                include_logger=self.config.include_logger,
                include_trace_id=self.config.include_trace_id,
                include_span_id=self.config.include_span_id,
                include_caller=self.config.include_caller,
                mask_fields=self.config.mask_fields,
            )
        elif self.config.format == "console":
            formatter = ConsoleFormatter()
        else:
            formatter = TextFormatter()

        # Console handler
        if self.config.console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

        # File handler
        if self.config.file_enabled:
            file_handler = RotatingFileHandler(
                self.config.file_path,
                max_bytes=self.config.file_max_size_mb * 1024 * 1024,
                backup_count=self.config.file_backup_count,
            )
            file_handler.setFormatter(JSONFormatter(
                include_timestamp=self.config.include_timestamp,
                include_level=self.config.include_level,
                include_logger=self.config.include_logger,
                include_trace_id=self.config.include_trace_id,
                include_span_id=self.config.include_span_id,
                mask_fields=self.config.mask_fields,
            ))
            self._logger.addHandler(file_handler)

    def _log(
        self,
        level: int,
        message: str,
        exc_info: bool = False,
        **kwargs,
    ) -> None:
        """Internal log method."""
        # Add extra context
        extra = kwargs.copy()

        # Add trace context if not present
        span = get_current_span()
        if span:
            extra.setdefault("trace_id", span.trace_id)
            extra.setdefault("span_id", span.span_id)

        self._logger.log(level, message, exc_info=exc_info, extra=extra)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, exc_info=exc_info, **kwargs)

    def critical(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, exc_info=exc_info, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self._log(logging.ERROR, message, exc_info=True, **kwargs)

    def bind(self, **context) -> "BoundLogger":
        """Create logger with bound context."""
        return BoundLogger(self, context)

    def with_context(self, **context) -> "BoundLogger":
        """Alias for bind."""
        return self.bind(**context)


class BoundLogger:
    """Logger with bound context."""

    def __init__(self, logger: StructuredLogger, context: dict[str, Any]):
        self._logger = logger
        self._context = context

    def _merge_context(self, kwargs: dict) -> dict:
        """Merge bound context with kwargs."""
        merged = self._context.copy()
        merged.update(kwargs)
        return merged

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._logger.debug(message, **self._merge_context(kwargs))

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._logger.info(message, **self._merge_context(kwargs))

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._logger.warning(message, **self._merge_context(kwargs))

    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message."""
        self._logger.error(message, exc_info=exc_info, **self._merge_context(kwargs))

    def critical(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log critical message."""
        self._logger.critical(message, exc_info=exc_info, **self._merge_context(kwargs))

    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self._logger.exception(message, **self._merge_context(kwargs))

    def bind(self, **context) -> "BoundLogger":
        """Add more context."""
        merged = self._context.copy()
        merged.update(context)
        return BoundLogger(self._logger, merged)


# =============================================================================
# Global Logger Registry
# =============================================================================

_loggers: dict[str, StructuredLogger] = {}
_default_config: Optional[LoggingConfig] = None
_lock = threading.Lock()


def configure_logging(config: LoggingConfig) -> None:
    """Configure global logging."""
    global _default_config
    _default_config = config

    # Update existing loggers
    with _lock:
        for logger in _loggers.values():
            logger.config = config
            logger._setup_handlers()


def get_logger(name: str) -> StructuredLogger:
    """Get or create logger."""
    with _lock:
        if name not in _loggers:
            _loggers[name] = StructuredLogger(name, _default_config)
        return _loggers[name]


# =============================================================================
# Decorators
# =============================================================================

def log_call(
    logger: Optional[StructuredLogger] = None,
    level: str = "info",
    include_args: bool = False,
    include_result: bool = False,
):
    """
    Decorator to log function calls.

    Usage:
        @log_call(logger=my_logger)
        def my_function(x, y):
            return x + y
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        log_func = logger or get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            log_method = getattr(log_func, level)

            # Log entry
            entry_data = {"function": func.__name__}
            if include_args:
                entry_data["args"] = str(args)
                entry_data["kwargs"] = str(kwargs)
            log_method(f"Calling {func.__name__}", **entry_data)

            try:
                result = func(*args, **kwargs)

                # Log success
                exit_data = {"function": func.__name__, "status": "success"}
                if include_result:
                    exit_data["result"] = str(result)[:200]
                log_method(f"Completed {func.__name__}", **exit_data)

                return result

            except Exception as e:
                log_func.error(
                    f"Failed {func.__name__}",
                    function=func.__name__,
                    error=str(e),
                    exc_info=True,
                )
                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            log_method = getattr(log_func, level)

            # Log entry
            entry_data = {"function": func.__name__}
            if include_args:
                entry_data["args"] = str(args)
                entry_data["kwargs"] = str(kwargs)
            log_method(f"Calling {func.__name__}", **entry_data)

            try:
                result = await func(*args, **kwargs)

                # Log success
                exit_data = {"function": func.__name__, "status": "success"}
                if include_result:
                    exit_data["result"] = str(result)[:200]
                log_method(f"Completed {func.__name__}", **exit_data)

                return result

            except Exception as e:
                log_func.error(
                    f"Failed {func.__name__}",
                    function=func.__name__,
                    error=str(e),
                    exc_info=True,
                )
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


# =============================================================================
# Logging Middleware
# =============================================================================

class LoggingMiddleware:
    """
    ASGI middleware for request logging.

    Usage with FastAPI:
        app.add_middleware(LoggingMiddleware, logger=logger)
    """

    def __init__(
        self,
        app,
        logger: Optional[StructuredLogger] = None,
        log_request_body: bool = False,
        log_response_body: bool = False,
    ):
        self.app = app
        self.logger = logger or get_logger("http")
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body

    async def __call__(self, scope, receive, send):
        """Handle request."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "/")

        # Log request
        self.logger.info(
            f"Request: {method} {path}",
            http_method=method,
            http_path=path,
            http_scheme=scope.get("scheme", "http"),
        )

        status_code = 500
        start_time = datetime.utcnow()

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            self.logger.error(
                f"Request failed: {method} {path}",
                http_method=method,
                http_path=path,
                http_status=500,
                error=str(e),
                exc_info=True,
            )
            raise
        finally:
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.logger.info(
                f"Response: {method} {path} {status_code}",
                http_method=method,
                http_path=path,
                http_status=status_code,
                duration_ms=round(duration, 2),
            )
