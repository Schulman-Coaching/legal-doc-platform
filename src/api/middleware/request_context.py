"""
Request Context Middleware
==========================
Request tracking, correlation IDs, and context propagation.
"""

from __future__ import annotations

import asyncio
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, TYPE_CHECKING
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

if TYPE_CHECKING:
    from starlette.types import ASGIApp


# =============================================================================
# Context Variables
# =============================================================================

# Context variables for async-safe request tracking
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
_request_context: ContextVar[Optional["RequestContext"]] = ContextVar("request_context", default=None)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RequestContext:
    """Full request context."""
    request_id: str
    correlation_id: str
    method: str
    path: str
    query_string: str
    client_ip: str
    user_agent: str
    start_time: float
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    session_id: Optional[str] = None
    api_version: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "method": self.method,
            "path": self.path,
            "query_string": self.query_string,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "elapsed_ms": self.elapsed_ms,
            "user_id": self.user_id,
            "organization_id": self.organization_id,
            "session_id": self.session_id,
            "api_version": self.api_version,
            **self.extra,
        }

    def to_log_dict(self) -> dict[str, Any]:
        """Convert to logging-friendly dictionary."""
        return {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "method": self.method,
            "path": self.path,
            "client_ip": self.client_ip,
            "user_id": self.user_id,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


@dataclass
class ResponseContext:
    """Response context for logging."""
    status_code: int
    content_length: Optional[int] = None
    content_type: Optional[str] = None
    elapsed_ms: float = 0


# =============================================================================
# Context Accessors
# =============================================================================

def get_request_id() -> Optional[str]:
    """Get current request ID."""
    return _request_id.get()


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return _correlation_id.get()


def get_request_context() -> Optional[RequestContext]:
    """Get full request context."""
    return _request_context.get()


def set_request_context(context: RequestContext) -> None:
    """Set request context."""
    _request_id.set(context.request_id)
    _correlation_id.set(context.correlation_id)
    _request_context.set(context)


def update_context(**kwargs) -> None:
    """Update current request context."""
    context = _request_context.get()
    if context:
        for key, value in kwargs.items():
            if hasattr(context, key):
                setattr(context, key, value)
            else:
                context.extra[key] = value


def clear_request_context() -> None:
    """Clear request context."""
    _request_id.set(None)
    _correlation_id.set(None)
    _request_context.set(None)


# =============================================================================
# Request Context Middleware
# =============================================================================

class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request context management.

    Features:
    - Generates unique request IDs
    - Propagates correlation IDs across services
    - Tracks request timing
    - Captures client information
    - Integrates with logging
    """

    # Header names
    REQUEST_ID_HEADER = "X-Request-ID"
    CORRELATION_ID_HEADER = "X-Correlation-ID"
    CLIENT_IP_HEADERS = ["X-Forwarded-For", "X-Real-IP"]

    def __init__(
        self,
        app: "ASGIApp",
        generate_request_id: Optional[Callable[[], str]] = None,
        on_request_start: Optional[Callable[[RequestContext], None]] = None,
        on_request_end: Optional[Callable[[RequestContext, ResponseContext], None]] = None,
    ):
        super().__init__(app)
        self._generate_request_id = generate_request_id or self._default_generate_id
        self._on_request_start = on_request_start
        self._on_request_end = on_request_end

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request context for each request."""
        # Generate or extract IDs
        request_id = (
            request.headers.get(self.REQUEST_ID_HEADER) or
            self._generate_request_id()
        )

        correlation_id = (
            request.headers.get(self.CORRELATION_ID_HEADER) or
            request_id  # Use request ID as correlation if not provided
        )

        # Get client IP
        client_ip = self._get_client_ip(request)

        # Get user agent
        user_agent = request.headers.get("User-Agent", "")

        # Extract API version from path
        api_version = self._extract_api_version(request.url.path)

        # Create context
        context = RequestContext(
            request_id=request_id,
            correlation_id=correlation_id,
            method=request.method,
            path=request.url.path,
            query_string=str(request.url.query),
            client_ip=client_ip,
            user_agent=user_agent,
            start_time=time.time(),
            api_version=api_version,
        )

        # Set in context vars
        set_request_context(context)

        # Store in request state for easy access
        request.state.request_id = request_id
        request.state.correlation_id = correlation_id
        request.state.context = context

        # Callback for request start
        if self._on_request_start:
            try:
                self._on_request_start(context)
            except Exception:
                pass

        # Process request
        try:
            response = await call_next(request)

            # Create response context
            response_ctx = ResponseContext(
                status_code=response.status_code,
                content_length=int(response.headers.get("Content-Length", 0)),
                content_type=response.headers.get("Content-Type"),
                elapsed_ms=context.elapsed_ms,
            )

            # Add context headers to response
            response.headers[self.REQUEST_ID_HEADER] = request_id
            response.headers[self.CORRELATION_ID_HEADER] = correlation_id
            response.headers["X-Response-Time"] = f"{context.elapsed_ms:.2f}ms"

            # Callback for request end
            if self._on_request_end:
                try:
                    self._on_request_end(context, response_ctx)
                except Exception:
                    pass

            return response

        finally:
            # Clear context
            clear_request_context()

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request headers."""
        for header in self.CLIENT_IP_HEADERS:
            value = request.headers.get(header)
            if value:
                # X-Forwarded-For can contain multiple IPs
                return value.split(",")[0].strip()

        # Fall back to direct connection
        if request.client:
            return request.client.host

        return "unknown"

    def _extract_api_version(self, path: str) -> Optional[str]:
        """Extract API version from path."""
        parts = path.strip("/").split("/")
        for part in parts:
            if part.startswith("v") and len(part) >= 2:
                try:
                    int(part[1:])
                    return part
                except ValueError:
                    continue
        return None

    @staticmethod
    def _default_generate_id() -> str:
        """Generate unique request ID."""
        return str(uuid4())


# =============================================================================
# Logging Integration
# =============================================================================

class ContextLogFilter:
    """
    Logging filter that adds request context to log records.

    Usage:
        import logging
        handler = logging.StreamHandler()
        handler.addFilter(ContextLogFilter())
    """

    def filter(self, record) -> bool:
        """Add context to log record."""
        context = get_request_context()

        if context:
            record.request_id = context.request_id
            record.correlation_id = context.correlation_id
            record.client_ip = context.client_ip
            record.user_id = context.user_id or "-"
            record.method = context.method
            record.path = context.path
        else:
            record.request_id = "-"
            record.correlation_id = "-"
            record.client_ip = "-"
            record.user_id = "-"
            record.method = "-"
            record.path = "-"

        return True


class ContextLogFormatter:
    """
    Log formatter with request context.

    Usage:
        formatter = ContextLogFormatter(
            "%(asctime)s [%(request_id)s] %(levelname)s %(message)s"
        )
    """

    def __init__(self, fmt: str):
        import logging
        self._formatter = logging.Formatter(fmt)

    def format(self, record) -> str:
        """Format log record with context."""
        # Ensure context fields exist
        context = get_request_context()
        if context:
            record.request_id = context.request_id
            record.correlation_id = context.correlation_id
        else:
            record.request_id = "-"
            record.correlation_id = "-"

        return self._formatter.format(record)


# =============================================================================
# Distributed Tracing Support
# =============================================================================

@dataclass
class TraceContext:
    """Distributed tracing context (W3C Trace Context compatible)."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    sampled: bool = True
    trace_state: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> Optional["TraceContext"]:
        """Extract trace context from headers."""
        traceparent = headers.get("traceparent")
        if not traceparent:
            return None

        try:
            parts = traceparent.split("-")
            if len(parts) != 4:
                return None

            version, trace_id, span_id, flags = parts
            sampled = int(flags, 16) & 0x01 == 0x01

            # Parse tracestate
            trace_state = {}
            tracestate = headers.get("tracestate", "")
            for item in tracestate.split(","):
                if "=" in item:
                    key, value = item.split("=", 1)
                    trace_state[key.strip()] = value.strip()

            return cls(
                trace_id=trace_id,
                span_id=span_id,
                sampled=sampled,
                trace_state=trace_state,
            )

        except Exception:
            return None

    def to_headers(self) -> dict[str, str]:
        """Convert to headers."""
        flags = "01" if self.sampled else "00"
        traceparent = f"00-{self.trace_id}-{self.span_id}-{flags}"

        headers = {"traceparent": traceparent}

        if self.trace_state:
            tracestate = ",".join(f"{k}={v}" for k, v in self.trace_state.items())
            headers["tracestate"] = tracestate

        return headers

    def create_child_span(self) -> "TraceContext":
        """Create child span context."""
        import secrets
        new_span_id = secrets.token_hex(8)

        return TraceContext(
            trace_id=self.trace_id,
            span_id=new_span_id,
            parent_span_id=self.span_id,
            sampled=self.sampled,
            trace_state=self.trace_state.copy(),
        )


# =============================================================================
# FastAPI Dependencies
# =============================================================================

async def get_request_id_dep(request: Request) -> str:
    """
    FastAPI dependency to get request ID.

    Usage:
        @app.get("/items")
        async def get_items(request_id: str = Depends(get_request_id_dep)):
            logger.info(f"Request {request_id}: Getting items")
    """
    return getattr(request.state, "request_id", None) or get_request_id() or str(uuid4())


async def get_context_dep(request: Request) -> RequestContext:
    """
    FastAPI dependency to get full request context.

    Usage:
        @app.get("/items")
        async def get_items(ctx: RequestContext = Depends(get_context_dep)):
            logger.info(f"User {ctx.user_id} requesting items")
    """
    return getattr(request.state, "context", None) or get_request_context()


# =============================================================================
# Context Manager
# =============================================================================

class RequestContextManager:
    """
    Context manager for manual request context.

    Usage:
        async with RequestContextManager(request_id="test-123") as ctx:
            # Code with context
            pass
    """

    def __init__(
        self,
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **extra,
    ):
        self.context = RequestContext(
            request_id=request_id or str(uuid4()),
            correlation_id=correlation_id or request_id or str(uuid4()),
            method="INTERNAL",
            path="/",
            query_string="",
            client_ip="127.0.0.1",
            user_agent="internal",
            start_time=time.time(),
            user_id=user_id,
            extra=extra,
        )
        self._token = None

    async def __aenter__(self) -> RequestContext:
        """Enter context."""
        set_request_context(self.context)
        return self.context

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context."""
        clear_request_context()

    def __enter__(self) -> RequestContext:
        """Sync enter."""
        set_request_context(self.context)
        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Sync exit."""
        clear_request_context()
