"""
Distributed Tracing Service
===========================
OpenTelemetry-compatible distributed tracing with context propagation.
"""

from __future__ import annotations

import asyncio
import secrets
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Iterator, Optional, TypeVar

from ..config import TracingConfig
from ..models import Span, SpanContext, SpanEvent, SpanStatus


T = TypeVar("T")


# =============================================================================
# Context Management
# =============================================================================

# Context variable for current span
_current_span: ContextVar[Optional[Span]] = ContextVar("current_span", default=None)


def get_current_span() -> Optional[Span]:
    """Get current active span."""
    return _current_span.get()


def set_current_span(span: Optional[Span]) -> None:
    """Set current active span."""
    _current_span.set(span)


# =============================================================================
# Span Builder
# =============================================================================

@dataclass
class SpanBuilder:
    """Builder for creating spans."""
    tracer: "Tracer"
    name: str
    kind: str = "internal"
    parent: Optional[Span] = None
    links: list[SpanContext] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None

    def set_kind(self, kind: str) -> "SpanBuilder":
        """Set span kind."""
        self.kind = kind
        return self

    def set_parent(self, parent: Span) -> "SpanBuilder":
        """Set parent span."""
        self.parent = parent
        return self

    def add_link(self, context: SpanContext) -> "SpanBuilder":
        """Add link to another span."""
        self.links.append(context)
        return self

    def set_attribute(self, key: str, value: Any) -> "SpanBuilder":
        """Set attribute."""
        self.attributes[key] = value
        return self

    def set_attributes(self, attributes: dict[str, Any]) -> "SpanBuilder":
        """Set multiple attributes."""
        self.attributes.update(attributes)
        return self

    def start(self) -> Span:
        """Start the span."""
        return self.tracer._start_span(self)


# =============================================================================
# Tracer
# =============================================================================

class Tracer:
    """
    Distributed tracer for creating and managing spans.

    Compatible with OpenTelemetry concepts.
    """

    def __init__(self, config: Optional[TracingConfig] = None):
        self.config = config or TracingConfig()
        self._spans: list[Span] = []
        self._exporters: list["SpanExporter"] = []
        self._lock = threading.Lock()
        self._sampling_rate = self.config.sample_rate

    def span(self, name: str) -> SpanBuilder:
        """Create a span builder."""
        return SpanBuilder(tracer=self, name=name)

    @contextmanager
    def start_span(
        self,
        name: str,
        kind: str = "internal",
        attributes: Optional[dict[str, Any]] = None,
    ) -> Iterator[Span]:
        """
        Context manager for creating spans.

        Usage:
            with tracer.start_span("operation_name") as span:
                span.set_attribute("key", "value")
                # ... do work ...
        """
        builder = SpanBuilder(
            tracer=self,
            name=name,
            kind=kind,
            attributes=attributes or {},
        )
        span = self._start_span(builder)

        # Set as current
        token = _current_span.set(span)

        try:
            yield span
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.set_attribute("exception.type", type(e).__name__)
            span.set_attribute("exception.message", str(e))
            raise
        finally:
            span.end()
            _current_span.reset(token)
            self._record_span(span)

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        kind: str = "internal",
        attributes: Optional[dict[str, Any]] = None,
    ) -> Iterator[Span]:
        """Alias for start_span that sets span as current."""
        with self.start_span(name, kind, attributes) as span:
            yield span

    def _start_span(self, builder: SpanBuilder) -> Span:
        """Internal span creation."""
        # Check sampling
        if not self._should_sample():
            # Return no-op span
            return self._create_noop_span(builder.name)

        # Get parent context
        parent_context = None
        if builder.parent:
            parent_context = builder.parent.context
        else:
            # Check for current span
            current = get_current_span()
            if current:
                parent_context = current.context

        # Create span context
        if parent_context:
            context = SpanContext(
                trace_id=parent_context.trace_id,
                span_id=secrets.token_hex(8),
                trace_flags=parent_context.trace_flags,
            )
        else:
            context = SpanContext.generate()

        span = Span(
            name=builder.name,
            context=context,
            parent_context=parent_context,
            kind=builder.kind,
            start_time=builder.start_time or datetime.utcnow(),
            attributes=builder.attributes.copy(),
            service_name=self.config.service_name,
            resource_attributes={
                "service.name": self.config.service_name,
                "service.version": self.config.service_version,
                "deployment.environment": self.config.environment,
                **self.config.resource_attributes,
            },
        )

        return span

    def _create_noop_span(self, name: str) -> Span:
        """Create a no-op span for when sampling is off."""
        return Span(
            name=name,
            context=SpanContext(
                trace_id="0" * 32,
                span_id="0" * 16,
                trace_flags=0,
            ),
        )

    def _should_sample(self) -> bool:
        """Check if this trace should be sampled."""
        if self._sampling_rate >= 1.0:
            return True
        if self._sampling_rate <= 0:
            return False
        import random
        return random.random() < self._sampling_rate

    def _record_span(self, span: Span) -> None:
        """Record completed span."""
        with self._lock:
            self._spans.append(span)

            # Export if we have exporters
            if self._exporters:
                for exporter in self._exporters:
                    try:
                        exporter.export([span])
                    except Exception:
                        pass

            # Limit stored spans
            if len(self._spans) > 10000:
                self._spans = self._spans[-5000:]

    def add_exporter(self, exporter: "SpanExporter") -> None:
        """Add span exporter."""
        with self._lock:
            self._exporters.append(exporter)

    def get_spans(self, limit: int = 100) -> list[Span]:
        """Get recorded spans."""
        with self._lock:
            return self._spans[-limit:]

    def clear_spans(self) -> None:
        """Clear recorded spans."""
        with self._lock:
            self._spans.clear()

    def extract_context(self, headers: dict[str, str]) -> Optional[SpanContext]:
        """Extract span context from headers."""
        traceparent = headers.get("traceparent")
        if traceparent:
            return SpanContext.from_traceparent(traceparent)
        return None

    def inject_context(self, span: Span, headers: dict[str, str]) -> None:
        """Inject span context into headers."""
        headers["traceparent"] = span.context.to_traceparent()
        if span.context.trace_state:
            tracestate = ",".join(
                f"{k}={v}" for k, v in span.context.trace_state.items()
            )
            headers["tracestate"] = tracestate


# =============================================================================
# Span Exporters
# =============================================================================

class SpanExporter:
    """Base class for span exporters."""

    def export(self, spans: list[Span]) -> None:
        """Export spans."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Shutdown exporter."""
        pass


class ConsoleSpanExporter(SpanExporter):
    """Export spans to console."""

    def export(self, spans: list[Span]) -> None:
        """Export spans to console."""
        for span in spans:
            print(f"[TRACE] {span.service_name} | {span.name} | "
                  f"{span.trace_id[:8]}:{span.span_id} | "
                  f"{span.duration_ms:.2f}ms | {span.status.value}")


class InMemorySpanExporter(SpanExporter):
    """Export spans to memory (for testing)."""

    def __init__(self):
        self.spans: list[Span] = []
        self._lock = threading.Lock()

    def export(self, spans: list[Span]) -> None:
        """Export spans to memory."""
        with self._lock:
            self.spans.extend(spans)

    def get_spans(self) -> list[Span]:
        """Get exported spans."""
        with self._lock:
            return self.spans.copy()

    def clear(self) -> None:
        """Clear exported spans."""
        with self._lock:
            self.spans.clear()


class OTLPSpanExporter(SpanExporter):
    """Export spans via OTLP (OpenTelemetry Protocol)."""

    def __init__(self, endpoint: str, headers: Optional[dict[str, str]] = None):
        self.endpoint = endpoint
        self.headers = headers or {}
        self._batch: list[Span] = []
        self._lock = threading.Lock()

    def export(self, spans: list[Span]) -> None:
        """Export spans via OTLP."""
        import json

        # Convert spans to OTLP format
        resource_spans = self._to_otlp(spans)

        try:
            import urllib.request

            data = json.dumps({"resourceSpans": resource_spans}).encode()
            req = urllib.request.Request(
                self.endpoint,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    **self.headers,
                },
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass  # Silent fail for non-blocking export

    def _to_otlp(self, spans: list[Span]) -> list[dict]:
        """Convert spans to OTLP format."""
        # Group by resource
        by_resource: dict[str, list[Span]] = {}
        for span in spans:
            key = span.service_name
            if key not in by_resource:
                by_resource[key] = []
            by_resource[key].append(span)

        resource_spans = []
        for service_name, service_spans in by_resource.items():
            resource_spans.append({
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": service_name}},
                    ],
                },
                "scopeSpans": [{
                    "scope": {"name": "legal-doc-platform"},
                    "spans": [s.to_dict() for s in service_spans],
                }],
            })

        return resource_spans


# =============================================================================
# Decorators
# =============================================================================

def trace(
    name: Optional[str] = None,
    kind: str = "internal",
    attributes: Optional[dict[str, Any]] = None,
):
    """
    Decorator to trace function execution.

    Usage:
        @trace("operation_name")
        def my_function():
            ...

        @trace()  # Uses function name
        async def my_async_function():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            tracer = get_default_tracer()
            with tracer.start_span(span_name, kind, attributes) as span:
                span.set_attribute("code.function", func.__name__)
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            tracer = get_default_tracer()
            with tracer.start_span(span_name, kind, attributes) as span:
                span.set_attribute("code.function", func.__name__)
                return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


def trace_method(
    kind: str = "internal",
    attributes: Optional[dict[str, Any]] = None,
):
    """
    Decorator to trace class method execution.

    Uses class.method as span name.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> T:
            span_name = f"{self.__class__.__name__}.{func.__name__}"
            tracer = get_default_tracer()
            with tracer.start_span(span_name, kind, attributes) as span:
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.class", self.__class__.__name__)
                return func(self, *args, **kwargs)

        @wraps(func)
        async def async_wrapper(self, *args, **kwargs) -> T:
            span_name = f"{self.__class__.__name__}.{func.__name__}"
            tracer = get_default_tracer()
            with tracer.start_span(span_name, kind, attributes) as span:
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.class", self.__class__.__name__)
                return await func(self, *args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


# =============================================================================
# Global Tracer
# =============================================================================

_default_tracer: Optional[Tracer] = None


def get_default_tracer() -> Tracer:
    """Get or create default tracer."""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = Tracer()
    return _default_tracer


def set_default_tracer(tracer: Tracer) -> None:
    """Set default tracer."""
    global _default_tracer
    _default_tracer = tracer


# =============================================================================
# Middleware Support
# =============================================================================

class TracingMiddleware:
    """
    ASGI middleware for request tracing.

    Usage with FastAPI:
        app.add_middleware(TracingMiddleware, tracer=tracer)
    """

    def __init__(self, app, tracer: Optional[Tracer] = None):
        self.app = app
        self.tracer = tracer or get_default_tracer()

    async def __call__(self, scope, receive, send):
        """Handle request."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract context from headers
        headers = dict(scope.get("headers", []))
        headers = {k.decode(): v.decode() for k, v in headers.items()}
        parent_context = self.tracer.extract_context(headers)

        # Create span
        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "/")
        span_name = f"{method} {path}"

        builder = SpanBuilder(
            tracer=self.tracer,
            name=span_name,
            kind="server",
            attributes={
                "http.method": method,
                "http.url": path,
                "http.scheme": scope.get("scheme", "http"),
            },
        )

        if parent_context:
            builder.parent = Span(
                name="",
                context=parent_context,
            )

        span = self.tracer._start_span(builder)
        token = _current_span.set(span)

        status_code = 500
        try:
            async def send_wrapper(message):
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                await send(message)

            await self.app(scope, receive, send_wrapper)

            span.set_attribute("http.status_code", status_code)
            if status_code >= 400:
                span.set_status(SpanStatus.ERROR, f"HTTP {status_code}")
            else:
                span.set_status(SpanStatus.OK)

        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.set_attribute("exception.type", type(e).__name__)
            raise
        finally:
            span.end()
            _current_span.reset(token)
            self.tracer._record_span(span)
