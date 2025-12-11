"""
Tests for distributed tracing service.
"""

import pytest
import asyncio
from datetime import datetime
from ..services.tracing import (
    Tracer,
    SpanBuilder,
    get_current_span,
    set_current_span,
    trace,
    trace_method,
    get_default_tracer,
    set_default_tracer,
    ConsoleSpanExporter,
    InMemorySpanExporter,
    OTLPSpanExporter,
    TracingMiddleware,
)
from ..models import Span, SpanContext, SpanStatus
from ..config import TracingConfig


class TestSpanContext:
    """Tests for SpanContext."""

    def test_generate_context(self):
        """Test generating new span context."""
        ctx = SpanContext.generate()
        assert len(ctx.trace_id) == 32
        assert len(ctx.span_id) == 16
        assert ctx.trace_flags == 1

    def test_traceparent_parsing(self):
        """Test W3C traceparent parsing."""
        traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        ctx = SpanContext.from_traceparent(traceparent)

        assert ctx.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert ctx.span_id == "b7ad6b7169203331"
        assert ctx.trace_flags == 1

    def test_traceparent_generation(self):
        """Test W3C traceparent generation."""
        ctx = SpanContext(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            trace_flags=1,
        )
        traceparent = ctx.to_traceparent()
        assert traceparent == "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

    def test_invalid_traceparent(self):
        """Test invalid traceparent handling."""
        ctx = SpanContext.from_traceparent("invalid")
        assert ctx is None

        ctx = SpanContext.from_traceparent("")
        assert ctx is None


class TestSpan:
    """Tests for Span."""

    def test_span_creation(self):
        """Test basic span creation."""
        ctx = SpanContext.generate()
        span = Span(name="test_span", context=ctx)

        assert span.name == "test_span"
        assert span.context == ctx
        assert span.status == SpanStatus.UNSET

    def test_span_attributes(self):
        """Test span attribute setting."""
        ctx = SpanContext.generate()
        span = Span(name="test", context=ctx)

        span.set_attribute("string_key", "value")
        span.set_attribute("int_key", 42)
        span.set_attribute("float_key", 3.14)
        span.set_attribute("bool_key", True)

        assert span.attributes["string_key"] == "value"
        assert span.attributes["int_key"] == 42
        assert span.attributes["float_key"] == 3.14
        assert span.attributes["bool_key"] is True

    def test_span_events(self):
        """Test span event adding."""
        ctx = SpanContext.generate()
        span = Span(name="test", context=ctx)

        span.add_event("event1")
        span.add_event("event2", {"key": "value"})

        assert len(span.events) == 2
        assert span.events[0].name == "event1"
        assert span.events[1].attributes == {"key": "value"}

    def test_span_status(self):
        """Test span status setting."""
        ctx = SpanContext.generate()
        span = Span(name="test", context=ctx)

        span.set_status(SpanStatus.OK)
        assert span.status == SpanStatus.OK
        assert span.status_message is None

        span.set_status(SpanStatus.ERROR, "Something failed")
        assert span.status == SpanStatus.ERROR
        assert span.status_message == "Something failed"

    def test_span_duration(self):
        """Test span duration calculation."""
        import time

        ctx = SpanContext.generate()
        span = Span(name="test", context=ctx)
        time.sleep(0.01)
        span.end()

        assert span.end_time is not None
        assert span.duration_ms >= 10  # At least 10ms

    def test_span_serialization(self):
        """Test span to_dict serialization."""
        ctx = SpanContext.generate()
        span = Span(
            name="test",
            context=ctx,
            kind="server",
            service_name="test_service",
        )
        span.set_attribute("key", "value")
        span.end()

        data = span.to_dict()
        assert data["name"] == "test"
        assert data["traceId"] == ctx.trace_id
        assert data["spanId"] == ctx.span_id
        assert "startTimeUnixNano" in data


class TestTracer:
    """Tests for Tracer."""

    def test_tracer_creation(self):
        """Test tracer creation."""
        tracer = Tracer()
        assert tracer is not None

    def test_tracer_with_config(self):
        """Test tracer with configuration."""
        config = TracingConfig(
            service_name="test_service",
            service_version="1.0.0",
            sample_rate=1.0,
        )
        tracer = Tracer(config)
        assert tracer.config.service_name == "test_service"

    def test_tracer_start_span(self):
        """Test tracer start_span context manager."""
        tracer = Tracer()

        with tracer.start_span("test_operation") as span:
            span.set_attribute("key", "value")

        assert span.end_time is not None
        assert span.status != SpanStatus.ERROR

    def test_tracer_nested_spans(self):
        """Test nested spans maintain parent-child relationship."""
        tracer = Tracer()

        with tracer.start_span("parent") as parent:
            parent_ctx = parent.context

            with tracer.start_span("child") as child:
                assert child.parent_context is not None
                assert child.context.trace_id == parent_ctx.trace_id
                assert child.parent_context.span_id == parent_ctx.span_id

    def test_tracer_span_on_exception(self):
        """Test span captures exception."""
        tracer = Tracer()

        with pytest.raises(ValueError):
            with tracer.start_span("failing_operation") as span:
                raise ValueError("Test error")

        assert span.status == SpanStatus.ERROR
        assert span.attributes.get("exception.type") == "ValueError"
        assert "Test error" in span.attributes.get("exception.message", "")

    def test_tracer_current_span(self):
        """Test current span tracking."""
        tracer = Tracer()

        assert get_current_span() is None

        with tracer.start_span("operation") as span:
            current = get_current_span()
            assert current is span

        assert get_current_span() is None

    def test_tracer_span_builder(self):
        """Test span builder."""
        tracer = Tracer()

        builder = tracer.span("custom_span")
        assert isinstance(builder, SpanBuilder)

        span = (
            builder
            .set_kind("client")
            .set_attribute("key", "value")
            .start()
        )

        assert span.name == "custom_span"
        assert span.kind == "client"
        assert span.attributes["key"] == "value"

    def test_tracer_context_extraction(self):
        """Test context extraction from headers."""
        tracer = Tracer()

        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        }
        ctx = tracer.extract_context(headers)

        assert ctx is not None
        assert ctx.trace_id == "0af7651916cd43dd8448eb211c80319c"

    def test_tracer_context_injection(self):
        """Test context injection into headers."""
        tracer = Tracer()

        with tracer.start_span("operation") as span:
            headers = {}
            tracer.inject_context(span, headers)

            assert "traceparent" in headers
            assert span.context.trace_id in headers["traceparent"]

    def test_tracer_sampling(self):
        """Test tracer sampling."""
        config = TracingConfig(sample_rate=0.0)  # Never sample
        tracer = Tracer(config)

        with tracer.start_span("operation") as span:
            # Should be a no-op span
            assert span.context.trace_id == "0" * 32

    def test_tracer_get_spans(self):
        """Test getting recorded spans."""
        tracer = Tracer()

        with tracer.start_span("span1"):
            pass
        with tracer.start_span("span2"):
            pass

        spans = tracer.get_spans()
        assert len(spans) >= 2

    def test_tracer_clear_spans(self):
        """Test clearing recorded spans."""
        tracer = Tracer()

        with tracer.start_span("span1"):
            pass

        tracer.clear_spans()
        spans = tracer.get_spans()
        assert len(spans) == 0


class TestSpanExporters:
    """Tests for span exporters."""

    def test_in_memory_exporter(self):
        """Test InMemorySpanExporter."""
        exporter = InMemorySpanExporter()
        tracer = Tracer()
        tracer.add_exporter(exporter)

        with tracer.start_span("test"):
            pass

        spans = exporter.get_spans()
        assert len(spans) == 1
        assert spans[0].name == "test"

    def test_in_memory_exporter_clear(self):
        """Test clearing InMemorySpanExporter."""
        exporter = InMemorySpanExporter()
        tracer = Tracer()
        tracer.add_exporter(exporter)

        with tracer.start_span("test"):
            pass

        exporter.clear()
        spans = exporter.get_spans()
        assert len(spans) == 0

    def test_console_exporter(self, capsys):
        """Test ConsoleSpanExporter."""
        exporter = ConsoleSpanExporter()
        tracer = Tracer()
        tracer.add_exporter(exporter)

        with tracer.start_span("test_operation"):
            pass

        captured = capsys.readouterr()
        assert "TRACE" in captured.out
        assert "test_operation" in captured.out

    def test_otlp_exporter_creation(self):
        """Test OTLPSpanExporter creation."""
        exporter = OTLPSpanExporter(
            endpoint="http://localhost:4318/v1/traces",
            headers={"Authorization": "Bearer token"},
        )
        assert exporter.endpoint == "http://localhost:4318/v1/traces"
        assert "Authorization" in exporter.headers


class TestTraceDecorators:
    """Tests for tracing decorators."""

    def test_trace_decorator(self):
        """Test @trace decorator."""
        tracer = Tracer()
        set_default_tracer(tracer)
        exporter = InMemorySpanExporter()
        tracer.add_exporter(exporter)

        @trace("my_operation")
        def my_function():
            return "result"

        result = my_function()
        assert result == "result"

        spans = exporter.get_spans()
        assert len(spans) >= 1
        assert any(s.name == "my_operation" for s in spans)

    def test_trace_decorator_default_name(self):
        """Test @trace decorator uses function name."""
        tracer = Tracer()
        set_default_tracer(tracer)
        exporter = InMemorySpanExporter()
        tracer.add_exporter(exporter)

        @trace()
        def another_function():
            return "done"

        another_function()

        spans = exporter.get_spans()
        assert any(s.name == "another_function" for s in spans)

    def test_trace_decorator_with_attributes(self):
        """Test @trace decorator with attributes."""
        tracer = Tracer()
        set_default_tracer(tracer)
        exporter = InMemorySpanExporter()
        tracer.add_exporter(exporter)

        @trace("operation", attributes={"custom": "value"})
        def func_with_attrs():
            return True

        func_with_attrs()

        spans = exporter.get_spans()
        operation_span = next(s for s in spans if s.name == "operation")
        assert operation_span.attributes.get("custom") == "value"

    @pytest.mark.asyncio
    async def test_trace_async_function(self):
        """Test @trace decorator with async function."""
        tracer = Tracer()
        set_default_tracer(tracer)
        exporter = InMemorySpanExporter()
        tracer.add_exporter(exporter)

        @trace("async_operation")
        async def async_func():
            await asyncio.sleep(0.01)
            return "async result"

        result = await async_func()
        assert result == "async result"

        spans = exporter.get_spans()
        assert any(s.name == "async_operation" for s in spans)

    def test_trace_method_decorator(self):
        """Test @trace_method decorator."""
        tracer = Tracer()
        set_default_tracer(tracer)
        exporter = InMemorySpanExporter()
        tracer.add_exporter(exporter)

        class MyService:
            @trace_method()
            def process(self):
                return "processed"

        service = MyService()
        result = service.process()
        assert result == "processed"

        spans = exporter.get_spans()
        assert any(s.name == "MyService.process" for s in spans)


class TestGlobalTracer:
    """Tests for global tracer functions."""

    def test_get_default_tracer(self):
        """Test get_default_tracer."""
        tracer = get_default_tracer()
        assert isinstance(tracer, Tracer)

    def test_set_default_tracer(self):
        """Test set_default_tracer."""
        custom_tracer = Tracer(TracingConfig(service_name="custom"))
        set_default_tracer(custom_tracer)

        assert get_default_tracer() is custom_tracer


class TestTracingMiddleware:
    """Tests for TracingMiddleware."""

    @pytest.mark.asyncio
    async def test_middleware_creates_span(self):
        """Test middleware creates span for request."""
        tracer = Tracer()
        exporter = InMemorySpanExporter()
        tracer.add_exporter(exporter)

        async def app(scope, receive, send):
            await send({
                "type": "http.response.start",
                "status": 200,
            })
            await send({
                "type": "http.response.body",
                "body": b"OK",
            })

        middleware = TracingMiddleware(app, tracer)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/test",
            "scheme": "https",
            "headers": [],
        }

        async def receive():
            return {"type": "http.request", "body": b""}

        sent_messages = []

        async def send(message):
            sent_messages.append(message)

        await middleware(scope, receive, send)

        spans = exporter.get_spans()
        assert len(spans) >= 1

        request_span = spans[-1]
        assert "GET /api/test" in request_span.name
        assert request_span.attributes["http.method"] == "GET"
        assert request_span.attributes["http.url"] == "/api/test"

    @pytest.mark.asyncio
    async def test_middleware_propagates_context(self):
        """Test middleware propagates trace context."""
        tracer = Tracer()
        exporter = InMemorySpanExporter()
        tracer.add_exporter(exporter)

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b""})

        middleware = TracingMiddleware(app, tracer)

        # Include traceparent header
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "scheme": "http",
            "headers": [
                (b"traceparent", b"00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"),
            ],
        }

        await middleware(scope, lambda: {"type": "http.request"}, lambda m: None)

        spans = exporter.get_spans()
        assert len(spans) >= 1

        # Should have the same trace ID
        assert spans[-1].context.trace_id == "0af7651916cd43dd8448eb211c80319c"

    @pytest.mark.asyncio
    async def test_middleware_ignores_non_http(self):
        """Test middleware ignores non-HTTP requests."""
        tracer = Tracer()
        exporter = InMemorySpanExporter()
        tracer.add_exporter(exporter)

        called = False

        async def app(scope, receive, send):
            nonlocal called
            called = True

        middleware = TracingMiddleware(app, tracer)

        scope = {"type": "websocket"}
        await middleware(scope, None, None)

        assert called
        # No spans should be created for non-HTTP
        spans = exporter.get_spans()
        assert len(spans) == 0
