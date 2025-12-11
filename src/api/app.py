"""
FastAPI Application
===================
Unified API application with versioning, middleware, and routes.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import APIConfig
from .models import ErrorResponse, HealthResponse
from .middleware.auth import AuthMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.request_context import RequestContextMiddleware
from .routes import (
    admin_router,
    analytics_router,
    auth_router,
    clients_router,
    documents_router,
    users_router,
    webhooks_router,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Application Factory
# =============================================================================

def create_app(config: Optional[APIConfig] = None) -> FastAPI:
    """
    Create FastAPI application with full configuration.

    Args:
        config: API configuration. Uses defaults if not provided.

    Returns:
        Configured FastAPI application.
    """
    config = config or APIConfig()

    # Create app with lifecycle
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifecycle manager."""
        logger.info("Starting Legal Document Platform API...")
        logger.info(f"Environment: {config.environment}")
        logger.info(f"Debug mode: {config.debug}")

        # Store config in app state
        app.state.config = config

        yield

        logger.info("Shutting down API...")

    app = FastAPI(
        title=config.title,
        description=config.description,
        version=config.version,
        docs_url=config.docs_url if config.debug else None,
        redoc_url=config.redoc_url,
        openapi_url=config.openapi_url,
        lifespan=lifespan,
    )

    # Configure middleware
    _setup_middleware(app, config)

    # Configure exception handlers
    _setup_exception_handlers(app)

    # Configure routes
    _setup_routes(app, config)

    return app


def _setup_middleware(app: FastAPI, config: APIConfig) -> None:
    """Configure middleware stack."""
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors.allow_origins,
        allow_credentials=config.cors.allow_credentials,
        allow_methods=config.cors.allow_methods,
        allow_headers=config.cors.allow_headers,
        max_age=config.cors.max_age,
    )

    # Request context (outermost - runs first)
    app.add_middleware(RequestContextMiddleware)

    # Rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        redis_url=config.redis_url,
        limits=config.rate_limits,
    )

    # Authentication
    app.add_middleware(
        AuthMiddleware,
        jwt_config=config.jwt,
        api_key_config=config.api_key,
    )


def _setup_exception_handlers(app: FastAPI) -> None:
    """Configure global exception handlers."""

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        """Handle HTTP exceptions."""
        request_id = getattr(request.state, "request_id", str(uuid4()))

        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=_status_to_error_code(exc.status_code),
                message=str(exc.detail),
                request_id=request_id,
            ).model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Handle validation errors."""
        request_id = getattr(request.state, "request_id", str(uuid4()))

        # Format validation errors
        errors = []
        for error in exc.errors():
            loc = ".".join(str(l) for l in error["loc"])
            errors.append(f"{loc}: {error['msg']}")

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="validation_error",
                message="Request validation failed",
                details={"errors": errors},
                request_id=request_id,
            ).model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        request_id = getattr(request.state, "request_id", str(uuid4()))

        logger.exception(f"Unhandled exception in request {request_id}: {exc}")

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_error",
                message="An unexpected error occurred",
                request_id=request_id,
            ).model_dump(mode="json"),
        )


def _setup_routes(app: FastAPI, config: APIConfig) -> None:
    """Configure API routes."""
    # Health endpoints (no prefix)
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check() -> HealthResponse:
        """Basic health check."""
        return HealthResponse(
            status="healthy",
            version=config.version,
            services={},
        )

    @app.get("/ready", response_model=HealthResponse, tags=["Health"])
    async def readiness_check() -> HealthResponse:
        """Readiness check with service status."""
        # In production, would check actual service health
        services = {
            "database": {"status": "healthy", "latency_ms": 1},
            "storage": {"status": "healthy", "latency_ms": 2},
            "cache": {"status": "healthy", "latency_ms": 1},
        }

        all_healthy = all(s["status"] == "healthy" for s in services.values())

        return HealthResponse(
            status="ready" if all_healthy else "degraded",
            version=config.version,
            services=services,
        )

    @app.get("/", tags=["Root"])
    async def root() -> dict[str, Any]:
        """API root with information."""
        return {
            "name": config.title,
            "version": config.version,
            "description": config.description,
            "docs": config.docs_url,
            "redoc": config.redoc_url,
            "api_versions": ["v1", "v2"],
        }

    # API v1 routes
    v1_prefix = f"{config.api_prefix}/v1"

    app.include_router(auth_router, prefix=v1_prefix)
    app.include_router(users_router, prefix=v1_prefix)
    app.include_router(documents_router, prefix=v1_prefix)
    app.include_router(clients_router, prefix=v1_prefix)
    app.include_router(webhooks_router, prefix=v1_prefix)
    app.include_router(admin_router, prefix=v1_prefix)
    app.include_router(analytics_router, prefix=v1_prefix)

    # API v2 routes (same routes for now, would have breaking changes)
    v2_prefix = f"{config.api_prefix}/v2"

    app.include_router(auth_router, prefix=v2_prefix, tags=["v2"])
    app.include_router(users_router, prefix=v2_prefix, tags=["v2"])
    app.include_router(documents_router, prefix=v2_prefix, tags=["v2"])
    app.include_router(clients_router, prefix=v2_prefix, tags=["v2"])
    app.include_router(webhooks_router, prefix=v2_prefix, tags=["v2"])
    app.include_router(admin_router, prefix=v2_prefix, tags=["v2"])
    app.include_router(analytics_router, prefix=v2_prefix, tags=["v2"])


def _status_to_error_code(status_code: int) -> str:
    """Convert HTTP status code to error code string."""
    mapping = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        409: "conflict",
        422: "validation_error",
        429: "rate_limit_exceeded",
        500: "internal_error",
        502: "bad_gateway",
        503: "service_unavailable",
        504: "gateway_timeout",
    }
    return mapping.get(status_code, "error")


# =============================================================================
# API Gateway Class
# =============================================================================

class APIGateway:
    """
    API Gateway wrapper for programmatic control.

    Provides methods for starting, stopping, and configuring the API.
    """

    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self.app = create_app(self.config)
        self._server = None

    def get_app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self.app

    async def start(self) -> None:
        """Start the API server."""
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            reload=self.config.reload,
            workers=self.config.workers,
            log_level="info" if not self.config.debug else "debug",
        )

        self._server = uvicorn.Server(config)
        await self._server.serve()

    async def stop(self) -> None:
        """Stop the API server."""
        if self._server:
            self._server.should_exit = True

    def run(self) -> None:
        """Run the API server (blocking)."""
        import uvicorn

        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            reload=self.config.reload,
            workers=self.config.workers,
            log_level="info" if not self.config.debug else "debug",
        )


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    """CLI entry point for running the API."""
    import argparse

    parser = argparse.ArgumentParser(description="Legal Document Platform API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")

    args = parser.parse_args()

    config = APIConfig(
        host=args.host,
        port=args.port,
        reload=args.reload,
        debug=args.debug,
        workers=args.workers,
    )

    gateway = APIGateway(config)
    gateway.run()


if __name__ == "__main__":
    main()
