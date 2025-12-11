"""
Tests for API Models
====================
Tests for Pydantic request/response models.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from ..models import (
    APIVersion,
    ClientCreate,
    ClientResponse,
    DocumentResponse,
    DocumentSearchRequest,
    DocumentStatus,
    DocumentUpdate,
    ErrorResponse,
    ExportRequest,
    HealthResponse,
    LegalHoldCreate,
    MatterCreate,
    PaginatedResponse,
    RateLimitTier,
    SortOrder,
    SuccessResponse,
    TokenRequest,
    TokenResponse,
    UserCreate,
    UserResponse,
    WebhookCreate,
    WebhookEventType,
    WebhookResponse,
)


class TestEnums:
    """Test enum definitions."""

    def test_api_version_values(self):
        """Test API version enum values."""
        assert APIVersion.V1.value == "v1"
        assert APIVersion.V2.value == "v2"

    def test_rate_limit_tier_values(self):
        """Test rate limit tier enum values."""
        assert RateLimitTier.FREE.value == "free"
        assert RateLimitTier.PROFESSIONAL.value == "professional"
        assert RateLimitTier.ENTERPRISE.value == "enterprise"
        assert RateLimitTier.UNLIMITED.value == "unlimited"

    def test_document_status_values(self):
        """Test document status enum values."""
        assert DocumentStatus.PENDING.value == "pending"
        assert DocumentStatus.PROCESSING.value == "processing"
        assert DocumentStatus.ANALYZED.value == "analyzed"
        assert DocumentStatus.FAILED.value == "failed"

    def test_sort_order_values(self):
        """Test sort order enum values."""
        assert SortOrder.ASC.value == "asc"
        assert SortOrder.DESC.value == "desc"

    def test_webhook_event_types(self):
        """Test webhook event type enum values."""
        assert WebhookEventType.DOCUMENT_UPLOADED.value == "document.uploaded"
        assert WebhookEventType.DOCUMENT_PROCESSED.value == "document.processed"
        assert WebhookEventType.ALERT_TRIGGERED.value == "alert.triggered"


class TestGenericResponses:
    """Test generic response models."""

    def test_paginated_response_create(self):
        """Test paginated response creation."""
        items = ["a", "b", "c"]
        response = PaginatedResponse.create(items=items, total=10, page=1, size=3)

        assert response.items == items
        assert response.total == 10
        assert response.page == 1
        assert response.size == 3
        assert response.pages == 4

    def test_paginated_response_empty(self):
        """Test paginated response with empty items."""
        response = PaginatedResponse.create(items=[], total=0, page=1, size=20)

        assert response.items == []
        assert response.total == 0
        assert response.pages == 0

    def test_error_response(self):
        """Test error response model."""
        error = ErrorResponse(
            error="not_found",
            message="Document not found",
            details={"document_id": "123"},
            request_id="req-456",
        )

        assert error.error == "not_found"
        assert error.message == "Document not found"
        assert error.details == {"document_id": "123"}
        assert error.request_id == "req-456"
        assert isinstance(error.timestamp, datetime)

    def test_success_response_defaults(self):
        """Test success response defaults."""
        response = SuccessResponse()

        assert response.success is True
        assert response.message == "Operation completed successfully"
        assert response.data is None

    def test_success_response_custom(self):
        """Test success response with custom values."""
        response = SuccessResponse(
            success=True,
            message="Document deleted",
            data={"deleted_id": "123"},
        )

        assert response.message == "Document deleted"
        assert response.data == {"deleted_id": "123"}

    def test_health_response(self):
        """Test health response model."""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            services={"database": {"status": "up"}},
        )

        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.services["database"]["status"] == "up"


class TestAuthModels:
    """Test authentication models."""

    def test_token_request_valid(self):
        """Test valid token request."""
        request = TokenRequest(username="testuser", password="password123")

        assert request.username == "testuser"
        assert request.password == "password123"

    def test_token_request_username_too_short(self):
        """Test token request with short username."""
        with pytest.raises(ValidationError):
            TokenRequest(username="ab", password="password123")

    def test_token_request_password_too_short(self):
        """Test token request with short password."""
        with pytest.raises(ValidationError):
            TokenRequest(username="testuser", password="short")

    def test_token_response(self):
        """Test token response model."""
        response = TokenResponse(
            access_token="access.token.here",
            refresh_token="refresh.token.here",
            expires_in=1800,
        )

        assert response.access_token == "access.token.here"
        assert response.refresh_token == "refresh.token.here"
        assert response.token_type == "bearer"
        assert response.expires_in == 1800
        assert response.scope == "read write"


class TestUserModels:
    """Test user models."""

    def test_user_create_valid(self):
        """Test valid user creation."""
        user = UserCreate(
            username="newuser",
            email="user@example.com",
            password="securepass123",
            first_name="John",
            last_name="Doe",
        )

        assert user.username == "newuser"
        assert user.email == "user@example.com"
        assert user.first_name == "John"

    def test_user_create_invalid_email(self):
        """Test user creation with invalid email."""
        with pytest.raises(ValidationError):
            UserCreate(
                username="newuser",
                email="not-an-email",
                password="securepass123",
            )

    def test_user_response(self):
        """Test user response model."""
        now = datetime.utcnow()
        response = UserResponse(
            id="user-123",
            username="testuser",
            email="test@example.com",
            roles=["user", "admin"],
            is_active=True,
            created_at=now,
        )

        assert response.id == "user-123"
        assert response.username == "testuser"
        assert "admin" in response.roles
        assert response.is_active is True


class TestDocumentModels:
    """Test document models."""

    def test_document_response(self):
        """Test document response model."""
        now = datetime.utcnow()
        response = DocumentResponse(
            id="doc-123",
            filename="document.pdf",
            original_filename="my-document.pdf",
            content_type="application/pdf",
            file_size=1024000,
            checksum="abc123",
            status=DocumentStatus.ANALYZED,
            tags=["contract", "legal"],
            created_at=now,
            updated_at=now,
            created_by="user-456",
        )

        assert response.id == "doc-123"
        assert response.file_size == 1024000
        assert response.status == DocumentStatus.ANALYZED
        assert "contract" in response.tags

    def test_document_update(self):
        """Test document update model."""
        update = DocumentUpdate(
            tags=["updated", "tag"],
            classification="confidential",
        )

        assert update.tags == ["updated", "tag"]
        assert update.classification == "confidential"
        assert update.client_id is None

    def test_document_search_request(self):
        """Test document search request model."""
        request = DocumentSearchRequest(
            query="contract agreement",
            client_id="client-123",
            sort_by="relevance",
            sort_order=SortOrder.DESC,
        )

        assert request.query == "contract agreement"
        assert request.client_id == "client-123"
        assert request.sort_order == SortOrder.DESC

    def test_document_search_request_min_query(self):
        """Test document search requires minimum query length."""
        with pytest.raises(ValidationError):
            DocumentSearchRequest(query="")


class TestClientMatterModels:
    """Test client and matter models."""

    def test_client_create(self):
        """Test client creation model."""
        client = ClientCreate(
            name="Acme Corporation",
            code="ACME",
            industry="Technology",
            contact_email="contact@acme.com",
        )

        assert client.name == "Acme Corporation"
        assert client.code == "ACME"
        assert client.industry == "Technology"

    def test_client_response(self):
        """Test client response model."""
        now = datetime.utcnow()
        response = ClientResponse(
            id="client-123",
            name="Test Client",
            document_count=50,
            matter_count=5,
            created_at=now,
            updated_at=now,
        )

        assert response.id == "client-123"
        assert response.document_count == 50
        assert response.matter_count == 5

    def test_matter_create(self):
        """Test matter creation model."""
        matter = MatterCreate(
            client_id="client-123",
            name="Case 2024-001",
            description="Litigation matter",
            practice_area="Litigation",
        )

        assert matter.client_id == "client-123"
        assert matter.name == "Case 2024-001"
        assert matter.status == "active"


class TestWebhookModels:
    """Test webhook models."""

    def test_webhook_create(self):
        """Test webhook creation model."""
        webhook = WebhookCreate(
            url="https://example.com/webhook",
            events=[
                WebhookEventType.DOCUMENT_UPLOADED,
                WebhookEventType.DOCUMENT_PROCESSED,
            ],
            description="Document notifications",
        )

        assert webhook.url == "https://example.com/webhook"
        assert len(webhook.events) == 2
        assert webhook.is_active is True

    def test_webhook_create_invalid_url(self):
        """Test webhook creation with invalid URL."""
        with pytest.raises(ValidationError):
            WebhookCreate(
                url="not-a-url",
                events=[WebhookEventType.DOCUMENT_UPLOADED],
            )

    def test_webhook_response(self):
        """Test webhook response model."""
        now = datetime.utcnow()
        response = WebhookResponse(
            id="webhook-123",
            url="https://example.com/webhook",
            events=[WebhookEventType.DOCUMENT_UPLOADED],
            secret_preview="abc12345***",
            is_active=True,
            created_at=now,
            failure_count=0,
        )

        assert response.id == "webhook-123"
        assert response.secret_preview == "abc12345***"
        assert response.failure_count == 0


class TestLegalHoldModels:
    """Test legal hold models."""

    def test_legal_hold_create(self):
        """Test legal hold creation model."""
        hold = LegalHoldCreate(
            matter_id="matter-123",
            matter_name="Important Case",
            document_ids=["doc-1", "doc-2", "doc-3"],
            custodians=["user-1", "user-2"],
            reason="Pending litigation",
        )

        assert hold.matter_id == "matter-123"
        assert len(hold.document_ids) == 3
        assert len(hold.custodians) == 2
        assert hold.reason == "Pending litigation"


class TestExportModels:
    """Test export models."""

    def test_export_request_defaults(self):
        """Test export request defaults."""
        request = ExportRequest(document_ids=["doc-1", "doc-2"])

        assert request.format == "zip"
        assert request.include_metadata is True
        assert request.include_analysis is False

    def test_export_request_custom(self):
        """Test export request with custom options."""
        request = ExportRequest(
            document_ids=["doc-1"],
            format="pdf",
            include_metadata=False,
            include_analysis=True,
        )

        assert request.format == "pdf"
        assert request.include_metadata is False
        assert request.include_analysis is True
