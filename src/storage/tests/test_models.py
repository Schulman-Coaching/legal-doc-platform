"""
Tests for storage layer models.
"""

from __future__ import annotations

import pytest
from datetime import datetime
from uuid import uuid4

from ..models import (
    StorageBackend,
    DocumentStorageClass,
    RetentionPolicy,
    StorageMetrics,
    DocumentRecord,
    SearchDocument,
    GraphNode,
    GraphRelationship,
    AnalyticsEvent,
)


class TestStorageBackendEnum:
    """Tests for StorageBackend enum."""

    def test_backend_values(self):
        """Test enum values."""
        assert StorageBackend.POSTGRESQL == "postgresql"
        assert StorageBackend.ELASTICSEARCH == "elasticsearch"
        assert StorageBackend.MINIO == "minio"
        assert StorageBackend.REDIS == "redis"
        assert StorageBackend.NEO4J == "neo4j"
        assert StorageBackend.CLICKHOUSE == "clickhouse"

    def test_backend_from_string(self):
        """Test creating enum from string."""
        assert StorageBackend("postgresql") == StorageBackend.POSTGRESQL
        assert StorageBackend("elasticsearch") == StorageBackend.ELASTICSEARCH


class TestDocumentStorageClass:
    """Tests for DocumentStorageClass enum."""

    def test_storage_class_values(self):
        """Test enum values."""
        assert DocumentStorageClass.HOT == "hot"
        assert DocumentStorageClass.WARM == "warm"
        assert DocumentStorageClass.COLD == "cold"
        assert DocumentStorageClass.GLACIER == "glacier"


class TestRetentionPolicy:
    """Tests for RetentionPolicy enum."""

    def test_retention_policy_values(self):
        """Test enum values."""
        assert RetentionPolicy.STANDARD == "standard"
        assert RetentionPolicy.LITIGATION_HOLD == "litigation_hold"
        assert RetentionPolicy.SHORT_TERM == "short_term"
        assert RetentionPolicy.REGULATORY == "regulatory"


class TestStorageMetrics:
    """Tests for StorageMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating storage metrics."""
        metrics = StorageMetrics(
            operation="upload",
            backend=StorageBackend.MINIO,
            duration_ms=150.5,
            success=True,
            bytes_processed=1024,
        )

        assert metrics.operation == "upload"
        assert metrics.backend == StorageBackend.MINIO
        assert metrics.duration_ms == 150.5
        assert metrics.success is True
        assert metrics.bytes_processed == 1024
        assert metrics.error_message is None
        assert isinstance(metrics.timestamp, datetime)

    def test_metrics_with_error(self):
        """Test creating metrics with error."""
        metrics = StorageMetrics(
            operation="download",
            backend=StorageBackend.MINIO,
            duration_ms=50.0,
            success=False,
            error_message="File not found",
        )

        assert metrics.success is False
        assert metrics.error_message == "File not found"


class TestDocumentRecord:
    """Tests for DocumentRecord model."""

    def test_create_document_record(self):
        """Test creating a document record."""
        doc = DocumentRecord(
            original_filename="contract.pdf",
            mime_type="application/pdf",
            file_size=50000,
            checksum_sha256="abc123",
            storage_path="hot/doc-123",
            user_id="user-001",
        )

        assert doc.original_filename == "contract.pdf"
        assert doc.mime_type == "application/pdf"
        assert doc.file_size == 50000
        assert doc.checksum_sha256 == "abc123"
        assert doc.storage_path == "hot/doc-123"
        assert doc.user_id == "user-001"
        assert doc.id is not None
        assert doc.storage_class == DocumentStorageClass.HOT
        assert doc.classification == "internal"
        assert doc.processing_status == "pending"
        assert doc.tags == []
        assert doc.practice_areas == []
        assert doc.legal_hold is False
        assert isinstance(doc.created_at, datetime)
        assert isinstance(doc.updated_at, datetime)

    def test_document_record_with_all_fields(self):
        """Test creating a document record with all fields."""
        doc = DocumentRecord(
            id="doc-123",
            original_filename="nda.docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            file_size=100000,
            checksum_sha256="def456",
            storage_path="hot/doc-123",
            storage_class=DocumentStorageClass.WARM,
            encryption_key_id="key-001",
            client_id="client-001",
            matter_id="matter-001",
            user_id="user-001",
            classification="confidential",
            tags=["nda", "confidential"],
            processing_status="completed",
            document_type="Contract",
            practice_areas=["Corporate", "M&A"],
            entity_count=15,
            clause_count=8,
            citation_count=3,
            page_count=5,
            word_count=2500,
            retention_policy=RetentionPolicy.LITIGATION_HOLD,
            legal_hold=True,
            custom_metadata={"key": "value"},
        )

        assert doc.id == "doc-123"
        assert doc.storage_class == "warm"  # Enum converted to value
        assert doc.encryption_key_id == "key-001"
        assert doc.client_id == "client-001"
        assert doc.classification == "confidential"
        assert len(doc.tags) == 2
        assert doc.entity_count == 15
        assert doc.legal_hold is True
        assert doc.retention_policy == "litigation_hold"

    def test_document_record_serialization(self):
        """Test document record serialization."""
        doc = DocumentRecord(
            original_filename="test.pdf",
            mime_type="application/pdf",
            file_size=1000,
            checksum_sha256="hash123",
            storage_path="hot/test",
            user_id="user-001",
        )

        data = doc.model_dump()

        assert isinstance(data, dict)
        assert data["original_filename"] == "test.pdf"
        assert data["storage_class"] == "hot"
        assert "created_at" in data


class TestSearchDocument:
    """Tests for SearchDocument model."""

    def test_create_search_document(self):
        """Test creating a search document."""
        doc = SearchDocument(
            id="doc-123",
            content="This is the document content.",
        )

        assert doc.id == "doc-123"
        assert doc.content == "This is the document content."
        assert doc.title is None
        assert doc.content_vector is None
        assert doc.tags == []
        assert doc.persons == []
        assert doc.organizations == []
        assert doc.language == "en"

    def test_search_document_with_entities(self):
        """Test search document with entities."""
        doc = SearchDocument(
            id="doc-456",
            title="Contract Agreement",
            content="Agreement between Acme Corp and Beta Inc.",
            document_type="Contract",
            practice_areas=["Corporate"],
            persons=["John Doe", "Jane Smith"],
            organizations=["Acme Corp", "Beta Inc."],
            case_numbers=["2024-CV-001"],
            citations=["15 U.S.C. 1"],
            monetary_amounts=["$1,000,000"],
        )

        assert doc.title == "Contract Agreement"
        assert len(doc.persons) == 2
        assert len(doc.organizations) == 2
        assert len(doc.case_numbers) == 1
        assert len(doc.citations) == 1
        assert len(doc.monetary_amounts) == 1

    def test_search_document_with_vector(self):
        """Test search document with embedding vector."""
        vector = [0.1] * 384
        doc = SearchDocument(
            id="doc-789",
            content="Test content",
            content_vector=vector,
        )

        assert doc.content_vector is not None
        assert len(doc.content_vector) == 384


class TestGraphNode:
    """Tests for GraphNode model."""

    def test_create_graph_node(self):
        """Test creating a graph node."""
        node = GraphNode(
            id="node-123",
            label="Document",
            properties={"title": "Contract", "type": "NDA"},
        )

        assert node.id == "node-123"
        assert node.label == "Document"
        assert node.properties["title"] == "Contract"

    def test_graph_node_default_properties(self):
        """Test graph node with default properties."""
        node = GraphNode(
            id="node-456",
            label="Entity",
        )

        assert node.properties == {}


class TestGraphRelationship:
    """Tests for GraphRelationship model."""

    def test_create_relationship(self):
        """Test creating a graph relationship."""
        rel = GraphRelationship(
            source_id="doc-001",
            target_id="entity-001",
            relationship_type="MENTIONS",
            properties={"confidence": 0.95},
        )

        assert rel.source_id == "doc-001"
        assert rel.target_id == "entity-001"
        assert rel.relationship_type == "MENTIONS"
        assert rel.properties["confidence"] == 0.95


class TestAnalyticsEvent:
    """Tests for AnalyticsEvent model."""

    def test_create_analytics_event(self):
        """Test creating an analytics event."""
        event = AnalyticsEvent(
            event_type="document_viewed",
            user_id="user-001",
            document_id="doc-123",
        )

        assert event.event_type == "document_viewed"
        assert event.user_id == "user-001"
        assert event.document_id == "doc-123"
        assert event.event_id is not None
        assert isinstance(event.timestamp, datetime)
        assert event.properties == {}

    def test_analytics_event_with_properties(self):
        """Test analytics event with properties."""
        event = AnalyticsEvent(
            event_type="search_performed",
            user_id="user-001",
            client_id="client-001",
            properties={
                "query": "contract termination",
                "results_count": 15,
                "response_time_ms": 125,
            },
        )

        assert event.client_id == "client-001"
        assert event.properties["query"] == "contract termination"
        assert event.properties["results_count"] == 15
