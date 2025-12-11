"""
Tests for Compliance Service.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from ..services.compliance import ComplianceService
from ..models import (
    PIIType,
    SecurityClassification,
    ComplianceFramework,
    LegalHold,
    GDPRRequest,
)


class TestPIIDetection:
    """Tests for PII detection."""

    @pytest.mark.asyncio
    async def test_detect_ssn(self, compliance_service: ComplianceService):
        """Test SSN detection."""
        text = "Customer SSN is 123-45-6789"
        matches = await compliance_service.detect_pii(text)

        ssn_matches = [m for m in matches if m.pii_type == PIIType.SSN]
        assert len(ssn_matches) == 1
        assert ssn_matches[0].value == "123-45-6789"

    @pytest.mark.asyncio
    async def test_detect_credit_card(self, compliance_service: ComplianceService):
        """Test credit card detection."""
        text = "Payment card: 4111111111111111"
        matches = await compliance_service.detect_pii(text)

        cc_matches = [m for m in matches if m.pii_type == PIIType.CREDIT_CARD]
        assert len(cc_matches) == 1

    @pytest.mark.asyncio
    async def test_detect_email(self, compliance_service: ComplianceService):
        """Test email detection."""
        text = "Contact: john.doe@example.com"
        matches = await compliance_service.detect_pii(text)

        email_matches = [m for m in matches if m.pii_type == PIIType.EMAIL]
        assert len(email_matches) == 1
        assert "john.doe@example.com" in email_matches[0].value

    @pytest.mark.asyncio
    async def test_detect_phone(self, compliance_service: ComplianceService):
        """Test phone number detection."""
        text = "Call us at (555) 123-4567"
        matches = await compliance_service.detect_pii(text)

        phone_matches = [m for m in matches if m.pii_type == PIIType.PHONE]
        assert len(phone_matches) == 1

    @pytest.mark.asyncio
    async def test_detect_multiple_pii(self, compliance_service: ComplianceService, sample_document_content: str):
        """Test detecting multiple PII types in a document."""
        matches = await compliance_service.detect_pii(sample_document_content)

        pii_types = {m.pii_type for m in matches}

        assert PIIType.SSN in pii_types
        assert PIIType.EMAIL in pii_types
        assert PIIType.PHONE in pii_types
        assert PIIType.CREDIT_CARD in pii_types

    @pytest.mark.asyncio
    async def test_no_pii_in_clean_content(self, compliance_service: ComplianceService, sample_clean_content: str):
        """Test that clean content has minimal PII matches."""
        matches = await compliance_service.detect_pii(sample_clean_content)

        # Should have no high-confidence PII matches
        high_conf_matches = [m for m in matches if m.confidence > 0.8]
        assert len(high_conf_matches) == 0

    @pytest.mark.asyncio
    async def test_pii_masking(self, compliance_service: ComplianceService):
        """Test PII masking."""
        text = "SSN: 123-45-6789"
        matches = await compliance_service.detect_pii(text)

        ssn_match = [m for m in matches if m.pii_type == PIIType.SSN][0]

        # Masked value should hide most digits
        assert "6789" in ssn_match.masked_value
        assert "123" not in ssn_match.masked_value

    @pytest.mark.asyncio
    async def test_redact_pii(self, compliance_service: ComplianceService):
        """Test PII redaction."""
        text = "SSN: 123-45-6789, Email: test@test.com"

        redacted, matches = await compliance_service.redact_pii(text)

        assert "123-45-6789" not in redacted
        assert "test@test.com" not in redacted
        assert "[REDACTED:" in redacted


class TestDocumentClassification:
    """Tests for document classification."""

    @pytest.mark.asyncio
    async def test_classify_privileged(self, compliance_service: ComplianceService):
        """Test attorney-client privilege classification."""
        content = "This is privileged and confidential attorney-client communication."

        classification = await compliance_service.classify_document(content, {})

        assert classification == SecurityClassification.ATTORNEY_CLIENT_PRIVILEGED

    @pytest.mark.asyncio
    async def test_classify_restricted_with_pii(self, compliance_service: ComplianceService):
        """Test restricted classification for high-risk PII."""
        content = "SSN: 123-45-6789, Credit Card: 4111111111111111"

        classification = await compliance_service.classify_document(content, {})

        assert classification == SecurityClassification.RESTRICTED

    @pytest.mark.asyncio
    async def test_classify_confidential_with_pii(self, compliance_service: ComplianceService):
        """Test confidential classification for PII."""
        content = "Contact email: john@example.com"

        classification = await compliance_service.classify_document(content, {})

        assert classification in [
            SecurityClassification.CONFIDENTIAL,
            SecurityClassification.RESTRICTED,
        ]

    @pytest.mark.asyncio
    async def test_classify_internal(self, compliance_service: ComplianceService, sample_clean_content: str):
        """Test internal classification for clean content."""
        classification = await compliance_service.classify_document(sample_clean_content, {})

        assert classification == SecurityClassification.INTERNAL


class TestLegalHold:
    """Tests for legal hold management."""

    @pytest.mark.asyncio
    async def test_apply_legal_hold(self, compliance_service: ComplianceService):
        """Test applying a legal hold."""
        hold = await compliance_service.apply_legal_hold(
            matter_id="matter-001",
            matter_name="Test Matter",
            document_ids=["doc-001", "doc-002"],
            custodians=["user-001"],
            reason="Litigation pending",
            applied_by="admin-001",
        )

        assert hold.matter_id == "matter-001"
        assert len(hold.document_ids) == 2
        assert hold.status == "active"

    @pytest.mark.asyncio
    async def test_check_legal_hold(self, compliance_service: ComplianceService):
        """Test checking if document is under legal hold."""
        await compliance_service.apply_legal_hold(
            matter_id="matter-002",
            matter_name="Test Matter 2",
            document_ids=["doc-003"],
            custodians=["user-001"],
            reason="Discovery",
            applied_by="admin-001",
        )

        hold = await compliance_service.check_legal_hold("doc-003")

        assert hold is not None
        assert hold.matter_id == "matter-002"

    @pytest.mark.asyncio
    async def test_no_legal_hold(self, compliance_service: ComplianceService):
        """Test that document without hold returns None."""
        hold = await compliance_service.check_legal_hold("doc-no-hold")

        assert hold is None

    @pytest.mark.asyncio
    async def test_release_legal_hold(self, compliance_service: ComplianceService):
        """Test releasing a legal hold."""
        hold = await compliance_service.apply_legal_hold(
            matter_id="matter-003",
            matter_name="Test Matter 3",
            document_ids=["doc-004"],
            custodians=["user-001"],
            reason="Litigation",
            applied_by="admin-001",
        )

        released = await compliance_service.release_legal_hold(hold.id, "admin-001")

        assert released.status == "released"
        assert released.released_by == "admin-001"

    @pytest.mark.asyncio
    async def test_get_legal_holds(self, compliance_service: ComplianceService):
        """Test getting legal holds."""
        # Create some holds
        await compliance_service.apply_legal_hold(
            matter_id="matter-list-1",
            matter_name="Matter 1",
            document_ids=["doc-list-1"],
            custodians=["user-001"],
            reason="Test",
            applied_by="admin-001",
        )

        holds = await compliance_service.get_legal_holds(status="active")

        assert len(holds) >= 1


class TestGDPRRequests:
    """Tests for GDPR data subject requests."""

    @pytest.mark.asyncio
    async def test_create_gdpr_access_request(self, compliance_service: ComplianceService):
        """Test creating GDPR access request."""
        request = await compliance_service.create_gdpr_request(
            request_type="access",
            data_subject_id="user-gdpr-001",
            data_subject_email="user@example.com",
        )

        assert request.request_type == "access"
        assert request.status == "pending"
        assert request.deadline > datetime.utcnow()

    @pytest.mark.asyncio
    async def test_gdpr_request_deadline(self, compliance_service: ComplianceService):
        """Test GDPR request has proper deadline."""
        request = await compliance_service.create_gdpr_request(
            request_type="erasure",
            data_subject_id="user-gdpr-002",
        )

        # Default deadline is 30 days - allow for timing variance
        expected_deadline = datetime.utcnow() + timedelta(days=30)
        delta = abs((request.deadline - expected_deadline).total_seconds())
        assert delta < 60  # Within 1 minute tolerance

    @pytest.mark.asyncio
    async def test_process_gdpr_request(self, compliance_service: ComplianceService):
        """Test processing a GDPR request."""
        request = await compliance_service.create_gdpr_request(
            request_type="access",
            data_subject_id="user-gdpr-003",
        )

        processed = await compliance_service.process_gdpr_request(
            request.id,
            response_data={"documents": ["doc1", "doc2"]},
            handled_by="admin-001",
        )

        assert processed.status == "completed"
        assert processed.completed_at is not None

    @pytest.mark.asyncio
    async def test_get_gdpr_requests(self, compliance_service: ComplianceService):
        """Test getting GDPR requests."""
        await compliance_service.create_gdpr_request(
            request_type="portability",
            data_subject_id="user-gdpr-004",
        )

        requests = await compliance_service.get_gdpr_requests(status="pending")

        assert len(requests) >= 1


class TestComplianceViolations:
    """Tests for compliance violations."""

    @pytest.mark.asyncio
    async def test_record_violation(self, compliance_service: ComplianceService):
        """Test recording a compliance violation."""
        violation = await compliance_service.record_violation(
            policy_id="soc2-encryption",
            resource_type="document",
            resource_id="doc-unencrypted",
            violation_type="missing_encryption",
            description="Document lacks required encryption",
            severity="high",
        )

        assert violation.policy_id == "soc2-encryption"
        assert violation.severity == "high"
        assert violation.status == "open"

    @pytest.mark.asyncio
    async def test_resolve_violation(self, compliance_service: ComplianceService):
        """Test resolving a violation."""
        violation = await compliance_service.record_violation(
            policy_id="gdpr-data-protection",
            resource_type="document",
            resource_id="doc-001",
            violation_type="missing_consent",
            description="Missing consent for PII",
        )

        resolved = await compliance_service.resolve_violation(
            violation.id,
            resolution="Consent obtained",
            resolved_by="admin-001",
        )

        assert resolved.status == "resolved"

    @pytest.mark.asyncio
    async def test_get_violations(self, compliance_service: ComplianceService):
        """Test getting violations with filters."""
        await compliance_service.record_violation(
            policy_id="test-policy",
            resource_type="document",
            resource_id="doc-test",
            violation_type="test_violation",
            description="Test",
            severity="critical",
        )

        critical = await compliance_service.get_violations(severity="critical")

        assert len(critical) >= 1


class TestComplianceEvaluation:
    """Tests for compliance evaluation."""

    @pytest.mark.asyncio
    async def test_evaluate_document_compliance(
        self,
        compliance_service: ComplianceService,
        sample_document_content: str,
        sample_document_metadata: dict,
    ):
        """Test evaluating document compliance."""
        result = await compliance_service.evaluate_document_compliance(
            document_id="doc-eval-001",
            content=sample_document_content,
            metadata=sample_document_metadata,
        )

        assert "compliant" in result
        assert "pii_detected" in result
        assert "classification" in result
        assert len(result["pii_detected"]) > 0

    @pytest.mark.asyncio
    async def test_compliance_violations_for_unencrypted_pii(
        self,
        compliance_service: ComplianceService,
        sample_document_content: str,
    ):
        """Test that unencrypted PII generates violations."""
        result = await compliance_service.evaluate_document_compliance(
            document_id="doc-eval-002",
            content=sample_document_content,
            metadata={"encrypted": False},
        )

        # Should have violations for unencrypted PII
        assert result["compliant"] is False or len(result["violations"]) > 0 or len(result["warnings"]) > 0


class TestRetentionPolicies:
    """Tests for retention policies."""

    @pytest.mark.asyncio
    async def test_get_retention_policy(self, compliance_service: ComplianceService):
        """Test getting applicable retention policy."""
        policy = await compliance_service.get_retention_policy(
            document_type="contract",
            classification=SecurityClassification.CONFIDENTIAL,
        )

        assert policy is not None
        assert policy.retention_days > 0

    @pytest.mark.asyncio
    async def test_check_retention_expiry(self, compliance_service: ComplianceService):
        """Test checking retention expiry."""
        old_date = datetime.utcnow() - timedelta(days=3000)  # ~8 years ago

        result = await compliance_service.check_retention_expiry(
            document_id="old-doc",
            created_at=old_date,
            document_type="contract",
            classification=SecurityClassification.CONFIDENTIAL,
        )

        assert result["expired"] is True


class TestHealthCheck:
    """Tests for health check."""

    @pytest.mark.asyncio
    async def test_health_check(self, compliance_service: ComplianceService):
        """Test health check."""
        health = await compliance_service.health_check()

        assert health["status"] == "healthy"
        assert "frameworks_enabled" in health
        assert "policies_loaded" in health
