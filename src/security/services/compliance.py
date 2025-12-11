"""
Compliance Engine Service
=========================
Compliance policy enforcement for SOC2, HIPAA, GDPR, CCPA.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
from uuid import uuid4

from ..config import ComplianceConfig
from ..models import (
    AuditEventType,
    ComplianceFramework,
    CompliancePolicy,
    ComplianceViolation,
    DataResidency,
    DataRetentionPolicy,
    GDPRRequest,
    LegalHold,
    PIIMatch,
    PIIType,
    SecurityClassification,
)

logger = logging.getLogger(__name__)


class ComplianceService:
    """
    Compliance policy enforcement service.

    Features:
    - Multi-framework support (SOC2, HIPAA, GDPR, CCPA, PCI-DSS)
    - PII detection and classification
    - Data retention management
    - Legal hold enforcement
    - GDPR data subject request handling
    - Compliance violation tracking
    """

    def __init__(self, config: ComplianceConfig, audit_service: Optional[Any] = None):
        self.config = config
        self.audit = audit_service
        self._policies: dict[str, CompliancePolicy] = {}
        self._retention_policies: dict[str, DataRetentionPolicy] = {}
        self._legal_holds: dict[str, LegalHold] = {}
        self._gdpr_requests: dict[str, GDPRRequest] = {}
        self._violations: list[ComplianceViolation] = []
        self._initialized = False

        # PII detection patterns
        self._pii_patterns = self._build_pii_patterns()

    def _build_pii_patterns(self) -> dict[PIIType, list[tuple[re.Pattern, float]]]:
        """Build regex patterns for PII detection."""
        return {
            PIIType.SSN: [
                (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), 0.95),
                (re.compile(r'\b\d{9}\b'), 0.6),
            ],
            PIIType.CREDIT_CARD: [
                (re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'), 0.95),
            ],
            PIIType.EMAIL: [
                (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), 0.98),
            ],
            PIIType.PHONE: [
                (re.compile(r'\b(?:\+1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'), 0.85),
            ],
            PIIType.DATE_OF_BIRTH: [
                (re.compile(r'\b(?:DOB|Date of Birth|Born)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', re.I), 0.9),
            ],
            PIIType.DRIVERS_LICENSE: [
                (re.compile(r'\b(?:DL|Driver\'?s?\s*License)[:\s#]*([A-Z0-9]{5,15})\b', re.I), 0.8),
            ],
            PIIType.PASSPORT: [
                (re.compile(r'\b(?:Passport)[:\s#]*([A-Z0-9]{6,12})\b', re.I), 0.85),
            ],
            PIIType.BANK_ACCOUNT: [
                (re.compile(r'\b(?:Account|Acct)[:\s#]*(\d{8,17})\b', re.I), 0.7),
                (re.compile(r'\b(?:Routing)[:\s#]*(\d{9})\b', re.I), 0.9),
            ],
            PIIType.IP_ADDRESS: [
                (re.compile(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'), 0.95),
            ],
            PIIType.ADDRESS: [
                (re.compile(r'\b\d{1,5}\s+[\w\s]{1,30}\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct)\b', re.I), 0.75),
            ],
        }

    async def initialize(self) -> None:
        """Initialize compliance service with default policies."""
        # Load default policies for enabled frameworks
        for framework in self.config.enabled_frameworks:
            await self._load_framework_policies(ComplianceFramework(framework))

        # Load default retention policies
        await self._load_default_retention_policies()

        self._initialized = True
        logger.info("Compliance service initialized with frameworks: %s", self.config.enabled_frameworks)

    async def _load_framework_policies(self, framework: ComplianceFramework) -> None:
        """Load default policies for a compliance framework."""
        policies = self._get_default_policies(framework)
        for policy in policies:
            self._policies[policy.id] = policy

    def _get_default_policies(self, framework: ComplianceFramework) -> list[CompliancePolicy]:
        """Get default policies for a framework."""
        if framework == ComplianceFramework.SOC2:
            return [
                CompliancePolicy(
                    id="soc2-access-control",
                    framework=framework,
                    name="SOC2 Access Control",
                    description="Ensure proper access controls are in place",
                    rules=[
                        {"type": "mfa_required", "conditions": {"classification": ["restricted", "confidential"]}},
                        {"type": "audit_required", "conditions": {"all": True}},
                    ],
                ),
                CompliancePolicy(
                    id="soc2-encryption",
                    framework=framework,
                    name="SOC2 Encryption",
                    description="Ensure data is encrypted at rest and in transit",
                    rules=[
                        {"type": "encryption_required", "conditions": {"classification": ["restricted", "confidential"]}},
                    ],
                ),
            ]
        elif framework == ComplianceFramework.HIPAA:
            return [
                CompliancePolicy(
                    id="hipaa-phi-protection",
                    framework=framework,
                    name="HIPAA PHI Protection",
                    description="Protect Protected Health Information",
                    rules=[
                        {"type": "pii_detection", "conditions": {"types": ["medical_record"]}},
                        {"type": "encryption_required", "conditions": {"all": True}},
                        {"type": "access_logging", "conditions": {"all": True}},
                    ],
                ),
            ]
        elif framework == ComplianceFramework.GDPR:
            return [
                CompliancePolicy(
                    id="gdpr-data-protection",
                    framework=framework,
                    name="GDPR Data Protection",
                    description="GDPR personal data protection requirements",
                    rules=[
                        {"type": "consent_required", "conditions": {"pii_types": ["any"]}},
                        {"type": "data_minimization", "conditions": {"all": True}},
                        {"type": "right_to_erasure", "conditions": {"all": True}},
                    ],
                ),
                CompliancePolicy(
                    id="gdpr-data-portability",
                    framework=framework,
                    name="GDPR Data Portability",
                    description="Support data portability requests",
                    rules=[
                        {"type": "export_format", "conditions": {"formats": ["json", "csv", "xml"]}},
                    ],
                ),
            ]
        elif framework == ComplianceFramework.CCPA:
            return [
                CompliancePolicy(
                    id="ccpa-disclosure",
                    framework=framework,
                    name="CCPA Disclosure",
                    description="CCPA disclosure requirements",
                    rules=[
                        {"type": "disclosure_notice", "conditions": {"california_residents": True}},
                        {"type": "opt_out_support", "conditions": {"all": True}},
                    ],
                ),
            ]
        return []

    async def _load_default_retention_policies(self) -> None:
        """Load default data retention policies."""
        default_policies = [
            DataRetentionPolicy(
                id="legal-7-years",
                name="Legal Documents - 7 Years",
                description="Retain legal documents for 7 years",
                document_types=["contract", "agreement", "litigation"],
                classifications=[SecurityClassification.CONFIDENTIAL, SecurityClassification.ATTORNEY_CLIENT_PRIVILEGED],
                retention_days=2555,
                action="archive",
            ),
            DataRetentionPolicy(
                id="general-3-years",
                name="General Documents - 3 Years",
                description="Retain general documents for 3 years",
                document_types=[],
                classifications=[SecurityClassification.INTERNAL],
                retention_days=1095,
                action="archive",
            ),
            DataRetentionPolicy(
                id="public-1-year",
                name="Public Documents - 1 Year",
                description="Retain public documents for 1 year",
                document_types=[],
                classifications=[SecurityClassification.PUBLIC],
                retention_days=365,
                action="delete",
            ),
        ]
        for policy in default_policies:
            self._retention_policies[policy.id] = policy

    # =========================================================================
    # PII Detection
    # =========================================================================

    async def detect_pii(
        self,
        text: str,
        confidence_threshold: Optional[float] = None,
    ) -> list[PIIMatch]:
        """
        Detect PII in text content.

        Args:
            text: Text to scan for PII
            confidence_threshold: Minimum confidence (default from config)

        Returns:
            List of PII matches found
        """
        if confidence_threshold is None:
            confidence_threshold = self.config.pii_detection_confidence_threshold

        matches = []

        for pii_type, patterns in self._pii_patterns.items():
            for pattern, base_confidence in patterns:
                for match in pattern.finditer(text):
                    if base_confidence >= confidence_threshold:
                        value = match.group(0)
                        matches.append(PIIMatch(
                            pii_type=pii_type,
                            value=value,
                            masked_value=self._mask_pii(value, pii_type),
                            start_position=match.start(),
                            end_position=match.end(),
                            confidence=base_confidence,
                            context=text[max(0, match.start()-20):min(len(text), match.end()+20)],
                        ))

        return matches

    def _mask_pii(self, value: str, pii_type: PIIType) -> str:
        """Mask PII value for safe display."""
        if pii_type == PIIType.SSN:
            return f"***-**-{value[-4:]}" if len(value) >= 4 else "***"
        elif pii_type == PIIType.CREDIT_CARD:
            return f"****-****-****-{value[-4:]}" if len(value) >= 4 else "****"
        elif pii_type == PIIType.EMAIL:
            parts = value.split('@')
            if len(parts) == 2:
                return f"{parts[0][0]}***@{parts[1]}"
            return "***@***.***"
        elif pii_type == PIIType.PHONE:
            return f"(***) ***-{value[-4:]}" if len(value) >= 4 else "(***) ***-****"
        else:
            # Generic masking
            if len(value) <= 4:
                return "****"
            return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"

    async def redact_pii(
        self,
        text: str,
        pii_types: Optional[list[PIIType]] = None,
    ) -> tuple[str, list[PIIMatch]]:
        """
        Redact PII from text.

        Args:
            text: Text to redact
            pii_types: Specific PII types to redact (None for all)

        Returns:
            Tuple of (redacted_text, matches_found)
        """
        matches = await self.detect_pii(text)

        if pii_types:
            matches = [m for m in matches if m.pii_type in pii_types]

        # Sort by position (reverse) to redact without messing up positions
        matches.sort(key=lambda m: m.start_position, reverse=True)

        redacted = text
        for match in matches:
            redacted = (
                redacted[:match.start_position] +
                f"[REDACTED:{match.pii_type.value}]" +
                redacted[match.end_position:]
            )

        return redacted, matches

    # =========================================================================
    # Data Classification
    # =========================================================================

    async def classify_document(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SecurityClassification:
        """
        Classify document security level based on content.

        Args:
            content: Document content
            metadata: Document metadata

        Returns:
            Recommended security classification
        """
        # Check for attorney-client privilege indicators
        privilege_indicators = [
            r'attorney.client\s*privilege',
            r'confidential\s*communication',
            r'legal\s*advice',
            r'privileged\s*and\s*confidential',
        ]
        for pattern in privilege_indicators:
            if re.search(pattern, content, re.I):
                return SecurityClassification.ATTORNEY_CLIENT_PRIVILEGED

        # Check for PII
        pii_matches = await self.detect_pii(content)
        high_risk_pii = [m for m in pii_matches if m.pii_type in [
            PIIType.SSN, PIIType.CREDIT_CARD, PIIType.BANK_ACCOUNT,
            PIIType.MEDICAL_RECORD
        ]]

        if high_risk_pii:
            return SecurityClassification.RESTRICTED

        if pii_matches:
            return SecurityClassification.CONFIDENTIAL

        # Check metadata
        if metadata:
            if metadata.get('confidential') or metadata.get('restricted'):
                return SecurityClassification.CONFIDENTIAL

        # Check for confidential markers
        if re.search(r'\b(confidential|proprietary|trade\s*secret)\b', content, re.I):
            return SecurityClassification.CONFIDENTIAL

        if re.search(r'\b(internal\s*use\s*only|internal)\b', content, re.I):
            return SecurityClassification.INTERNAL

        return SecurityClassification.INTERNAL  # Default

    # =========================================================================
    # Legal Hold Management
    # =========================================================================

    async def apply_legal_hold(
        self,
        matter_id: str,
        matter_name: str,
        document_ids: list[str],
        custodians: list[str],
        reason: str,
        applied_by: str,
        search_criteria: Optional[dict[str, Any]] = None,
    ) -> LegalHold:
        """
        Apply a legal hold to documents.

        Args:
            matter_id: Legal matter identifier
            matter_name: Matter name/description
            document_ids: Documents to hold
            custodians: Custodian user IDs
            reason: Reason for hold
            applied_by: User applying the hold
            search_criteria: Optional search criteria for dynamic holds

        Returns:
            Created LegalHold
        """
        hold = LegalHold(
            id=str(uuid4()),
            matter_id=matter_id,
            matter_name=matter_name,
            document_ids=document_ids,
            custodians=custodians,
            reason=reason,
            applied_by=applied_by,
            search_criteria=search_criteria,
            applied_at=datetime.utcnow(),
            status="active",
        )

        self._legal_holds[hold.id] = hold

        # Log audit event
        if self.audit:
            await self.audit.log_compliance_event(
                event_type=AuditEventType.LEGAL_HOLD_APPLIED,
                resource_id=hold.id,
                user_id=applied_by,
                details={
                    "matter_id": matter_id,
                    "document_count": len(document_ids),
                    "custodian_count": len(custodians),
                },
            )

        logger.info("Legal hold applied: %s for matter %s", hold.id, matter_id)
        return hold

    async def release_legal_hold(
        self,
        hold_id: str,
        released_by: str,
    ) -> LegalHold:
        """Release a legal hold."""
        hold = self._legal_holds.get(hold_id)
        if not hold:
            raise ValueError(f"Legal hold not found: {hold_id}")

        hold.status = "released"
        hold.released_by = released_by
        hold.released_at = datetime.utcnow()

        # Log audit event
        if self.audit:
            await self.audit.log_compliance_event(
                event_type=AuditEventType.LEGAL_HOLD_RELEASED,
                resource_id=hold_id,
                user_id=released_by,
                details={"matter_id": hold.matter_id},
            )

        logger.info("Legal hold released: %s", hold_id)
        return hold

    async def check_legal_hold(self, document_id: str) -> Optional[LegalHold]:
        """Check if a document is under legal hold."""
        for hold in self._legal_holds.values():
            if hold.status == "active" and document_id in hold.document_ids:
                return hold
        return None

    async def get_legal_holds(
        self,
        status: Optional[str] = None,
        matter_id: Optional[str] = None,
    ) -> list[LegalHold]:
        """Get legal holds with optional filters."""
        holds = list(self._legal_holds.values())

        if status:
            holds = [h for h in holds if h.status == status]
        if matter_id:
            holds = [h for h in holds if h.matter_id == matter_id]

        return holds

    # =========================================================================
    # GDPR Data Subject Requests
    # =========================================================================

    async def create_gdpr_request(
        self,
        request_type: str,
        data_subject_id: str,
        data_subject_email: Optional[str] = None,
        handled_by: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> GDPRRequest:
        """
        Create a GDPR data subject request.

        Args:
            request_type: Type (access, rectification, erasure, portability, restriction)
            data_subject_id: Data subject identifier
            data_subject_email: Contact email
            handled_by: Assigned handler
            notes: Additional notes

        Returns:
            Created GDPRRequest
        """
        deadline = datetime.utcnow() + timedelta(days=self.config.gdpr_request_deadline_days)

        request = GDPRRequest(
            id=str(uuid4()),
            request_type=request_type,
            data_subject_id=data_subject_id,
            data_subject_email=data_subject_email,
            requested_at=datetime.utcnow(),
            deadline=deadline,
            status="pending",
            handled_by=handled_by,
            notes=notes,
        )

        self._gdpr_requests[request.id] = request

        # Log audit event
        event_type_map = {
            'access': AuditEventType.GDPR_REQUEST_ACCESS,
            'rectification': AuditEventType.GDPR_REQUEST_RECTIFY,
            'erasure': AuditEventType.GDPR_REQUEST_ERASE,
            'portability': AuditEventType.GDPR_REQUEST_PORTABILITY,
        }
        event_type = event_type_map.get(request_type, AuditEventType.GDPR_REQUEST_ACCESS)

        if self.audit:
            await self.audit.log_compliance_event(
                event_type=event_type,
                resource_id=request.id,
                user_id=data_subject_id,
                details={"request_type": request_type, "deadline": deadline.isoformat()},
            )

        logger.info("GDPR request created: %s (%s)", request.id, request_type)
        return request

    async def process_gdpr_request(
        self,
        request_id: str,
        response_data: Optional[dict[str, Any]] = None,
        handled_by: Optional[str] = None,
    ) -> GDPRRequest:
        """
        Process and complete a GDPR request.

        Args:
            request_id: Request to process
            response_data: Data to return (for access/portability)
            handled_by: Handler completing the request

        Returns:
            Updated GDPRRequest
        """
        request = self._gdpr_requests.get(request_id)
        if not request:
            raise ValueError(f"GDPR request not found: {request_id}")

        request.status = "completed"
        request.completed_at = datetime.utcnow()
        request.handled_by = handled_by
        request.response_data = response_data

        logger.info("GDPR request completed: %s", request_id)
        return request

    async def get_gdpr_requests(
        self,
        status: Optional[str] = None,
        data_subject_id: Optional[str] = None,
        overdue_only: bool = False,
    ) -> list[GDPRRequest]:
        """Get GDPR requests with filters."""
        requests = list(self._gdpr_requests.values())

        if status:
            requests = [r for r in requests if r.status == status]
        if data_subject_id:
            requests = [r for r in requests if r.data_subject_id == data_subject_id]
        if overdue_only:
            now = datetime.utcnow()
            requests = [
                r for r in requests
                if r.status == "pending" and r.deadline < now
            ]

        return requests

    # =========================================================================
    # Compliance Violations
    # =========================================================================

    async def record_violation(
        self,
        policy_id: str,
        resource_type: str,
        resource_id: str,
        violation_type: str,
        description: str,
        severity: str = "medium",
        remediation: Optional[str] = None,
    ) -> ComplianceViolation:
        """
        Record a compliance violation.

        Args:
            policy_id: Policy that was violated
            resource_type: Type of resource
            resource_id: Resource identifier
            violation_type: Type of violation
            description: Violation description
            severity: Severity level (low, medium, high, critical)
            remediation: Suggested remediation

        Returns:
            Created ComplianceViolation
        """
        policy = self._policies.get(policy_id)

        violation = ComplianceViolation(
            id=str(uuid4()),
            policy_id=policy_id,
            policy_name=policy.name if policy else policy_id,
            framework=policy.framework if policy else ComplianceFramework.SOC2,
            resource_type=resource_type,
            resource_id=resource_id,
            violation_type=violation_type,
            severity=severity,
            description=description,
            remediation=remediation,
            detected_at=datetime.utcnow(),
            status="open",
        )

        self._violations.append(violation)

        # Log audit event
        if self.audit:
            await self.audit.log_compliance_event(
                event_type=AuditEventType.COMPLIANCE_VIOLATION,
                resource_id=violation.id,
                details={
                    "policy_id": policy_id,
                    "violation_type": violation_type,
                    "severity": severity,
                },
            )

        logger.warning(
            "Compliance violation: %s - %s (%s)",
            violation_type, description, severity
        )
        return violation

    async def resolve_violation(
        self,
        violation_id: str,
        resolution: str,
        resolved_by: str,
    ) -> ComplianceViolation:
        """Resolve a compliance violation."""
        for violation in self._violations:
            if violation.id == violation_id:
                violation.status = "resolved"
                violation.resolved_at = datetime.utcnow()
                violation.remediation = resolution
                logger.info("Violation resolved: %s", violation_id)
                return violation

        raise ValueError(f"Violation not found: {violation_id}")

    async def get_violations(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        framework: Optional[ComplianceFramework] = None,
    ) -> list[ComplianceViolation]:
        """Get violations with filters."""
        violations = self._violations.copy()

        if status:
            violations = [v for v in violations if v.status == status]
        if severity:
            violations = [v for v in violations if v.severity == severity]
        if framework:
            violations = [v for v in violations if v.framework == framework]

        return violations

    # =========================================================================
    # Policy Evaluation
    # =========================================================================

    async def evaluate_document_compliance(
        self,
        document_id: str,
        content: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Evaluate document against all applicable compliance policies.

        Returns:
            Compliance evaluation result
        """
        results = {
            "document_id": document_id,
            "compliant": True,
            "violations": [],
            "warnings": [],
            "pii_detected": [],
            "classification": None,
            "evaluated_at": datetime.utcnow().isoformat(),
        }

        # Detect PII
        pii_matches = await self.detect_pii(content)
        results["pii_detected"] = [
            {"type": m.pii_type.value, "confidence": m.confidence}
            for m in pii_matches
        ]

        # Classify document
        classification = await self.classify_document(content, metadata)
        results["classification"] = classification.value

        # Check each enabled framework
        for framework_name in self.config.enabled_frameworks:
            framework = ComplianceFramework(framework_name)
            framework_result = await self._evaluate_framework(
                framework, document_id, content, metadata, pii_matches
            )
            if framework_result["violations"]:
                results["compliant"] = False
                results["violations"].extend(framework_result["violations"])
            results["warnings"].extend(framework_result.get("warnings", []))

        return results

    async def _evaluate_framework(
        self,
        framework: ComplianceFramework,
        document_id: str,
        content: str,
        metadata: dict[str, Any],
        pii_matches: list[PIIMatch],
    ) -> dict[str, Any]:
        """Evaluate document against a specific framework."""
        violations = []
        warnings = []

        if framework == ComplianceFramework.GDPR:
            # Check for unprotected PII
            if pii_matches and not metadata.get('encrypted'):
                violations.append({
                    "framework": "gdpr",
                    "rule": "pii_encryption",
                    "message": "Document contains PII but is not encrypted",
                })

            # Check for consent
            if pii_matches and not metadata.get('consent_obtained'):
                warnings.append({
                    "framework": "gdpr",
                    "rule": "consent_required",
                    "message": "Document contains PII - ensure consent is obtained",
                })

        elif framework == ComplianceFramework.HIPAA:
            # Check for PHI
            medical_pii = [m for m in pii_matches if m.pii_type == PIIType.MEDICAL_RECORD]
            if medical_pii:
                if not metadata.get('encrypted'):
                    violations.append({
                        "framework": "hipaa",
                        "rule": "phi_encryption",
                        "message": "Document contains PHI but is not encrypted",
                    })
                if not metadata.get('access_logged'):
                    warnings.append({
                        "framework": "hipaa",
                        "rule": "access_logging",
                        "message": "Ensure all access to PHI is logged",
                    })

        elif framework == ComplianceFramework.SOC2:
            # Check encryption for confidential data
            classification = await self.classify_document(content, metadata)
            if classification in [SecurityClassification.CONFIDENTIAL, SecurityClassification.RESTRICTED]:
                if not metadata.get('encrypted'):
                    violations.append({
                        "framework": "soc2",
                        "rule": "encryption_required",
                        "message": f"Document classified as {classification.value} requires encryption",
                    })

        return {"violations": violations, "warnings": warnings}

    # =========================================================================
    # Retention Policy
    # =========================================================================

    async def get_retention_policy(
        self,
        document_type: Optional[str] = None,
        classification: Optional[SecurityClassification] = None,
    ) -> Optional[DataRetentionPolicy]:
        """Get applicable retention policy for a document."""
        for policy in self._retention_policies.values():
            if not policy.enabled:
                continue

            # Check document type match
            if policy.document_types and document_type not in policy.document_types:
                continue

            # Check classification match
            if policy.classifications and classification not in policy.classifications:
                continue

            return policy

        return None

    async def check_retention_expiry(
        self,
        document_id: str,
        created_at: datetime,
        document_type: Optional[str] = None,
        classification: Optional[SecurityClassification] = None,
    ) -> dict[str, Any]:
        """Check if a document has exceeded its retention period."""
        policy = await self.get_retention_policy(document_type, classification)

        if not policy:
            return {
                "expired": False,
                "policy_id": None,
                "message": "No retention policy applies",
            }

        expiry_date = created_at + timedelta(days=policy.retention_days)
        expired = datetime.utcnow() > expiry_date

        return {
            "expired": expired,
            "policy_id": policy.id,
            "policy_name": policy.name,
            "retention_days": policy.retention_days,
            "expiry_date": expiry_date.isoformat(),
            "action": policy.action if expired else None,
        }

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> dict[str, Any]:
        """Check compliance service health."""
        if not self._initialized:
            return {"status": "not_initialized"}

        # Check for overdue GDPR requests
        overdue_requests = await self.get_gdpr_requests(overdue_only=True)

        # Check for open violations
        open_violations = await self.get_violations(status="open")
        critical_violations = [v for v in open_violations if v.severity == "critical"]

        return {
            "status": "healthy",
            "frameworks_enabled": self.config.enabled_frameworks,
            "policies_loaded": len(self._policies),
            "retention_policies": len(self._retention_policies),
            "active_legal_holds": len([h for h in self._legal_holds.values() if h.status == "active"]),
            "pending_gdpr_requests": len(await self.get_gdpr_requests(status="pending")),
            "overdue_gdpr_requests": len(overdue_requests),
            "open_violations": len(open_violations),
            "critical_violations": len(critical_violations),
        }
