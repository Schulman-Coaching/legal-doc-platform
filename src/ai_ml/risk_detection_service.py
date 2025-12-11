"""
Risk Detection Service
=======================
Detect and assess risks in legal documents.
"""

import logging
import re
from typing import Any, Optional

from .llm_gateway import LLMGateway
from .models import RiskAssessment, RiskLevel

logger = logging.getLogger(__name__)


class RiskDetectionService:
    """Service for detecting risks and anomalies in legal documents."""

    # Risk indicators with weights (0-1 scale)
    RISK_INDICATORS = {
        # Liability risks (high weight)
        "unlimited liability": 0.9,
        "uncapped liability": 0.9,
        "unlimited damages": 0.85,
        "waive all rights": 0.85,
        "sole discretion": 0.7,
        "absolute discretion": 0.75,
        "waives any claim": 0.8,

        # Indemnification risks
        "broad indemnification": 0.7,
        "indemnify and hold harmless": 0.5,
        "defend and indemnify": 0.5,
        "third party claims": 0.4,

        # Term/termination risks
        "perpetual": 0.6,
        "irrevocable": 0.7,
        "automatic renewal": 0.5,
        "no termination": 0.8,
        "termination for convenience": 0.4,
        "immediate termination": 0.5,

        # Dispute risks
        "binding arbitration": 0.4,
        "class action waiver": 0.6,
        "mandatory arbitration": 0.5,
        "exclusive venue": 0.3,
        "waive jury trial": 0.5,

        # Remedy limitations
        "exclusive remedy": 0.5,
        "sole remedy": 0.5,
        "liquidated damages": 0.4,
        "penalty": 0.5,
        "consequential damages waiver": 0.4,

        # Financial risks
        "material adverse change": 0.4,
        "change of control": 0.3,
        "acceleration clause": 0.4,
        "cross-default": 0.5,

        # IP/Confidentiality risks
        "work for hire": 0.4,
        "assignment of ip": 0.5,
        "perpetual license": 0.4,
        "survival of confidentiality": 0.3,

        # One-sided terms
        "at its option": 0.4,
        "in its sole judgment": 0.6,
        "without cause": 0.5,
        "without notice": 0.6,
        "non-negotiable": 0.4,
    }

    # Patterns that indicate favorable terms (negative risk)
    FAVORABLE_PATTERNS = {
        "mutual indemnification": -0.3,
        "mutual termination": -0.2,
        "reasonable notice": -0.2,
        "good faith": -0.1,
        "commercially reasonable": -0.1,
        "liability cap": -0.3,
        "limitation of liability": -0.2,
    }

    def __init__(self, llm_gateway: LLMGateway):
        """Initialize risk detection service."""
        self.llm = llm_gateway

    async def assess_document_risk(
        self,
        document_text: str,
        document_type: str = "contract",
        perspective: str = "neutral",
    ) -> RiskAssessment:
        """
        Assess overall risk of a document.

        Args:
            document_text: Full document text
            document_type: Type of document (contract, agreement, etc.)
            perspective: Risk assessment perspective

        Returns:
            Comprehensive risk assessment
        """
        # Keyword-based risk scoring
        keyword_risks = self._keyword_risk_analysis(document_text)

        # AI-based risk analysis
        ai_risks = await self._ai_risk_analysis(document_text, document_type, perspective)

        # Combine assessments (weighted average)
        keyword_weight = 0.3
        ai_weight = 0.7

        combined_score = (
            keyword_risks["score"] * keyword_weight +
            ai_risks["score"] * ai_weight
        )

        # Merge risk factors, removing duplicates
        all_factors = []
        seen_factors = set()

        for factor in ai_risks["factors"] + keyword_risks["factors"]:
            factor_key = factor.get("factor", "").lower()
            if factor_key not in seen_factors:
                seen_factors.add(factor_key)
                all_factors.append(factor)

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_factors.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 4))

        # Determine overall risk level
        if combined_score >= 0.8:
            overall_risk = RiskLevel.CRITICAL
        elif combined_score >= 0.6:
            overall_risk = RiskLevel.HIGH
        elif combined_score >= 0.4:
            overall_risk = RiskLevel.MEDIUM
        else:
            overall_risk = RiskLevel.LOW

        # Generate recommendations
        recommendations = self._generate_recommendations(all_factors)

        return RiskAssessment(
            overall_risk=overall_risk,
            risk_score=combined_score,
            risk_factors=all_factors,
            recommendations=recommendations,
            confidence=0.85,
        )

    def _keyword_risk_analysis(self, text: str) -> dict[str, Any]:
        """Perform keyword-based risk analysis."""
        text_lower = text.lower()
        factors = []
        total_weight = sum(self.RISK_INDICATORS.values())
        matched_weight = 0

        for indicator, weight in self.RISK_INDICATORS.items():
            if indicator in text_lower:
                # Find context around the match
                idx = text_lower.find(indicator)
                start = max(0, idx - 50)
                end = min(len(text), idx + len(indicator) + 50)
                context = text[start:end]

                severity = "critical" if weight > 0.8 else "high" if weight > 0.6 else "medium" if weight > 0.4 else "low"

                factors.append({
                    "factor": indicator.replace("_", " ").title(),
                    "severity": severity,
                    "weight": weight,
                    "type": "keyword_match",
                    "context": f"...{context}...",
                })
                matched_weight += weight

        # Check for favorable patterns
        for pattern, adjustment in self.FAVORABLE_PATTERNS.items():
            if pattern in text_lower:
                matched_weight += adjustment  # Negative adjustment reduces risk

        score = min(1.0, max(0.0, matched_weight / (total_weight * 0.3)))  # Normalize

        return {
            "score": score,
            "factors": factors,
        }

    async def _ai_risk_analysis(
        self,
        text: str,
        document_type: str,
        perspective: str,
    ) -> dict[str, Any]:
        """Perform AI-based risk analysis."""
        prompt = f"""Analyze this {document_type} for legal and business risks.
Perspective: {perspective}

Consider:
1. Liability exposure and limitations
2. Indemnification scope and balance
3. Termination provisions and consequences
4. IP ownership and licensing issues
5. Confidentiality and data protection
6. Regulatory compliance requirements
7. Financial terms and payment risks
8. One-sided or unusual provisions
9. Missing standard protections
10. Ambiguous language creating uncertainty

Document:
{text[:12000]}

Return as JSON:
{{
  "overall_score": 0.0-1.0,
  "risk_factors": [
    {{
      "factor": "description",
      "severity": "critical|high|medium|low",
      "clause_reference": "section or location",
      "explanation": "why this is risky",
      "mitigation": "how to address"
    }}
  ]
}}"""

        try:
            result = await self.llm.generate_json(
                prompt,
                system_prompt="You are an expert legal risk analyst. Identify and assess risks objectively and thoroughly.",
            )

            factors = result.get("risk_factors", [])
            score = result.get("overall_score", 0.5)

            # Add type marker
            for factor in factors:
                factor["type"] = "ai_analysis"

            return {
                "score": score,
                "factors": factors,
            }

        except Exception as e:
            logger.error(f"AI risk analysis failed: {e}")
            return {
                "score": 0.5,
                "factors": [],
            }

    def _generate_recommendations(
        self,
        risk_factors: list[dict[str, Any]],
    ) -> list[str]:
        """Generate actionable recommendations based on risk factors."""
        recommendations = set()

        recommendation_map = {
            "liability": [
                "Negotiate a liability cap (e.g., contract value or annual fees)",
                "Add mutual limitation of liability clause",
            ],
            "indemnif": [
                "Request mutual indemnification provisions",
                "Limit indemnification to direct claims, not third-party",
                "Add indemnification cap",
            ],
            "terminat": [
                "Add mutual termination rights with reasonable notice",
                "Include termination for convenience clause",
                "Clarify post-termination obligations",
            ],
            "arbitration": [
                "Consider preserving litigation option",
                "Negotiate arbitration venue and rules",
                "Ensure arbitration costs are reasonable",
            ],
            "renewal": [
                "Add renewal notice requirement (e.g., 60 days prior)",
                "Include right to not renew without penalty",
            ],
            "discretion": [
                "Change 'sole discretion' to 'reasonable discretion'",
                "Add objective criteria for discretionary decisions",
            ],
            "perpetual": [
                "Negotiate finite term with renewal options",
                "Add termination rights even for perpetual provisions",
            ],
            "waiv": [
                "Review waiver scope carefully",
                "Ensure waiver is mutual where appropriate",
            ],
            "exclusive": [
                "Consider non-exclusive alternatives",
                "Add performance requirements for exclusive rights",
            ],
            "damages": [
                "Negotiate carve-outs for fraud, gross negligence",
                "Ensure consequential damages waiver is mutual",
            ],
        }

        for factor in risk_factors:
            factor_lower = factor.get("factor", "").lower()

            for keyword, recs in recommendation_map.items():
                if keyword in factor_lower:
                    recommendations.update(recs)

            # Also add mitigation from AI analysis
            if factor.get("mitigation"):
                recommendations.add(factor["mitigation"])

        # Prioritize and limit recommendations
        return list(recommendations)[:10]

    async def detect_anomalies(
        self,
        document_text: str,
        reference_documents: Optional[list[str]] = None,
        document_type: str = "contract",
    ) -> list[dict[str, Any]]:
        """
        Detect anomalies compared to standard documents.

        Args:
            document_text: Document to analyze
            reference_documents: Optional reference documents for comparison
            document_type: Type of document

        Returns:
            List of detected anomalies
        """
        reference_context = ""
        if reference_documents:
            reference_context = f"""
Reference documents for comparison:
{' '.join(ref[:1000] for ref in reference_documents[:3])}
"""

        prompt = f"""Identify anomalies and unusual provisions in this {document_type}.
Compare against standard industry practices and typical contract language.

{reference_context}

Document to analyze:
{document_text[:10000]}

Return as JSON array:
[
  {{
    "anomaly": "description of unusual element",
    "typical_practice": "what would normally be expected",
    "location": "where in document",
    "severity": "high|medium|low",
    "recommendation": "how to address"
  }}
]"""

        try:
            result = await self.llm.generate_json(
                prompt,
                system_prompt="You are an expert at identifying unusual or non-standard contract provisions.",
            )

            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "items" in result:
                return result["items"]
            return []

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []

    async def compare_risk_profiles(
        self,
        document1_text: str,
        document2_text: str,
    ) -> dict[str, Any]:
        """Compare risk profiles of two documents."""
        risk1 = await self.assess_document_risk(document1_text)
        risk2 = await self.assess_document_risk(document2_text)

        return {
            "document1": {
                "overall_risk": risk1.overall_risk.value,
                "risk_score": risk1.risk_score,
                "factor_count": len(risk1.risk_factors),
            },
            "document2": {
                "overall_risk": risk2.overall_risk.value,
                "risk_score": risk2.risk_score,
                "factor_count": len(risk2.risk_factors),
            },
            "comparison": {
                "safer_document": 1 if risk1.risk_score < risk2.risk_score else 2,
                "score_difference": abs(risk1.risk_score - risk2.risk_score),
                "unique_risks_doc1": len([f for f in risk1.risk_factors if f not in risk2.risk_factors]),
                "unique_risks_doc2": len([f for f in risk2.risk_factors if f not in risk1.risk_factors]),
            },
        }

    def quick_risk_scan(self, document_text: str) -> dict[str, Any]:
        """
        Perform quick keyword-only risk scan (no LLM call).
        Useful for initial screening of many documents.
        """
        result = self._keyword_risk_analysis(document_text)

        if result["score"] >= 0.6:
            risk_level = "high"
        elif result["score"] >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "risk_level": risk_level,
            "risk_score": result["score"],
            "indicators_found": len(result["factors"]),
            "top_concerns": [f["factor"] for f in result["factors"][:5]],
            "needs_detailed_review": result["score"] >= 0.4,
        }
