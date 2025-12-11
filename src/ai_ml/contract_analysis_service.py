"""
Contract Analysis Service
==========================
Comprehensive analysis of legal contracts using LLM.
"""

import asyncio
import logging
import time
from typing import Any, Optional

from .llm_gateway import LLMGateway
from .models import ContractReviewResult, RiskAssessment, RiskLevel

logger = logging.getLogger(__name__)


class ContractAnalysisService:
    """Service for comprehensive contract analysis."""

    SYSTEM_PROMPT = """You are an expert contract attorney with extensive experience
in commercial contracts, corporate law, and risk assessment. Provide thorough,
accurate analysis focused on protecting your client's interests. Be specific
about clause locations and exact language when identifying issues."""

    OBLIGATION_EXTRACTION_PROMPT = """Extract all obligations from this contract.
For each obligation, identify:
1. The obligated party (who must perform)
2. The nature of the obligation (what must be done)
3. Any conditions or triggers (when it applies)
4. Deadlines or timeframes
5. Consequences of non-compliance
6. Section/clause reference

Contract text:
{contract_text}

Return as JSON array:
[
  {{
    "party": "party name",
    "obligation": "description",
    "condition": "trigger condition or null",
    "deadline": "timeframe or null",
    "consequence": "penalty/remedy or null",
    "section": "section reference"
  }}
]"""

    RISK_ANALYSIS_PROMPT = """Analyze this contract for legal and business risks.
Consider from the {perspective}'s perspective:

1. Liability exposure (unlimited liability, indemnification scope)
2. Financial risks (payment terms, penalties, liquidated damages)
3. Operational risks (performance standards, SLAs, termination)
4. IP/Confidentiality risks (ownership, permitted use, disclosure)
5. Regulatory/Compliance risks
6. Missing standard protections
7. Unusual or one-sided provisions

Contract text:
{contract_text}

Return as JSON:
{{
  "risk_factors": [
    {{
      "factor": "risk description",
      "severity": "low|medium|high|critical",
      "clause": "section reference",
      "current_language": "problematic text",
      "recommendation": "suggested change"
    }}
  ],
  "overall_risk_score": 0.0-1.0,
  "summary": "brief risk summary"
}}"""

    KEY_TERMS_PROMPT = """Extract all defined terms and key financial/legal terms from this contract.

Contract text:
{contract_text}

Return as JSON array:
[
  {{
    "term": "term name",
    "definition": "definition text",
    "section": "where defined",
    "category": "defined_term|financial|date|party|other"
  }}
]"""

    DEADLINES_PROMPT = """Extract all important dates, deadlines, and time-sensitive requirements.
Include:
- Contract term dates (effective, expiration)
- Notice periods (for termination, renewal, etc.)
- Payment due dates
- Milestone deadlines
- Renewal/option exercise dates
- Reporting deadlines

Contract text:
{contract_text}

Return as JSON array:
[
  {{
    "type": "term_start|term_end|renewal|notice|payment|milestone|reporting|other",
    "date": "specific date or null",
    "timeframe": "relative timeframe (e.g., '30 days after notice')",
    "description": "what this deadline is for",
    "is_recurring": true/false,
    "section": "section reference"
  }}
]"""

    NEGOTIATION_POINTS_PROMPT = """Identify key points to negotiate in this contract.
Analyze from the {perspective}'s perspective. Focus on terms that:
1. Favor the other party disproportionately
2. Create unnecessary risk or liability
3. Lack standard protections
4. Have ambiguous language that could be exploited
5. Include unusual or non-standard requirements
6. Have unreasonable timeframes or penalties

Contract text:
{contract_text}

Return as JSON array with priority (high/medium/low):
[
  {{
    "clause": "clause name",
    "section": "section reference",
    "issue": "what's wrong",
    "current_language": "problematic text",
    "suggested_language": "recommended revision",
    "priority": "high|medium|low",
    "rationale": "why this matters"
  }}
]"""

    NON_STANDARD_PROMPT = """Identify unusual or non-standard provisions in this contract.
Compare against typical contract language and industry standards.
Flag anything that deviates from standard practice.

Contract text:
{contract_text}

Return as JSON array:
[
  {{
    "provision": "description of unusual provision",
    "section": "section reference",
    "typical_approach": "what's usually done",
    "concern": "why this is problematic",
    "severity": "low|medium|high"
  }}
]"""

    MISSING_CLAUSES = [
        "force majeure",
        "severability",
        "entire agreement",
        "governing law",
        "dispute resolution",
        "confidentiality",
        "data protection",
        "insurance",
        "assignment",
        "notices",
        "amendments",
        "waiver",
        "survival",
        "representations and warranties",
        "limitation of liability",
    ]

    def __init__(self, llm_gateway: LLMGateway):
        """Initialize contract analysis service."""
        self.llm = llm_gateway

    async def analyze_contract(
        self,
        document_id: str,
        contract_text: str,
        party_perspective: str = "neutral",
    ) -> ContractReviewResult:
        """
        Perform comprehensive contract analysis.

        Args:
            document_id: Document identifier
            contract_text: Full contract text
            party_perspective: Analysis perspective ("neutral", "buyer", "seller", etc.)

        Returns:
            Complete contract review result
        """
        start_time = time.time()

        # Truncate if too long (keep most important parts)
        max_chars = 30000
        if len(contract_text) > max_chars:
            # Keep beginning and end, which usually have key terms
            contract_text = contract_text[:max_chars // 2] + "\n...[truncated]...\n" + contract_text[-max_chars // 2:]

        # Run analyses in parallel
        tasks = [
            self._generate_summary(contract_text),
            self._extract_obligations(contract_text),
            self._analyze_risks(contract_text, party_perspective),
            self._extract_key_terms(contract_text),
            self._extract_deadlines(contract_text),
            self._identify_negotiation_points(contract_text, party_perspective),
            self._identify_non_standard_provisions(contract_text),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, handling any failures
        summary = results[0] if not isinstance(results[0], Exception) else "Summary generation failed"
        obligations = results[1] if not isinstance(results[1], Exception) else []
        risk_data = results[2] if not isinstance(results[2], Exception) else {"risk_factors": [], "overall_risk_score": 0.5}
        key_terms = results[3] if not isinstance(results[3], Exception) else []
        deadlines = results[4] if not isinstance(results[4], Exception) else []
        negotiation_points = results[5] if not isinstance(results[5], Exception) else []
        non_standard = results[6] if not isinstance(results[6], Exception) else []

        # Check for missing clauses
        missing_clauses = self._check_missing_clauses(contract_text)

        # Build risk assessment
        risk_score = risk_data.get("overall_risk_score", 0.5)
        if risk_score >= 0.8:
            overall_risk = RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            overall_risk = RiskLevel.HIGH
        elif risk_score >= 0.4:
            overall_risk = RiskLevel.MEDIUM
        else:
            overall_risk = RiskLevel.LOW

        risk_assessment = RiskAssessment(
            overall_risk=overall_risk,
            risk_score=risk_score,
            risk_factors=risk_data.get("risk_factors", []),
            recommendations=[
                rf.get("recommendation", "")
                for rf in risk_data.get("risk_factors", [])
                if rf.get("recommendation")
            ],
            confidence=0.85,
        )

        processing_time = int((time.time() - start_time) * 1000)

        return ContractReviewResult(
            document_id=document_id,
            summary=summary,
            key_terms=key_terms,
            obligations=obligations,
            deadlines=deadlines,
            risk_assessment=risk_assessment,
            missing_clauses=missing_clauses,
            non_standard_provisions=non_standard,
            negotiation_points=negotiation_points,
            processing_time_ms=processing_time,
        )

    async def _generate_summary(self, contract_text: str) -> str:
        """Generate contract summary."""
        prompt = f"""Provide a concise summary of this contract including:
- Type of contract/agreement
- Parties involved
- Main purpose and subject matter
- Key terms and conditions
- Contract duration/term
- Notable provisions or concerns

Contract:
{contract_text[:10000]}

Summary:"""

        response = await self.llm.generate(prompt, system_prompt=self.SYSTEM_PROMPT)
        return response.text

    async def _extract_obligations(self, contract_text: str) -> list[dict[str, Any]]:
        """Extract obligations from contract."""
        prompt = self.OBLIGATION_EXTRACTION_PROMPT.format(
            contract_text=contract_text[:12000]
        )

        try:
            result = await self.llm.generate_json(prompt, system_prompt=self.SYSTEM_PROMPT)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "items" in result:
                return result["items"]
            return []
        except Exception as e:
            logger.error(f"Failed to extract obligations: {e}")
            return []

    async def _analyze_risks(
        self,
        contract_text: str,
        perspective: str,
    ) -> dict[str, Any]:
        """Analyze contract risks."""
        prompt = self.RISK_ANALYSIS_PROMPT.format(
            contract_text=contract_text[:12000],
            perspective=perspective,
        )

        try:
            result = await self.llm.generate_json(prompt, system_prompt=self.SYSTEM_PROMPT)
            return result
        except Exception as e:
            logger.error(f"Failed to analyze risks: {e}")
            return {"risk_factors": [], "overall_risk_score": 0.5}

    async def _extract_key_terms(self, contract_text: str) -> list[dict[str, Any]]:
        """Extract key terms and definitions."""
        prompt = self.KEY_TERMS_PROMPT.format(contract_text=contract_text[:12000])

        try:
            result = await self.llm.generate_json(prompt, system_prompt=self.SYSTEM_PROMPT)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "items" in result:
                return result["items"]
            return []
        except Exception as e:
            logger.error(f"Failed to extract key terms: {e}")
            return []

    async def _extract_deadlines(self, contract_text: str) -> list[dict[str, Any]]:
        """Extract important dates and deadlines."""
        prompt = self.DEADLINES_PROMPT.format(contract_text=contract_text[:12000])

        try:
            result = await self.llm.generate_json(prompt, system_prompt=self.SYSTEM_PROMPT)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "items" in result:
                return result["items"]
            return []
        except Exception as e:
            logger.error(f"Failed to extract deadlines: {e}")
            return []

    async def _identify_negotiation_points(
        self,
        contract_text: str,
        perspective: str,
    ) -> list[dict[str, Any]]:
        """Identify negotiation points."""
        prompt = self.NEGOTIATION_POINTS_PROMPT.format(
            contract_text=contract_text[:12000],
            perspective=perspective,
        )

        try:
            result = await self.llm.generate_json(prompt, system_prompt=self.SYSTEM_PROMPT)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "items" in result:
                return result["items"]
            return []
        except Exception as e:
            logger.error(f"Failed to identify negotiation points: {e}")
            return []

    async def _identify_non_standard_provisions(
        self,
        contract_text: str,
    ) -> list[dict[str, Any]]:
        """Identify non-standard provisions."""
        prompt = self.NON_STANDARD_PROMPT.format(contract_text=contract_text[:12000])

        try:
            result = await self.llm.generate_json(prompt, system_prompt=self.SYSTEM_PROMPT)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "items" in result:
                return result["items"]
            return []
        except Exception as e:
            logger.error(f"Failed to identify non-standard provisions: {e}")
            return []

    def _check_missing_clauses(self, contract_text: str) -> list[str]:
        """Check for commonly expected but missing clauses."""
        text_lower = contract_text.lower()
        missing = []

        for clause in self.MISSING_CLAUSES:
            # Check for various forms of the clause name
            variations = [
                clause,
                clause.replace(" ", "-"),
                clause.replace(" ", "_"),
            ]

            found = any(var in text_lower for var in variations)
            if not found:
                missing.append(clause)

        return missing

    async def compare_contracts(
        self,
        contract1_text: str,
        contract2_text: str,
        contract1_name: str = "Contract 1",
        contract2_name: str = "Contract 2",
    ) -> dict[str, Any]:
        """Compare two contracts."""
        prompt = f"""Compare these two contracts and identify:

1. KEY DIFFERENCES in terms, conditions, and provisions
2. MISSING CLAUSES that appear in one but not the other
3. CONFLICTING TERMS between the two
4. Which contract is MORE FAVORABLE overall (and to whom)
5. RECOMMENDATIONS for harmonizing the contracts

{contract1_name}:
{contract1_text[:8000]}

{contract2_name}:
{contract2_text[:8000]}

Provide detailed comparison:"""

        response = await self.llm.generate(prompt, system_prompt=self.SYSTEM_PROMPT)

        return {
            "comparison": response.text,
            "contract1_name": contract1_name,
            "contract2_name": contract2_name,
        }

    async def generate_redline_suggestions(
        self,
        contract_text: str,
        party_perspective: str,
    ) -> list[dict[str, Any]]:
        """Generate specific redline suggestions for the contract."""
        prompt = f"""As counsel for the {party_perspective}, provide specific redline suggestions
for this contract. For each suggestion:
- Quote the original language
- Provide the proposed revision
- Explain why this change protects your client

Focus on the most impactful changes.

Contract:
{contract_text[:10000]}

Return as JSON array:
[
  {{
    "section": "section reference",
    "original": "original language",
    "proposed": "proposed revision",
    "rationale": "why this change",
    "priority": "high|medium|low"
  }}
]"""

        try:
            result = await self.llm.generate_json(prompt, system_prompt=self.SYSTEM_PROMPT)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "items" in result:
                return result["items"]
            return []
        except Exception as e:
            logger.error(f"Failed to generate redline suggestions: {e}")
            return []
