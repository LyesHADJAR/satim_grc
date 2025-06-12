from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class PolicySection:
    """Represents a section of a policy document."""
    document_id: str
    content: str
    domain: str
    policy_type: str  # "company" or "reference"
    relevance_score: float
    section_title: Optional[str] = None
    source_document: Optional[str] = None
    page_number: Optional[int] = None
    compliance_framework: Optional[str] = None
    authority_level: Optional[int] = None

@dataclass
class Gap:
    """Represents a gap between company and reference policies."""
    type: str  # "missing", "insufficient", "outdated"
    severity: str  # "critical", "high", "medium", "low"
    description: str
    reference_requirement: str
    suggested_action: str
    domain: str

@dataclass
class Overlap:
    """Represents areas where company policies exceed reference requirements."""
    description: str
    company_provision: str
    reference_requirement: str
    value_assessment: str  # "positive", "neutral", "excessive"
    domain: str

@dataclass
class Recommendation:
    """Actionable recommendation for policy improvement."""
    priority: str  # "high", "medium", "low"
    action_type: str  # "policy_creation", "policy_enhancement", "policy_review"
    description: str
    suggested_action: str
    domain: str
    estimated_effort: str  # "low", "medium", "high"
    compliance_impact: str  # "critical", "high", "medium", "low"

@dataclass
class ComparisonResult:
    """Complete result of a policy comparison analysis."""
    domain: str
    company_policy_sections: List[PolicySection]
    reference_sections: List[PolicySection]
    gaps: List[Gap]
    overlaps: List[Overlap]
    coverage_score: float  # 0-100
    implementation_score: float  # 0-100
    recommendations: List[Recommendation]
    timestamp: datetime