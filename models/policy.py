from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class Policy:
    """Represents a policy document."""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]

@dataclass
class PolicySection:
    """Represents a section of a policy document."""
    policy_id: str
    title: str
    content: str
    domain_relevance: Dict[str, float]  # Domain name -> relevance score

@dataclass
class PolicySectionMatch:
    """Represents a match between company and reference policy sections."""
    company_section_id: str
    reference_section_id: str
    match_score: float
    alignment_notes: str