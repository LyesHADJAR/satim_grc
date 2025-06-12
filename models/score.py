from typing import List
from dataclasses import dataclass

@dataclass
class ScoreCriteria:
    """Individual scoring criteria."""
    name: str
    weight: float
    score: float
    
    @property
    def weighted_score(self) -> float:
        """Calculate the weighted score."""
        return self.weight * self.score

@dataclass
class ComplianceScore:
    """Compliance score for a domain or overall."""
    domain: str
    score: float
    criteria: List[ScoreCriteria]
    max_score: float
    recommendations: List[str]