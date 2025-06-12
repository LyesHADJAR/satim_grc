import asyncio
import json
import hashlib
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

from ..models.comparison_models import PolicySection, Gap, Overlap, Recommendation
from ..prompts.comparison_prompts import GAP_ANALYSIS_PROMPT, OVERLAP_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    gaps: List[Gap]
    overlaps: List[Overlap]
    coverage_score: float
    implementation_score: float
    recommendations: List[Recommendation]

class GapAnalysisProcessor:
    """
    Processes gap analysis between company and reference policies using LLM.
    """
    
    def __init__(self, llm_client, cache_client=None):
        self.llm = llm_client
        self.cache = cache_client
        
    async def analyze_gaps(self, 
                          company_sections: List[PolicySection], 
                          reference_sections: List[PolicySection],
                          domain: str) -> AnalysisResult:
        """
        Analyze gaps between company and reference policies.
        
        Args:
            company_sections: Company policy sections
            reference_sections: Reference framework sections
            domain: Policy domain being analyzed
            
        Returns:
            AnalysisResult with identified gaps and recommendations
        """
        logger.info(f"Starting gap analysis for domain: {domain}")
        
        # Create cache key for this analysis
        cache_key = self._create_cache_key(company_sections, reference_sections, domain)
        
        # Check cache first
        if self.cache:
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Using cached analysis for domain: {domain}")
                return cached_result
        
        # Prepare content for LLM analysis
        company_content = self._aggregate_policy_content(company_sections)
        reference_content = self._aggregate_policy_content(reference_sections)
        
        # Perform gap analysis using LLM
        gap_analysis = await self._perform_gap_analysis(
            company_content, reference_content, domain
        )
        
        # Perform overlap analysis
        overlap_analysis = await self._perform_overlap_analysis(
            company_content, reference_content, domain
        )
        
        # Calculate scores
        coverage_score = self._calculate_coverage_score(gap_analysis, reference_sections)
        implementation_score = self._calculate_implementation_score(gap_analysis, company_sections)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(gap_analysis, overlap_analysis, domain)
        
        # Create result
        result = AnalysisResult(
            gaps=gap_analysis,
            overlaps=overlap_analysis,
            coverage_score=coverage_score,
            implementation_score=implementation_score,
            recommendations=recommendations
        )
        
        # Cache the result
        if self.cache:
            await self._cache_result(cache_key, result)
        
        logger.info(f"Completed gap analysis for {domain}: {len(gap_analysis)} gaps identified")
        return result
    
    async def _perform_gap_analysis(self, 
                                   company_content: str, 
                                   reference_content: str, 
                                   domain: str) -> List[Gap]:
        """
        Use LLM to identify gaps between company and reference policies.
        """
        prompt = GAP_ANALYSIS_PROMPT.format(
            company_policy_content=company_content,
            reference_framework_content=reference_content,
            domain=domain
        )
        
        try:
            response = await self.llm.acomplete(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.1
            )
            
            # Parse LLM response into structured gaps
            gaps_data = json.loads(response.text)
            gaps = []
            
            for gap_data in gaps_data.get("gaps", []):
                gap = Gap(
                    type=gap_data.get("type", "missing"),
                    severity=gap_data.get("severity", "medium"),
                    description=gap_data.get("description", ""),
                    reference_requirement=gap_data.get("reference_requirement", ""),
                    suggested_action=gap_data.get("suggested_action", ""),
                    domain=domain
                )
                gaps.append(gap)
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error in gap analysis: {str(e)}")
            return []
    
    async def _perform_overlap_analysis(self, 
                                       company_content: str, 
                                       reference_content: str, 
                                       domain: str) -> List[Overlap]:
        """
        Identify areas where company policies exceed reference requirements.
        """
        prompt = OVERLAP_ANALYSIS_PROMPT.format(
            company_policy_content=company_content,
            reference_framework_content=reference_content,
            domain=domain
        )
        
        try:
            response = await self.llm.acomplete(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.1
            )
            
            overlaps_data = json.loads(response.text)
            overlaps = []
            
            for overlap_data in overlaps_data.get("overlaps", []):
                overlap = Overlap(
                    description=overlap_data.get("description", ""),
                    company_provision=overlap_data.get("company_provision", ""),
                    reference_requirement=overlap_data.get("reference_requirement", ""),
                    value_assessment=overlap_data.get("value_assessment", "positive"),
                    domain=domain
                )
                overlaps.append(overlap)
            
            return overlaps
            
        except Exception as e:
            logger.error(f"Error in overlap analysis: {str(e)}")
            return []
    
    def _aggregate_policy_content(self, sections: List[PolicySection]) -> str:
        """
        Combine policy sections into a single content string for analysis.
        """
        if not sections:
            return "No policy content found for this domain."
        
        content_parts = []
        for section in sections:
            header = f"## {section.section_title or 'Policy Section'}"
            if section.source_document:
                header += f" (Source: {section.source_document})"
            
            content_parts.append(f"{header}\n{section.content}\n")
        
        return "\n".join(content_parts)
    
    def _calculate_coverage_score(self, gaps: List[Gap], reference_sections: List[PolicySection]) -> float:
        """
        Calculate how well company policies cover reference requirements.
        """
        if not reference_sections:
            return 100.0
        
        # Simple scoring: reduce score based on gap severity
        total_deductions = 0
        for gap in gaps:
            if gap.severity == "critical":
                total_deductions += 20
            elif gap.severity == "high":
                total_deductions += 15
            elif gap.severity == "medium":
                total_deductions += 10
            elif gap.severity == "low":
                total_deductions += 5
        
        # Base score starts at 100, deduct points for gaps
        score = max(0, 100 - total_deductions)
        return round(score, 2)
    
    def _calculate_implementation_score(self, gaps: List[Gap], company_sections: List[PolicySection]) -> float:
        """
        Calculate implementation quality of existing company policies.
        """
        if not company_sections:
            return 0.0
        
        # Score based on policy completeness and gap types
        insufficient_gaps = len([g for g in gaps if g.type == "insufficient"])
        missing_gaps = len([g for g in gaps if g.type == "missing"])
        
        # Start with base score based on policy existence
        base_score = min(80, len(company_sections) * 10)
        
        # Deduct for implementation issues
        implementation_deductions = insufficient_gaps * 8 + missing_gaps * 5
        
        score = max(0, base_score - implementation_deductions)
        return round(score, 2)
    
    def _generate_recommendations(self, 
                                 gaps: List[Gap], 
                                 overlaps: List[Overlap], 
                                 domain: str) -> List[Recommendation]:
        """
        Generate actionable recommendations based on analysis results.
        """
        recommendations = []
        
        # High-priority recommendations for critical gaps
        critical_gaps = [g for g in gaps if g.severity == "critical"]
        for gap in critical_gaps:
            rec = Recommendation(
                priority="high",
                action_type="policy_creation" if gap.type == "missing" else "policy_enhancement",
                description=f"Address critical gap: {gap.description}",
                suggested_action=gap.suggested_action,
                domain=domain,
                estimated_effort="high",
                compliance_impact="critical"
            )
            recommendations.append(rec)
        
        # Medium-priority recommendations for other gaps
        other_gaps = [g for g in gaps if g.severity in ["high", "medium"]]
        for gap in other_gaps[:3]:  # Limit to top 3 to avoid overwhelming
            rec = Recommendation(
                priority="medium",
                action_type="policy_enhancement",
                description=f"Improve policy coverage: {gap.description}",
                suggested_action=gap.suggested_action,
                domain=domain,
                estimated_effort="medium",
                compliance_impact="high" if gap.severity == "high" else "medium"
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _create_cache_key(self, 
                         company_sections: List[PolicySection], 
                         reference_sections: List[PolicySection],
                         domain: str) -> str:
        """
        Create a cache key for the analysis result.
        """
        content_hash = hashlib.md5()
        
        # Hash company content
        for section in company_sections:
            content_hash.update(section.content.encode('utf-8'))
        
        # Hash reference content
        for section in reference_sections:
            content_hash.update(section.content.encode('utf-8'))
        
        # Include domain
        content_hash.update(domain.encode('utf-8'))
        
        return f"gap_analysis_{content_hash.hexdigest()}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[AnalysisResult]:
        """
        Retrieve cached analysis result.
        """
        try:
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                # Deserialize cached result
                return AnalysisResult(**json.loads(cached_data))
        except Exception as e:
            logger.warning(f"Failed to retrieve cached result: {str(e)}")
        
        return None
    
    async def _cache_result(self, cache_key: str, result: AnalysisResult):
        """
        Cache analysis result for future use.
        """
        try:
            # Serialize result (convert dataclasses to dict)
            result_dict = {
                "gaps": [gap.__dict__ for gap in result.gaps],
                "overlaps": [overlap.__dict__ for overlap in result.overlaps],
                "coverage_score": result.coverage_score,
                "implementation_score": result.implementation_score,
                "recommendations": [rec.__dict__ for rec in result.recommendations]
            }
            
            await self.cache.set(cache_key, json.dumps(result_dict), expire=3600)  # 1 hour TTL
        except Exception as e:
            logger.warning(f"Failed to cache result: {str(e)}")