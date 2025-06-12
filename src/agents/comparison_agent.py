import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging
from dataclasses import dataclass, asdict

from .domain_querier import PolicyDomainQuerier
from .gap_analyzer import GapAnalysisProcessor
from ..models.comparison_models import ComparisonResult, Gap, Overlap, Recommendation
from ..prompts.comparison_prompts import GAP_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)

@dataclass
class AgentState:
    current_domain: Optional[str] = None
    comparison_results: List[ComparisonResult] = None
    processing_status: Dict[str, str] = None
    
    def __post_init__(self):
        if self.comparison_results is None:
            self.comparison_results = []
        if self.processing_status is None:
            self.processing_status = {}

class PolicyComparisonAgent:
    """
    Main agent for comparing company policies against reference frameworks.
    Uses RAG system to retrieve relevant policy content and LLM for analysis.
    """
    
    def __init__(self, rag_system, llm_client, cache_client=None):
        self.rag = rag_system
        self.llm = llm_client
        self.cache = cache_client
        self.state = AgentState()
        self.domain_querier = PolicyDomainQuerier(rag_system)
        self.gap_analyzer = GapAnalysisProcessor(llm_client, cache_client)
        
    async def compare_policies(self, domain: str) -> ComparisonResult:
        """
        Compare company policies against reference frameworks for a specific domain.
        
        Args:
            domain: Policy domain to analyze (e.g., 'data_protection', 'access_control')
            
        Returns:
            ComparisonResult with gaps, overlaps, and recommendations
        """
        logger.info(f"Starting policy comparison for domain: {domain}")
        
        try:
            # Update agent state
            self.state.current_domain = domain
            self.state.processing_status[domain] = "in_progress"
            
            # Step 1: Query company policies for this domain
            company_policies = await self.domain_querier.query_company_policies(domain)
            logger.info(f"Retrieved {len(company_policies)} company policy sections")
            
            # Step 2: Query reference frameworks for this domain
            reference_policies = await self.domain_querier.query_reference_policies(domain)
            logger.info(f"Retrieved {len(reference_policies)} reference policy sections")
            
            # Step 3: Perform gap analysis
            analysis_result = await self.gap_analyzer.analyze_gaps(
                company_policies, 
                reference_policies, 
                domain
            )
            
            # Step 4: Create structured comparison result
            comparison_result = ComparisonResult(
                domain=domain,
                company_policy_sections=company_policies,
                reference_sections=reference_policies,
                gaps=analysis_result.gaps,
                overlaps=analysis_result.overlaps,
                coverage_score=analysis_result.coverage_score,
                implementation_score=analysis_result.implementation_score,
                recommendations=analysis_result.recommendations,
                timestamp=datetime.utcnow()
            )
            
            # Update state
            self.state.comparison_results.append(comparison_result)
            self.state.processing_status[domain] = "completed"
            
            logger.info(f"Completed policy comparison for {domain}")
            return comparison_result
            
        except Exception as e:
            logger.error(f"Error comparing policies for domain {domain}: {str(e)}")
            self.state.processing_status[domain] = "failed"
            raise
    
    async def compare_multiple_domains(self, domains: List[str]) -> List[ComparisonResult]:
        """
        Compare policies across multiple domains in parallel.
        
        Args:
            domains: List of policy domains to analyze
            
        Returns:
            List of ComparisonResult objects
        """
        logger.info(f"Starting parallel comparison for {len(domains)} domains")
        
        # Create tasks for parallel processing
        tasks = [self.compare_policies(domain) for domain in domains]
        
        # Execute all comparisons concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process domain {domains[i]}: {str(result)}")
            else:
                successful_results.append(result)
        
        logger.info(f"Completed {len(successful_results)}/{len(domains)} domain comparisons")
        return successful_results
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of all completed comparisons.
        """
        if not self.state.comparison_results:
            return {"message": "No comparisons completed yet"}
        
        total_gaps = sum(len(result.gaps) for result in self.state.comparison_results)
        avg_coverage = sum(result.coverage_score for result in self.state.comparison_results) / len(self.state.comparison_results)
        avg_implementation = sum(result.implementation_score for result in self.state.comparison_results) / len(self.state.comparison_results)
        
        critical_gaps = []
        for result in self.state.comparison_results:
            critical_gaps.extend([gap for gap in result.gaps if gap.severity == "critical"])
        
        return {
            "total_domains_analyzed": len(self.state.comparison_results),
            "total_gaps_identified": total_gaps,
            "critical_gaps_count": len(critical_gaps),
            "average_coverage_score": round(avg_coverage, 2),
            "average_implementation_score": round(avg_implementation, 2),
            "domains_processed": [result.domain for result in self.state.comparison_results],
            "processing_status": self.state.processing_status
        }
    
    def export_results(self, format: str = "json") -> str:
        """
        Export comparison results in specified format.
        """
        if format.lower() == "json":
            return json.dumps([asdict(result) for result in self.state.comparison_results], 
                            default=str, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")