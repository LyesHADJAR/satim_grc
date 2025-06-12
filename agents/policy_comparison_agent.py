from typing import Dict, Any, List
import asyncio
from agents.base_agent import BaseAgent
from models.policy import Policy, PolicySectionMatch
from models.score import ComplianceScore, ScoreCriteria

class PolicyComparisonAgent(BaseAgent):
    """
    Agent responsible for comparing company policies against reference frameworks
    and generating compliance scores.
    """
    
    def __init__(self, name: str, llm_config: Dict[str, Any], rag_engine: Any):
        """
        Initialize the policy comparison agent.
        
        Args:
            name: Agent identifier
            llm_config: Configuration for the language model
            rag_engine: Interface to the RAG system
        """
        super().__init__(name, llm_config)
        self.rag_engine = rag_engine
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process policy comparison requests.
        
        Args:
            input_data: Dictionary containing:
                - company_policy_ids: List of company policy document IDs
                - reference_policy_ids: List of reference framework document IDs
                - domains: List of policy domains to compare (e.g., "data protection")
                
        Returns:
            Dictionary containing comparison results and scores
        """
        company_policy_ids = input_data.get("company_policy_ids", [])
        reference_policy_ids = input_data.get("reference_policy_ids", [])
        domains = input_data.get("domains", [])
        
        self.logger.info(f"Comparing policies for domains: {domains}")
        
        # Retrieve policy documents from RAG system
        company_policies = await self._retrieve_policies(company_policy_ids)
        reference_policies = await self._retrieve_policies(reference_policy_ids)
        
        # Compare policies for each domain
        results = {}
        for domain in domains:
            domain_result = await self._compare_domain(domain, company_policies, reference_policies)
            results[domain] = domain_result
        
        # Calculate overall compliance score
        overall_score = self._calculate_overall_score(results)
        
        return {
            "domain_results": results,
            "overall_score": overall_score
        }
    
    async def _retrieve_policies(self, policy_ids: List[str]) -> List[Policy]:
        """Retrieve policies from the RAG system."""
        policies = []
        for policy_id in policy_ids:
            # Query the RAG engine for the policy document
            policy_data = await self.rag_engine.get_document(policy_id)
            policy = Policy(
                id=policy_id,
                title=policy_data.get("title", ""),
                content=policy_data.get("content", ""),
                metadata=policy_data.get("metadata", {})
            )
            policies.append(policy)
        return policies
    
    async def _compare_domain(self, domain: str, company_policies: List[Policy], 
                              reference_policies: List[Policy]) -> Dict[str, Any]:
        """Compare company policies against reference frameworks for a specific domain."""
        # Extract relevant sections from company policies for this domain
        company_sections = await self._extract_domain_sections(domain, company_policies)
        
        # Extract relevant sections from reference policies for this domain
        reference_sections = await self._extract_domain_sections(domain, reference_policies)
        
        # Find gaps and overlaps
        gaps = await self._identify_gaps(domain, company_sections, reference_sections)
        coverage = await self._calculate_coverage(domain, company_sections, reference_sections)
        
        # Generate section matches
        section_matches = await self._match_sections(company_sections, reference_sections)
        
        # Calculate domain-specific score
        domain_score = await self._score_domain(domain, gaps, coverage, section_matches)
        
        return {
            "domain": domain,
            "gaps": gaps,
            "coverage": coverage,
            "section_matches": section_matches,
            "score": domain_score
        }
    
    async def _extract_domain_sections(self, domain: str, policies: List[Policy]) -> List[Dict[str, Any]]:
        """Extract sections relevant to a specific domain from policies."""
        # Prepare a prompt for the LLM to identify relevant sections
        prompt = f"""
        Identify and extract sections from the policies that are relevant to the domain: {domain}.
        For each relevant section, provide:
        1. The section title or identifier
        2. The full section text
        3. A confidence score (0-100) indicating how relevant this section is to the domain
        
        Policies to analyze:
        {[policy.title for policy in policies]}
        """
        
        # Use the RAG engine to process the policies and extract relevant sections
        sections = []
        for policy in policies:
            # This would call the LLM through your RAG system
            # For now, we'll simulate the response
            response = await self.rag_engine.query_llm(
                query=prompt,
                context=policy.content,
                max_tokens=2000
            )
            
            # Parse the response to extract sections
            # In a real implementation, you'd have a more robust parsing logic
            # or a structured output format from your LLM
            parsed_sections = self._parse_sections_from_llm_response(response, policy.id)
            sections.extend(parsed_sections)
            
        return sections
    
    def _parse_sections_from_llm_response(self, response: str, policy_id: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract policy sections."""
        # This is a simplified implementation
        # In a real system, you'd have more robust parsing
        sections = []
        # Mock implementation - in reality this would parse structured output from the LLM
        # For demo purposes, let's assume we got back a basic structure
        mock_sections = [
            {"title": "Data Classification", "text": "Sample text about data classification...", "confidence": 95},
            {"title": "Access Controls", "text": "Sample text about access controls...", "confidence": 90}
        ]
        
        for section in mock_sections:
            section["policy_id"] = policy_id
            sections.append(section)
            
        return sections
    
    async def _identify_gaps(self, domain: str, company_sections: List[Dict[str, Any]], 
                            reference_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify gaps in company policies compared to reference frameworks."""
        # Create a prompt for the LLM to identify gaps
        company_section_texts = "\n\n".join([f"{s['title']}: {s['text']}" for s in company_sections])
        reference_section_texts = "\n\n".join([f"{s['title']}: {s['text']}" for s in reference_sections])
        
        prompt = f"""
        Compare the company policy sections with the reference framework sections for the domain: {domain}.
        
        Company policy sections:
        {company_section_texts}
        
        Reference framework sections:
        {reference_section_texts}
        
        Identify gaps where the company policies do not adequately address requirements or best practices 
        defined in the reference frameworks. For each gap, provide:
        1. A description of the gap
        2. The severity (High, Medium, Low)
        3. Recommendation to address the gap
        """
        
        # Query the LLM through the RAG engine
        response = await self.rag_engine.query_llm(prompt, max_tokens=1500)
        
        # Parse the gaps from the LLM response
        # In a real implementation, you'd have a more structured approach
        gaps = self._parse_gaps_from_llm_response(response)
        return gaps
    
    def _parse_gaps_from_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract identified gaps."""
        # Mock implementation for demo purposes
        gaps = [
            {
                "description": "Company policy lacks specific guidance on data encryption at rest",
                "severity": "High",
                "recommendation": "Add section on encryption requirements for stored data"
            },
            {
                "description": "Policy on third-party access lacks detail on validation procedures",
                "severity": "Medium",
                "recommendation": "Enhance third-party validation process with specific controls"
            }
        ]
        return gaps
    
    async def _calculate_coverage(self, domain: str, company_sections: List[Dict[str, Any]], 
                                reference_sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate how well company policies cover reference framework requirements."""
        # This would involve a detailed comparison of policy coverage
        # For the demo, we'll provide a simplified implementation
        
        # Count the number of reference topics covered by company policies
        total_reference_topics = len(reference_sections)
        covered_topics = 0
        
        # Create a mapping of company sections to reference sections
        # For a real implementation, this would be a more sophisticated semantic matching
        company_section_texts = [s['title'].lower() for s in company_sections]
        
        for ref_section in reference_sections:
            for company_text in company_section_texts:
                if ref_section['title'].lower() in company_text or company_text in ref_section['title'].lower():
                    covered_topics += 1
                    break
        
        coverage_percentage = (covered_topics / total_reference_topics * 100) if total_reference_topics > 0 else 0
        
        return {
            "total_reference_topics": total_reference_topics,
            "covered_topics": covered_topics,
            "coverage_percentage": coverage_percentage
        }
    
    async def _match_sections(self, company_sections: List[Dict[str, Any]], 
                             reference_sections: List[Dict[str, Any]]) -> List[PolicySectionMatch]:
        """Match company policy sections with reference framework sections."""
        matches = []
        
        # For each reference section, find the best matching company section
        for ref_section in reference_sections:
            best_match = None
            best_score = 0
            
            for comp_section in company_sections:
                # In a real implementation, use semantic similarity between sections
                # For demo, we'll use a simple keyword match
                similarity_score = self._calculate_similarity(ref_section['text'], comp_section['text'])
                
                if similarity_score > best_score:
                    best_score = similarity_score
                    best_match = comp_section
            
            if best_match and best_score > 0.5:  # Set a threshold for matches
                match = PolicySectionMatch(
                    company_section_id=best_match.get('policy_id', '') + ':' + best_match.get('title', ''),
                    reference_section_id=ref_section.get('policy_id', '') + ':' + ref_section.get('title', ''),
                    match_score=best_score,
                    alignment_notes=f"Alignment score: {best_score:.2f}"
                )
                matches.append(match)
        
        return matches
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text sections."""
        # This is a placeholder - in a real implementation, 
        # you would use embeddings and cosine similarity
        # For demo purposes, we'll return a random similarity
        import random
        return random.uniform(0.5, 0.95)
    
    async def _score_domain(self, domain: str, gaps: List[Dict[str, Any]], 
                           coverage: Dict[str, Any], 
                           section_matches: List[PolicySectionMatch]) -> ComplianceScore:
        """Calculate compliance score for a specific domain."""
        # Count severe gaps
        high_severity_gaps = sum(1 for gap in gaps if gap['severity'] == 'High')
        
        # Base score starts at 100 and is reduced based on gaps and coverage
        base_score = 100
        
        # Reduce score based on coverage
        coverage_percentage = coverage.get('coverage_percentage', 0)
        coverage_score_impact = (100 - coverage_percentage) * 0.5  # 50% weight for coverage
        
        # Reduce score based on high severity gaps
        gap_impact = high_severity_gaps * 10  # Each high severity gap reduces score by 10 points
        
        final_score = max(0, base_score - coverage_score_impact - gap_impact)
        
        # Create criteria for the score
        criteria = [
            ScoreCriteria(name="Coverage", weight=0.5, score=coverage_percentage),
            ScoreCriteria(name="Gap Severity", weight=0.3, score=max(0, 100 - gap_impact)),
            ScoreCriteria(name="Policy Quality", weight=0.2, score=85)  # Placeholder
        ]
        
        return ComplianceScore(
            domain=domain,
            score=final_score,
            criteria=criteria,
            max_score=100,
            recommendations=[gap['recommendation'] for gap in gaps]
        )
    
    def _calculate_overall_score(self, domain_results: Dict[str, Dict]) -> ComplianceScore:
        """Calculate the overall compliance score across all domains."""
        # Extract scores from each domain
        domain_scores = [result['score'] for result in domain_results.values()]
        
        # Calculate the average score
        if domain_scores:
            avg_score = sum(score.score for score in domain_scores) / len(domain_scores)
        else:
            avg_score = 0
        
        # Combine recommendations from all domains
        all_recommendations = []
        for result in domain_results.values():
            all_recommendations.extend(result['score'].recommendations)
        
        # Create an overall score with combined recommendations
        return ComplianceScore(
            domain="Overall",
            score=avg_score,
            criteria=[
                ScoreCriteria(name="Average Domain Compliance", weight=1.0, score=avg_score)
            ],
            max_score=100,
            recommendations=all_recommendations
        )