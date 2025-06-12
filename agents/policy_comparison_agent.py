from typing import Dict, Any, List
import asyncio
import re
import logging
from agents.base_agent import BaseAgent
from models.policy import Policy, PolicySectionMatch
from models.score import ComplianceScore, ScoreCriteria

class PolicyComparisonAgent(BaseAgent):
    """
    Agent responsible for comparing company policies against reference frameworks
    and generating compliance scores using real data.
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
        Process policy comparison requests using real data.
        
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
        
        # Find gaps and overlaps using real data
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
        """Extract sections relevant to a specific domain from policies using real data."""
        sections = []
        
        # Use semantic search to find relevant sections
        search_results = await self.rag_engine.semantic_search(domain, top_k=10)
        
        # Filter results by policy IDs
        policy_titles = [policy.title.lower() for policy in policies]
        
        for result in search_results:
            document_id = result.get("document_id", "").lower()
            
            # Check if this result belongs to one of our policies
            if any(policy_title in document_id for policy_title in policy_titles):
                sections.append({
                    "title": result.get("section", "Unknown Section"),
                    "text": result.get("content", ""),
                    "confidence": int(result.get("similarity_score", 0) * 100),
                    "policy_id": result.get("document_id", ""),
                    "document_type": result.get("document_type", "unknown")
                })
        
        return sections
    
    async def _identify_gaps(self, domain: str, company_sections: List[Dict[str, Any]], 
                            reference_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify gaps in company policies compared to reference frameworks using real analysis."""
        
        # Get the actual content for analysis
        company_content = "\n\n".join([
            f"**{s['title']}**: {s['text'][:500]}..." 
            for s in company_sections[:5]  # Limit to prevent token overflow
        ])
        
        reference_content = "\n\n".join([
            f"**{s['title']}**: {s['text'][:500]}..." 
            for s in reference_sections[:5]  # Limit to prevent token overflow
        ])
        
        # Create a structured prompt for gap analysis
        prompt = f"""
        Analyze the following company policy sections against reference framework sections for the domain: {domain}.
        
        COMPANY POLICY SECTIONS:
        {company_content}
        
        REFERENCE FRAMEWORK SECTIONS:
        {reference_content}
        
        Please identify specific gaps where the company policies do not adequately address requirements found in the reference frameworks. 
        
        For each gap, provide:
        1. A clear description of what is missing
        2. Severity level (High/Medium/Low)
        3. Specific recommendation to address the gap
        
        Format your response as:
        GAP: [description]
        SEVERITY: [High/Medium/Low]
        RECOMMENDATION: [specific recommendation]
        ---
        """
        
        # Query the LLM for gap analysis
        response = await self.rag_engine.query_llm(prompt, max_tokens=1500)
        
        # Parse the structured response
        gaps = self._parse_structured_gaps(response)
        
        # If no gaps found from LLM, perform basic keyword analysis
        if not gaps:
            gaps = self._perform_basic_gap_analysis(company_sections, reference_sections)
        
        return gaps
    
    def _parse_structured_gaps(self, response: str) -> List[Dict[str, Any]]:
        """Parse structured gap analysis response from LLM."""
        gaps = []
        
        # Split response into individual gap blocks
        gap_blocks = response.split("---")
        
        for block in gap_blocks:
            if not block.strip():
                continue
                
            # Extract gap information using regex
            gap_match = re.search(r'GAP:\s*(.+?)(?=SEVERITY:|$)', block, re.DOTALL | re.IGNORECASE)
            severity_match = re.search(r'SEVERITY:\s*(.+?)(?=RECOMMENDATION:|$)', block, re.DOTALL | re.IGNORECASE)
            rec_match = re.search(r'RECOMMENDATION:\s*(.+?)(?=---|$)', block, re.DOTALL | re.IGNORECASE)
            
            if gap_match:
                gap_desc = gap_match.group(1).strip()
                severity = severity_match.group(1).strip() if severity_match else "Medium"
                recommendation = rec_match.group(1).strip() if rec_match else "Review and enhance policy coverage"
                
                # Clean up the extracted text
                gap_desc = re.sub(r'\s+', ' ', gap_desc)
                recommendation = re.sub(r'\s+', ' ', recommendation)
                
                gaps.append({
                    "description": gap_desc,
                    "severity": severity,
                    "recommendation": recommendation
                })
        
        return gaps
    
    def _perform_basic_gap_analysis(self, company_sections: List[Dict[str, Any]], 
                                   reference_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform basic gap analysis using keyword matching."""
        gaps = []
        
        # Extract key topics from reference sections
        reference_topics = set()
        company_topics = set()
        
        for section in reference_sections:
            title = section.get('title', '').lower()
            # Extract key terms from titles
            key_terms = re.findall(r'\b[a-z]{4,}\b', title)
            reference_topics.update(key_terms)
        
        for section in company_sections:
            title = section.get('title', '').lower()
            text = section.get('text', '').lower()
            # Extract key terms from titles and content
            key_terms = re.findall(r'\b[a-z]{4,}\b', title + ' ' + text)
            company_topics.update(key_terms)
        
        # Find missing topics
        missing_topics = reference_topics - company_topics
        
        # Generate gap descriptions for missing topics
        for topic in list(missing_topics)[:3]:  # Limit to top 3
            gaps.append({
                "description": f"Company policy lacks specific coverage of {topic.replace('_', ' ')} requirements",
                "severity": "Medium",
                "recommendation": f"Add or enhance policy sections addressing {topic.replace('_', ' ')} controls and procedures"
            })
        
        # If no gaps found, add a generic analysis
        if not gaps:
            gaps.append({
                "description": "Policy coverage appears adequate but may benefit from more detailed implementation guidance",
                "severity": "Low",
                "recommendation": "Review policy implementation procedures and add specific operational guidance"
            })
        
        return gaps
    
    async def _calculate_coverage(self, domain: str, company_sections: List[Dict[str, Any]], 
                                reference_sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate coverage using real section analysis."""
        
        if not reference_sections:
            return {
                "total_reference_topics": 0,
                "covered_topics": 0,
                "coverage_percentage": 0
            }
        
        total_reference_topics = len(reference_sections)
        covered_topics = 0
        
        # Create sets of key terms from each section type
        company_terms = set()
        for section in company_sections:
            text = (section.get('title', '') + ' ' + section.get('text', '')).lower()
            # Extract meaningful terms
            terms = re.findall(r'\b[a-z]{4,}\b', text)
            company_terms.update(terms)
        
        # Check coverage for each reference section
        for ref_section in reference_sections:
            ref_text = (ref_section.get('title', '') + ' ' + ref_section.get('text', '')).lower()
            ref_terms = set(re.findall(r'\b[a-z]{4,}\b', ref_text))
            
            # Check if there's significant overlap in terms
            overlap = len(company_terms.intersection(ref_terms))
            if overlap >= 2:  # Require at least 2 matching terms
                covered_topics += 1
        
        coverage_percentage = (covered_topics / total_reference_topics * 100) if total_reference_topics > 0 else 0
        
        return {
            "total_reference_topics": total_reference_topics,
            "covered_topics": covered_topics,
            "coverage_percentage": coverage_percentage
        }
    
    async def _match_sections(self, company_sections: List[Dict[str, Any]], 
                             reference_sections: List[Dict[str, Any]]) -> List[PolicySectionMatch]:
        """Match company policy sections with reference framework sections using real similarity."""
        matches = []
        
        for ref_section in reference_sections:
            best_match = None
            best_score = 0
            
            ref_text = ref_section.get('text', '').lower()
            ref_title = ref_section.get('title', '').lower()
            ref_terms = set(re.findall(r'\b[a-z]{4,}\b', ref_text + ' ' + ref_title))
            
            for comp_section in company_sections:
                comp_text = comp_section.get('text', '').lower()
                comp_title = comp_section.get('title', '').lower()
                comp_terms = set(re.findall(r'\b[a-z]{4,}\b', comp_text + ' ' + comp_title))
                
                # Calculate similarity based on term overlap
                if ref_terms and comp_terms:
                    overlap = len(ref_terms.intersection(comp_terms))
                    union = len(ref_terms.union(comp_terms))
                    similarity_score = overlap / union if union > 0 else 0
                    
                    if similarity_score > best_score:
                        best_score = similarity_score
                        best_match = comp_section
            
            if best_match and best_score > 0.1:  # Lower threshold for real data
                match = PolicySectionMatch(
                    company_section_id=f"{best_match.get('policy_id', 'unknown')}:{best_match.get('title', 'Unknown')}",
                    reference_section_id=f"{ref_section.get('policy_id', 'reference')}:{ref_section.get('title', 'Unknown')}",
                    match_score=best_score,
                    alignment_notes=f"Semantic similarity: {best_score:.2f} based on term overlap"
                )
                matches.append(match)
        
        return matches
    
    async def _score_domain(self, domain: str, gaps: List[Dict[str, Any]], 
                           coverage: Dict[str, Any], 
                           section_matches: List[PolicySectionMatch]) -> ComplianceScore:
        """Calculate compliance score based on real analysis."""
        
        # Count gaps by severity
        high_severity_gaps = sum(1 for gap in gaps if gap['severity'].lower() == 'high')
        medium_severity_gaps = sum(1 for gap in gaps if gap['severity'].lower() == 'medium')
        low_severity_gaps = sum(1 for gap in gaps if gap['severity'].lower() == 'low')
        
        # Base score calculation
        base_score = 100
        
        # Coverage impact (40% weight)
        coverage_percentage = coverage.get('coverage_percentage', 0)
        coverage_score = coverage_percentage
        
        # Gap impact (40% weight)
        gap_penalty = (high_severity_gaps * 15) + (medium_severity_gaps * 8) + (low_severity_gaps * 3)
        gap_score = max(0, 100 - gap_penalty)
        
        # Section matching quality (20% weight)
        if section_matches:
            avg_match_score = sum(match.match_score for match in section_matches) / len(section_matches)
            match_score = avg_match_score * 100
        else:
            match_score = 50  # Default if no matches found
        
        # Weighted final score
        final_score = (coverage_score * 0.4) + (gap_score * 0.4) + (match_score * 0.2)
        
        # Create detailed criteria
        criteria = [
            ScoreCriteria(name="Policy Coverage", weight=0.4, score=coverage_score),
            ScoreCriteria(name="Gap Analysis", weight=0.4, score=gap_score),
            ScoreCriteria(name="Section Alignment", weight=0.2, score=match_score)
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
        if not domain_results:
            return ComplianceScore(
                domain="Overall",
                score=0,
                criteria=[],
                max_score=100,
                recommendations=["No domains analyzed"]
            )
        
        # Extract scores from each domain
        domain_scores = [result['score'] for result in domain_results.values()]
        
        # Calculate weighted average (can be enhanced with domain-specific weights)
        avg_score = sum(score.score for score in domain_scores) / len(domain_scores)
        
        # Combine all recommendations
        all_recommendations = []
        for result in domain_results.values():
            all_recommendations.extend(result['score'].recommendations)
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        seen = set()
        for rec in all_recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)
        
        # Create overall criteria
        overall_criteria = [
            ScoreCriteria(name="Average Domain Compliance", weight=1.0, score=avg_score)
        ]
        
        return ComplianceScore(
            domain="Overall",
            score=avg_score,
            criteria=overall_criteria,
            max_score=100,
            recommendations=unique_recommendations[:10]  # Limit to top 10 recommendations
        )