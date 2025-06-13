from typing import Dict, Any, List, Optional
import asyncio
import logging
import json
from dataclasses import dataclass

@dataclass
class PolicyContent:
    """Structured policy content for analysis."""
    document_id: str
    section_title: str
    content: str
    domain_relevance: float
    content_type: str  # 'company' or 'reference'
    metadata: Dict[str, Any]

@dataclass
class ComparisonContext:
    """Structured context for policy comparison."""
    domain: str
    company_content: List[PolicyContent]
    reference_content: List[PolicyContent]
    comparison_matrix: Dict[str, Any]
    analysis_metadata: Dict[str, Any]

class ContextBuilder:
    """Builds rich, structured context for LLM analysis."""
    
    def __init__(self, rag_engine: Any):
        self.rag_engine = rag_engine
        self.logger = logging.getLogger("context_builder")
    
    async def build_comparison_context(self, domain: str, 
                                     company_policies: List[str], 
                                     reference_policies: List[str]) -> ComparisonContext:
        """Build comprehensive comparison context for a domain."""
        
        # Extract domain-specific content
        company_content = await self._extract_domain_content(
            domain, company_policies, "company"
        )
        reference_content = await self._extract_domain_content(
            domain, reference_policies, "reference"
        )
        
        # Create comparison matrix
        comparison_matrix = await self._create_comparison_matrix(
            company_content, reference_content
        )
        
        # Gather analysis metadata
        analysis_metadata = {
            "domain": domain,
            "company_sections_count": len(company_content),
            "reference_sections_count": len(reference_content),
            "extraction_timestamp": "2025-06-13 00:25:26",
            "analysis_scope": "comprehensive_domain_analysis"
        }
        
        return ComparisonContext(
            domain=domain,
            company_content=company_content,
            reference_content=reference_content,
            comparison_matrix=comparison_matrix,
            analysis_metadata=analysis_metadata
        )
    
    async def _extract_domain_content(self, domain: str, policy_ids: List[str], 
                                    content_type: str) -> List[PolicyContent]:
        """Extract domain-relevant content from policies."""
        
        domain_content = []
        
        # Enhanced search queries for better content extraction
        search_queries = [
            f"{domain}",
            f"{domain} policy",
            f"{domain} procedures", 
            f"{domain} controls",
            f"{domain} requirements",
            f"{domain} implementation"
        ]
        
        for policy_id in policy_ids:
            for query in search_queries:
                search_results = await self.rag_engine.semantic_search(
                    f"{query} {policy_id}", top_k=15
                )
                
                for result in search_results:
                    if result['similarity_score'] > 0.25:  # Quality threshold
                        
                        # Calculate domain relevance
                        relevance_score = self._calculate_domain_relevance(
                            domain, result['content'], result['section']
                        )
                        
                        if relevance_score > 0.3:  # Relevance threshold
                            policy_content = PolicyContent(
                                document_id=result['document_id'],
                                section_title=result['section'],
                                content=result['content'],
                                domain_relevance=relevance_score,
                                content_type=content_type,
                                metadata={
                                    "similarity_score": result['similarity_score'],
                                    "document_type": result.get('document_type', 'unknown'),
                                    "extraction_query": query
                                }
                            )
                            domain_content.append(policy_content)
        
        # Remove duplicates and sort by relevance
        unique_content = self._deduplicate_content(domain_content)
        unique_content.sort(key=lambda x: x.domain_relevance, reverse=True)
        
        return unique_content[:12]  # Top 12 most relevant sections
    
    def _calculate_domain_relevance(self, domain: str, content: str, section_title: str) -> float:
        """Calculate how relevant content is to the domain."""
        
        domain_keywords = {
            "access control": [
                "access", "authentication", "authorization", "login", "password", 
                "mfa", "multi-factor", "rbac", "role", "permission", "privilege"
            ],
            "incident response": [
                "incident", "response", "emergency", "escalation", "notification",
                "containment", "recovery", "forensics", "breach", "security event"
            ],
            "data protection": [
                "data", "protection", "encryption", "backup", "privacy", "confidential",
                "classification", "retention", "disposal", "pii", "sensitive"
            ]
        }
        
        keywords = domain_keywords.get(domain.lower(), [domain.lower()])
        content_lower = content.lower()
        section_lower = section_title.lower()
        
        # Calculate keyword density
        keyword_matches = 0
        total_keywords = len(keywords)
        
        for keyword in keywords:
            if keyword in content_lower:
                keyword_matches += content_lower.count(keyword) * 0.1
            if keyword in section_lower:
                keyword_matches += 2  # Higher weight for section titles
        
        # Normalize score
        relevance_score = min(keyword_matches / (total_keywords * 2), 1.0)
        
        return relevance_score
    
    def _deduplicate_content(self, content_list: List[PolicyContent]) -> List[PolicyContent]:
        """Remove duplicate content based on similarity."""
        
        unique_content = []
        seen_content = set()
        
        for content in content_list:
            # Create a hash of the content for deduplication
            content_hash = hash(content.content[:200])  # First 200 chars
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_content.append(content)
        
        return unique_content
    
    async def _create_comparison_matrix(self, company_content: List[PolicyContent], 
                                      reference_content: List[PolicyContent]) -> Dict[str, Any]:
        """Create a comparison matrix between company and reference content."""
        
        matrix = {
            "coverage_mapping": {},
            "gap_indicators": [],
            "strength_areas": [],
            "alignment_scores": {}
        }
        
        # Map company content to reference content
        for ref_content in reference_content:
            best_match = None
            best_score = 0.0
            
            for comp_content in company_content:
                # Calculate semantic similarity (simplified)
                similarity = self._calculate_content_similarity(
                    comp_content.content, ref_content.content
                )
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = comp_content
            
            matrix["coverage_mapping"][ref_content.section_title] = {
                "matched_company_section": best_match.section_title if best_match else None,
                "match_score": best_score,
                "coverage_quality": "High" if best_score > 0.7 else "Medium" if best_score > 0.4 else "Low"
            }
        
        # Identify gaps (reference content without good matches)
        for ref_section, mapping in matrix["coverage_mapping"].items():
            if mapping["match_score"] < 0.3:
                matrix["gap_indicators"].append({
                    "missing_area": ref_section,
                    "reference_content": next(
                        (rc.content[:200] for rc in reference_content if rc.section_title == ref_section), 
                        ""
                    ),
                    "gap_severity": "High" if mapping["match_score"] < 0.1 else "Medium"
                })
        
        # Identify strength areas (high alignment)
        for ref_section, mapping in matrix["coverage_mapping"].items():
            if mapping["match_score"] > 0.6:
                matrix["strength_areas"].append({
                    "area": ref_section,
                    "alignment_quality": mapping["coverage_quality"],
                    "match_score": mapping["match_score"]
                })
        
        return matrix
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate semantic similarity between two content pieces."""
        
        # Simple keyword-based similarity (can be enhanced with embeddings)
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        jaccard_similarity = len(intersection) / len(union)
        
        # Boost for common compliance terms
        compliance_terms = {
            'policy', 'procedure', 'control', 'requirement', 'compliance',
            'audit', 'review', 'approval', 'documentation', 'monitoring'
        }
        
        common_compliance_terms = intersection.intersection(compliance_terms)
        compliance_boost = len(common_compliance_terms) * 0.1
        
        return min(jaccard_similarity + compliance_boost, 1.0)
    
    async def format_for_llm_analysis(self, context: ComparisonContext) -> str:
        """Format comparison context for LLM analysis."""
        
        formatted_context = f"""
# POLICY ANALYSIS CONTEXT: {context.domain.upper()}

## ANALYSIS METADATA
- Domain: {context.domain}
- Company Sections: {len(context.company_content)}
- Reference Sections: {len(context.reference_content)}
- Analysis Timestamp: {context.analysis_metadata['extraction_timestamp']}

## COMPANY POLICY CONTENT
"""
        
        for i, content in enumerate(context.company_content, 1):
            formatted_context += f"""
### Company Section {i}: {content.section_title}
**Relevance Score:** {content.domain_relevance:.2f}
**Content:** {content.content[:800]}...

"""
        
        formatted_context += """
## REFERENCE STANDARD CONTENT (PCI DSS)
"""
        
        for i, content in enumerate(context.reference_content, 1):
            formatted_context += f"""
### Reference Section {i}: {content.section_title}
**Relevance Score:** {content.domain_relevance:.2f}
**Content:** {content.content[:800]}...

"""
        
        # Add comparison matrix insights
        formatted_context += f"""
## COMPARISON ANALYSIS
**Coverage Mapping:** {len(context.comparison_matrix['coverage_mapping'])} reference sections analyzed
**Identified Gaps:** {len(context.comparison_matrix['gap_indicators'])} potential gaps
**Strength Areas:** {len(context.comparison_matrix['strength_areas'])} areas of good alignment

### Key Gaps Identified:
"""
        
        for gap in context.comparison_matrix['gap_indicators'][:5]:
            formatted_context += f"- **{gap['missing_area']}** (Severity: {gap['gap_severity']})\n"
        
        formatted_context += """
### Strength Areas:
"""
        
        for strength in context.comparison_matrix['strength_areas'][:5]:
            formatted_context += f"- **{strength['area']}** (Quality: {strength['alignment_quality']})\n"
        
        return formatted_context