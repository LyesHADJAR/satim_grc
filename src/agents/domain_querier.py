from typing import List, Dict, Any
import logging
from ..models.comparison_models import PolicySection

logger = logging.getLogger(__name__)

class PolicyDomainQuerier:
    """
    Handles domain-specific querying of policies from the RAG system.
    """
    
    def __init__(self, rag_system):
        self.rag = rag_system
        
        # Domain-specific query patterns
        self.domain_queries = {
            "data_protection": [
                "data protection policy",
                "personal data handling",
                "privacy policy",
                "data retention",
                "data breach response"
            ],
            "access_control": [
                "access control policy",
                "user authentication",
                "authorization procedures",
                "identity management",
                "privileged access"
            ],
            "incident_response": [
                "incident response plan",
                "security incident handling",
                "breach notification",
                "incident management",
                "emergency procedures"
            ],
            "business_continuity": [
                "business continuity plan",
                "disaster recovery",
                "backup procedures",
                "continuity planning",
                "recovery procedures"
            ],
            "risk_management": [
                "risk assessment",
                "risk management policy",
                "threat analysis",
                "vulnerability management",
                "risk mitigation"
            ]
        }
    
    async def query_company_policies(self, domain: str) -> List[PolicySection]:
        """
        Query company policies for a specific domain.
        
        Args:
            domain: The policy domain to query
            
        Returns:
            List of relevant policy sections
        """
        logger.info(f"Querying company policies for domain: {domain}")
        
        if domain not in self.domain_queries:
            logger.warning(f"Unknown domain: {domain}. Using generic query.")
            query_terms = [domain]
        else:
            query_terms = self.domain_queries[domain]
        
        policy_sections = []
        
        for query_term in query_terms:
            # Query RAG system with metadata filtering for company policies
            results = await self.rag.query(
                query=query_term,
                filter_metadata={"policy_type": "company"},
                top_k=5
            )
            
            for result in results:
                policy_section = PolicySection(
                    document_id=result.metadata.get("document_id"),
                    section_title=result.metadata.get("section_title", ""),
                    content=result.content,
                    domain=domain,
                    policy_type="company",
                    relevance_score=result.score,
                    source_document=result.metadata.get("source_document"),
                    page_number=result.metadata.get("page_number")
                )
                policy_sections.append(policy_section)
        
        # Remove duplicates and sort by relevance
        unique_sections = self._deduplicate_sections(policy_sections)
        sorted_sections = sorted(unique_sections, key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"Found {len(sorted_sections)} unique company policy sections for {domain}")
        return sorted_sections[:10]  # Return top 10 most relevant
    
    async def query_reference_policies(self, domain: str) -> List[PolicySection]:
        """
        Query reference framework policies for a specific domain.
        
        Args:
            domain: The policy domain to query
            
        Returns:
            List of relevant reference policy sections
        """
        logger.info(f"Querying reference policies for domain: {domain}")
        
        if domain not in self.domain_queries:
            query_terms = [domain]
        else:
            query_terms = self.domain_queries[domain]
        
        policy_sections = []
        
        for query_term in query_terms:
            # Query RAG system with metadata filtering for reference policies
            results = await self.rag.query(
                query=query_term,
                filter_metadata={"policy_type": "reference"},
                top_k=5
            )
            
            for result in results:
                policy_section = PolicySection(
                    document_id=result.metadata.get("document_id"),
                    section_title=result.metadata.get("section_title", ""),
                    content=result.content,
                    domain=domain,
                    policy_type="reference",
                    relevance_score=result.score,
                    source_document=result.metadata.get("source_document"),
                    compliance_framework=result.metadata.get("compliance_framework"),
                    authority_level=result.metadata.get("authority_level", 3)
                )
                policy_sections.append(policy_section)
        
        # Remove duplicates and sort by authority level and relevance
        unique_sections = self._deduplicate_sections(policy_sections)
        sorted_sections = sorted(unique_sections, 
                               key=lambda x: (x.authority_level, x.relevance_score), 
                               reverse=True)
        
        logger.info(f"Found {len(sorted_sections)} unique reference policy sections for {domain}")
        return sorted_sections[:10]  # Return top 10 most relevant
    
    def _deduplicate_sections(self, sections: List[PolicySection]) -> List[PolicySection]:
        """
        Remove duplicate policy sections based on content similarity.
        """
        seen_content = set()
        unique_sections = []
        
        for section in sections:
            # Use first 100 characters as a simple deduplication key
            content_key = section.content[:100].strip().lower()
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_sections.append(section)
        
        return unique_sections
    
    async def get_policy_context(self, domain: str, section_id: str) -> Dict[str, Any]:
        """
        Get additional context for a specific policy section.
        """
        # Query for related sections in the same document
        related_results = await self.rag.query(
            query=f"document_id:{section_id}",
            filter_metadata={"domain": domain},
            top_k=3
        )
        
        return {
            "related_sections": len(related_results),
            "document_coverage": "partial" if len(related_results) < 3 else "comprehensive"
        }