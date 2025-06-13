"""
Enhanced RAG Query Engine with Vector Search and International Law Context
Current Date: 2025-06-13 20:26:29 UTC
Current User: LyesHADJAR
"""
from typing import Dict, Any, List, Optional
import asyncio
import logging
import json
import os
import time
from datetime import datetime, timezone

from .vector_search_service import EnhancedVectorSearchService
from utils.logging_config import log_llm_interaction, log_performance, log_analysis_stage
# Import Google Gemini
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

class InternationalLawEnhancedRAGEngine:
    """Enhanced RAG system with vector search and international law expertise."""
    
    def __init__(self, 
                 llm_config: Dict[str, Any], 
                 data_paths: Dict[str, str] = None):
        
        self.llm_config = llm_config
        self.logger = logging.getLogger("rag.enhanced_engine")
        
        # Set up data paths
        if data_paths is None:
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_paths = {
                "company_policies": os.path.join(base_path, "preprocessing", "policies", "satim_chunks_cleaned.json"),
                "reference_policies": os.path.join(base_path, "preprocessing", "norms", "international_norms", "pci_dss_chunks.json")
            }
        
        # Initialize vector search service
        log_analysis_stage("INITIALIZATION", "Setting up vector search service")
        start_time = time.time()
        self.vector_search = EnhancedVectorSearchService(data_paths)
        log_performance("Vector Search Initialization", time.time() - start_time, 
                       self.vector_search.get_document_stats())
        
        # Initialize LLM client
        self._init_gemini_client()
        
        if not self.gemini_available:
            raise RuntimeError("Gemini LLM is required for analysis.")
    
    def _init_gemini_client(self):
        """Initialize Gemini client with enhanced configuration."""
        self.gemini_model = None
        self.gemini_available = False
        
        if not HAS_GEMINI:
            self.logger.error("google-generativeai package not installed")
            return
        
        try:
            api_key = (
                self.llm_config.get("api_key") or 
                os.getenv("GEMINI_API_KEY") or 
                os.getenv("GOOGLE_AI_API_KEY") or
                os.getenv("GOOGLE_API_KEY")
            )
            
            if not api_key:
                self.logger.error("No Gemini API key found")
                return
            
            genai.configure(api_key=api_key)
            
            model_name = self.llm_config.get("model", "gemini-1.5-flash")
            
            # Enhanced system instruction with international law expertise
            system_instruction = """You are an expert GRC compliance analyst with deep expertise in international regulatory frameworks and compliance standards.

YOUR EXPERTISE INCLUDES:
- International compliance frameworks (PCI DSS, ISO 27001, SOX, GDPR, etc.)
- French regulatory requirements and compliance standards
- Cross-border compliance and regulatory harmonization
- Risk assessment and gap analysis methodologies
- Policy development and implementation best practices
- Regulatory interpretation and practical application

ANALYSIS APPROACH:
1. Always reference specific regulatory requirements and standards when making recommendations
2. Provide quantitative assessments with clear scoring rationale
3. Consider international best practices and benchmark against global standards
4. Identify specific regulatory citations and compliance obligations
5. Offer actionable, implementable recommendations with clear timelines
6. Consider cultural and jurisdictional context (especially French regulatory environment)

OUTPUT REQUIREMENTS:
- Be specific and actionable in all recommendations
- Quantify risks and compliance gaps with clear metrics
- Reference international standards and regulatory requirements
- Provide implementation guidance with realistic timelines
- Consider resource requirements and organizational constraints
- Maintain professional, authoritative tone suitable for executive presentation

FRENCH COMPLIANCE CONTEXT:
- Understand French regulatory environment and compliance expectations
- Consider CNIL, ANSSI, and other French regulatory bodies
- Align with French data protection and cybersecurity requirements
- Respect French business culture and organizational structures"""

            self.gemini_model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.llm_config.get("temperature", 0.2),
                    max_output_tokens=self.llm_config.get("max_tokens", 4000),
                    top_p=self.llm_config.get("top_p", 0.8),
                    top_k=self.llm_config.get("top_k", 40)
                ),
                system_instruction=system_instruction
            )
            
            self.gemini_available = True
            self.logger.info(f"✅ Enhanced Gemini initialized with international law expertise: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Gemini initialization failed: {e}")
            self.gemini_available = False
    
    async def semantic_search(self, query: str, top_k: int = 5, 
                            doc_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Enhanced semantic search with international law context.
        
        Args:
            query: Search query
            top_k: Number of results to return
            doc_types: Document types to search
            
        Returns:
            List of enhanced search results with regulatory context
        """
        start_time = time.time()
        
        # Enhance query with international law context
        enhanced_query = self._enhance_query_with_regulatory_context(query)
        
        # Perform vector search
        results = self.vector_search.semantic_search(
            enhanced_query, 
            doc_types=doc_types, 
            top_k=top_k * 2,  # Get more results for better filtering
            method='hybrid'
        )
        
        # Enhance results with regulatory context
        enhanced_results = []
        for result in results[:top_k]:
            enhanced_result = self._enhance_result_with_context(result, query)
            enhanced_results.append(enhanced_result)
        
        log_performance("Enhanced Semantic Search", time.time() - start_time, {
            "query_length": len(query),
            "results_found": len(enhanced_results),
            "doc_types": doc_types or "all"
        })
        
        return enhanced_results
    
    def _enhance_query_with_regulatory_context(self, query: str) -> str:
        """Enhance query with regulatory and compliance context."""
        
        # Add regulatory context keywords based on query content
        regulatory_enhancements = []
        
        # Detect domain-specific context
        if "access" in query.lower() or "authentication" in query.lower():
            regulatory_enhancements.extend([
                "access control standards", "authentication requirements", 
                "identity management", "authorization controls"
            ])
        
        if "data" in query.lower() or "information" in query.lower():
            regulatory_enhancements.extend([
                "data protection", "information security", "privacy requirements",
                "data classification", "encryption standards"
            ])
        
        if "incident" in query.lower() or "response" in query.lower():
            regulatory_enhancements.extend([
                "incident response procedures", "breach notification",
                "security incident management", "forensics"
            ])
        
        if "risk" in query.lower():
            regulatory_enhancements.extend([
                "risk assessment", "risk management", "vulnerability management",
                "threat analysis", "risk mitigation"
            ])
        
        # Add international framework context
        framework_keywords = [
            "ISO 27001", "PCI DSS", "NIST", "regulatory compliance",
            "international standards", "best practices"
        ]
        
        # Combine original query with enhancements
        if regulatory_enhancements:
            enhanced_query = f"{query} {' '.join(regulatory_enhancements[:3])} {' '.join(framework_keywords[:2])}"
        else:
            enhanced_query = f"{query} regulatory compliance international standards"
        
        return enhanced_query
    
    def _enhance_result_with_context(self, result: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Enhance search result with regulatory context and analysis."""
        
        content = result.get('content', '')
        
        # Identify regulatory references in content
        regulatory_refs = self._extract_regulatory_references(content)
        
        # Assess compliance relevance
        compliance_relevance = self._assess_compliance_relevance(content, original_query)
        
        # Enhanced result
        enhanced_result = {
            **result,
            'regulatory_context': {
                'identified_standards': regulatory_refs.get('standards', []),
                'compliance_areas': regulatory_refs.get('compliance_areas', []),
                'regulatory_citations': regulatory_refs.get('citations', []),
                'compliance_relevance_score': compliance_relevance
            },
            'international_law_context': self._get_international_law_context(content),
            'enhanced_similarity_score': result.get('similarity_score', 0) * (1 + compliance_relevance * 0.3)
        }
        
        return enhanced_result
    
    def _extract_regulatory_references(self, content: str) -> Dict[str, List[str]]:
        """Extract regulatory references and standards from content."""
        
        import re
        
        # Standard patterns
        standards = []
        compliance_areas = []
        citations = []
        
        # ISO standards
        iso_pattern = r'ISO\s*(\d+)'
        iso_matches = re.findall(iso_pattern, content, re.IGNORECASE)
        standards.extend([f"ISO {match}" for match in iso_matches])
        
        # PCI DSS references
        if re.search(r'PCI\s*DSS', content, re.IGNORECASE):
            standards.append("PCI DSS")
        
        # NIST references
        if re.search(r'NIST', content, re.IGNORECASE):
            standards.append("NIST")
        
        # GDPR references
        if re.search(r'GDPR', content, re.IGNORECASE):
            standards.append("GDPR")
        
        # Compliance areas
        compliance_patterns = {
            'access_control': r'\b(access\s+control|authentication|authorization)\b',
            'data_protection': r'\b(data\s+protection|encryption|privacy)\b',
            'incident_response': r'\b(incident\s+response|breach|forensics)\b',
            'risk_management': r'\b(risk\s+management|vulnerability|threat)\b',
            'audit_monitoring': r'\b(audit|monitoring|logging)\b',
            'business_continuity': r'\b(business\s+continuity|disaster\s+recovery)\b'
        }
        
        for area, pattern in compliance_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                compliance_areas.append(area)
        
        # Extract specific requirements/citations
        requirement_patterns = [
            r'requirement\s+(\d+\.?\d*)',
            r'section\s+(\d+\.?\d*)',
            r'article\s+(\d+)',
            r'clause\s+(\d+\.?\d*)'
        ]
        
        for pattern in requirement_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            citations.extend(matches)
        
        return {
            'standards': list(set(standards)),
            'compliance_areas': list(set(compliance_areas)),
            'citations': list(set(citations))
        }
    
    def _assess_compliance_relevance(self, content: str, query: str) -> float:
        """Assess how relevant content is to compliance requirements."""
        
        # Compliance keywords
        compliance_keywords = [
            'requirement', 'compliance', 'mandatory', 'shall', 'must',
            'policy', 'procedure', 'control', 'standard', 'guideline',
            'audit', 'assessment', 'review', 'monitor', 'implement'
        ]
        
        regulatory_keywords = [
            'regulatory', 'regulation', 'law', 'legal', 'statute',
            'framework', 'directive', 'ordinance', 'act'
        ]
        
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Count keyword matches
        compliance_score = sum(1 for keyword in compliance_keywords if keyword in content_lower)
        regulatory_score = sum(1 for keyword in regulatory_keywords if keyword in content_lower)
        
        # Query relevance
        query_words = query_lower.split()
        query_match_score = sum(1 for word in query_words if word in content_lower)
        
        # Normalize scores
        max_compliance = len(compliance_keywords)
        max_regulatory = len(regulatory_keywords)
        max_query = len(query_words)
        
        normalized_compliance = compliance_score / max_compliance
        normalized_regulatory = regulatory_score / max_regulatory
        normalized_query = query_match_score / max_query if max_query > 0 else 0
        
        # Combined relevance score
        relevance_score = (
            normalized_compliance * 0.4 +
            normalized_regulatory * 0.3 +
            normalized_query * 0.3
        )
        
        return min(relevance_score, 1.0)
    
    def _get_international_law_context(self, content: str) -> Dict[str, Any]:
        """Get international law and regulatory context for content."""
        
        context = {
            'jurisdictions': [],
            'applicable_laws': [],
            'cross_border_considerations': [],
            'harmonization_notes': []
        }
        
        content_lower = content.lower()
        
        # Detect jurisdictions
        if any(word in content_lower for word in ['france', 'french', 'français']):
            context['jurisdictions'].append('France')
            context['applicable_laws'].extend(['GDPR', 'French Data Protection Law'])
        
        if any(word in content_lower for word in ['europe', 'european', 'eu']):
            context['jurisdictions'].append('European Union')
            context['applicable_laws'].extend(['GDPR', 'NIS Directive'])
        
        if any(word in content_lower for word in ['international', 'global', 'worldwide']):
            context['cross_border_considerations'].append(
                'International data transfer requirements'
            )
            context['harmonization_notes'].append(
                'Consider regulatory harmonization across jurisdictions'
            )
        
        # Detect specific regulatory frameworks
        if 'pci' in content_lower:
            context['applicable_laws'].append('PCI DSS')
            context['harmonization_notes'].append(
                'PCI DSS applies globally to card payment processing'
            )
        
        return context
    
    async def query_llm(self, query: str, context: str = "", max_tokens: int = 3000) -> str:
        """
        Enhanced LLM query with international law context and detailed analysis.
        
        Args:
            query: Analysis query
            context: Regulatory and policy context
            max_tokens: Maximum response tokens
            
        Returns:
            Enhanced LLM response with international law insights
        """
        start_time = time.time()
        
        if not self.gemini_available:
            raise RuntimeError("Gemini LLM not available")
        
        # Enhance prompt with international law context
        enhanced_prompt = self._build_enhanced_prompt(query, context)
        
        try:
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                enhanced_prompt
            )
            
            duration = time.time() - start_time
            
            if response and response.text:
                response_text = response.text.strip()
                
                # Log interaction
                log_llm_interaction(
                    len(enhanced_prompt),
                    len(response_text),
                    self.llm_config.get("model", "gemini-1.5-flash"),
                    duration
                )
                
                return response_text
            else:
                self.logger.warning("Empty response from LLM")
                return "Analysis could not be completed - empty response from LLM."
                
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"LLM query failed after {duration:.2f}s: {e}")
            raise RuntimeError(f"LLM query failed: {e}")
    
    def _build_enhanced_prompt(self, query: str, context: str) -> str:
        """Build enhanced prompt with international law context."""
        
        current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        
        enhanced_prompt = f"""
INTERNATIONAL COMPLIANCE ANALYSIS REQUEST
Analysis Date: {current_time}
Organization: SATIM
Regulatory Context: French GRC Framework with International Standards
User: LyesHADJAR

REGULATORY CONTEXT & REFERENCE MATERIAL:
{context[:4000] if context else "No specific regulatory context provided"}

ANALYSIS REQUEST:
{query}

ANALYSIS FRAMEWORK:
Please provide a comprehensive analysis that addresses:

1. REGULATORY COMPLIANCE ASSESSMENT
   - Identify specific international standards and frameworks applicable
   - Reference relevant regulatory requirements (ISO, NIST, PCI DSS, GDPR, etc.)
   - Assess compliance gaps against international best practices
   - Provide quantitative compliance scoring with clear rationale

2. INTERNATIONAL LAW CONSIDERATIONS
   - Consider cross-border regulatory requirements
   - Address harmonization between French and international standards
   - Identify jurisdiction-specific compliance obligations
   - Note any conflicting requirements and recommended resolutions

3. RISK AND IMPACT ANALYSIS
   - Quantify risks associated with current state
   - Assess potential regulatory penalties and business impact
   - Consider reputational and operational risks
   - Provide risk mitigation prioritization

4. IMPLEMENTATION GUIDANCE
   - Provide specific, actionable recommendations
   - Include realistic timelines and resource requirements
   - Reference implementation best practices from international experience
   - Consider French organizational and cultural context

5. BENCHMARKING AND STANDARDS
   - Compare against international benchmarks
   - Reference peer organization implementations
   - Cite specific regulatory guidance and interpretations
   - Provide industry-specific considerations

RESPONSE REQUIREMENTS:
- Be specific and quantitative in all assessments
- Reference specific regulatory citations and standards
- Provide actionable recommendations with clear implementation steps
- Consider resource constraints and organizational capabilities
- Maintain professional tone suitable for executive presentation
- Include relevant French regulatory context where applicable

Please ensure your analysis is evidence-based, actionable, and aligned with international compliance best practices.
"""
        
        return enhanced_prompt
    
    async def get_document_by_id(self, doc_id: str) -> Dict[str, Any]:
        """Get document with enhanced regulatory context."""
        
        # Use vector search to find relevant content
        results = await self.semantic_search(doc_id, top_k=10)
        
        if results:
            # Combine top results
            combined_content = "\n\n".join([
                f"## {result['section']}\n{result['content']}"
                for result in results[:5]
            ])
            
            return {
                "title": results[0].get('document_id', 'Unknown'),
                "content": combined_content,
                "metadata": {
                    "search_results": len(results),
                    "regulatory_context": results[0].get('regulatory_context', {}),
                    "source": "enhanced_vector_search"
                }
            }
        else:
            return {"title": "Not Found", "content": "", "metadata": {}}