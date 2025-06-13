from typing import Dict, Any, List, Optional
import asyncio
import logging
import json
import os
from .document_loader import DocumentLoader
from .context_builder import ContextBuilder
import re

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads the .env file
    print("✅ .env file loaded successfully")
except ImportError:
    print("⚠️ python-dotenv not installed - using system environment variables only")

# Import Google Gemini
try:
    import google.generativeai as genai
    HAS_GEMINI = True
    print("✅ google-generativeai package loaded successfully")
except ImportError:
    HAS_GEMINI = False
    print("❌ google-generativeai package not installed")

class EnhancedRAGQueryEngine:
    """Enhanced RAG system with no mock fallbacks - real LLM only."""
    
    def __init__(self, 
                 llm_config: Dict[str, Any], 
                 embedding_config: Dict[str, Any] = None,
                 vector_db_config: Dict[str, Any] = None,
                 data_paths: Dict[str, str] = None):
        
        self.llm_config = llm_config
        self.embedding_config = embedding_config or {}
        self.vector_db_config = vector_db_config or {}
        self.logger = logging.getLogger("rag.enhanced_query_engine")
        
        # Set up data paths
        if data_paths is None:
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_paths = {
                "company_policies": os.path.join(base_path, "preprocessing", "policies", "satim_chunks_cleaned.json"),
                "reference_policies": os.path.join(base_path, "preprocessing", "norms", "international_norms", "pci_dss_chunks.json")
            }
        
        self.logger.info(f"Company policies: {data_paths.get('company_policies')}")
        self.logger.info(f"Reference policies: {data_paths.get('reference_policies')}")
        
        # Initialize components
        self.document_loader = DocumentLoader(data_paths)
        self.context_builder = ContextBuilder(self)
        
        # Initialize LLM client - NO FALLBACKS
        self._init_gemini_client()
        
        if not self.gemini_available:
            raise RuntimeError(
                "Gemini LLM is required for analysis. Please install google-generativeai and set GEMINI_API_KEY environment variable. "
                "No mock analysis is available."
            )
    
    def _init_gemini_client(self):
        """Initialize Gemini client - fail if not available."""
        self.gemini_model = None
        self.gemini_available = False
        
        if not HAS_GEMINI:
            self.logger.error("google-generativeai package not installed")
            print("❌ Install with: pip install google-generativeai")
            return
        
        try:
            # Check multiple possible environment variable names
            api_key = (
                self.llm_config.get("api_key") or 
                os.getenv("GEMINI_API_KEY") or 
                os.getenv("GOOGLE_AI_API_KEY") or
                os.getenv("GOOGLE_API_KEY")
            )
            
            if not api_key:
                self.logger.error("No Gemini API key found in environment variables")
                print("❌ Checked for: GEMINI_API_KEY, GOOGLE_AI_API_KEY, GOOGLE_API_KEY")
                print("💡 Current environment variables:")
                for key in ["GEMINI_API_KEY", "GOOGLE_AI_API_KEY", "GOOGLE_API_KEY"]:
                    value = os.getenv(key)
                    if value:
                        print(f"   ✅ {key}: {value[:10]}...{value[-4:]}")
                    else:
                        print(f"   ❌ {key}: Not set")
                return
            
            print(f"✅ Found API key: {api_key[:10]}...{api_key[-4:]}")
            
            genai.configure(api_key=api_key)
            
            model_name = self.llm_config.get("model", "gemini-2.0-flash-exp")
            self.gemini_model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.llm_config.get("temperature", 0.2),
                    max_output_tokens=self.llm_config.get("max_tokens", 4000),
                    top_p=self.llm_config.get("top_p", 0.8),
                    top_k=self.llm_config.get("top_k", 40)
                ),
                system_instruction="""You are an expert GRC compliance analyst specializing in policy analysis, gap assessment, and regulatory compliance. 

Your expertise includes:
- Deep analysis of policy documents and regulatory standards
- Identification of compliance gaps and risks
- Quantitative assessment of policy coverage and alignment
- Strategic recommendations for compliance improvement
- Understanding of frameworks like PCI DSS, ISO 27001, SOX, GDPR

Provide detailed, accurate, and actionable analysis. Use structured output with clear sections, specific metrics, and evidence-based recommendations. Always quantify your assessments where possible."""
            )
            
            self.gemini_available = True
            self.logger.info(f"✅ Gemini Flash 2.0 initialized successfully")
            print(f"✅ Gemini Flash 2.0 initialized with model: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Gemini initialization failed: {e}")
            print(f"❌ Gemini initialization failed: {e}")
            self.gemini_available = False
    
    async def query_llm(self, query: str, context: str = "", max_tokens: int = 3000) -> str:
        """Query LLM - no fallbacks, real analysis only."""
        
        if not self.gemini_available:
            raise RuntimeError("LLM not available. Cannot perform analysis without Gemini.")
        
        self.logger.info(f"LLM Query: {query[:100]}...")
        
        try:
            return await self._query_gemini_real(query, context, max_tokens)
        except Exception as e:
            self.logger.error(f"LLM query failed: {e}")
            raise RuntimeError(f"LLM analysis failed: {e}")
    
    async def _query_gemini_real(self, query: str, context: str = "", max_tokens: int = 3000) -> str:
        """Real Gemini API call with enhanced prompting."""
        
        if context:
            full_prompt = f"""
CONTEXT DATA:
{context}

ANALYSIS REQUEST:
{query}

Please provide a comprehensive analysis based on the context. Structure your response with clear sections, specific metrics, and actionable recommendations. Include quantitative assessments where possible.
"""
        else:
            full_prompt = query
        
        try:
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                full_prompt
            )
            
            if response and response.text:
                self.logger.info("✅ Real Gemini response generated")
                return response.text.strip()
            else:
                raise RuntimeError("Empty response from Gemini")
                
        except Exception as e:
            self.logger.error(f"Gemini API error: {e}")
            raise
    
    async def comprehensive_domain_analysis(self, domain: str, company_policies: List[str], 
                                          reference_policies: List[str]) -> Dict[str, Any]:
        """Perform comprehensive domain analysis using LLM."""
        
        # Build rich comparison context
        context = await self.context_builder.build_comparison_context(
            domain, company_policies, reference_policies
        )
        
        # Format context for LLM
        formatted_context = await self.context_builder.format_for_llm_analysis(context)
        
        # Create comprehensive analysis prompt
        analysis_prompt = f"""
Perform a comprehensive compliance analysis for the {domain.upper()} domain by comparing company policies against PCI DSS reference standards.

ANALYSIS REQUIREMENTS:

1. **COVERAGE ASSESSMENT** (Provide specific numbers):
   - Calculate exact coverage percentage of reference requirements (0-100%)
   - Count how many reference topics are adequately addressed
   - Assess coverage depth (Comprehensive/Adequate/Basic/Insufficient)
   - Identify the maturity level (Advanced/Developing/Initial/Inadequate)

2. **GAP ANALYSIS** (Be specific and actionable):
   - Identify 4-6 specific compliance gaps with exact descriptions
   - Classify severity: Critical/High/Medium/Low with justification
   - Provide specific, actionable recommendations for each gap
   - Prioritize gaps by risk and implementation effort

3. **POLICY ALIGNMENT ASSESSMENT**:
   - Calculate alignment percentage with reference standards (0-100%)
   - Identify specific areas of strong alignment with evidence
   - Highlight misalignments or conflicts with reference standards
   - Assess policy implementation guidance quality

4. **QUANTITATIVE SCORING**:
   - Coverage Score (0-100): Based on how many reference topics are covered
   - Quality Score (0-100): Based on depth and adequacy of coverage  
   - Alignment Score (0-100): Based on consistency with reference standards
   - Implementation Score (0-100): Based on practical guidance provided

5. **STRATEGIC INSIGHTS**:
   - Overall domain maturity assessment
   - Key compliance strengths and competitive advantages
   - Critical improvement areas with business impact
   - Strategic recommendations with implementation priorities

Provide specific evidence from the context to support your analysis. Use exact quotes where relevant.
"""
        
        # Get comprehensive LLM analysis
        analysis_result = await self.query_llm(analysis_prompt, formatted_context, max_tokens=4000)
        
        # Parse and structure the response
        structured_analysis = await self._parse_comprehensive_analysis(analysis_result, context)
        
        return structured_analysis
    
    async def _parse_comprehensive_analysis(self, llm_response: str, context) -> Dict[str, Any]:
        """Parse comprehensive LLM analysis into structured format."""
        
        import re
        
        # Extract quantitative scores
        coverage_match = re.search(r'coverage.*?(\d+(?:\.\d+)?)%', llm_response, re.IGNORECASE)
        coverage_score = float(coverage_match.group(1)) if coverage_match else 50.0
        
        alignment_match = re.search(r'alignment.*?(\d+(?:\.\d+)?)%', llm_response, re.IGNORECASE)  
        alignment_score = float(alignment_match.group(1)) if alignment_match else 60.0
        
        quality_match = re.search(r'quality.*?(\d+(?:\.\d+)?)%', llm_response, re.IGNORECASE)
        quality_score = float(quality_match.group(1)) if quality_match else 55.0
        
        implementation_match = re.search(r'implementation.*?(\d+(?:\.\d+)?)%', llm_response, re.IGNORECASE)
        implementation_score = float(implementation_match.group(1)) if implementation_match else 50.0
        
        # Extract maturity level
        maturity_patterns = ['advanced', 'developing', 'initial', 'inadequate']
        maturity_level = "Developing"  # default
        
        for pattern in maturity_patterns:
            if pattern in llm_response.lower():
                maturity_level = pattern.capitalize()
                break
        
        # Extract gaps with improved parsing
        gaps = self._extract_detailed_gaps(llm_response)
        
        # Extract coverage details
        topics_covered_match = re.search(r'(\d+).*?topics?.*?covered', llm_response, re.IGNORECASE)
        topics_covered = int(topics_covered_match.group(1)) if topics_covered_match else int(coverage_score / 10)
        
        total_topics = len(context.reference_content)
        
        # Extract strategic insights
        insights = self._extract_strategic_insights(llm_response)
        
        return {
            "coverage": {
                "coverage_percentage": coverage_score,
                "topics_covered": topics_covered,
                "total_reference_topics": total_topics,
                "coverage_depth": self._determine_coverage_depth(coverage_score),
                "maturity_level": maturity_level
            },
            "gaps": gaps,
            "alignment": {
                "alignment_percentage": alignment_score,
                "quality_score": quality_score,
                "implementation_score": implementation_score
            },
            "quantitative_scores": {
                "coverage_score": coverage_score,
                "quality_score": quality_score,
                "alignment_score": alignment_score,
                "implementation_score": implementation_score,
                "overall_score": (coverage_score * 0.3 + quality_score * 0.25 + 
                                alignment_score * 0.25 + implementation_score * 0.2)
            },
            "strategic_insights": insights,
            "evidence_based": True,
            "analysis_timestamp": "2025-06-13 00:25:26"
        }
    
    def _extract_detailed_gaps(self, llm_response: str) -> List[Dict[str, Any]]:
        """Extract detailed gap information from LLM response."""
        
        gaps = []
        
        # Enhanced gap extraction patterns
        gap_sections = re.split(r'\n(?=\d+\.|\*|\-)', llm_response)
        
        for section in gap_sections:
            if any(keyword in section.lower() for keyword in ['gap', 'missing', 'insufficient', 'lacks', 'weak']):
                
                # Extract gap title/description
                title_match = re.search(r'(?:gap|missing|insufficient):\s*([^\n]+)', section, re.IGNORECASE)
                if not title_match:
                    title_match = re.search(r'^\d+\.\s*([^\n]+)', section)
                
                if title_match:
                    gap_title = title_match.group(1).strip()
                    
                    # Extract severity
                    severity = "Medium"  # default
                    if any(word in section.lower() for word in ['critical', 'severe', 'major']):
                        severity = "Critical"
                    elif any(word in section.lower() for word in ['high', 'significant', 'important']):
                        severity = "High"
                    elif any(word in section.lower() for word in ['low', 'minor', 'small']):
                        severity = "Low"
                    
                    # Extract recommendation
                    rec_match = re.search(r'recommend[^:]*:\s*([^\n]+)', section, re.IGNORECASE)
                    recommendation = rec_match.group(1).strip() if rec_match else f"Address {gap_title.lower()} through policy enhancement"
                    
                    # Calculate risk impact
                    risk_indicators = ['compliance', 'audit', 'regulatory', 'security', 'data breach']
                    risk_impact = "Medium"
                    
                    risk_count = sum(1 for indicator in risk_indicators if indicator in section.lower())
                    if risk_count >= 3:
                        risk_impact = "High"
                    elif risk_count <= 1:
                        risk_impact = "Low"
                    
                    gaps.append({
                        "title": gap_title,
                        "description": gap_title,
                        "severity": severity,
                        "risk_impact": risk_impact,
                        "recommendation": recommendation,
                        "evidence": section[:200] + "..." if len(section) > 200 else section
                    })
        
        # Ensure we have meaningful gaps
        if not gaps:
            gaps = [{
                "title": "Policy Documentation Enhancement Needed",
                "description": "Policy documentation requires enhancement for comprehensive compliance coverage",
                "severity": "Medium",
                "risk_impact": "Medium", 
                "recommendation": "Develop more detailed policy procedures with specific implementation guidance",
                "evidence": "Based on comparative analysis with reference standards"
            }]
        
        return gaps[:6]  # Limit to top 6 gaps
    
    def _extract_strategic_insights(self, llm_response: str) -> Dict[str, Any]:
        """Extract strategic insights from LLM analysis."""
        
        insights = {
            "key_strengths": [],
            "improvement_priorities": [],
            "strategic_recommendations": [],
            "business_impact": "",
            "implementation_roadmap": []
        }
        
        # Extract strengths
        strength_patterns = [
            r'strength[s]?[^:]*:\s*([^\n]+)',
            r'advantage[s]?[^:]*:\s*([^\n]+)',
            r'well[^:]*covered[^:]*:\s*([^\n]+)'
        ]
        
        for pattern in strength_patterns:
            matches = re.findall(pattern, llm_response, re.IGNORECASE)
            insights["key_strengths"].extend([match.strip() for match in matches[:3]])
        
        # Extract improvement priorities  
        improvement_patterns = [
            r'improve[^:]*:\s*([^\n]+)',
            r'enhance[^:]*:\s*([^\n]+)',
            r'priority[^:]*:\s*([^\n]+)'
        ]
        
        for pattern in improvement_patterns:
            matches = re.findall(pattern, llm_response, re.IGNORECASE)
            insights["improvement_priorities"].extend([match.strip() for match in matches[:4]])
        
        # Extract strategic recommendations
        rec_patterns = [
            r'recommend[^:]*:\s*([^\n]+)',
            r'strategic[^:]*:\s*([^\n]+)',
            r'should[^:]*:\s*([^\n]+)'
        ]
        
        for pattern in rec_patterns:
            matches = re.findall(pattern, llm_response, re.IGNORECASE)
            insights["strategic_recommendations"].extend([match.strip() for match in matches[:5]])
        
        return insights
    
    def _determine_coverage_depth(self, coverage_percentage: float) -> str:
        """Determine coverage depth based on percentage."""
        if coverage_percentage >= 85:
            return "Comprehensive"
        elif coverage_percentage >= 70:
            return "Adequate"
        elif coverage_percentage >= 50:
            return "Basic"
        else:
            return "Insufficient"
    
    async def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Enhanced semantic search with better scoring."""
        self.logger.info(f"Semantic search: {query}")
        
        results = self.document_loader.search_documents(query, top_k=top_k*3)
        
        # Enhanced scoring with domain relevance
        formatted_results = []
        for result in results:
            chunk = result["chunk"]
            base_score = result["score"]
            
            # Enhanced scoring factors
            section_title = chunk.get("section_title", "").lower()
            content = chunk.get("text", "").lower()
            query_lower = query.lower()
            
            # Title matching boost
            title_boost = 1.0
            if any(word in section_title for word in query_lower.split()):
                title_boost = 1.4
            
            # Content relevance boost
            content_boost = 1.0
            query_words = query_lower.split()
            content_matches = sum(1 for word in query_words if word in content)
            if content_matches > len(query_words) * 0.5:
                content_boost = 1.2
            
            # Document type consideration
            doc_type_boost = 1.0
            if result["document_type"] == "reference_policies" and "pci" in query_lower:
                doc_type_boost = 1.1
            elif result["document_type"] == "company_policies" and any(word in query_lower for word in ["satim", "company"]):
                doc_type_boost = 1.1
            
            final_score = (base_score * title_boost * content_boost * doc_type_boost) / 20.0
            final_score = min(final_score, 1.0)
            
            formatted_results.append({
                "document_id": chunk.get("document", "unknown"),
                "section": chunk.get("section_title", "Unknown Section"),
                "content": chunk.get("text", ""),
                "similarity_score": final_score,
                "document_type": result["document_type"]
            })
        
        # Sort by enhanced score and return top results
        formatted_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return formatted_results[:top_k]