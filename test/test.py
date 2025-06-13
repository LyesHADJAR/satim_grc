"""
Enhanced Test System with Vector Search and International Law Context
Current Date: 2025-06-13 20:26:29 UTC
Current User: LyesHADJAR
"""
import asyncio
import sys
import os
import time
from datetime import datetime, timezone

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.policy_comparison_agent import EnhancedPolicyComparisonAgent
from agents.policy_feedback_agent import IntelligentPolicyFeedbackAgent
from agents.communication_protocol import AgentCommunicationProtocol
from rag.query_engine import InternationalLawEnhancedRAGEngine
from utils.logging_config import setup_logging, log_analysis_stage, log_domain_analysis, log_performance

# Try to import rich for enhanced output
try:
    from utils.rich_output import EnhancedRichDisplay
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

async def test_enhanced_vector_system():
    """Test the enhanced system with vector search and international law context."""
    
    # Setup enhanced logging
    setup_logging(log_level="INFO", enable_console=True)
    
    # Initialize display
    if RICH_AVAILABLE:
        display = EnhancedRichDisplay()
        display.display_startup_banner()
    else:
        print("🚀 ENHANCED SATIM GRC SYSTEM WITH VECTOR SEARCH")
        print("=" * 80)
    
    log_analysis_stage("STARTUP", "Enhanced SATIM GRC Analysis with Vector Search and International Law Context")
    
    # API key verification
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        log_analysis_stage("ERROR", "CRITICAL: Gemini API key not found!", "ERROR")
        return
    
    log_analysis_stage("CONFIG", f"Gemini API configured: {api_key[:10]}...{api_key[-4:]}")
    
    # Set up data paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_paths = {
        "company_policies": os.path.join(base_path, "preprocessing", "policies", "satim_chunks_cleaned.json"),
        "reference_policies": os.path.join(base_path, "preprocessing", "norms", "international_norms", "pci_dss_chunks.json")
    }
    
    # Verify data files
    for name, path in data_paths.items():
        if os.path.exists(path):
            log_analysis_stage("DATA", f"✅ {name}: {path}")
        else:
            log_analysis_stage("ERROR", f"❌ {name}: FILE NOT FOUND - {path}", "ERROR")
            return
    
    try:
        overall_start_time = time.time()
        
        # Initialize enhanced systems
        log_analysis_stage("INIT", "Initializing Enhanced RAG Engine with Vector Search")
        rag_engine = InternationalLawEnhancedRAGEngine(
            llm_config={
                "provider": "gemini",
                "model": "gemini-1.5-flash",
                "temperature": 0.2,
                "max_tokens": 4000,
                "api_key": api_key
            },
            data_paths=data_paths
        )
        
        log_analysis_stage("INIT", "Initializing Communication Protocol")
        communication_protocol = AgentCommunicationProtocol()
        
        log_analysis_stage("INIT", "Initializing Policy Comparison Agent")
        policy_agent = EnhancedPolicyComparisonAgent(
            name="enhanced_policy_agent",
            llm_config={"provider": "gemini", "model": "gemini-1.5-flash", "api_key": api_key},
            rag_engine=rag_engine,
            communication_protocol=communication_protocol
        )
        
        log_analysis_stage("INIT", "Initializing Intelligent Feedback Agent")
        feedback_agent = IntelligentPolicyFeedbackAgent(
            name="enhanced_feedback_agent",
            llm_config={"provider": "gemini", "model": "gemini-1.5-flash", "api_key": api_key},
            rag_engine=rag_engine,
            communication_protocol=communication_protocol
        )
        
        # Test configuration
        input_data = {
            "company_policy_ids": ["satim"],
            "reference_policy_ids": ["pci-dss"],
            "domains": ["access_control", "data_protection", "incident_response"]
        }
        
        log_analysis_stage("ANALYSIS", "Starting comprehensive GRC analysis with vector search")
        
        # PHASE 1: Enhanced Policy Analysis with Vector Search
        log_analysis_stage("PHASE_1", "Policy Analysis with International Law Context")
        phase1_start = time.time()
        
        results = await policy_agent.process(input_data)
        
        log_performance("Phase 1 - Policy Analysis", time.time() - phase1_start, {
            "domains_analyzed": len(results.get("domain_results", {})),
            "discovered_domains": len(results.get("discovered_domains", {}))
        })
        
        # Display results if Rich is available
        if RICH_AVAILABLE and 'display' in locals():
            display.display_domain_discovery_results(results.get("discovered_domains", {}))
            display.display_french_compliance_analysis(results.get("domain_results", {}))
            display.display_gap_analysis(results.get("domain_results", {}))
        
        # PHASE 2: Enhanced Feedback Generation
        log_analysis_stage("PHASE_2", "Intelligent Policy Improvement Feedback")
        phase2_start = time.time()
        
        feedback_input = {
            "domain_results": results.get("domain_results", {}),
            "overall_score": results.get("overall_score", {}),
            "organization_context": "SATIM"
        }
        
        feedback_results = await feedback_agent.process(feedback_input)
        
        log_performance("Phase 2 - Feedback Generation", time.time() - phase2_start, {
            "recommendations_generated": len(feedback_results.get("improvement_recommendations", {}))
        })
        
        # PHASE 3: Vector Search Demonstration
        log_analysis_stage("PHASE_3", "Vector Search Demonstration")
        phase3_start = time.time()
        
        # Test vector search capabilities
        search_queries = [
            "access control authentication requirements",
            "data encryption protection standards",
            "incident response procedures",
            "PCI DSS compliance requirements",
            "French regulatory requirements"
        ]
        
        for query in search_queries:
            log_analysis_stage("SEARCH", f"Testing vector search: {query}")
            search_results = await rag_engine.semantic_search(query, top_k=3)
            
            log_performance(f"Vector Search - {query[:20]}...", 0.1, {
                "results_found": len(search_results),
                "avg_relevance": sum(r.get('enhanced_similarity_score', 0) for r in search_results) / len(search_results) if search_results else 0
            })
            
            for i, result in enumerate(search_results[:2], 1):
                regulatory_context = result.get('regulatory_context', {})
                standards = regulatory_context.get('identified_standards', [])
                log_analysis_stage("SEARCH", f"  Result {i}: {result['section'][:50]}... | Standards: {', '.join(standards)}")
        
        log_performance("Phase 3 - Vector Search Demo", time.time() - phase3_start)
        
        # Summary and Results
        total_duration = time.time() - overall_start_time
        
        log_analysis_stage("SUMMARY", "Analysis Complete - Generating Summary")
        
        # Generate summary statistics
        domain_results = results.get("domain_results", {})
        total_domains = len(domain_results)
        total_gaps = sum(len(domain_result.get("gaps", [])) for domain_result in domain_results.values())
        
        overall_score = results.get("overall_score", {})
        enterprise_score = overall_score.score if hasattr(overall_score, 'score') else 0
        
        french_summary = results.get("french_compliance_summary", {})
        overall_french = french_summary.get("overall_french_compliance", {})
        compliance_level = overall_french.get("overall_compliance_level", 0)
        
        # Log final summary
        log_analysis_stage("RESULTS", f"Enterprise Compliance Score: {enterprise_score:.1f}/100")
        log_analysis_stage("RESULTS", f"French Compliance Level: {compliance_level}/5")
        log_analysis_stage("RESULTS", f"Domains Analyzed: {total_domains}")
        log_analysis_stage("RESULTS", f"Total Gaps Identified: {total_gaps}")
        
        log_performance("Complete Analysis", total_duration, {
            "domains": total_domains,
            "gaps": total_gaps,
            "compliance_score": enterprise_score,
            "french_level": compliance_level
        })
        
        # Display final results if Rich is available
        if RICH_AVAILABLE and 'display' in locals():
            display.display_overall_assessment(overall_score, french_summary)
            display.display_completion_summary(total_duration, total_domains, total_gaps)
        else:
            print(f"\n✅ ANALYSIS COMPLETE")
            print(f"📊 Enterprise Score: {enterprise_score:.1f}/100")
            print(f"🇫🇷 French Level: {compliance_level}/5")
            print(f"⏱️ Duration: {total_duration:.1f}s")
            print(f"🎯 Domains: {total_domains} | Gaps: {total_gaps}")
        
        log_analysis_stage("SUCCESS", "🎉 SATIM GRC Analysis Successfully Completed with Enhanced Vector Search!")
        
        return results, feedback_results
        
    except Exception as e:
        log_analysis_stage("ERROR", f"Analysis failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 ENHANCED SATIM GRC SYSTEM")
    print("✅ Vector search with FAISS integration")
    print("✅ International law and regulatory context")
    print("✅ Enhanced logging with performance tracking")
    print("✅ LLM responses with regulatory expertise")
    print("")
    
    asyncio.run(test_enhanced_vector_system())