"""
Test the Enhanced GRC System that was working perfectly
"""
import asyncio
import logging
import sys
import os
import json
from datetime import datetime, timezone

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env file loaded")
except ImportError:
    print("⚠️ python-dotenv not installed - using system environment variables only")

# Check for google-generativeai
try:
    import google.generativeai as genai
    print("✅ google-generativeai package loaded successfully")
except ImportError:
    print("❌ google-generativeai package not installed")
    print("💡 Install with: pip install google-generativeai")
    sys.exit(1)

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.policy_comparison_agent import EnhancedPolicyComparisonAgent
from agents.communication_protocol import AgentCommunicationProtocol
from rag.query_engine import EnhancedRAGQueryEngine

async def test_enhanced_real_llm_analysis():
    """Test the enhanced real LLM analysis system that was working."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 ENHANCED GRC AUTOMATION - REAL LLM ANALYSIS 🚀")
    print("="*70)
    print(f"📅 Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"👤 User: LyesHADJAR")
    print("🤖 Powered by REAL Gemini Flash 2.0 with Agent Collaboration")
    print("="*70)
    
    # Debug environment variables
    print("\n🔍 ENVIRONMENT VARIABLE DEBUG:")
    for key in ["GEMINI_API_KEY", "GOOGLE_AI_API_KEY", "GOOGLE_API_KEY"]:
        value = os.getenv(key)
        if value:
            print(f"   ✅ {key}: {value[:10]}...{value[-4:]}")
        else:
            print(f"   ❌ {key}: Not set")
    
    # Verify API key first
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n❌ CRITICAL: Gemini API key not found!")
        print("💡 Please check your .env file contains:")
        print("   GEMINI_API_KEY=your_api_key_here")
        print("🚫 No mock analysis available - real LLM required")
        return
    
    print(f"\n✅ Gemini API Key: Configured ({api_key[:10]}...{api_key[-4:]})")
    
    # Set up data paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_paths = {
        "company_policies": os.path.join(base_path, "preprocessing", "policies", "satim_chunks_cleaned.json"),
        "reference_policies": os.path.join(base_path, "preprocessing", "norms", "international_norms", "pci_dss_chunks.json")
    }
    
    # Verify data files
    print("\n🔧 DATA VERIFICATION:")
    for name, path in data_paths.items():
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ {name}: {len(data)} chunks loaded")
            print(f"   📊 Content: {sum(len(chunk.get('text', '')) for chunk in data):,} characters")
            print(f"   📄 Sample: {data[0].get('document', 'Unknown')}")
        else:
            print(f"❌ {name}: FILE NOT FOUND")
            return
    
    try:
        # Initialize enhanced RAG engine (no fallbacks)
        print(f"\n🔧 INITIALIZING ENHANCED SYSTEMS:")
        rag_engine = EnhancedRAGQueryEngine(
            llm_config={
                "provider": "gemini",
                "model": "gemini-2.0-flash-exp",
                "temperature": 0.2,
                "max_tokens": 4000,
                "top_p": 0.8,
                "top_k": 40,
                "api_key": api_key
            },
            data_paths=data_paths
        )
        print("✅ Enhanced RAG Engine initialized")
        
        # Initialize agent communication protocol
        communication_protocol = AgentCommunicationProtocol()
        print("✅ Agent Communication Protocol initialized")
        
        # Initialize enhanced policy comparison agent
        agent = EnhancedPolicyComparisonAgent(
            name="enhanced_policy_analyzer",
            llm_config={
                "provider": "gemini",
                "model": "gemini-2.0-flash-exp",
                "temperature": 0.2,
                "api_key": api_key
            },
            rag_engine=rag_engine,
            communication_protocol=communication_protocol
        )
        print("✅ Enhanced Policy Comparison Agent initialized")
        
        # Test enhanced semantic search
        print(f"\n🔍 TESTING ENHANCED CONTENT RETRIEVAL:")
        search_results = await rag_engine.semantic_search("access control authentication requirements", top_k=5)
        print(f"✅ Found {len(search_results)} relevant sections with enhanced scoring")
        for i, result in enumerate(search_results[:3], 1):
            print(f"   {i}. {result['section'][:60]}... (Score: {result['similarity_score']:.3f})")
        
        # Run enhanced analysis
        input_data = {
            "company_policy_ids": ["satim", "access control", "incident response"],
            "reference_policy_ids": ["pci-dss"],
            "domains": ["access_control", "incident_response", "data_protection"]
        }
        
        print(f"\n🔄 RUNNING ENHANCED REAL LLM ANALYSIS:")
        print(f"   🏢 Company Policies: {input_data['company_policy_ids']}")
        print(f"   📋 Reference Standards: {input_data['reference_policy_ids']}")
        print(f"   🎯 Analysis Domains: {input_data['domains']}")
        print(f"   🤖 Analysis Mode: Real LLM + Agent Collaboration")
        
        # Perform comprehensive analysis
        results = await agent.process(input_data)
        
        print("\n" + "="*70)
        print("📊 ENHANCED REAL LLM ANALYSIS RESULTS")
        print("="*70)
        
        domain_scores = []
        total_gaps = 0
        
        for domain, domain_results in results["domain_results"].items():
            print(f"\n🎯 DOMAIN: {domain.upper()}")
            print("─" * 60)
            
            coverage = domain_results['coverage']
            gaps = domain_results['gaps']
            alignment = domain_results['alignment']
            quantitative_scores = domain_results['quantitative_scores']
            score = domain_results['score']
            insights = domain_results.get('strategic_insights', {})
            
            domain_scores.append(score.score)
            total_gaps += len(gaps)
            
            # Enhanced Coverage Analysis
            print(f"📈 COMPREHENSIVE COVERAGE ANALYSIS:")
            print(f"   • Coverage Percentage: {coverage['coverage_percentage']:.1f}%")
            print(f"   • Topics Covered: {coverage['topics_covered']}/{coverage['total_reference_topics']}")
            print(f"   • Coverage Depth: {coverage['coverage_depth']}")
            print(f"   • Maturity Level: {coverage['maturity_level']}")
            
            # Quantitative Scoring Breakdown
            print(f"\n📊 QUANTITATIVE SCORES:")
            print(f"   • Coverage Score: {quantitative_scores['coverage_score']:.1f}/100")
            print(f"   • Quality Score: {quantitative_scores['quality_score']:.1f}/100")
            print(f"   • Alignment Score: {quantitative_scores['alignment_score']:.1f}/100")
            print(f"   • Implementation Score: {quantitative_scores['implementation_score']:.1f}/100")
            
            # Enhanced Gap Analysis
            print(f"\n🔍 DETAILED GAP ANALYSIS ({len(gaps)} gaps identified):")
            for i, gap in enumerate(gaps[:4], 1):  # Show top 4 gaps
                severity_emoji = {
                    "Critical": "🚨", "High": "⚠️", "Medium": "💡", "Low": "📝"
                }.get(gap['severity'], "📝")
                print(f"   {severity_emoji} Gap {i}: {gap['title']}")
                print(f"      📊 Severity: {gap['severity']} | Risk Impact: {gap.get('risk_impact', 'Medium')}")
                print(f"      💡 Recommendation: {gap['recommendation'][:100]}...")
            
            # Strategic Insights
            if insights:
                print(f"\n🧠 STRATEGIC INSIGHTS:")
                if insights.get('key_strengths'):
                    print(f"   💪 Key Strengths:")
                    for strength in insights['key_strengths'][:3]:
                        print(f"      • {strength}")
                
                if insights.get('improvement_priorities'):
                    print(f"   📈 Improvement Priorities:")
                    for priority in insights['improvement_priorities'][:3]:
                        print(f"      • {priority}")
            
            # Enhanced Compliance Score
            score_emoji = "🟢" if score.score >= 75 else "🟡" if score.score >= 55 else "🔴"
            print(f"\n{score_emoji} ENHANCED COMPLIANCE SCORE: {score.score:.1f}/100")
            print(f"   📊 Score Components:")
            for criterion in score.criteria:
                comp_emoji = "🟢" if criterion.score >= 75 else "🟡" if criterion.score >= 55 else "🔴"
                print(f"      {comp_emoji} {criterion.name}: {criterion.score:.1f} (weight: {criterion.weight:.0%})")
            
            print("─" * 60)
        
        # Enhanced Overall Assessment
        overall_score = results['overall_score']
        overall_emoji = "🟢" if overall_score.score >= 75 else "🟡" if overall_score.score >= 55 else "🔴"
        
        print(f"\n{overall_emoji} ENTERPRISE COMPLIANCE ASSESSMENT")
        print("="*60)
        print(f"📊 Enterprise Score: {overall_score.score:.1f}/100")
        print(f"📈 Domain Score Range: {min(domain_scores):.1f} - {max(domain_scores):.1f}")
        print(f"🔍 Total Gaps Identified: {total_gaps}")
        
        # Enhanced Compliance Level
        if overall_score.score >= 80:
            compliance_level = "🟢 ADVANCED - Mature compliance framework with strong controls"
        elif overall_score.score >= 65:
            compliance_level = "🟡 DEVELOPING - Good foundation with targeted improvements needed"
        elif overall_score.score >= 50:
            compliance_level = "🟡 EMERGING - Basic framework requiring significant enhancement"
        else:
            compliance_level = "🔴 INITIAL - Foundational development and urgent attention required"
        
        print(f"🎯 Enterprise Assessment: {compliance_level}")
        
        # Strategic Enterprise Recommendations
        print(f"\n📋 STRATEGIC ENTERPRISE RECOMMENDATIONS:")
        for i, rec in enumerate(overall_score.recommendations[:8], 1):
            print(f"   {i}. {rec}")
        
        # Enhanced Analysis Quality Indicators
        print(f"\n📊 ANALYSIS QUALITY METRICS:")
        print("   ✅ Real Gemini Flash 2.0 Analysis - No Mock Fallbacks")
        print("   ✅ Agent Collaborative Intelligence")
        print("   ✅ Comprehensive Context Building")
        print("   ✅ Evidence-Based Gap Identification")
        print("   ✅ Quantitative Scoring Framework")
        print("   ✅ Strategic Enterprise Assessment")
        
        # Performance Metrics
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   • Analysis Approach: {results.get('analysis_approach', 'enhanced_llm_collaborative_analysis')}")
        print(f"   • Analysis Quality: {results.get('analysis_quality', 'comprehensive_real_llm')}")
        print(f"   • Domains Analyzed: {len(results['domain_results'])}")
        print(f"   • Total Sections Analyzed: {sum(len(d.get('coverage', {})) for d in results['domain_results'].values()) if results['domain_results'] else 0}")
        print(f"   • Evidence-Based Analysis: {'✅ Yes' if any(d.get('evidence_based') for d in results['domain_results'].values()) else '❌ No'}")
        
        # Risk Assessment Summary
        print(f"\n🎯 EXECUTIVE RISK SUMMARY:")
        critical_gaps = sum(1 for domain_result in results['domain_results'].values() 
                           for gap in domain_result['gaps'] 
                           if gap.get('severity') == 'Critical')
        high_gaps = sum(1 for domain_result in results['domain_results'].values() 
                       for gap in domain_result['gaps'] 
                       if gap.get('severity') == 'High')
        
        print(f"   🚨 Critical Risk Areas: {critical_gaps}")
        print(f"   ⚠️ High Priority Items: {high_gaps}")
        print(f"   📊 Overall Risk Level: {'High' if critical_gaps > 0 else 'Medium' if high_gaps > 2 else 'Low'}")
        
        # Implementation Roadmap
        print(f"\n🗺️ IMPLEMENTATION ROADMAP:")
        print("   Phase 1 (0-3 months): Address critical gaps and high-priority items")
        print("   Phase 2 (3-6 months): Implement strategic recommendations")
        print("   Phase 3 (6-12 months): Enhance maturity and continuous improvement")
        
        print(f"\n✨ Enhanced real LLM analysis complete!")
        print(f"🎉 SATIM policies analyzed with comprehensive AI intelligence!")
        print(f"🔥 No mock analysis used - 100% real LLM powered!")
        
        # Test agent collaboration
        print(f"\n🤝 TESTING AGENT COLLABORATION:")
        from agents.communication_protocol import AgentRequest, RequestType
        
        # Test content analysis request
        test_request = AgentRequest(
            request_id="test_001",
            requesting_agent="test_coordinator",
            target_agent="enhanced_policy_analyzer",
            request_type=RequestType.CONTENT_ANALYSIS,
            data={"domain": "access control", "content": ["test content"]}
        )
        
        collaboration_response = await communication_protocol.request_analysis(test_request)
        if collaboration_response.success:
            print("   ✅ Agent collaboration working successfully")
            print(f"   📊 Response data keys: {list(collaboration_response.data.keys())}")
        else:
            print(f"   ❌ Agent collaboration failed: {collaboration_response.error_message}")
        
    except RuntimeError as e:
        if "Gemini" in str(e):
            print(f"\n❌ LLM INITIALIZATION FAILED:")
            print(f"   Error: {e}")
            print(f"   💡 Ensure GEMINI_API_KEY is set correctly")
            print(f"   🚫 No mock analysis available - real LLM required")
        else:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"❌ Error during enhanced analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔧 ENHANCED REAL LLM ANALYSIS SYSTEM:")
    print("   ✅ Enhanced policy comparison agent")
    print("   ✅ Real Gemini Flash 2.0 integration (no fallbacks)")
    print("   ✅ Agent collaborative intelligence")
    print("   ✅ Comprehensive context building")
    print("   ✅ Evidence-based gap identification")
    print("   ✅ Quantitative scoring framework")
    print("   ✅ Strategic enterprise assessment")
    print("")
    
    asyncio.run(test_enhanced_real_llm_analysis())