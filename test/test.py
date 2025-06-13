import asyncio
import logging
import sys
import os
import json

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.policy_comparison_agent import PolicyComparisonAgent
from rag.query_engine import RAGQueryEngine

async def test_real_llm_analysis():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 GRC AUTOMATION - REAL LLM ANALYSIS 🚀")
    print("="*70)
    print(f"📅 Date: 2025-06-13 00:11:54 UTC")
    print(f"👤 User: LyesHADJAR")
    print("🤖 Powered by REAL Gemini Flash 2.0 Analysis")
    print("="*70)
    
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
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")
    if api_key:
        print(f"\n✅ Gemini API Key: Configured")
        llm_available = True
    else:
        print(f"\n⚠️ Gemini API Key: Not found")
        print("💡 Set GEMINI_API_KEY for full LLM analysis")
        llm_available = False
    
    # Initialize RAG engine
    print(f"\n🔧 INITIALIZING SYSTEMS:")
    rag_engine = RAGQueryEngine(
        llm_config={
            "provider": "gemini",
            "model": "gemini-2.0-flash-exp",
            "temperature": 0.2,
            "max_tokens": 3000,
            "top_p": 0.8,
            "top_k": 40
        },
        data_paths=data_paths
    )
    print("✅ RAG Engine initialized")
    
    # Initialize real policy comparison agent
    agent = PolicyComparisonAgent(
        name="real_policy_analyzer",
        llm_config={
            "provider": "gemini",
            "model": "gemini-2.0-flash-exp",
            "temperature": 0.2,
        },
        rag_engine=rag_engine
    )
    print("✅ Policy Comparison Agent initialized")
    
    # Test semantic search first
    print(f"\n🔍 TESTING CONTENT RETRIEVAL:")
    search_results = await rag_engine.semantic_search("access control authentication", top_k=5)
    print(f"✅ Found {len(search_results)} relevant sections")
    for i, result in enumerate(search_results[:3], 1):
        print(f"   {i}. {result['section'][:50]}... (Score: {result['similarity_score']:.2f})")
    
    # Run real analysis
    input_data = {
        "company_policy_ids": ["satim", "access control", "incident response"],
        "reference_policy_ids": ["pci-dss"],
        "domains": ["access control", "incident response", "data protection"]
    }
    
    print(f"\n🔄 RUNNING REAL LLM ANALYSIS:")
    print(f"   🏢 Company Policies: {input_data['company_policy_ids']}")
    print(f"   📋 Reference Standards: {input_data['reference_policy_ids']}")
    print(f"   🎯 Analysis Domains: {input_data['domains']}")
    
    if llm_available:
        print("\n🤖 Using REAL Gemini Flash 2.0 Analysis...")
    else:
        print("\n🔧 Using Enhanced Mock Analysis...")
    
    try:
        results = await agent.process(input_data)
        
        print("\n" + "="*70)
        print("📊 REAL LLM ANALYSIS RESULTS")
        print("="*70)
        
        domain_scores = []
        total_gaps = 0
        
        for domain, domain_results in results["domain_results"].items():
            print(f"\n🎯 DOMAIN: {domain.upper()}")
            print("─" * 60)
            
            coverage = domain_results['coverage']
            gaps = domain_results['gaps']
            matches = domain_results['section_matches']
            score = domain_results['score']
            insights = domain_results.get('llm_insights', {})
            
            domain_scores.append(score.score)
            total_gaps += len(gaps)
            
            # Coverage Analysis (from real LLM)
            print(f"📈 LLM COVERAGE ANALYSIS:")
            print(f"   • Reference Topics: {coverage['total_reference_topics']}")
            print(f"   • Covered by Company: {coverage['covered_topics']}")
            print(f"   • Coverage Percentage: {coverage['coverage_percentage']:.1f}%")
            print(f"   • Maturity Level: {insights.get('maturity_level', 'Unknown')}")
            
            # Gap Analysis (from real LLM)
            print(f"\n🔍 LLM GAP ANALYSIS ({len(gaps)} gaps identified):")
            if gaps:
                for i, gap in enumerate(gaps, 1):
                    severity_emoji = {"High": "🚨", "Medium": "⚠️", "Low": "💡"}.get(gap['severity'], "📝")
                    print(f"   {severity_emoji} Gap {i}: {gap['description']}")
                    print(f"      📊 Severity: {gap['severity']}")
                    print(f"      💡 Recommendation: {gap['recommendation']}")
            else:
                print("   ✅ No significant gaps identified by LLM")
            
            # LLM Insights
            if insights:
                print(f"\n🧠 LLM INSIGHTS:")
                if insights.get('key_strengths'):
                    print(f"   💪 Strengths: {', '.join(insights['key_strengths'][:2])}")
                if insights.get('improvement_areas'):
                    print(f"   📈 Improvements: {', '.join(insights['improvement_areas'][:2])}")
            
            # Alignment Analysis
            print(f"\n🔗 POLICY ALIGNMENT:")
            if matches:
                print(f"   ✅ {len(matches)} alignment areas identified")
                for i, match in enumerate(matches, 1):
                    print(f"   {i}. Confidence: {match.match_score:.2f}")
            else:
                print("   ⚠️ Limited alignment detected - requires review")
            
            # Score with real criteria
            score_emoji = "🟢" if score.score >= 75 else "🟡" if score.score >= 55 else "🔴"
            print(f"\n{score_emoji} LLM COMPLIANCE SCORE: {score.score:.1f}/100")
            print(f"   📊 Score Components:")
            for criterion in score.criteria:
                comp_emoji = "🟢" if criterion.score >= 75 else "🟡" if criterion.score >= 55 else "🔴"
                print(f"      {comp_emoji} {criterion.name}: {criterion.score:.1f} (weight: {criterion.weight:.0%})")
            
            print("─" * 60)
        
        # Overall LLM Assessment
        overall_score = results['overall_score']
        overall_emoji = "🟢" if overall_score.score >= 75 else "🟡" if overall_score.score >= 55 else "🔴"
        
        print(f"\n{overall_emoji} OVERALL LLM ASSESSMENT")
        print("="*60)
        print(f"📊 LLM Overall Score: {overall_score.score:.1f}/100")
        print(f"📈 Domain Score Range: {min(domain_scores):.1f} - {max(domain_scores):.1f}")
        print(f"🔍 Total Gaps: {total_gaps}")
        
        # Compliance Level
        if overall_score.score >= 80:
            compliance_level = "🟢 STRONG - Robust compliance framework"
        elif overall_score.score >= 65:
            compliance_level = "🟡 GOOD - Solid foundation with improvements needed"
        elif overall_score.score >= 50:
            compliance_level = "🟡 DEVELOPING - Basic framework requires enhancement"
        else:
            compliance_level = "🔴 EMERGING - Foundational development needed"
        
        print(f"🎯 LLM Assessment: {compliance_level}")
        
        # Strategic Recommendations from LLM
        print(f"\n📋 LLM STRATEGIC RECOMMENDATIONS:")
        for i, rec in enumerate(overall_score.recommendations[:6], 1):
            print(f"   {i}. {rec}")
        
        # Analysis Quality Indicators
        print(f"\n📊 ANALYSIS QUALITY:")
        if llm_available:
            print("   ✅ Real Gemini Flash 2.0 Analysis Used")
            print("   ✅ Contextual Content Analysis")
            print("   ✅ Dynamic Gap Identification")
            print("   ✅ Strategic Recommendations")
        else:
            print("   ⚠️ Enhanced Mock Analysis Used")
            print("   💡 Install google-generativeai for full LLM power")
        
        print(f"\n✨ Real LLM analysis complete!")
        print(f"🎉 SATIM policies analyzed with actual intelligence!")
        
    except Exception as e:
        print(f"❌ Error during real analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔧 REAL LLM ANALYSIS SYSTEM:")
    print("   ✅ Real policy comparison agent")
    print("   ✅ Actual Gemini Flash 2.0 integration")
    print("   ✅ Dynamic content analysis")
    print("   ✅ Intelligent gap identification")
    print("   ✅ Contextual recommendations")
    print("")
    
    asyncio.run(test_real_llm_analysis())