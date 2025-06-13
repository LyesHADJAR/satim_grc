"""
Test Dynamic Domain Discovery and Expertise Extraction
"""
import asyncio
import logging
import sys
import os
import json
from datetime import datetime, timezone

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.policy_comparison_agent import EnhancedPolicyComparisonAgent
from agents.communication_protocol import AgentCommunicationProtocol
from rag.query_engine import EnhancedRAGQueryEngine

async def test_dynamic_domain_discovery():
    """Test dynamic domain discovery and analysis."""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("🚀 DYNAMIC DOMAIN DISCOVERY - GRC ANALYSIS 🚀")
    print("="*70)
    print(f"📅 Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"👤 User: LyesHADJAR")
    print("🤖 Powered by Dynamic Domain Discovery + Real Gemini LLM")
    print("="*70)
    
    # API key verification
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n❌ CRITICAL: Gemini API key not found!")
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
        else:
            print(f"❌ {name}: FILE NOT FOUND")
            return
    
    try:
        # Initialize systems
        print(f"\n🔧 INITIALIZING DYNAMIC DISCOVERY SYSTEM:")
        
        rag_engine = EnhancedRAGQueryEngine(
            llm_config={
                "provider": "gemini",
                "model": "gemini-2.0-flash-exp",
                "temperature": 0.2,
                "max_tokens": 4000,
                "api_key": api_key
            },
            data_paths=data_paths
        )
        print("✅ Enhanced RAG Engine initialized")
        
        communication_protocol = AgentCommunicationProtocol()
        print("✅ Agent Communication Protocol initialized")
        
        agent = EnhancedPolicyComparisonAgent(
            name="dynamic_discovery_agent",
            llm_config={
                "provider": "gemini",
                "model": "gemini-2.0-flash-exp",
                "temperature": 0.2,
                "api_key": api_key
            },
            rag_engine=rag_engine,
            communication_protocol=communication_protocol
        )
        print("✅ Dynamic Domain Discovery Agent initialized")
        
        # Test 1: Auto-discovery (no domains provided)
        print(f"\n🔄 TEST 1: AUTOMATIC DOMAIN DISCOVERY:")
        input_data_auto = {
            "company_policy_ids": ["satim"],
            "reference_policy_ids": ["pci-dss"],
            # No domains provided - should auto-discover
        }
        
        print("   🔍 Running automatic domain discovery...")
        results_auto = await agent.process(input_data_auto)
        
        print(f"\n📊 DISCOVERED DOMAINS:")
        discovered = results_auto.get("discovered_domains", {})
        for domain, info in discovered.items():
            confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(info.get("confidence", "low"), "🔴")
            print(f"   {confidence_emoji} {domain}: {info.get('confidence', 'unknown')} confidence")
            print(f"      📝 Evidence: {info.get('evidence', 'No evidence')[:80]}...")
            print(f"      🔖 Topics: {', '.join(info.get('key_topics', [])[:5])}")
        
        print(f"\n📊 EXTRACTED DOMAIN EXPERTISE:")
        expertise = results_auto.get("extracted_domain_expertise", {})
        for domain, exp in expertise.items():
            print(f"\n🎯 {domain.upper()}:")
            print(f"   🔑 Key Topics ({len(exp.get('key_topics', []))}): {', '.join(exp.get('key_topics', [])[:6])}")
            print(f"   🛡️ Critical Controls ({len(exp.get('critical_controls', []))}): {', '.join(exp.get('critical_controls', [])[:4])}")
            print(f"   📋 Frameworks: {', '.join(exp.get('compliance_frameworks', []))}")
            print(f"   ⚠️ Risk Factors: {', '.join(exp.get('risk_factors', [])[:3])}")
        
        print(f"\n📊 ANALYSIS RESULTS:")
        for domain, domain_result in results_auto["domain_results"].items():
            score = domain_result['score']
            coverage = domain_result['coverage']
            score_emoji = "🟢" if score.score >= 75 else "🟡" if score.score >= 55 else "🔴"
            print(f"\n{score_emoji} {domain.upper()}: {score.score:.1f}/100")
            print(f"   📈 Coverage: {coverage['coverage_percentage']:.1f}%")
            print(f"   📊 Maturity: {coverage['maturity_level']}")
            print(f"   🔍 Gaps: {len(domain_result['gaps'])}")
        
        # Overall Results
        overall = results_auto['overall_score']
        overall_emoji = "🟢" if overall.score >= 75 else "🟡" if overall.score >= 55 else "🔴"
        print(f"\n{overall_emoji} OVERALL ENTERPRISE SCORE: {overall.score:.1f}/100")
        
        # Test 2: Manual domain specification
        print(f"\n🔄 TEST 2: MANUAL DOMAIN SPECIFICATION:")
        input_data_manual = {
            "company_policy_ids": ["satim"],
            "reference_policy_ids": ["pci-dss"], 
            "domains": ["access_control", "data_protection"]  # Manually specified
        }
        
        print("   🎯 Using manually specified domains...")
        results_manual = await agent.process(input_data_manual)
        
        print(f"\n📊 MANUAL DOMAIN ANALYSIS:")
        for domain in results_manual["domains_analyzed"]:
            domain_result = results_manual["domain_results"][domain]
            score = domain_result['score']
            score_emoji = "🟢" if score.score >= 75 else "🟡" if score.score >= 55 else "🔴"
            print(f"   {score_emoji} {domain}: {score.score:.1f}/100")
        
        # Performance Comparison
        print(f"\n📊 DISCOVERY vs MANUAL COMPARISON:")
        auto_domains = len(results_auto["domain_results"])
        manual_domains = len(results_manual["domain_results"])
        
        print(f"   🔍 Auto-discovered domains: {auto_domains}")
        print(f"   🎯 Manual domains: {manual_domains}")
        print(f"   📈 Discovery effectiveness: {'✅ Good' if auto_domains >= 3 else '⚠️ Limited'}")
        
        print(f"\n✨ Dynamic domain discovery complete!")
        print(f"🎉 System successfully adapted to document content!")
        
    except Exception as e:
        print(f"❌ Error during dynamic discovery: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔧 DYNAMIC DOMAIN DISCOVERY SYSTEM:")
    print("   ✅ Automatic domain identification from documents")
    print("   ✅ Dynamic expertise extraction from reference standards")
    print("   ✅ Adaptive analysis based on discovered content")
    print("   ✅ Real Gemini LLM integration")
    print("")
    
    asyncio.run(test_dynamic_domain_discovery())