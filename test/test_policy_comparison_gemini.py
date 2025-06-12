import asyncio
import logging
from pprint import pprint
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.policy_comparison_agent import PolicyComparisonAgent
from rag.query_engine import RAGQueryEngine

async def test_policy_comparison_with_gemini():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up paths to your real data
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_paths = {
        "company_policies": os.path.join(base_path, "preprocessing", "policies", "satim_chunks_cleaned.json"),
        "reference_policies": os.path.join(base_path, "preprocessing", "norms", "pci_dss_chunks.json")
    }
    
    # Initialize the RAG engine with Gemini
    rag_engine = RAGQueryEngine(
        llm_config={
            "provider": "gemini",
            "model": "gemini-2.0-flash-exp",  # Using Gemini 2.0 Flash
            "temperature": 0.2,
            "max_tokens": 2000,
            "top_p": 0.8,
            "top_k": 40
        },
        data_paths=data_paths
    )
    
    # Initialize the policy comparison agent
    agent = PolicyComparisonAgent(
        name="policy_comparison_gemini",
        llm_config={
            "provider": "gemini",
            "model": "gemini-2.0-flash-exp",
            "temperature": 0.2,
        },
        rag_engine=rag_engine
    )
    
    # Test data for comparison using real document identifiers
    input_data = {
        "company_policy_ids": ["access control", "incident response"],  # These will match your SATIM policies
        "reference_policy_ids": ["pci_dss"],  # This will match your PCI DSS chunks
        "domains": ["access control", "incident response", "data protection"]
    }
    
    # Process the comparison
    print("\n🚀 Testing GRC Policy Comparison with Google Gemini Flash 2.0 🚀\n")
    print(f"📅 Current Date: 2025-06-12 22:07:51 UTC")
    print(f"👤 User: LyesHADJAR")
    print(f"🔍 Analyzing domains: {input_data['domains']}")
    print(f"🏢 Company policies: {input_data['company_policy_ids']}")
    print(f"📋 Reference frameworks: {input_data['reference_policy_ids']}")
    print("="*80)
    
    try:
        # Test the RAG engine first
        print("\n🔧 Testing RAG Engine...")
        test_doc = await rag_engine.get_document("access control")
        print(f"✅ Successfully loaded document: {test_doc['title']}")
        
        # Test LLM query
        print("\n🤖 Testing Gemini LLM...")
        test_query = "What are the key components of an access control policy?"
        test_response = await rag_engine.query_llm(test_query, test_doc['content'][:1000])
        print(f"✅ Gemini response length: {len(test_response)} characters")
        
        print("\n🔄 Processing policy comparison...")
        results = await agent.process(input_data)
        
        print("\n" + "="*80)
        print("📊 POLICY COMPARISON RESULTS")
        print("="*80)
        
        for domain, domain_results in results["domain_results"].items():
            print(f"\n🎯 DOMAIN: {domain.upper()}")
            print(f"{'─'*50}")
            
            # Coverage Analysis
            coverage = domain_results['coverage']
            print(f"📈 Coverage Analysis:")
            print(f"   • Total Reference Topics: {coverage['total_reference_topics']}")
            print(f"   • Covered Topics: {coverage['covered_topics']}")
            print(f"   • Coverage Percentage: {coverage['coverage_percentage']:.1f}%")
            
            # Gap Analysis
            gaps = domain_results['gaps']
            print(f"\n🔍 Gap Analysis ({len(gaps)} gaps identified):")
            for i, gap in enumerate(gaps, 1):
                severity_emoji = {"High": "🚨", "Medium": "⚠️", "Low": "💡"}.get(gap['severity'], "📝")
                print(f"   {severity_emoji} Gap {i}: {gap['description']}")
                print(f"      Severity: {gap['severity']}")
                print(f"      💡 Recommendation: {gap['recommendation']}")
            
            # Section Matches
            matches = domain_results['section_matches']
            print(f"\n🔗 Section Alignment ({len(matches)} matches found):")
            for i, match in enumerate(matches, 1):
                print(f"   🎯 Match {i}: {match.match_score:.2f} similarity")
                print(f"      Company: {match.company_section_id}")
                print(f"      Reference: {match.reference_section_id}")
            
            # Domain Score
            score = domain_results['score']
            score_emoji = "🟢" if score.score >= 80 else "🟡" if score.score >= 60 else "🔴"
            print(f"\n{score_emoji} DOMAIN SCORE: {score.score:.1f}/100")
            
            # Score Breakdown
            print(f"   📊 Score Breakdown:")
            for criterion in score.criteria:
                print(f"      • {criterion.name}: {criterion.score:.1f} (weight: {criterion.weight:.1f})")
            
            print(f"{'─'*50}")
        
        # Overall Results
        overall_score = results['overall_score']
        overall_emoji = "🟢" if overall_score.score >= 80 else "🟡" if overall_score.score >= 60 else "🔴"
        
        print(f"\n{overall_emoji} OVERALL COMPLIANCE SCORE: {overall_score.score:.1f}/100")
        print(f"{'='*50}")
        
        print(f"\n📋 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(overall_score.recommendations[:7], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n✨ Analysis completed successfully using Gemini Flash 2.0!")
        print(f"🎉 Your SATIM policies have been analyzed against PCI DSS standards.")
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set up environment variables
    print("🔧 Setup Instructions:")
    print("1. Set your Gemini API key:")
    print("   export GEMINI_API_KEY=your_gemini_api_key_here")
    print("   or")
    print("   export GOOGLE_AI_API_KEY=your_gemini_api_key_here")
    print("\n2. Get your API key from: https://makersuite.google.com/app/apikey")
    print("\n3. Run the test!")
    print("="*60)
    
    asyncio.run(test_policy_comparison_with_gemini())