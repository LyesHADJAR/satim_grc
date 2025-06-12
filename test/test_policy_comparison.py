import asyncio
import logging
from pprint import pprint
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.policy_comparison_agent import PolicyComparisonAgent
from rag.query_engine import RAGQueryEngine

async def test_policy_comparison():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize the RAG engine
    rag_engine = RAGQueryEngine(
        llm_config={
            "model": "gpt-4",
            "temperature": 0.2,
            "max_tokens": 2000
        }
    )
    
    # Initialize the policy comparison agent
    agent = PolicyComparisonAgent(
        name="policy_comparison",
        llm_config={
            "model": "gpt-4",
            "temperature": 0.2,
        },
        rag_engine=rag_engine
    )
    
    # Test data for comparison
    input_data = {
        "company_policy_ids": ["company_data_policy"],
        "reference_policy_ids": ["iso27001_data_security"],
        "domains": ["data protection"]
    }
    
    # Process the comparison
    print("\n=== Testing Policy Comparison Agent ===\n")
    print(f"Comparing policies for domains: {input_data['domains']}")
    print(f"Company policies: {input_data['company_policy_ids']}")
    print(f"Reference frameworks: {input_data['reference_policy_ids']}\n")
    
    results = await agent.process(input_data)
    
    print("=== Comparison Results ===\n")
    for domain, domain_results in results["domain_results"].items():
        print(f"Domain: {domain}")
        print(f"Coverage: {domain_results['coverage']['coverage_percentage']:.1f}%")
        print("Gaps identified:")
        for i, gap in enumerate(domain_results['gaps'], 1):
            print(f"  {i}. {gap['description']} (Severity: {gap['severity']})")
            print(f"     Recommendation: {gap['recommendation']}")
        print(f"Score: {domain_results['score'].score:.1f}/100\n")
    
    print(f"Overall compliance score: {results['overall_score'].score:.1f}/100")
    print("\nRecommendations:")
    for i, rec in enumerate(results['overall_score'].recommendations, 1):
        print(f"  {i}. {rec}")

if __name__ == "__main__":
    asyncio.run(test_policy_comparison())