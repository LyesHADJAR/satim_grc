#!/usr/bin/env python3
"""
GRC Automation with Google Gemini Flash 2.0
Main application entry point for LyesHADJAR's hackathon project.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.policy_comparison_agent import PolicyComparisonAgent
from rag.query_engine import RAGQueryEngine
from utils.config import ConfigManager

class GRCAutomationApp:
    """Main GRC Automation application."""
    
    def __init__(self):
        """Initialize the GRC Automation application."""
        self.logger = logging.getLogger("grc_app")
        self.config = ConfigManager()
        self.rag_engine = None
        self.policy_agent = None
        
    async def initialize(self):
        """Initialize the application components."""
        print("🚀 Initializing GRC Automation with Gemini Flash 2.0...")
        
        # Set up data paths
        base_path = os.path.dirname(os.path.abspath(__file__))
        data_paths = {
            "company_policies": os.path.join(base_path, "preprocessing", "policies", "satim_chunks_cleaned.json"),
            "reference_policies": os.path.join(base_path, "preprocessing", "norms", "pci_dss_chunks.json")
        }
        
        # Initialize RAG engine
        self.rag_engine = RAGQueryEngine(
            llm_config={
                "provider": "gemini",
                "model": "gemini-2.0-flash-exp",
                "temperature": 0.2,
                "max_tokens": 2000,
                "top_p": 0.8,
                "top_k": 40
            },
            data_paths=data_paths
        )
        
        # Initialize policy comparison agent
        self.policy_agent = PolicyComparisonAgent(
            name="policy_comparison_gemini",
            llm_config={"provider": "gemini"},
            rag_engine=self.rag_engine
        )
        
        print("✅ Initialization complete!")
    
    async def run_policy_analysis(self, domains: list = None) -> Dict[str, Any]:
        """Run comprehensive policy analysis."""
        if not self.policy_agent:
            await self.initialize()
        
        domains = domains or ["access control", "incident response", "data protection"]
        
        input_data = {
            "company_policy_ids": ["access control", "incident response"],
            "reference_policy_ids": ["pci_dss"],
            "domains": domains
        }
        
        print(f"🔍 Analyzing domains: {domains}")
        return await self.policy_agent.process(input_data)
    
    def display_results(self, results: Dict[str, Any]):
        """Display results in a formatted way."""
        print("\n" + "="*80)
        print("📊 GRC POLICY ANALYSIS RESULTS")
        print("="*80)
        
        for domain, domain_results in results["domain_results"].items():
            print(f"\n🎯 {domain.upper()}")
            print("─" * 50)
            
            # Quick summary
            coverage = domain_results['coverage']['coverage_percentage']
            score = domain_results['score'].score
            gap_count = len(domain_results['gaps'])
            
            status = "🟢 GOOD" if score >= 80 else "🟡 NEEDS WORK" if score >= 60 else "🔴 CRITICAL"
            
            print(f"Status: {status}")
            print(f"Score: {score:.1f}/100")
            print(f"Coverage: {coverage:.1f}%")
            print(f"Gaps: {gap_count}")
        
        overall_score = results['overall_score'].score
        overall_status = "🟢 COMPLIANT" if overall_score >= 80 else "🟡 PARTIAL" if overall_score >= 60 else "🔴 NON-COMPLIANT"
        
        print(f"\n{overall_status} OVERALL SCORE: {overall_score:.1f}/100")

async def main():
    """Main application function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🎯 GRC Automation System")
    print("👤 User: LyesHADJAR")
    print(f"📅 Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("🤖 Powered by Google Gemini Flash 2.0")
    print("="*60)
    
    try:
        app = GRCAutomationApp()
        await app.initialize()
        
        # Run analysis
        results = await app.run_policy_analysis()
        
        # Display results
        app.display_results(results)
        
        print("\n🎉 Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 