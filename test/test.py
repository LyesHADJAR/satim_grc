"""
Enhanced Test System with Rich Output and LLM Feedback
Current Date: 2025-06-13 15:28:46 UTC
Current User: LyesHADJAR
"""
import asyncio
import logging
import sys
import os
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.policy_comparison_agent import EnhancedPolicyComparisonAgent
from agents.policy_feedback_agent import IntelligentPolicyFeedbackAgent
from agents.communication_protocol import AgentCommunicationProtocol
from rag.query_engine import EnhancedRAGQueryEngine
from utils.rich_output import EnhancedRichDisplay

async def test_enhanced_grc_system():
    """Test the complete enhanced GRC system with rich output and LLM feedback."""
    
    # Initialize rich display
    rich_display = EnhancedRichDisplay()
    
    # Display startup
    rich_display.display_startup_banner()
    
    # System initialization
    components = [
        "Enhanced RAG Query Engine",
        "French Compliance Framework",
        "Dynamic Domain Discovery",
        "Policy Comparison Agent", 
        "Intelligent Feedback Agent",
        "Communication Protocol",
        "Real Gemini LLM Integration"
    ]
    
    rich_display.display_system_initialization(components)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('grc_analysis.log'), logging.StreamHandler()]
    )
    
    # API key verification
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        rich_display.console.print("❌ [red]CRITICAL: Gemini API key not found![/red]")
        return
    
    # Set up data paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_paths = {
        "company_policies": os.path.join(base_path, "preprocessing", "policies", "satim_chunks_cleaned.json"),
        "reference_policies": os.path.join(base_path, "preprocessing", "norms", "international_norms", "pci_dss_chunks.json")
    }
    
    try:
        analysis_start_time = time.time()
        
        # Initialize systems
        rag_engine = EnhancedRAGQueryEngine(
            llm_config={
                "provider": "gemini",
                "model": "gemini-1.5-flash",  # Using stable model for reliability
                "temperature": 0.2,
                "max_tokens": 3000,
                "api_key": api_key
            },
            data_paths=data_paths
        )
        
        communication_protocol = AgentCommunicationProtocol()
        
        # Initialize main analysis agent
        policy_agent = EnhancedPolicyComparisonAgent(
            name="french_compliance_agent",
            llm_config={"provider": "gemini", "model": "gemini-1.5-flash", "api_key": api_key},
            rag_engine=rag_engine,
            communication_protocol=communication_protocol
        )
        
        # Initialize feedback agent
        feedback_agent = IntelligentPolicyFeedbackAgent(
            name="policy_feedback_agent",
            llm_config={"provider": "gemini", "model": "gemini-1.5-flash", "api_key": api_key},
            rag_engine=rag_engine,
            communication_protocol=communication_protocol
        )
        
        # Test configuration - limit domains for speed
        input_data = {
            "company_policy_ids": ["satim"],
            "reference_policy_ids": ["pci-dss"],
            "domains": ["access_control", "data_protection", "incident_response"]  # 3 domains for comprehensive analysis
        }
        
        rich_display.console.print("🔄 [cyan]Running comprehensive GRC analysis...[/cyan]")
        
        # PHASE 1: Domain Discovery and Analysis
        rich_display.console.print("\n📊 [bold]PHASE 1: Policy Analysis[/bold]")
        results = await policy_agent.process(input_data)
        
        # Display discovery results
        rich_display.display_domain_discovery_results(results.get("discovered_domains", {}))
        rich_display.display_expertise_extraction(results.get("extracted_domain_expertise", {}))
        
        # Display French compliance analysis
        rich_display.display_french_compliance_analysis(results.get("domain_results", {}))
        
        # Display gap analysis
        rich_display.display_gap_analysis(results.get("domain_results", {}))
        
        # Display strategic insights
        rich_display.display_strategic_insights(results.get("domain_results", {}))
        
        # Display overall assessment
        rich_display.display_overall_assessment(
            results.get("overall_score", {}),
            results.get("french_compliance_summary", {})
        )
        
        # PHASE 2: Intelligent LLM Feedback Generation
        rich_display.console.print("\n🤖 [bold]PHASE 2: Intelligent Policy Improvement Feedback[/bold]")
        rich_display.display_llm_feedback_generation()
        
        # Generate intelligent feedback
        feedback_input = {
            "domain_results": results.get("domain_results", {}),
            "overall_score": results.get("overall_score", {}),
            "organization_context": "SATIM"
        }
        
        feedback_results = await feedback_agent.process(feedback_input)
        
        # Display feedback results with rich formatting
        rich_display.console.print("✅ [green]Policy improvement recommendations generated![/green]\n")
        
        # Display improvement recommendations
        await display_improvement_recommendations(rich_display, feedback_results)
        
        # Display executive action plan
        await display_executive_action_plan(rich_display, feedback_results)
        
        # Display implementation roadmap
        await display_implementation_roadmap(rich_display, feedback_results)
        
        # Calculate analysis duration
        analysis_duration = time.time() - analysis_start_time
        
        # Display completion summary
        total_domains = len(results.get("domain_results", {}))
        total_gaps = sum(len(domain_result.get("gaps", [])) for domain_result in results.get("domain_results", {}).values())
        
        rich_display.display_completion_summary(analysis_duration, total_domains, total_gaps)
        
        # Save results to file
        await save_results_to_file(results, feedback_results)
        
        return results, feedback_results
        
    except Exception as e:
        rich_display.console.print(f"❌ [red]Error during enhanced analysis: {e}[/red]")
        import traceback
        traceback.print_exc()

async def display_improvement_recommendations(rich_display: EnhancedRichDisplay, feedback_results: Dict[str, Any]):
    """Display improvement recommendations with rich formatting."""
    
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    
    rich_display.console.print(rich_display.console.rule("[bold green]💡 Policy Improvement Recommendations[/bold green]"))
    
    recommendations = feedback_results.get("improvement_recommendations", {})
    
    for domain, domain_recs in recommendations.items():
        if not domain_recs:
            continue
        
        # Create table for this domain
        rec_table = Table(title=f"{domain.replace('_', ' ').title()} Recommendations", box=box.ROUNDED)
        rec_table.add_column("ID", style="cyan", width=15)
        rec_table.add_column("Priority", justify="center", width=10)
        rec_table.add_column("Target State", style="green", width=40)
        rec_table.add_column("Timeline", justify="center", width=15)
        rec_table.add_column("Expected Impact", style="yellow", width=25)
        
        for rec in domain_recs[:3]:  # Show top 3 recommendations
            priority_emoji = {
                "Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"
            }.get(rec.priority, "⚪")
            
            rec_table.add_row(
                rec.recommendation_id,
                f"{priority_emoji} {rec.priority}",
                rec.target_state[:37] + "..." if len(rec.target_state) > 40 else rec.target_state,
                rec.timeline,
                rec.expected_impact[:22] + "..." if len(rec.expected_impact) > 25 else rec.expected_impact
            )
        
        rich_display.console.print(rec_table)
        rich_display.console.print()

async def display_executive_action_plan(rich_display: EnhancedRichDisplay, feedback_results: Dict[str, Any]):
    """Display executive action plan."""
    
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    
    rich_display.console.print(rich_display.console.rule("[bold purple]🏆 Executive Action Plan[/bold purple]"))
    
    action_plan = feedback_results.get("executive_action_plan", {})
    
    # Executive summary panel
    exec_summary = action_plan.get("executive_summary", "No executive summary available")
    rich_display.console.print(Panel(
        exec_summary,
        title="[bold purple]Executive Summary[/bold purple]",
        border_style="purple"
    ))
    
    # Action plan details table
    plan_table = Table(title="Action Plan Overview", box=box.DOUBLE_EDGE)
    plan_table.add_column("Metric", style="bold cyan", width=30)
    plan_table.add_column("Value", style="green", width=50)
    
    plan_table.add_row("📊 Total Recommendations", str(action_plan.get("total_recommendations", 0)))
    plan_table.add_row("🔴 High Priority Items", str(action_plan.get("high_priority_count", 0)))
    plan_table.add_row("⏱️ Implementation Timeline", action_plan.get("estimated_timeline", "Unknown"))
    plan_table.add_row("💰 Investment Level", action_plan.get("estimated_investment", "Unknown"))
    plan_table.add_row("📈 Expected Improvement", action_plan.get("expected_compliance_improvement", "Unknown"))
    
    rich_display.console.print(plan_table)
    rich_display.console.print()

async def display_implementation_roadmap(rich_display: EnhancedRichDisplay, feedback_results: Dict[str, Any]):
    """Display implementation roadmap."""
    
    from rich.tree import Tree
    from rich.panel import Panel
    
    rich_display.console.print(rich_display.console.rule("[bold blue]🗺️ Implementation Roadmap[/bold blue]"))
    
    roadmap = feedback_results.get("implementation_roadmap", {})
    phases = roadmap.get("roadmap_phases", {})
    
    # Create roadmap tree
    roadmap_tree = Tree("🗺️ [bold]SATIM Policy Implementation Roadmap[/bold]")
    
    phase_config = {
        "critical_immediate": ("🔴 Critical & Immediate (0-3 months)", "red"),
        "high_short_term": ("🟠 High Priority Short-term (3-6 months)", "orange3"),
        "medium_medium_term": ("🟡 Medium Priority Medium-term (6-12 months)", "yellow"),
        "low_long_term": ("🟢 Low Priority Long-term (12+ months)", "green")
    }
    
    for phase_key, (phase_name, style) in phase_config.items():
        phase_recs = phases.get(phase_key, [])
        if phase_recs:
            phase_branch = roadmap_tree.add(f"[{style}]{phase_name}[/{style}] ({len(phase_recs)} items)")
            
            for rec in phase_recs[:3]:  # Show top 3 per phase
                phase_branch.add(f"[dim]{rec.recommendation_id}:[/dim] {rec.target_state[:50]}...")
    
    rich_display.console.print(roadmap_tree)
    
    # Resource requirements
    resources = roadmap.get("resource_requirements", {})
    most_needed = resources.get("most_needed_resources", [])
    
    if most_needed:
        resource_content = "\n".join([
            f"• {resource}: {count} recommendations" 
            for resource, count in most_needed[:5]
        ])
        
        rich_display.console.print(Panel(
            resource_content,
            title="[bold]🔧 Key Resource Requirements[/bold]",
            border_style="blue"
        ))
    
    rich_display.console.print()

async def save_results_to_file(analysis_results: Dict[str, Any], feedback_results: Dict[str, Any]):
    """Save comprehensive results to JSON file."""
    
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    filename = f"satim_grc_analysis_{timestamp}.json"
    
    combined_results = {
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "organization": "SATIM",
        "user": "LyesHADJAR",
        "analysis_results": analysis_results,
        "feedback_results": feedback_results,
        "system_version": "Enhanced GRC v2.0 with Rich Output & LLM Feedback"
    }
    
    try:
        # Convert objects to serializable format
        def make_serializable(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2, default=make_serializable, ensure_ascii=False)
        
        print(f"📄 Results saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

if __name__ == "__main__":
    print("🚀 ENHANCED GRC SYSTEM WITH RICH OUTPUT & LLM FEEDBACK")
    print("=" * 80)
    print("✅ Rich terminal output with tables, panels, and progress bars")
    print("✅ Intelligent policy improvement recommendations") 
    print("✅ Executive action plans and implementation roadmaps")
    print("✅ Real Gemini LLM feedback for SATIM policy enhancement")
    print("✅ French compliance framework integration")
    print("")
    
    asyncio.run(test_enhanced_grc_system())