"""
Rich Terminal Output System for Enhanced GRC Analysis
Current Date: 2025-06-13 15:28:46 UTC
Current User: LyesHADJAR
"""
from typing import Dict, Any, List, Optional
import time
from datetime import datetime, timezone

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.tree import Tree
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.align import Align
    from rich.columns import Columns
    from rich.rule import Rule
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

class EnhancedRichDisplay:
    """Enhanced rich terminal display for GRC analysis."""
    
    def __init__(self):
        if not RICH_AVAILABLE:
            raise ImportError("Rich package is required. Install with: pip install rich")
        
        self.console = Console(width=120)
        self.analysis_start_time = None
    
    def display_startup_banner(self):
        """Display enhanced startup banner."""
        banner_text = """
🚀 ENHANCED GRC ANALYSIS SYSTEM 🚀
══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
🤖 Real Gemini LLM • 📊 Dynamic Domain Discovery • 💡 Intelligent Policy Feedback
══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""
        
        # System info table
        info_table = Table(show_header=False, box=box.ROUNDED, expand=True)
        info_table.add_column("Property", style="bold cyan", width=25)
        info_table.add_column("Value", style="green")
        
        current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        info_table.add_row("📅 Date & Time (UTC)", current_time)
        info_table.add_row("👤 Current User", "LyesHADJAR")
        info_table.add_row("🏢 Organization", "SATIM")
        info_table.add_row("🤖 AI Engine", "Google Gemini Flash 2.0")
        info_table.add_row("🔍 Analysis Mode", "Dynamic Domain Discovery + LLM Feedback")
        
        self.console.print(Panel(banner_text, style="bold blue"))
        self.console.print(Panel(info_table, title="[bold]System Information[/bold]", style="cyan"))
        self.console.print()
    
    def display_system_initialization(self, components: List[str]):
        """Display system initialization with progress."""
        self.console.print(Rule("[bold blue]System Initialization[/bold blue]"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            expand=True
        ) as progress:
            
            main_task = progress.add_task("[cyan]Initializing GRC System...", total=len(components))
            
            for component in components:
                component_task = progress.add_task(f"[green]{component}", total=100)
                
                # Simulate component loading
                for i in range(0, 101, 20):
                    time.sleep(0.1)
                    progress.update(component_task, completed=i)
                
                progress.update(main_task, advance=1)
                self.console.print(f"✅ {component} initialized", style="green")
        
        self.console.print("\n🎉 [bold green]All systems initialized successfully![/bold green]\n")
    
    def display_domain_discovery_results(self, discovered_domains: Dict[str, Any]):
        """Display domain discovery results with rich formatting."""
        self.console.print(Rule("[bold magenta]🔍 Dynamic Domain Discovery Results[/bold magenta]"))
        
        if not discovered_domains:
            self.console.print("[yellow]⚠️ No domains discovered[/yellow]")
            return
        
        # Domain discovery table
        domain_table = Table(title="Discovered Compliance Domains", box=box.ROUNDED)
        domain_table.add_column("Domain", style="cyan", width=25)
        domain_table.add_column("Confidence", justify="center", width=15)
        domain_table.add_column("Evidence", style="dim", width=50)
        domain_table.add_column("Key Topics", style="green", width=25)
        
        for domain, info in discovered_domains.items():
            confidence = info.get('confidence', 'unknown')
            confidence_style = {
                'high': '[green]🟢 High[/green]',
                'medium': '[yellow]🟡 Medium[/yellow]',
                'low': '[red]🔴 Low[/red]'
            }.get(confidence, '[dim]⚪ Unknown[/dim]')
            
            evidence = info.get('evidence', 'No evidence')[:47] + "..." if len(info.get('evidence', '')) > 50 else info.get('evidence', 'No evidence')
            topics = ', '.join(info.get('key_topics', [])[:3])
            
            domain_table.add_row(
                f"[bold]{domain}[/bold]",
                confidence_style,
                evidence,
                topics
            )
        
        self.console.print(domain_table)
        self.console.print(f"\n📊 [bold]Total Domains Discovered:[/bold] {len(discovered_domains)}")
        self.console.print()
    
    def display_expertise_extraction(self, domain_expertise: Dict[str, Any]):
        """Display extracted domain expertise."""
        self.console.print(Rule("[bold blue]🧠 Extracted Domain Expertise[/bold blue]"))
        
        for domain, expertise in domain_expertise.items():
            # Create expertise panel for each domain
            expertise_content = []
            
            key_topics = expertise.get('key_topics', [])
            critical_controls = expertise.get('critical_controls', [])
            frameworks = expertise.get('compliance_frameworks', [])
            risk_factors = expertise.get('risk_factors', [])
            
            expertise_content.append(f"🔑 [bold]Key Topics ({len(key_topics)}):[/bold] {', '.join(key_topics[:6])}")
            expertise_content.append(f"🛡️ [bold]Critical Controls ({len(critical_controls)}):[/bold] {', '.join(critical_controls[:4])}")
            expertise_content.append(f"📋 [bold]Frameworks:[/bold] {', '.join(frameworks)}")
            expertise_content.append(f"⚠️ [bold]Risk Factors:[/bold] {', '.join(risk_factors[:3])}")
            
            expertise_text = "\n".join(expertise_content)
            
            self.console.print(Panel(
                expertise_text,
                title=f"[bold cyan]{domain.upper()}[/bold cyan]",
                border_style="blue"
            ))
        
        self.console.print()
    
    def display_gap_analysis(self, domain_results: Dict[str, Any]):
        """Display comprehensive gap analysis."""
        self.console.print(Rule("[bold yellow]🔍 Gap Analysis & Recommendations[/bold yellow]"))
        
        for domain, result in domain_results.items():
            gaps = result.get('gaps', [])
            
            if not gaps:
                continue
            
            # Gap analysis for this domain
            gap_panel_content = []
            
            for i, gap in enumerate(gaps[:3], 1):  # Show top 3 gaps
                severity = gap.get('severity', 'Medium')
                severity_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(severity, "⚪")
                
                gap_content = f"""
{severity_emoji} [bold]Gap {i}: {gap.get('title', 'Unknown Gap')}[/bold]
[dim]Severity: {severity}[/dim]
[italic]{gap.get('description', 'No description')}[/italic]

💡 [bold green]Recommendation:[/bold green]
{gap.get('recommendation', 'No recommendation provided')}
"""
                gap_panel_content.append(gap_content.strip())
            
            if gap_panel_content:
                self.console.print(Panel(
                    "\n\n".join(gap_panel_content),
                    title=f"[bold red]{domain.replace('_', ' ').title()} - Identified Gaps[/bold red]",
                    border_style="red"
                ))
        
        self.console.print()
    
    def display_strategic_insights(self, domain_results: Dict[str, Any]):
        """Display strategic insights for each domain."""
        self.console.print(Rule("[bold green]💡 Strategic Insights & Strengths[/bold green]"))
        
        # Create columns for better layout
        insights_panels = []
        
        for domain, result in domain_results.items():
            insights = result.get('strategic_insights', {})
            strengths = insights.get('key_strengths', [])
            priorities = insights.get('improvement_priorities', [])
            
            insight_content = []
            
            if strengths:
                insight_content.append("[bold green]💪 Key Strengths:[/bold green]")
                for strength in strengths[:3]:
                    insight_content.append(f"  ✅ {strength}")
            
            if priorities:
                insight_content.append("\n[bold yellow]📈 Improvement Priorities:[/bold yellow]")
                for priority in priorities[:3]:
                    insight_content.append(f"  🎯 {priority}")
            
            if insight_content:
                insights_panels.append(Panel(
                    "\n".join(insight_content),
                    title=f"[bold]{domain.replace('_', ' ').title()}[/bold]",
                    border_style="green"
                ))
        
        if insights_panels:
            self.console.print(Columns(insights_panels, equal=True))
        
        self.console.print()
    
    def display_overall_assessment(self, overall_score: Any, french_summary: Dict[str, Any]):
        """Display overall enterprise assessment."""
        self.console.print(Rule("[bold purple]🏆 Enterprise Assessment Summary[/bold purple]"))
        
        # Overall score panel
        score_value = overall_score.score if hasattr(overall_score, 'score') else 0
        
        score_emoji = "🟢" if score_value >= 75 else "🟡" if score_value >= 55 else "🔴"
        
        overall_content = f"""
{score_emoji} [bold]Enterprise Compliance Score:[/bold] {score_value:.1f}/100

📊 [bold]Compliance Distribution:[/bold]
"""
        
        self.console.print(Panel(
            overall_content,
            title="[bold purple]🏆 Enterprise Assessment[/bold purple]",
            border_style="purple"
        ))
        
        self.console.print()
    
    def display_llm_feedback_generation(self):
        """Display LLM feedback generation progress."""
        self.console.print(Rule("[bold blue]🤖 Generating Intelligent Policy Feedback[/bold blue]"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            feedback_task = progress.add_task("[cyan]Analyzing gaps and generating improvement recommendations...", total=None)
            time.sleep(2)  # Simulate processing
            
        self.console.print("✅ [green]LLM feedback generation completed[/green]\n")
    
    def display_completion_summary(self, analysis_duration: float, total_domains: int, total_gaps: int):
        """Display analysis completion summary."""
        self.console.print(Rule("[bold green]✨ Analysis Complete[/bold green]"))
        
        summary_table = Table(show_header=False, box=box.SIMPLE)
        summary_table.add_column("Metric", style="bold cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("⏱️ Analysis Duration", f"{analysis_duration:.1f} seconds")
        summary_table.add_row("🎯 Domains Analyzed", str(total_domains))
        summary_table.add_row("🔍 Total Gaps Identified", str(total_gaps))
        summary_table.add_row("🤖 AI Engine", "Google Gemini Flash 2.0")
        summary_table.add_row("📊 Analysis Quality", "Real LLM with Feedback")
        
        self.console.print(Panel(
            summary_table,
            title="[bold green]Analysis Summary[/bold green]",
            border_style="green"
        ))
        
        success_message = """
🎉 [bold green]SATIM GRC Analysis Successfully Completed![/bold green]
📋 Comprehensive compliance assessment
💡 Intelligent policy improvement recommendations generated
🚀 Ready for executive review and implementation planning
"""
        
        self.console.print(Panel(success_message, style="green"))