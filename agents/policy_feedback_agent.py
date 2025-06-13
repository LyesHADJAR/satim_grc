"""
Intelligent Policy Feedback Agent with Real LLM Analysis
Current Date: 2025-06-13 15:28:46 UTC
Current User: LyesHADJAR
"""
from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime, timezone

from numpy import number

from .base_agent import EnhancedBaseAgent
from .communication_protocol import AgentCommunicationProtocol

class PolicyImprovementRecommendation:
    """Structured policy improvement recommendation."""
    
    def __init__(self, recommendation_id: str, domain: str, priority: str, 
                 current_state: str, target_state: str, implementation_steps: List[str],
                 timeline: str, resources_needed: List[str], expected_impact: str):
        self.recommendation_id = recommendation_id
        self.domain = domain
        self.priority = priority
        self.current_state = current_state
        self.target_state = target_state
        self.implementation_steps = implementation_steps
        self.timeline = timeline
        self.resources_needed = resources_needed
        self.expected_impact = expected_impact
        self.creation_timestamp = datetime.now(timezone.utc)

class IntelligentPolicyFeedbackAgent(EnhancedBaseAgent):
    """Intelligent agent that provides actionable policy improvement feedback using real LLM analysis."""
    
    def __init__(self, name: str, llm_config: Dict[str, Any], 
                 rag_engine: Any, communication_protocol: AgentCommunicationProtocol):
        super().__init__(name, llm_config)
        self.rag_engine = rag_engine
        self.communication_protocol = communication_protocol
        
        # Register with communication protocol
        communication_protocol.register_agent(name, self)
        
        self.logger.info("Intelligent Policy Feedback Agent initialized")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process gap analysis and generate intelligent policy feedback."""
        
        self.logger.info("Starting intelligent policy feedback generation")
        
        domain_results = input_data.get("domain_results", {})
        overall_assessment = input_data.get("overall_score", {})
        organization_context = input_data.get("organization_context", "SATIM")
        
        # Generate comprehensive policy improvement recommendations
        improvement_recommendations = await self._generate_improvement_recommendations(
            domain_results, overall_assessment, organization_context
        )
        
        # Generate executive action plan
        executive_action_plan = await self._generate_executive_action_plan(
            improvement_recommendations, overall_assessment
        )
        
        # Generate specific policy templates and examples
        policy_templates = await self._generate_policy_templates(
            domain_results, improvement_recommendations
        )
        
        # Generate implementation roadmap
        implementation_roadmap = await self._generate_implementation_roadmap(
            improvement_recommendations, organization_context
        )
        
        return {
            "feedback_approach": "intelligent_llm_powered_policy_improvement",
            "organization": organization_context,
            "improvement_recommendations": improvement_recommendations,
            "executive_action_plan": executive_action_plan,
            "policy_templates": policy_templates,
            "implementation_roadmap": implementation_roadmap,
            "feedback_timestamp": datetime.now(timezone.utc).isoformat(),
            "user": "LyesHADJAR",
            "system_version": "Intelligent Policy Feedback v1.0"
        }
    
    async def _generate_improvement_recommendations(self, domain_results: Dict[str, Any], 
                                                  overall_assessment: Any, 
                                                  organization: str) -> Dict[str, List[PolicyImprovementRecommendation]]:
        """Generate detailed improvement recommendations for each domain."""
        
        recommendations = {}
        
        for domain, result in domain_results.items():
            self.logger.info(f"Generating recommendations for domain: {domain}")
            
            domain_recommendations = await self._analyze_domain_for_improvements(
                domain, result, organization
            )
            recommendations[domain] = domain_recommendations
            
            # Add delay to respect rate limits
            await asyncio.sleep(1)
        
        return recommendations
    
    async def _analyze_domain_for_improvements(self, domain: str, domain_result: Dict[str, Any], 
                                             organization: str) -> List[PolicyImprovementRecommendation]:
        """Analyze a specific domain and generate targeted improvement recommendations."""
        
        gaps = domain_result.get('gaps', [])
        coverage = domain_result.get('coverage', {})
        french_status = domain_result.get('french_compliance_status', {})
        score = domain_result.get('score', {})
        
        # Build comprehensive context for LLM analysis
        analysis_context = f"""
DOMAIN: {domain.upper()}
ORGANIZATION: {organization}

CURRENT STATE ANALYSIS:
- Coverage Percentage: {coverage.get('coverage_percentage', 0)}%
- Maturity Level: {coverage.get('maturity_level', 'Unknown')}
- French Policy Status: {french_status.get('policy_status', {}).get('description', 'Unknown')}
- Implementation Status: {french_status.get('implementation_status', {}).get('description', 'Unknown')}
- Automation Level: {french_status.get('automation_status', {}).get('description', 'Unknown')}
- Reporting Status: {french_status.get('reporting_status', {}).get('description', 'Unknown')}
- Overall Compliance Score: {getattr(score, 'score', 0) if hasattr(score, 'score') else 0}

IDENTIFIED GAPS:
"""
        
        for i, gap in enumerate(gaps, 1):
            analysis_context += f"""
Gap {i}: {gap.get('title', 'Unknown')}
- Severity: {gap.get('severity', 'Unknown')}
- Description: {gap.get('description', 'No description')}
- Current Recommendation: {gap.get('recommendation', 'No recommendation')}
"""
        
        improvement_prompt = f"""
As an expert GRC consultant specializing in {organization} and French compliance frameworks, analyze the current state and provide specific, actionable policy improvement recommendations.

{analysis_context}

ANALYSIS REQUIREMENTS:
Generate 3-5 specific policy improvement recommendations that:

1. Address the identified gaps with concrete solutions
2. Improve French compliance levels (policy, implementation, automation, reporting)
3. Are tailored specifically for {organization}'s context
4. Include practical implementation steps
5. Consider resource requirements and timelines

For each recommendation, provide:
- Recommendation ID (format: REC_{domain}_{number})
- Priority Level (Critical/High/Medium/Low)
- Current State (specific description)
- Target State (specific desired outcome)
- Implementation Steps (5-7 concrete actions)
- Timeline (realistic timeframe)
- Resources Needed (personnel, technology, budget)
- Expected Impact (quantified improvement)

FORMAT AS STRUCTURED OUTPUT:
RECOMMENDATION_1:
ID: REC_{domain}_001
PRIORITY: High
CURRENT_STATE: [specific current state]
TARGET_STATE: [specific target state]
IMPLEMENTATION_STEPS:
1. [step 1]
2. [step 2]
3. [step 3]
4. [step 4]
5. [step 5]
TIMELINE: [timeframe]
RESOURCES: [list of resources]
EXPECTED_IMPACT: [quantified improvement]

Continue for each recommendation...

Focus on practical, implementable solutions that will measurably improve {organization}'s compliance posture.
"""
        
        try:
            llm_response = await self.rag_engine.query_llm(improvement_prompt, max_tokens=4000)
            return self._parse_improvement_recommendations(llm_response, domain)
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations for {domain}: {e}")
            return self._create_fallback_recommendations(domain, gaps)
    
    def _parse_improvement_recommendations(self, llm_response: str, domain: str) -> List[PolicyImprovementRecommendation]:
        """Parse LLM response into structured recommendations."""
        
        recommendations = []
        recommendation_blocks = llm_response.split("RECOMMENDATION_")
        
        for block in recommendation_blocks[1:]:  # Skip first empty block
            try:
                rec = self._parse_single_recommendation(block, domain)
                if rec:
                    recommendations.append(rec)
            except Exception as e:
                self.logger.error(f"Failed to parse recommendation block: {e}")
                continue
        
        # Ensure we have at least some recommendations
        if len(recommendations) < 2:
            recommendations.extend(self._create_fallback_recommendations(domain, []))
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _parse_single_recommendation(self, block: str, domain: str) -> Optional[PolicyImprovementRecommendation]:
        """Parse a single recommendation block."""
        
        import re
        
        # Extract fields using regex
        patterns = {
            'id': r'ID:\s*([^\n]+)',
            'priority': r'PRIORITY:\s*([^\n]+)',
            'current_state': r'CURRENT_STATE:\s*([^\n]+)',
            'target_state': r'TARGET_STATE:\s*([^\n]+)',
            'timeline': r'TIMELINE:\s*([^\n]+)',
            'resources': r'RESOURCES:\s*([^\n]+)',
            'expected_impact': r'EXPECTED_IMPACT:\s*([^\n]+)'
        }
        
        fields = {}
        for field, pattern in patterns.items():
            match = re.search(pattern, block, re.IGNORECASE)
            fields[field] = match.group(1).strip() if match else f"Not specified for {field}"
        
        # Extract implementation steps
        steps_match = re.search(r'IMPLEMENTATION_STEPS:\s*((?:\d+\..*?\n?)+)', block, re.IGNORECASE | re.DOTALL)
        if steps_match:
            steps_text = steps_match.group(1)
            implementation_steps = [
                step.strip() for step in re.split(r'\d+\.', steps_text) 
                if step.strip()
            ]
        else:
            implementation_steps = [
                f"Define {domain} improvement strategy",
                f"Implement {domain} controls",
                f"Test and validate {domain} procedures",
                f"Train staff on {domain} requirements",
                f"Monitor and report {domain} compliance"
            ]
        
        # Parse resources
        resources_text = fields.get('resources', '')
        resources_needed = [r.strip() for r in resources_text.split(',') if r.strip()]
        if not resources_needed:
            resources_needed = ["Policy team", "Technical resources", "Management approval"]
        
        return PolicyImprovementRecommendation(
            recommendation_id=fields.get('id', f"REC_{domain}_{len(block)}"),
            domain=domain,
            priority=fields.get('priority', 'Medium'),
            current_state=fields.get('current_state'),
            target_state=fields.get('target_state'),
            implementation_steps=implementation_steps[:7],  # Limit to 7 steps
            timeline=fields.get('timeline'),
            resources_needed=resources_needed,
            expected_impact=fields.get('expected_impact')
        )
    
    def _create_fallback_recommendations(self, domain: str, gaps: List[Dict[str, Any]]) -> List[PolicyImprovementRecommendation]:
        """Create fallback recommendations when LLM parsing fails."""
        
        fallback_recommendations = [
            PolicyImprovementRecommendation(
                recommendation_id=f"REC_{domain}_FALLBACK_001",
                domain=domain,
                priority="High",
                current_state=f"Current {domain} policies lack comprehensive coverage and implementation details",
                target_state=f"Comprehensive {domain} framework with clear procedures and implementation guidance",
                implementation_steps=[
                    f"Conduct comprehensive {domain} policy review",
                    f"Develop detailed {domain} procedures and controls",
                    f"Implement {domain} monitoring and reporting mechanisms",
                    f"Train staff on new {domain} requirements",
                    f"Establish regular {domain} compliance reviews"
                ],
                timeline="3-6 months",
                resources_needed=["Policy team", "Subject matter experts", "Management approval"],
                expected_impact=f"Improve {domain} compliance score by 20-30 points"
            )
        ]
        
        return fallback_recommendations
    
    async def _generate_executive_action_plan(self, recommendations: Dict[str, List[PolicyImprovementRecommendation]], 
                                            overall_assessment: Any) -> Dict[str, Any]:
        """Generate executive action plan based on recommendations."""
        
        # Collect all high-priority recommendations
        high_priority_recs = []
        total_recs = 0
        
        for domain_recs in recommendations.values():
            total_recs += len(domain_recs)
            high_priority_recs.extend([
                rec for rec in domain_recs 
                if rec.priority.lower() in ['critical', 'high']
            ])
        
        # Generate executive summary using LLM
        executive_prompt = f"""
Generate an executive action plan for SATIM based on the following compliance analysis:

TOTAL RECOMMENDATIONS: {total_recs}
HIGH PRIORITY RECOMMENDATIONS: {len(high_priority_recs)}
CURRENT COMPLIANCE SCORE: {getattr(overall_assessment, 'score', 0) if hasattr(overall_assessment, 'score') else 'Unknown'}

HIGH PRIORITY ITEMS:
"""
        
        for rec in high_priority_recs[:10]:  # Top 10 high priority
            executive_prompt += f"""
- {rec.recommendation_id}: {rec.target_state}
  Timeline: {rec.timeline}
  Expected Impact: {rec.expected_impact}
"""
        
        executive_prompt += """

Create an executive action plan that includes:
1. Executive Summary (2-3 sentences)
2. Strategic Priorities (Top 5 actions)
3. Resource Requirements Summary
4. Timeline Overview
5. Expected ROI and Risk Mitigation
6. Key Success Metrics

Format as professional executive briefing.
"""
        
        try:
            exec_response = await self.rag_engine.query_llm(executive_prompt, max_tokens=2500)
            
            return {
                "executive_summary": exec_response,
                "total_recommendations": total_recs,
                "high_priority_count": len(high_priority_recs),
                "estimated_timeline": "6-12 months for full implementation",
                "estimated_investment": "Medium - primarily internal resources with some external consulting",
                "expected_compliance_improvement": "25-40 point improvement in overall score",
                "key_stakeholders": ["CISO", "Compliance Team", "IT Security", "Legal Department", "Executive Management"]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate executive action plan: {e}")
            return self._create_fallback_action_plan(total_recs, len(high_priority_recs))
    
    def _create_fallback_action_plan(self, total_recs: int, high_priority_count: int) -> Dict[str, Any]:
        """Create fallback executive action plan."""
        
        return {
            "executive_summary": f"SATIM requires comprehensive policy improvements across multiple domains. {high_priority_count} critical items identified out of {total_recs} total recommendations. Immediate action required for high-priority items.",
            "total_recommendations": total_recs,
            "high_priority_count": high_priority_count,
            "estimated_timeline": "6-12 months for full implementation",
            "estimated_investment": "Medium - primarily internal resources",
            "expected_compliance_improvement": "20-35 point improvement in overall score",
            "key_stakeholders": ["CISO", "Compliance Team", "IT Security", "Executive Management"]
        }
    
    async def _generate_policy_templates(self, domain_results: Dict[str, Any], 
                                       recommendations: Dict[str, List[PolicyImprovementRecommendation]]) -> Dict[str, str]:
        """Generate specific policy templates for high-priority improvements."""
        
        templates = {}
        
        # Generate templates for top priority domains
        priority_domains = sorted(
            domain_results.keys(),
            key=lambda d: getattr(domain_results[d].get('score', type('obj', (object,), {'score': 0})()), 'score', 0)
        )[:3]  # Top 3 lowest scoring domains
        
        for domain in priority_domains:
            domain_recs = recommendations.get(domain, [])
            if domain_recs:
                template = await self._generate_domain_policy_template(domain, domain_recs[0])
                templates[domain] = template
                await asyncio.sleep(1)  # Rate limiting
        
        return templates
    
    async def _generate_domain_policy_template(self, domain: str, recommendation: PolicyImprovementRecommendation) -> str:
        """Generate a specific policy template for a domain."""
        
        template_prompt = f"""
Generate a professional policy template for SATIM that addresses the following improvement recommendation:

DOMAIN: {domain}
TARGET STATE: {recommendation.target_state}
IMPLEMENTATION STEPS: {', '.join(recommendation.implementation_steps[:3])}

Create a policy template that includes:
1. Policy Title
2. Purpose and Scope
3. Policy Statement
4. Procedures (specific steps)
5. Roles and Responsibilities
6. Compliance Requirements
7. Review and Update Process

Format as a professional policy document template that SATIM can customize and implement.
Use clear, actionable language suitable for French compliance frameworks.
"""
        
        try:
            template_response = await self.rag_engine.query_llm(template_prompt, max_tokens=3000)
            return template_response
        except Exception as e:
            self.logger.error(f"Failed to generate template for {domain}: {e}")
            return self._create_fallback_template(domain, recommendation)
    
    def _create_fallback_template(self, domain: str, recommendation: PolicyImprovementRecommendation) -> str:
        """Create fallback policy template."""
        
        return f"""
# SATIM {domain.replace('_', ' ').title()} Policy Template

## 1. Purpose and Scope
This policy establishes requirements for {domain.replace('_', ' ')} at SATIM to ensure compliance with applicable regulations and standards.

## 2. Policy Statement
SATIM is committed to implementing comprehensive {domain.replace('_', ' ')} controls to achieve: {recommendation.target_state}

## 3. Procedures
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(recommendation.implementation_steps)])}

## 4. Roles and Responsibilities
- Policy Owner: [To be defined]
- Implementation Team: {', '.join(recommendation.resources_needed)}
- Review Authority: Compliance Team

## 5. Compliance Requirements
This policy supports compliance with French regulatory frameworks and industry standards.

## 6. Review Process
This policy will be reviewed annually or as required by regulatory changes.
"""
    
    async def _generate_implementation_roadmap(self, recommendations: Dict[str, List[PolicyImprovementRecommendation]], 
                                             organization: str) -> Dict[str, Any]:
        """Generate comprehensive implementation roadmap."""
        
        # Organize recommendations by priority and timeline
        roadmap_data = {
            "critical_immediate": [],    # 0-3 months
            "high_short_term": [],       # 3-6 months  
            "medium_medium_term": [],    # 6-12 months
            "low_long_term": []          # 12+ months
        }
        
        for domain_recs in recommendations.values():
            for rec in domain_recs:
                timeline = rec.timeline.lower()
                priority = rec.priority.lower()
                
                if priority == 'critical' or ('immediate' in timeline or 'urgent' in timeline):
                    roadmap_data["critical_immediate"].append(rec)
                elif priority == 'high' or ('3' in timeline and 'month' in timeline):
                    roadmap_data["high_short_term"].append(rec)
                elif priority == 'medium' or ('6' in timeline and 'month' in timeline):
                    roadmap_data["medium_medium_term"].append(rec)
                else:
                    roadmap_data["low_long_term"].append(rec)
        
        return {
            "roadmap_phases": roadmap_data,
            "total_timeline": "12-18 months for complete implementation",
            "resource_requirements": self._calculate_resource_requirements(recommendations),
            "success_metrics": self._define_success_metrics(recommendations),
            "risk_mitigation": "Phased approach reduces implementation risk and allows for course correction"
        }
    
    def _calculate_resource_requirements(self, recommendations: Dict[str, List[PolicyImprovementRecommendation]]) -> Dict[str, Any]:
        """Calculate overall resource requirements."""
        
        all_resources = []
        for domain_recs in recommendations.values():
            for rec in domain_recs:
                all_resources.extend(rec.resources_needed)
        
        # Count resource frequency
        resource_counts = {}
        for resource in all_resources:
            resource_counts[resource] = resource_counts.get(resource, 0) + 1
        
        return {
            "most_needed_resources": sorted(resource_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "estimated_team_size": "8-12 people across all domains",
            "external_support_needed": "Legal/compliance consulting, technical implementation support",
            "budget_estimate": "Medium investment - primarily internal effort with selective external expertise"
        }
    
    def _define_success_metrics(self, recommendations: Dict[str, List[PolicyImprovementRecommendation]]) -> List[str]:
        """Define success metrics for implementation."""
        
        return [
            "Overall compliance score improvement of 25+ points",
            "Achievement of Level 3+ French compliance across all domains",
            "100% of critical recommendations implemented within 6 months",
            "All policy templates developed and approved",
            "Staff training completion rate >95%",
            "Successful audit findings reduction by 50%",
            "Executive stakeholder satisfaction score >4.0/5.0"
        ]