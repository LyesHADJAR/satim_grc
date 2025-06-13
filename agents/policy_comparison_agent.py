"""
Enhanced Policy Comparison Agent - French Compliance Framework Integration - URGENT
Current Date: 2025-06-13 15:06:52 UTC
User: LyesHADJAR
"""
from typing import Dict, Any, List, Optional, Set
import asyncio
import logging
from datetime import datetime, timezone
import json
import re

from .base_agent import EnhancedBaseAgent
from .communication_protocol import AgentCommunicationProtocol

class FrenchComplianceFramework:
    """French compliance framework with specific status definitions."""
    
    # État de la politique (Policy Status)
    POLICY_STATUS = {
        0: "Aucune politique",  # No policy
        1: "Politique non formalisée",  # Non-formalized policy
        2: "Politique partiellement formalisée",  # Partially formalized policy
        3: "Politique formalisée",  # Formalized policy
        4: "Politique formalisée et approuvée"  # Formalized and approved policy
    }
    
    # Statut de l'implémentation (Implementation Status)
    IMPLEMENTATION_STATUS = {
        0: "Non implémenté",  # Not implemented
        1: "Implémentation partielle de la politique",  # Partial policy implementation
        2: "Implémenté sur certains systèmes",  # Implemented on some systems
        3: "Implémenté sur la plupart des systèmes",  # Implemented on most systems
        4: "Implémenté sur tous les systèmes"  # Implemented on all systems
    }
    
    # État de l'automatisation (Automation Status)
    AUTOMATION_STATUS = {
        0: "Non automatisé",  # Not automated
        1: "Automatisation partielle de la politique",  # Partial policy automation
        2: "Automatisé sur certains systèmes",  # Automated on some systems
        3: "Automatisé sur la plupart des systèmes",  # Automated on most systems
        4: "Automatisé sur tous les systèmes"  # Automated on all systems
    }
    
    # État des rapports (Reporting Status)
    REPORTING_STATUS = {
        0: "Non rapporté",  # Not reported
        1: "Rapport partiel de la politique",  # Partial policy reporting
        2: "Rapporté sur certains systèmes",  # Reported on some systems
        3: "Rapporté sur la plupart des systèmes",  # Reported on most systems
        4: "Rapporté sur tous systèmes"  # Reported on all systems
    }
    
    # Niveau de conformité du contrôle (Control Compliance Level)
    COMPLIANCE_LEVELS = {
        0: {"level": "Inexistant", "meaning": "La mesure n'existe pas"},
        1: {"level": "Non formalisé", "meaning": "La politique non formalisé"},
        2: {"level": "Formalisé", "meaning": "Existence d'une politique formalisé et approuvé"},
        3: {"level": "Formalisé et implémenté", "meaning": "Le contrôle est implémenté sur les systèmes"},
        4: {"level": "Formalisé, Implémenté et Automatisé", "meaning": "Le contrôle est implémenté et automatisé sur les systèmes"},
        5: {"level": "Formalisé, Implémenté, Automatisé et Rapporté", "meaning": "le contrôle est implémenté, automatisé et rapporter sur les systèmes"}
    }

class FrenchComplianceScore:
    """French compliance scoring with detailed status breakdown."""
    
    def __init__(self, score: float, french_assessment: Dict[str, Any], criteria: List[Dict[str, Any]] = None):
        self.score = score
        self.french_assessment = french_assessment
        self.criteria = criteria or []
        self.timestamp = datetime.now(timezone.utc)
        self.recommendations = []
        self.recommendations_french = []

class EnhancedPolicyComparisonAgent(EnhancedBaseAgent):
    """
    Enhanced policy comparison agent with French compliance framework and dynamic domain discovery.
    """
    
    def __init__(self, name: str, llm_config: Dict[str, Any], 
                 rag_engine: Any, communication_protocol: AgentCommunicationProtocol):
        super().__init__(name, llm_config)
        self.rag_engine = rag_engine
        self.communication_protocol = communication_protocol
        
        # French compliance framework
        self.french_framework = FrenchComplianceFramework()
        
        # Dynamic containers - populated from documents
        self.discovered_domains = {}
        self.domain_expertise = {}
        self.document_analysis_cache = {}
        
        # Register with communication protocol
        communication_protocol.register_agent(name, self)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process enhanced policy comparison analysis with French compliance framework."""
        
        self.logger.info(f"Starting enhanced LLM-powered policy analysis with French compliance framework")
        
        # Get analysis parameters
        company_policy_ids = input_data.get("company_policy_ids", [])
        reference_policy_ids = input_data.get("reference_policy_ids", [])
        
        # STEP 1: Discover domains dynamically from documents
        self.logger.info("🔍 STEP 1: Discovering domains from documents...")
        try:
            discovered_domains = await self._discover_domains_from_documents(company_policy_ids, reference_policy_ids)
            self.discovered_domains = discovered_domains
        except Exception as e:
            self.logger.error(f"Domain discovery failed: {e}")
            discovered_domains = self._get_fallback_domains()
            self.discovered_domains = discovered_domains
        
        # Use provided domains or discovered ones
        provided_domains = input_data.get("domains", [])
        if provided_domains:
            domains = provided_domains
            self.logger.info(f"Using provided domains: {domains}")
        else:
            domains = list(discovered_domains.keys())[:5]  # Limit to 5 domains to avoid rate limits
            self.logger.info(f"Using discovered domains (limited to 5): {domains}")
        
        # STEP 2: Extract domain expertise dynamically
        self.logger.info("🔍 STEP 2: Extracting domain expertise from documents...")
        for domain in domains:
            if domain not in self.domain_expertise:
                try:
                    self.domain_expertise[domain] = await self._extract_dynamic_domain_expertise(
                        domain, reference_policy_ids
                    )
                    # Add delay to avoid rate limits
                    await asyncio.sleep(2)
                except Exception as e:
                    self.logger.error(f"Failed to extract expertise for {domain}: {e}")
                    self.domain_expertise[domain] = self._get_fallback_expertise(domain)
        
        # STEP 3: Perform enhanced analysis for each domain with French framework
        self.logger.info("🔍 STEP 3: Analyzing domains with French compliance framework...")
        domain_results = {}
        for domain in domains:
            self.logger.info(f"Analyzing domain: {domain}")
            try:
                domain_result = await self._analyze_domain_with_french_framework(
                    domain, company_policy_ids, reference_policy_ids
                )
                domain_results[domain] = domain_result
                # Add delay to avoid rate limits
                await asyncio.sleep(3)
            except Exception as e:
                self.logger.error(f"Failed to analyze domain {domain}: {e}")
                # Create fallback result
                domain_results[domain] = self._create_fallback_domain_result_french(domain)
        
        # Generate overall enterprise assessment with French framework
        try:
            overall_score = await self._generate_overall_french_assessment(domain_results)
        except Exception as e:
            self.logger.error(f"Failed to generate overall assessment: {e}")
            overall_score = self._create_fallback_overall_score_french(domain_results)
        
        return {
            "analysis_approach": "enhanced_llm_collaborative_analysis_with_french_compliance_framework",
            "analysis_quality": "comprehensive_real_llm_with_french_expertise", 
            "compliance_framework": "French GRC Framework",
            "discovered_domains": discovered_domains,
            "domains_analyzed": domains,
            "extracted_domain_expertise": self.domain_expertise,
            "domain_results": domain_results,
            "overall_score": overall_score,
            "french_compliance_summary": self._generate_french_compliance_summary(domain_results),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "user": "LyesHADJAR",
            "system_version": "Enhanced Multi-Agent GRC v2.0 - French Compliance Framework"
        }
    
    async def _analyze_domain_with_french_framework(self, domain: str, company_policies: List[str], 
                                                  reference_policies: List[str]) -> Dict[str, Any]:
        """Perform enhanced domain analysis with French compliance framework."""
        
        # Get comprehensive content for the domain using extracted expertise
        domain_content = await self._get_comprehensive_domain_content(domain, company_policies, reference_policies)
        
        # Perform LLM-powered analysis with French compliance framework
        analysis_results = await self._perform_llm_french_analysis(domain, domain_content)
        
        # Extract structured results with French framework
        coverage_analysis = self._extract_coverage_analysis(analysis_results, domain)
        gap_analysis = self._extract_gap_analysis(analysis_results, domain)
        alignment_analysis = self._extract_alignment_analysis(analysis_results, domain)
        
        # NEW: Extract French compliance status
        french_compliance_status = await self._extract_french_compliance_status(analysis_results, domain)
        
        # Calculate quantitative scores
        quantitative_scores = self._calculate_quantitative_scores(coverage_analysis, gap_analysis, alignment_analysis)
        
        # Generate strategic insights
        strategic_insights = await self._generate_strategic_insights(domain, analysis_results)
        
        # Calculate domain compliance score with French framework
        domain_score = self._calculate_french_domain_score(quantitative_scores, french_compliance_status)
        
        return {
            "domain": domain,
            "coverage": coverage_analysis,
            "gaps": gap_analysis,
            "alignment": alignment_analysis,
            "quantitative_scores": quantitative_scores,
            "strategic_insights": strategic_insights,
            "french_compliance_status": french_compliance_status,
            "score": domain_score,
            "evidence_based": True,
            "extracted_expertise_used": self.domain_expertise.get(domain, {}),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _perform_llm_french_analysis(self, domain: str, domain_content: Dict[str, Any]) -> str:
        """Perform comprehensive LLM analysis with French compliance framework integration."""
        
        company_content = domain_content.get("company_content", [])
        reference_content = domain_content.get("reference_content", [])
        domain_expertise = domain_content.get("domain_expertise_applied", {})
        
        # Build comprehensive context
        company_context = "\n".join([
            f"Section {i+1}: {item.get('section', 'Unknown')}\nContent: {item.get('content', '')[:300]}..."
            for i, item in enumerate(company_content[:10])
        ])
        
        reference_context = "\n".join([
            f"Requirement {i+1}: {item.get('section', 'Unknown')}\nContent: {item.get('content', '')[:300]}..."
            for i, item in enumerate(reference_content[:10])
        ])
        
        key_topics = domain_expertise.get("key_topics", [])
        critical_controls = domain_expertise.get("critical_controls", [])
        compliance_frameworks = domain_expertise.get("compliance_frameworks", [])
        
        analysis_prompt = f"""
Perform a comprehensive compliance analysis for the {domain.upper()} domain using the French GRC compliance framework.

EXTRACTED DOMAIN EXPERTISE:
Key Topics: {', '.join(key_topics[:8])}
Critical Controls: {', '.join(critical_controls[:6])}
Compliance Frameworks: {', '.join(compliance_frameworks)}

COMPANY POLICY CONTENT:
{company_context}

REFERENCE STANDARD REQUIREMENTS:
{reference_context}

FRENCH COMPLIANCE FRAMEWORK ANALYSIS:
For each control/topic identified, assess according to the French compliance levels:

1. POLICY STATUS (État de la politique):
   - Aucune politique (0): No policy exists
   - Politique non formalisée (1): Non-formalized policy
   - Politique partiellement formalisée (2): Partially formalized policy
   - Politique formalisée (3): Formalized policy
   - Politique formalisée et approuvée (4): Formalized and approved policy

2. IMPLEMENTATION STATUS (Statut de l'implémentation):
   - Non implémenté (0): Not implemented
   - Implémentation partielle (1): Partial implementation
   - Implémenté sur certains systèmes (2): Implemented on some systems
   - Implémenté sur la plupart des systèmes (3): Implemented on most systems
   - Implémenté sur tous les systèmes (4): Implemented on all systems

3. AUTOMATION STATUS (État de l'automatisation):
   - Non automatisé (0): Not automated
   - Automatisation partielle (1): Partial automation
   - Automatisé sur certains systèmes (2): Automated on some systems
   - Automatisé sur la plupart des systèmes (3): Automated on most systems
   - Automatisé sur tous les systèmes (4): Automated on all systems

4. REPORTING STATUS (État des rapports):
   - Non rapporté (0): Not reported
   - Rapport partiel (1): Partial reporting
   - Rapporté sur certains systèmes (2): Reported on some systems
   - Rapporté sur la plupart des systèmes (3): Reported on most systems
   - Rapporté sur tous systèmes (4): Reported on all systems

ANALYSIS REQUIREMENTS:
1. For each key topic, provide French compliance assessment
2. Calculate overall compliance level (0-5) based on the French framework
3. Identify specific gaps according to French standards
4. Provide recommendations in French compliance context

FORMAT YOUR RESPONSE WITH:
- Coverage percentage and French compliance level assessment
- Specific policy status, implementation status, automation status, and reporting status
- Gap analysis with French compliance context
- Strategic recommendations for improving French compliance levels

Be specific about which French compliance level (0-5) applies to each domain area.
"""
        
        # Query LLM with French framework prompt
        analysis_response = await self.rag_engine.query_llm(analysis_prompt, max_tokens=4000)
        
        return analysis_response
    
    async def _extract_french_compliance_status(self, llm_analysis: str, domain: str) -> Dict[str, Any]:
        """Extract French compliance status from LLM analysis."""
        
        french_status = {
            "policy_status": {"level": 2, "description": "Politique partiellement formalisée"},
            "implementation_status": {"level": 2, "description": "Implémenté sur certains systèmes"},
            "automation_status": {"level": 1, "description": "Automatisation partielle de la politique"},
            "reporting_status": {"level": 1, "description": "Rapport partiel de la politique"},
            "overall_compliance_level": 2,
            "compliance_meaning": "Formalisé"
        }
        
        # Extract policy status
        policy_patterns = [
            (r'politique.*?formalisée.*?approuvée', 4),
            (r'politique.*?formalisée', 3),
            (r'politique.*?partiellement.*?formalisée', 2),
            (r'politique.*?non.*?formalisée', 1),
            (r'aucune.*?politique', 0)
        ]
        
        for pattern, level in policy_patterns:
            if re.search(pattern, llm_analysis, re.IGNORECASE):
                french_status["policy_status"] = {
                    "level": level,
                    "description": self.french_framework.POLICY_STATUS[level]
                }
                break
        
        # Extract implementation status
        impl_patterns = [
            (r'implémenté.*?tous.*?systèmes', 4),
            (r'implémenté.*?plupart.*?systèmes', 3),
            (r'implémenté.*?certains.*?systèmes', 2),
            (r'implémentation.*?partielle', 1),
            (r'non.*?implémenté', 0)
        ]
        
        for pattern, level in impl_patterns:
            if re.search(pattern, llm_analysis, re.IGNORECASE):
                french_status["implementation_status"] = {
                    "level": level,
                    "description": self.french_framework.IMPLEMENTATION_STATUS[level]
                }
                break
        
        # Extract automation status
        auto_patterns = [
            (r'automatisé.*?tous.*?systèmes', 4),
            (r'automatisé.*?plupart.*?systèmes', 3),
            (r'automatisé.*?certains.*?systèmes', 2),
            (r'automatisation.*?partielle', 1),
            (r'non.*?automatisé', 0)
        ]
        
        for pattern, level in auto_patterns:
            if re.search(pattern, llm_analysis, re.IGNORECASE):
                french_status["automation_status"] = {
                    "level": level,
                    "description": self.french_framework.AUTOMATION_STATUS[level]
                }
                break
        
        # Extract reporting status
        report_patterns = [
            (r'rapporté.*?tous.*?systèmes', 4),
            (r'rapporté.*?plupart.*?systèmes', 3),
            (r'rapporté.*?certains.*?systèmes', 2),
            (r'rapport.*?partiel', 1),
            (r'non.*?rapporté', 0)
        ]
        
        for pattern, level in report_patterns:
            if re.search(pattern, llm_analysis, re.IGNORECASE):
                french_status["reporting_status"] = {
                    "level": level,
                    "description": self.french_framework.REPORTING_STATUS[level]
                }
                break
        
        # Calculate overall compliance level
        levels = [
            french_status["policy_status"]["level"],
            french_status["implementation_status"]["level"],
            french_status["automation_status"]["level"],
            french_status["reporting_status"]["level"]
        ]
        
        # Overall compliance level calculation
        policy_level = french_status["policy_status"]["level"]
        impl_level = french_status["implementation_status"]["level"]
        auto_level = french_status["automation_status"]["level"]
        report_level = french_status["reporting_status"]["level"]
        
        # French compliance logic
        if policy_level >= 4 and impl_level >= 4 and auto_level >= 4 and report_level >= 4:
            overall_level = 5  # Formalisé, Implémenté, Automatisé et Rapporté
        elif policy_level >= 3 and impl_level >= 3 and auto_level >= 3:
            overall_level = 4  # Formalisé, Implémenté et Automatisé
        elif policy_level >= 3 and impl_level >= 2:
            overall_level = 3  # Formalisé et implémenté
        elif policy_level >= 3:
            overall_level = 2  # Formalisé
        elif policy_level >= 1:
            overall_level = 1  # Non formalisé
        else:
            overall_level = 0  # Inexistant
        
        french_status["overall_compliance_level"] = overall_level
        french_status["compliance_meaning"] = self.french_framework.COMPLIANCE_LEVELS[overall_level]["meaning"]
        french_status["compliance_level_description"] = self.french_framework.COMPLIANCE_LEVELS[overall_level]["level"]
        
        return french_status
    
    def _calculate_french_domain_score(self, quantitative_scores: Dict[str, Any], 
                                     french_status: Dict[str, Any]) -> FrenchComplianceScore:
        """Calculate domain compliance score with French framework integration."""
        
        # Base score calculation
        weights = {
            "coverage_score": 0.25,
            "quality_score": 0.20,
            "alignment_score": 0.20,
            "implementation_score": 0.15,
            "french_compliance": 0.20  # New French compliance weight
        }
        
        # Calculate French compliance score (0-100 based on 0-5 scale)
        french_compliance_score = (french_status["overall_compliance_level"] / 5.0) * 100
        
        # Combined quantitative scores
        base_score = sum(quantitative_scores[score] * weights[score] 
                        for score in quantitative_scores if score in weights)
        
        # Final weighted score including French compliance
        final_score = base_score + (french_compliance_score * weights["french_compliance"])
        
        # Create criteria breakdown with French framework
        criteria = []
        for score_name, weight in weights.items():
            if score_name == "french_compliance":
                score_value = french_compliance_score
                criteria.append({
                    "name": "French Compliance Level",
                    "score": score_value,
                    "weight": weight,
                    "status": french_status["compliance_level_description"],
                    "french_level": french_status["overall_compliance_level"]
                })
            elif score_name in quantitative_scores:
                score_value = quantitative_scores[score_name]
                criteria.append({
                    "name": score_name.replace("_", " ").title(),
                    "score": score_value,
                    "weight": weight,
                    "status": "Good" if score_value >= 70 else "Needs Improvement" if score_value >= 50 else "Poor"
                })
        
        return FrenchComplianceScore(final_score, french_status, criteria)
    
    def _generate_french_compliance_summary(self, domain_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate French compliance summary across all domains."""
        
        summary = {
            "overall_french_compliance": {
                "average_policy_level": 0.0,
                "average_implementation_level": 0.0,
                "average_automation_level": 0.0,
                "average_reporting_level": 0.0,
                "overall_compliance_level": 0.0
            },
            "domains_by_compliance_level": {
                0: [], 1: [], 2: [], 3: [], 4: [], 5: []
            },
            "compliance_distribution": {},
            "priority_improvements": []
        }
        
        if not domain_results:
            return summary
        
        # Calculate averages
        total_domains = len(domain_results)
        policy_levels = []
        impl_levels = []
        auto_levels = []
        report_levels = []
        overall_levels = []
        
        for domain, result in domain_results.items():
            french_status = result.get("french_compliance_status", {})
            
            policy_level = french_status.get("policy_status", {}).get("level", 0)
            impl_level = french_status.get("implementation_status", {}).get("level", 0)
            auto_level = french_status.get("automation_status", {}).get("level", 0)
            report_level = french_status.get("reporting_status", {}).get("level", 0)
            overall_level = french_status.get("overall_compliance_level", 0)
            
            policy_levels.append(policy_level)
            impl_levels.append(impl_level)
            auto_levels.append(auto_level)
            report_levels.append(report_level)
            overall_levels.append(overall_level)
            
            # Group by compliance level
            summary["domains_by_compliance_level"][overall_level].append(domain)
        
        # Calculate averages
        summary["overall_french_compliance"] = {
            "average_policy_level": sum(policy_levels) / total_domains,
            "average_implementation_level": sum(impl_levels) / total_domains,
            "average_automation_level": sum(auto_levels) / total_domains,
            "average_reporting_level": sum(report_levels) / total_domains,
            "overall_compliance_level": sum(overall_levels) / total_domains
        }
        
        # Compliance distribution
        for level in range(6):
            count = len(summary["domains_by_compliance_level"][level])
            percentage = (count / total_domains) * 100 if total_domains > 0 else 0
            summary["compliance_distribution"][level] = {
                "count": count,
                "percentage": percentage,
                "description": self.french_framework.COMPLIANCE_LEVELS[level]["level"]
            }
        
        # Priority improvements
        low_compliance_domains = []
        for level in range(3):  # Levels 0, 1, 2 need improvement
            low_compliance_domains.extend(summary["domains_by_compliance_level"][level])
        
        summary["priority_improvements"] = [
            f"Améliorer le niveau de conformité pour {domain}" 
            for domain in low_compliance_domains[:5]
        ]
        
        return summary
    
    async def _generate_overall_french_assessment(self, domain_results: Dict[str, Any]) -> FrenchComplianceScore:
        """Generate overall enterprise assessment with French compliance framework."""
        
        # Calculate average scores
        domain_scores = [result["score"].score for result in domain_results.values()]
        overall_score = sum(domain_scores) / len(domain_scores) if domain_scores else 50.0
        
        # Calculate overall French compliance status
        french_statuses = []
        for result in domain_results.values():
            french_status = result.get("french_compliance_status", {})
            french_statuses.append(french_status)
        
        # Average French compliance levels
        overall_french_status = self._calculate_overall_french_status(french_statuses)
        
        # Generate enterprise recommendations with French context
        enterprise_prompt = f"""
Based on comprehensive domain analysis using French compliance framework, provide enterprise strategic recommendations:

Domain Scores: {', '.join([f"{domain}: {result['score'].score:.1f}" for domain, result in domain_results.items()])}
Overall Score: {overall_score:.1f}
French Compliance Level: {overall_french_status['overall_compliance_level']}

Provide 8-10 strategic enterprise recommendations focusing on:
1. French compliance framework improvement priorities
2. Policy formalization and approval processes
3. Implementation roadmap for system deployment
4. Automation strategy for compliance controls
5. Reporting and monitoring enhancement
6. Resource allocation for compliance maturity

Format recommendations in both French and English context.
"""
        
        try:
            recommendations_response = await self.rag_engine.query_llm(enterprise_prompt, max_tokens=2000)
            recommendations = self._extract_recommendations(recommendations_response)
            recommendations_french = self._extract_french_recommendations(recommendations_response)
        except Exception as e:
            self.logger.error(f"Failed to generate enterprise recommendations: {e}")
            recommendations = self._get_default_recommendations()
            recommendations_french = self._get_default_french_recommendations()
        
        # Create overall criteria with French framework
        overall_criteria = [
            {
                "name": "Overall Compliance Maturity",
                "score": overall_score,
                "weight": 0.8,
                "status": "Advanced" if overall_score >= 80 else "Developing" if overall_score >= 60 else "Initial"
            },
            {
                "name": "French Compliance Level",
                "score": (overall_french_status['overall_compliance_level'] / 5.0) * 100,
                "weight": 0.2,
                "status": overall_french_status['compliance_level_description'],
                "french_level": overall_french_status['overall_compliance_level']
            }
        ]
        
        # Create compliance score with French framework
        compliance_score = FrenchComplianceScore(overall_score, overall_french_status, overall_criteria)
        compliance_score.recommendations = recommendations
        compliance_score.recommendations_french = recommendations_french
        
        return compliance_score
    
    def _calculate_overall_french_status(self, french_statuses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall French compliance status from all domains."""
        
        if not french_statuses:
            return {
                "policy_status": {"level": 1, "description": "Politique non formalisée"},
                "implementation_status": {"level": 1, "description": "Implémentation partielle de la politique"},
                "automation_status": {"level": 0, "description": "Non automatisé"},
                "reporting_status": {"level": 0, "description": "Non rapporté"},
                "overall_compliance_level": 1,
                "compliance_meaning": "Non formalisé",
                "compliance_level_description": "Non formalisé"
            }
        
        # Calculate averages
        policy_levels = [status.get("policy_status", {}).get("level", 0) for status in french_statuses]
        impl_levels = [status.get("implementation_status", {}).get("level", 0) for status in french_statuses]
        auto_levels = [status.get("automation_status", {}).get("level", 0) for status in french_statuses]
        report_levels = [status.get("reporting_status", {}).get("level", 0) for status in french_statuses]
        
        avg_policy = sum(policy_levels) / len(policy_levels)
        avg_impl = sum(impl_levels) / len(impl_levels)
        avg_auto = sum(auto_levels) / len(auto_levels)
        avg_report = sum(report_levels) / len(report_levels)
        
        # Determine overall level based on averages
        if avg_policy >= 3.5 and avg_impl >= 3.5 and avg_auto >= 3.5 and avg_report >= 3.5:
            overall_level = 5
        elif avg_policy >= 3 and avg_impl >= 3 and avg_auto >= 2.5:
            overall_level = 4
        elif avg_policy >= 3 and avg_impl >= 2:
            overall_level = 3
        elif avg_policy >= 2.5:
            overall_level = 2
        elif avg_policy >= 1:
            overall_level = 1
        else:
            overall_level = 0
        
        return {
            "policy_status": {
                "level": round(avg_policy),
                "description": self.french_framework.POLICY_STATUS[round(avg_policy)]
            },
            "implementation_status": {
                "level": round(avg_impl),
                "description": self.french_framework.IMPLEMENTATION_STATUS[round(avg_impl)]
            },
            "automation_status": {
                "level": round(avg_auto),
                "description": self.french_framework.AUTOMATION_STATUS[round(avg_auto)]
            },
            "reporting_status": {
                "level": round(avg_report),
                "description": self.french_framework.REPORTING_STATUS[round(avg_report)]
            },
            "overall_compliance_level": overall_level,
            "compliance_meaning": self.french_framework.COMPLIANCE_LEVELS[overall_level]["meaning"],
            "compliance_level_description": self.french_framework.COMPLIANCE_LEVELS[overall_level]["level"]
        }
    
    def _extract_french_recommendations(self, response: str) -> List[str]:
        """Extract French-specific recommendations."""
        
        french_patterns = [
            r'améliorer[^.]+',
            r'mettre en place[^.]+',
            r'renforcer[^.]+',
            r'développer[^.]+',
            r'établir[^.]+',
            r'implémenter[^.]+',
            r'formaliser[^.]+',
            r'automatiser[^.]+',
            r'rapporter[^.]+',
            r'conformité[^.]+',
            r'niveau[^.]+conformité[^.]+'
        ]
        
        french_recommendations = []
        for pattern in french_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            french_recommendations.extend(matches[:2])
        
        # Default French recommendations if extraction fails
        if len(french_recommendations) < 5:
            french_recommendations.extend(self._get_default_french_recommendations())
        
        return french_recommendations[:8]
    
    def _get_default_french_recommendations(self) -> List[str]:
        """Get default French recommendations."""
        return [
            "Formaliser et approuver toutes les politiques de sécurité selon le cadre français",
            "Améliorer le niveau d'implémentation sur tous les systèmes",
            "Développer l'automatisation des contrôles de conformité",
            "Mettre en place un système de rapportage complet",
            "Renforcer la gouvernance des politiques de sécurité",
            "Établir des processus d'audit et de surveillance continus",
            "Améliorer la formation et la sensibilisation du personnel",
            "Développer un tableau de bord de conformité exécutif"
        ]
    
    # ============================================================================
    # ALL PREVIOUS METHODS (Discovery, Expertise, Analysis) - UNCHANGED
    # ============================================================================
    
    async def _discover_domains_from_documents(self, company_policy_ids: List[str], 
                                             reference_policy_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Discover compliance domains by analyzing all available documents."""
        
        self.logger.info("Analyzing documents to discover compliance domains...")
        
        # Get overview of all documents
        all_content = []
        
        # Sample content from company policies
        for policy_id in company_policy_ids:
            results = await self.rag_engine.semantic_search(f"{policy_id}", top_k=15)
            all_content.extend(results)
        
        # Sample content from reference policies  
        for policy_id in reference_policy_ids:
            results = await self.rag_engine.semantic_search(f"{policy_id}", top_k=15)
            all_content.extend(results)
        
        # Additional broad searches to capture domain-related content
        domain_discovery_searches = [
            "security policy", "access control", "incident response", "data protection",
            "compliance requirements", "risk management", "audit", "governance",
            "privacy", "encryption", "authentication", "monitoring", "business continuity",
            "vendor management", "change management", "information security"
        ]
        
        for search_term in domain_discovery_searches:
            results = await self.rag_engine.semantic_search(search_term, top_k=8)
            all_content.extend(results)
        
        # Remove duplicates
        unique_content = self._deduplicate_content(all_content)
        
        # Build comprehensive context for domain discovery
        content_context = "\n".join([
            f"Section: {item.get('section', 'Unknown')}\nContent: {item.get('content', '')[:200]}..."
            for item in unique_content[:30]  # Top 30 most relevant sections
        ])
        
        # Use LLM to discover domains
        domain_discovery_prompt = f"""
Analyze the following policy and compliance content to discover the main compliance domains covered:

DOCUMENT CONTENT:
{content_context}

ANALYSIS TASK:
Identify and extract the main compliance/security domains that are covered in this content.

For each domain you identify, provide:
1. Domain name (use standard terminology like: access_control, incident_response, data_protection, etc.)
2. Evidence of coverage (brief description of what content exists for this domain)
3. Confidence level (high/medium/low based on amount of content found)
4. Key topics detected for this domain

FORMAT YOUR RESPONSE AS:
DOMAIN: domain_name
EVIDENCE: brief evidence description
CONFIDENCE: high/medium/low
KEY_TOPICS: topic1, topic2, topic3

DOMAIN: another_domain_name
EVIDENCE: brief evidence description
CONFIDENCE: high/medium/low
KEY_TOPICS: topic1, topic2, topic3

Focus on major compliance domains like access control, incident response, data protection, risk management, business continuity, vendor management, change management, etc.
Only include domains where you find substantial evidence in the content.
"""
        
        try:
            discovery_response = await self.rag_engine.query_llm(domain_discovery_prompt, max_tokens=3000)
            discovered_domains = self._parse_domain_discovery_response(discovery_response)
            
            self.logger.info(f"✅ Discovered {len(discovered_domains)} domains: {list(discovered_domains.keys())}")
            return discovered_domains
            
        except Exception as e:
            self.logger.error(f"Domain discovery failed: {e}")
            return self._get_fallback_domains()
    
    def _parse_domain_discovery_response(self, response: str) -> Dict[str, Dict[str, Any]]:
        """Parse LLM response to extract discovered domains."""
        
        domains = {}
        current_domain = None
        
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('DOMAIN:'):
                domain_name = line.replace('DOMAIN:', '').strip()
                # Normalize domain name
                domain_name = self._normalize_domain_name(domain_name)
                current_domain = domain_name
                domains[current_domain] = {
                    "evidence": "",
                    "confidence": "medium",
                    "key_topics": [],
                    "discovery_source": "document_analysis"
                }
            
            elif line.startswith('EVIDENCE:') and current_domain:
                evidence = line.replace('EVIDENCE:', '').strip()
                domains[current_domain]["evidence"] = evidence
            
            elif line.startswith('CONFIDENCE:') and current_domain:
                confidence = line.replace('CONFIDENCE:', '').strip().lower()
                domains[current_domain]["confidence"] = confidence
            
            elif line.startswith('KEY_TOPICS:') and current_domain:
                topics_text = line.replace('KEY_TOPICS:', '').strip()
                topics = [topic.strip() for topic in topics_text.split(',') if topic.strip()]
                domains[current_domain]["key_topics"] = topics
        
        # Filter out low confidence domains
        filtered_domains = {
            domain: info for domain, info in domains.items() 
            if info.get("confidence") in ["high", "medium"] and len(info.get("key_topics", [])) > 0
        }
        
        return filtered_domains
    
    def _normalize_domain_name(self, domain_name: str) -> str:
        """Normalize domain names to standard format."""
        
        domain_mapping = {
            # Access Control variants
            "access control": "access_control",
            "access management": "access_control", 
            "identity and access management": "access_control",
            "iam": "access_control",
            
            # Incident Response variants
            "incident response": "incident_response",
            "incident management": "incident_response",
            "security incident": "incident_response",
            
            # Data Protection variants
            "data protection": "data_protection",
            "data security": "data_protection",
            "information protection": "data_protection",
            "privacy": "data_protection",
            
            # Risk Management variants
            "risk management": "risk_management",
            "risk assessment": "risk_management",
            
            # Business Continuity variants
            "business continuity": "business_continuity",
            "disaster recovery": "business_continuity",
            "continuity planning": "business_continuity",
            
            # Vendor Management variants
            "vendor management": "vendor_management",
            "third party": "vendor_management",
            "supplier management": "vendor_management",
            
            # Change Management variants
            "change management": "change_management",
            "change control": "change_management",
            
            # Audit and Monitoring variants
            "audit": "audit_monitoring",
            "monitoring": "audit_monitoring",
            "logging": "audit_monitoring"
        }
        
        normalized = domain_name.lower().strip()
        return domain_mapping.get(normalized, normalized.replace(" ", "_"))
    
    def _get_fallback_domains(self) -> Dict[str, Dict[str, Any]]:
        """Provide fallback domains if discovery fails."""
        
        return {
            "access_control": {
                "evidence": "Fallback domain - standard compliance area",
                "confidence": "medium",
                "key_topics": ["authentication", "authorization", "access management"],
                "discovery_source": "fallback"
            },
            "incident_response": {
                "evidence": "Fallback domain - standard compliance area", 
                "confidence": "medium",
                "key_topics": ["incident detection", "response procedures", "recovery"],
                "discovery_source": "fallback"
            },
            "data_protection": {
                "evidence": "Fallback domain - standard compliance area",
                "confidence": "medium", 
                "key_topics": ["data classification", "encryption", "privacy"],
                "discovery_source": "fallback"
            }
        }
    
    async def _extract_dynamic_domain_expertise(self, domain: str, reference_policy_ids: List[str]) -> Dict[str, Any]:
        """Extract domain expertise dynamically from reference documents."""
        
        self.logger.info(f"Extracting expertise for domain: {domain}")
        
        # Use discovered domain info if available
        domain_info = self.discovered_domains.get(domain, {})
        initial_topics = domain_info.get("key_topics", [domain])
        
        # Search for domain-specific content in reference documents
        domain_content = []
        
        # Search using initial topics
        for topic in initial_topics[:10]:  # Limit searches
            for policy_id in reference_policy_ids:
                search_query = f"{topic} {policy_id}"
                results = await self.rag_engine.semantic_search(search_query, top_k=5)
                domain_content.extend(results)
        
        # Additional domain-specific searches
        domain_searches = [
            f"{domain} requirements",
            f"{domain} controls", 
            f"{domain} standards",
            f"{domain} implementation",
            f"{domain} procedures"
        ]
        
        for search_term in domain_searches:
            results = await self.rag_engine.semantic_search(search_term, top_k=6)
            domain_content.extend(results)
        
        # Remove duplicates and get top content
        unique_content = self._deduplicate_content(domain_content)
        top_content = unique_content[:25]  # Top 25 most relevant sections
        
        # Build context for expertise extraction
        content_text = "\n".join([
            f"Section: {item.get('section', 'Unknown')}\nContent: {item.get('content', '')[:400]}..."
            for item in top_content[:20]
        ])
        
        # Extract domain expertise using LLM
        expertise_prompt = f"""
Analyze the following reference standard content for the {domain.upper()} domain and extract comprehensive domain expertise:

REFERENCE CONTENT:
{content_text}

EXTRACTION TASK:
Based on the reference content provided, extract detailed domain expertise for {domain}:

1. KEY_TOPICS: Identify 10-15 specific topics/areas central to this domain (extract from actual content)
2. CRITICAL_CONTROLS: Identify 8-12 critical controls or mechanisms mentioned in the standards
3. COMPLIANCE_FRAMEWORKS: Identify which compliance frameworks/standards are referenced
4. REQUIREMENT_CATEGORIES: Identify main categories of requirements for this domain
5. IMPLEMENTATION_AREAS: Identify key areas where implementation guidance is provided
6. RISK_FACTORS: Identify main risk factors this domain addresses
7. MONITORING_ASPECTS: Identify what should be monitored for this domain

FORMAT YOUR RESPONSE AS:
KEY_TOPICS: topic1, topic2, topic3, ...
CRITICAL_CONTROLS: control1, control2, control3, ...
COMPLIANCE_FRAMEWORKS: framework1, framework2, ...
REQUIREMENT_CATEGORIES: category1, category2, ...
IMPLEMENTATION_AREAS: area1, area2, ...
RISK_FACTORS: risk1, risk2, risk3, ...
MONITORING_ASPECTS: aspect1, aspect2, aspect3, ...

Extract terms and concepts directly from the provided reference content. Be specific and detailed.
"""
        
        try:
            expertise_response = await self.rag_engine.query_llm(expertise_prompt, max_tokens=2500)
            expertise_data = self._parse_expertise_response(expertise_response, domain)
            
            # Add metadata
            expertise_data["content_analyzed"] = len(top_content)
            expertise_data["total_sections_found"] = len(unique_content)
            expertise_data["extraction_timestamp"] = datetime.now(timezone.utc).isoformat()
            expertise_data["discovery_info"] = domain_info
            
            self.logger.info(f"✅ Extracted expertise for {domain}: {len(expertise_data.get('key_topics', []))} topics")
            return expertise_data
            
        except Exception as e:
            self.logger.error(f"Failed to extract expertise for {domain}: {e}")
            return self._get_fallback_expertise(domain)
    
    def _parse_expertise_response(self, response: str, domain: str) -> Dict[str, Any]:
        """Parse LLM response to extract structured domain expertise."""
        
        expertise = {
            "key_topics": [],
            "critical_controls": [],
            "compliance_frameworks": [],
            "requirement_categories": [],
            "implementation_areas": [],
            "risk_factors": [],
            "monitoring_aspects": []
        }
        
        # Parse each section using regex
        sections = {
            "key_topics": r'KEY_TOPICS:\s*([^\n]+)',
            "critical_controls": r'CRITICAL_CONTROLS:\s*([^\n]+)',
            "compliance_frameworks": r'COMPLIANCE_FRAMEWORKS:\s*([^\n]+)',
            "requirement_categories": r'REQUIREMENT_CATEGORIES:\s*([^\n]+)',
            "implementation_areas": r'IMPLEMENTATION_AREAS:\s*([^\n]+)',
            "risk_factors": r'RISK_FACTORS:\s*([^\n]+)',
            "monitoring_aspects": r'MONITORING_ASPECTS:\s*([^\n]+)'
        }
        
        for key, pattern in sections.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                items_text = match.group(1).strip()
                # Split by comma and clean up
                items = [item.strip() for item in items_text.split(',') if item.strip()]
                expertise[key] = items[:15]  # Limit to 15 items max
        
        # Fallback extraction if structured format not found
        if not any(expertise.values()):
            expertise = self._extract_expertise_fallback(response, domain)
        
        # Ensure we have at least some content
        if not expertise["key_topics"]:
            expertise["key_topics"] = self._generate_default_topics(domain)
        
        return expertise
    
    def _extract_expertise_fallback(self, response: str, domain: str) -> Dict[str, Any]:
        """Fallback extraction method using general patterns."""
        
        lines = response.split('\n')
        
        key_topics = []
        critical_controls = []
        frameworks = []
        risk_factors = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 5:
                # Look for topic indicators
                if any(word in line.lower() for word in ['requirement', 'standard', 'policy', 'procedure', 'control']):
                    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', line)
                    key_topics.extend(words[:3])
                
                # Look for control indicators
                if any(word in line.lower() for word in ['control', 'mechanism', 'protection', 'security', 'monitor']):
                    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', line)
                    critical_controls.extend(words[:2])
                
                # Look for risk indicators
                if any(word in line.lower() for word in ['risk', 'threat', 'vulnerability', 'breach']):
                    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', line)
                    risk_factors.extend(words[:2])
                
                # Look for framework names
                framework_patterns = [r'PCI\s*DSS', r'ISO\s*\d+', r'NIST', r'GDPR', r'SOX', r'HIPAA']
                for pattern in framework_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        frameworks.append(re.search(pattern, line, re.IGNORECASE).group())
        
        return {
            "key_topics": list(set(key_topics))[:12],
            "critical_controls": list(set(critical_controls))[:10],
            "compliance_frameworks": list(set(frameworks)),
            "requirement_categories": [f"{domain} requirements", f"{domain} implementation"],
            "implementation_areas": [f"{domain} procedures", f"{domain} monitoring"],
            "risk_factors": list(set(risk_factors))[:8],
            "monitoring_aspects": [f"{domain} monitoring", f"{domain} audit"]
        }
    
    def _generate_default_topics(self, domain: str) -> List[str]:
        """Generate default topics if extraction fails."""
        
        default_topics = {
            "access_control": [
                "authentication", "authorization", "access management", "user accounts", 
                "password policy", "multi-factor authentication", "privileged access", "role-based access"
            ],
            "incident_response": [
                "incident detection", "response procedures", "escalation", "recovery", 
                "lessons learned", "forensics", "communication", "containment"
            ],
            "data_protection": [
                "data classification", "encryption", "backup", "retention", 
                "disposal", "privacy", "data loss prevention", "access controls"
            ],
            "risk_management": [
                "risk assessment", "risk mitigation", "risk monitoring", "threat analysis",
                "vulnerability management", "risk appetite", "risk reporting"
            ],
            "business_continuity": [
                "continuity planning", "disaster recovery", "backup procedures", "recovery testing",
                "business impact analysis", "continuity strategies"
            ]
        }
        
        return default_topics.get(domain, [domain, f"{domain} requirements", f"{domain} implementation"])
    
    def _get_fallback_expertise(self, domain: str) -> Dict[str, Any]:
        """Provide fallback expertise if extraction fails."""
        
        fallback_expertise = {
            "access_control": {
                "key_topics": ["authentication", "authorization", "access management", "user accounts", "password policy", "multi-factor"],
                "critical_controls": ["multi-factor authentication", "role-based access", "privilege management"],
                "compliance_frameworks": ["PCI DSS", "ISO 27001", "NIST"],
                "requirement_categories": ["authentication requirements", "authorization controls"],
                "implementation_areas": ["access procedures", "account management"],
                "risk_factors": ["unauthorized access", "privilege escalation", "account compromise"],
                "monitoring_aspects": ["access logs", "authentication failures", "privilege usage"]
            },
            "incident_response": {
                "key_topics": ["incident detection", "response procedures", "escalation", "recovery", "lessons learned", "forensics"],
                "critical_controls": ["incident classification", "response team", "communication plan"],
                "compliance_frameworks": ["PCI DSS", "NIST CSF", "ISO 27035"],
                "requirement_categories": ["detection requirements", "response procedures"],
                "implementation_areas": ["response playbooks", "escalation procedures"],
                "risk_factors": ["delayed response", "inadequate containment", "data breach"],
                "monitoring_aspects": ["incident metrics", "response times", "recovery effectiveness"]
            },
            "data_protection": {
                "key_topics": ["data classification", "encryption", "backup", "retention", "disposal", "privacy"],
                "critical_controls": ["data encryption", "access controls", "retention policies"],
                "compliance_frameworks": ["GDPR", "PCI DSS", "CCPA"],
                "requirement_categories": ["data security", "privacy protection"],
                "implementation_areas": ["encryption procedures", "data handling"],
                "risk_factors": ["data breach", "unauthorized disclosure", "data loss"],
                "monitoring_aspects": ["data access logs", "encryption status", "retention compliance"]
            }
        }
        
        return fallback_expertise.get(domain, {
            "key_topics": [domain],
            "critical_controls": [f"{domain} controls"],
            "compliance_frameworks": ["Unknown"],
            "requirement_categories": [f"{domain} requirements"],
            "implementation_areas": [f"{domain} implementation"],
            "risk_factors": [f"{domain} risks"],
            "monitoring_aspects": [f"{domain} monitoring"]
        })
    
    def _deduplicate_content(self, content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate content based on similarity."""
        unique_content = []
        seen_content = set()
        
        for item in content_list:
            content_key = item.get('content', '')[:100]  # First 100 chars as key
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_content.append(item)
        
        # Sort by similarity score
        unique_content.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        return unique_content
    
    async def _get_comprehensive_domain_content(self, domain: str, company_policies: List[str], 
                                              reference_policies: List[str]) -> Dict[str, Any]:
        """Get comprehensive content for domain analysis using extracted expertise."""
        
        domain_expertise = self.domain_expertise.get(domain, {})
        key_topics = domain_expertise.get("key_topics", [domain])
        critical_controls = domain_expertise.get("critical_controls", [])
        
        # Enhanced content retrieval with extracted topics
        company_content = []
        reference_content = []
        
        # Search using extracted key topics
        search_terms = (key_topics + critical_controls)[:15]  # Limit to prevent too many searches
        
        for policy_id in company_policies:
            for topic in search_terms:
                search_query = f"{topic} {policy_id}"
                results = await self.rag_engine.semantic_search(search_query, top_k=3)
                company_content.extend(results)
        
        for policy_id in reference_policies:
            for topic in search_terms:
                search_query = f"{topic} {policy_id}"
                results = await self.rag_engine.semantic_search(search_query, top_k=3)
                reference_content.extend(results)
        
        # Domain-specific searches using extracted categories
        requirement_categories = domain_expertise.get("requirement_categories", [])
        implementation_areas = domain_expertise.get("implementation_areas", [])
        
        domain_queries = (
            [f"{domain} policy", f"{domain} procedures", f"{domain} controls"] +
            requirement_categories[:3] +
            implementation_areas[:3]
        )
        
        for query in domain_queries:
            company_results = await self.rag_engine.semantic_search(query, top_k=3)
            reference_results = await self.rag_engine.semantic_search(query, top_k=3)
            company_content.extend(company_results)
            reference_content.extend(reference_results)
        
        # Remove duplicates and sort by relevance
        company_content = self._deduplicate_content(company_content)
        reference_content = self._deduplicate_content(reference_content)
        
        return {
            "domain": domain,
            "company_content": company_content[:15],  # Top 15 most relevant
            "reference_content": reference_content[:15],
            "total_company_sections": len(company_content),
            "total_reference_sections": len(reference_content),
            "search_terms_used": search_terms,
            "domain_expertise_applied": domain_expertise
        }
    
    def _extract_coverage_analysis(self, llm_analysis: str, domain: str) -> Dict[str, Any]:
        """Extract coverage analysis from LLM response with dynamic expertise."""
        
        # Extract coverage percentage using improved regex
        coverage_patterns = [
            r'coverage[^:]*:?\s*(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%[^.]*coverage',
            r'addresses[^:]*(\d+(?:\.\d+)?)\s*%',
            r'covers?\s*(\d+(?:\.\d+)?)\s*%'
        ]
        
        coverage_percentage = 75.0  # Default good coverage
        for pattern in coverage_patterns:
            match = re.search(pattern, llm_analysis, re.IGNORECASE)
            if match:
                try:
                    coverage_percentage = float(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
        
        # Use extracted domain expertise for topic counting
        domain_expertise = self.domain_expertise.get(domain, {})
        total_topics = len(domain_expertise.get("key_topics", [])) or 8
        
        # Extract topics covered from LLM analysis
        topics_covered = self._extract_number_from_text(llm_analysis, ["topics covered", "topics addressed", "areas covered"])
        if topics_covered == 0:
            topics_covered = max(1, int(total_topics * coverage_percentage / 100))
        
        # Determine coverage depth
        if "comprehensive" in llm_analysis.lower():
            coverage_depth = "Comprehensive"
        elif "adequate" in llm_analysis.lower():
            coverage_depth = "Adequate"
        elif "basic" in llm_analysis.lower():
            coverage_depth = "Basic"
        else:
            coverage_depth = "Adequate" if coverage_percentage >= 70 else "Basic"
        
        # Determine maturity level
        if coverage_percentage >= 85:
            maturity_level = "Advanced"
        elif coverage_percentage >= 70:
            maturity_level = "Developing"
        elif coverage_percentage >= 50:
            maturity_level = "Initial"
        else:
            maturity_level = "Inadequate"
        
        return {
            "coverage_percentage": coverage_percentage,
            "topics_covered": topics_covered,
            "total_reference_topics": total_topics,
            "coverage_depth": coverage_depth,
            "maturity_level": maturity_level,
            "evidence_source": "LLM analysis using extracted domain expertise",
            "extracted_key_topics": domain_expertise.get("key_topics", [])
        }
    
    def _extract_number_from_text(self, text: str, patterns: List[str]) -> int:
        """Extract numbers from text using patterns."""
        for pattern in patterns:
            # Look for pattern followed by number
            regex = rf'{pattern}[^:]*:?\s*(\d+)'
            match = re.search(regex, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        return 0
    
    def _extract_gap_analysis(self, llm_analysis: str, domain: str) -> List[Dict[str, Any]]:
        """Extract gap analysis from LLM response."""
        
        gaps = []
        
        # Enhanced gap extraction patterns
        gap_patterns = [
            r'gap[^:]*:?\s*([^.\n]+)',
            r'missing[^:]*:?\s*([^.\n]+)',
            r'lacks?[^:]*:?\s*([^.\n]+)',
            r'insufficient[^:]*:?\s*([^.\n]+)',
            r'inadequate[^:]*:?\s*([^.\n]+)'
        ]
        
        gap_id_counter = 1
        for pattern in gap_patterns:
            matches = re.findall(pattern, llm_analysis, re.IGNORECASE)
            for match in matches[:2]:  # Top 2 per pattern
                gap_text = match.strip()
                if len(gap_text) > 15:  # Quality filter
                    
                    # Clean gap text
                    gap_text = self._clean_gap_text(gap_text)
                    
                    # Determine severity from context
                    severity = "Medium"
                    if any(word in gap_text.lower() for word in ['critical', 'severe', 'major', 'essential']):
                        severity = "High"
                    elif any(word in gap_text.lower() for word in ['minor', 'small', 'low']):
                        severity = "Low"
                    
                    # Generate recommendation
                    recommendation = self._generate_gap_recommendation(gap_text, domain)
                    
                    gaps.append({
                        "gap_id": f"GAP_{domain}_{gap_id_counter:03d}",
                        "title": gap_text[:80] + "..." if len(gap_text) > 80 else gap_text,
                        "description": gap_text,
                                                "severity": severity,
                        "risk_impact": severity,
                        "domain": domain,
                        "recommendation": recommendation,
                        "evidence": f"LLM analysis identified: {gap_text[:100]}..."
                    })
                    gap_id_counter += 1
        
        # Ensure we have at least one meaningful gap
        if not gaps:
            gaps.append({
                "gap_id": f"GAP_{domain}_001",
                "title": "Policy Documentation Enhancement Needed",
                "description": f"Policy documentation for {domain} requires enhancement to meet comprehensive compliance standards",
                "severity": "Medium",
                "risk_impact": "Medium",
                "domain": domain,
                "recommendation": "Develop more detailed policy procedures with specific implementation guidance and measurable compliance criteria",
                "evidence": "Analysis indicates opportunity for policy enhancement"
            })
        
        return gaps[:5]  # Top 5 gaps
    
    def _clean_gap_text(self, text: str) -> str:
        """Clean gap text from markdown artifacts and incomplete sentences."""
        # Remove markdown artifacts
        text = re.sub(r'\*\*', '', text)
        text = re.sub(r'\*', '', text)
        
        # Remove incomplete sentences starting with special characters
        text = re.sub(r'^[^A-Za-z]*', '', text)
        
        # Remove trailing incomplete parts
        text = re.sub(r'\s*\([^)]*$', '', text)
        
        return text.strip()
    
    def _generate_gap_recommendation(self, gap_text: str, domain: str) -> str:
        """Generate specific recommendation for a gap."""
        
        gap_lower = gap_text.lower()
        
        # Domain-specific recommendations using extracted expertise
        domain_expertise = self.domain_expertise.get(domain, {})
        critical_controls = domain_expertise.get("critical_controls", [])
        
        # Try to match gap with critical controls
        for control in critical_controls:
            if any(word in gap_lower for word in control.lower().split()):
                return f"Implement {control} with specific procedures, technology standards, and monitoring capabilities"
        
        # Fallback domain-specific recommendations
        domain_recommendations = {
            "access_control": "Implement comprehensive access control framework with authentication, authorization, and monitoring",
            "incident_response": "Develop detailed incident response playbooks with clear roles, timelines, and communication procedures",
            "data_protection": "Establish data protection framework with classification, encryption, and retention policies",
            "risk_management": "Implement risk management program with assessment, mitigation, and monitoring processes",
            "vendor_management": "Establish vendor management program with due diligence, contracts, and monitoring"
        }
        
        return domain_recommendations.get(domain, f"Address {gap_text[:50]} by developing specific policies, procedures, and controls with measurable compliance criteria")
    
    def _extract_alignment_analysis(self, llm_analysis: str, domain: str) -> Dict[str, Any]:
        """Extract alignment analysis from LLM response."""
        
        # Extract alignment percentage
        alignment_patterns = [
            r'alignment[^:]*:?\s*(\d+(?:\.\d+)?)\s*%',
            r'aligns?[^:]*(\d+(?:\.\d+)?)\s*%',
            r'consistent[^:]*(\d+(?:\.\d+)?)\s*%'
        ]
        
        alignment_score = 75.0  # Default good alignment
        for pattern in alignment_patterns:
            match = re.search(pattern, llm_analysis, re.IGNORECASE)
            if match:
                try:
                    alignment_score = float(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
        
        # Extract strengths and weaknesses
        strengths = self._extract_list_items(llm_analysis, ["strength", "aligns well", "consistent", "meets"])
        weaknesses = self._extract_list_items(llm_analysis, ["weakness", "gap", "inconsistent", "lacks"])
        
        return {
            "alignment_score": alignment_score,
            "strong_areas": strengths[:3],
            "improvement_areas": weaknesses[:3],
            "consistency_level": "High" if alignment_score >= 80 else "Medium" if alignment_score >= 60 else "Low",
            "evidence_source": "LLM comparative analysis"
        }
    
    def _extract_list_items(self, text: str, keywords: List[str]) -> List[str]:
        """Extract list items related to keywords."""
        items = []
        
        for keyword in keywords:
            pattern = rf'{keyword}[^:]*:?\s*([^.\n]+)'
            matches = re.findall(pattern, text, re.IGNORECASE)
            cleaned_matches = [self._clean_insight_text(match.strip()) for match in matches if len(match.strip()) > 10]
            items.extend(cleaned_matches)
        
        return items[:5]  # Top 5 items
    
    def _clean_insight_text(self, text: str) -> str:
        """Clean insight text from artifacts."""
        # Remove markdown
        text = re.sub(r'\*\*', '', text)
        text = re.sub(r'\*', '', text)
        
        # Remove incomplete sentences
        if text.startswith('*') or text.startswith('#'):
            text = text[1:].strip()
        
        return text.strip()
    
    def _calculate_quantitative_scores(self, coverage: Dict[str, Any], gaps: List[Dict[str, Any]], 
                                     alignment: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quantitative scores for the domain."""
        
        # Coverage score from coverage analysis
        coverage_score = coverage.get("coverage_percentage", 75.0)
        
        # Quality score based on coverage depth and gaps
        quality_base = 70.0
        if coverage.get("coverage_depth") == "Comprehensive":
            quality_base = 85.0
        elif coverage.get("coverage_depth") == "Basic":
            quality_base = 55.0
        
        # Adjust for gaps
        critical_gaps = len([g for g in gaps if g.get("severity") == "High"])
        quality_score = max(quality_base - (critical_gaps * 10), 30.0)
        
        # Alignment score from alignment analysis
        alignment_score = alignment.get("alignment_score", 75.0)
        
        # Implementation score based on quality and specific criteria
        implementation_base = 65.0
        if "procedures" in str(gaps).lower() or "implementation" in str(gaps).lower():
            implementation_base = 50.0
        implementation_score = min(implementation_base + (quality_score - 70) / 2, 90.0)
        
        return {
            "coverage_score": coverage_score,
            "quality_score": quality_score,
            "alignment_score": alignment_score,
            "implementation_score": max(implementation_score, 40.0)
        }
    
    async def _generate_strategic_insights(self, domain: str, llm_analysis: str) -> Dict[str, Any]:
        """Generate strategic insights from analysis."""
        
        # Extract insights using LLM analysis
        insights_prompt = f"""
Based on the {domain} analysis, provide strategic insights:

Analysis: {llm_analysis[:1000]}...

Extract:
1. Key Strengths (3-4 items)
2. Improvement Priorities (3-4 items)
3. Strategic Recommendations (2-3 items)

Format as bullet points with specific, actionable insights.
"""
        
        try:
            insights_response = await self.rag_engine.query_llm(insights_prompt, max_tokens=1500)
            
            # Parse insights
            key_strengths = self._extract_list_items(insights_response, ["strength", "strong", "good", "effective"])
            improvement_priorities = self._extract_list_items(insights_response, ["improve", "priority", "enhance", "develop"])
            
        except Exception as e:
            self.logger.error(f"Failed to generate strategic insights: {e}")
            key_strengths = []
            improvement_priorities = []
        
        # Add domain-specific insights using extracted expertise
        domain_expertise = self.domain_expertise.get(domain, {})
        if len(key_strengths) < 2:
            key_topics = domain_expertise.get("key_topics", [])
            if key_topics:
                key_strengths.append(f"Foundation established for {', '.join(key_topics[:3])}")
            key_strengths.append("Policy framework shows compliance awareness")
        
        if len(improvement_priorities) < 2:
            critical_controls = domain_expertise.get("critical_controls", [])
            if critical_controls:
                improvement_priorities.append(f"Enhance implementation of {critical_controls[0]}")
            improvement_priorities.append("Strengthen policy detail and implementation guidance")
        
        return {
            "key_strengths": key_strengths[:4],
            "improvement_priorities": improvement_priorities[:4],
            "strategic_focus": domain,
            "insight_source": "LLM strategic analysis with extracted expertise"
        }
    
    def _extract_recommendations(self, response: str) -> List[str]:
        """Extract recommendations from LLM response."""
        
        # Split by lines and extract meaningful recommendations
        lines = response.split('\n')
        recommendations = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 20:
                # Remove bullet points and numbering
                cleaned = re.sub(r'^[\d\.\-\*\s]+', '', line)
                cleaned = self._clean_insight_text(cleaned)
                if len(cleaned) > 15:
                    recommendations.append(cleaned)
        
        # Use default recommendations if extraction fails
        if len(recommendations) < 5:
            recommendations.extend(self._get_default_recommendations())
        
        return recommendations[:10]  # Top 10 recommendations
    
    def _get_default_recommendations(self) -> List[str]:
        """Get default recommendations if LLM extraction fails."""
        return [
            "Establish comprehensive policy governance framework with regular review cycles",
            "Implement automated compliance monitoring and reporting capabilities",
            "Develop staff training programs for policy awareness and implementation",
            "Create executive dashboard for compliance status visibility",
            "Establish continuous improvement process for policy enhancement",
            "Implement risk-based approach to compliance priority setting",
            "Develop incident response capabilities with regular testing",
            "Create cross-functional compliance team with clear responsibilities"
        ]
    
    # ============================================================================
    # FALLBACK METHODS FOR ERROR HANDLING WITH FRENCH FRAMEWORK
    # ============================================================================
    
    def _create_fallback_domain_result_french(self, domain: str) -> Dict[str, Any]:
        """Create fallback domain result with French compliance framework when analysis fails."""
        
        fallback_french_status = {
            "policy_status": {"level": 2, "description": "Politique partiellement formalisée"},
            "implementation_status": {"level": 1, "description": "Implémentation partielle de la politique"},
            "automation_status": {"level": 0, "description": "Non automatisé"},
            "reporting_status": {"level": 0, "description": "Non rapporté"},
            "overall_compliance_level": 1,
            "compliance_meaning": "Non formalisé",
            "compliance_level_description": "Non formalisé"
        }
        
        fallback_score = FrenchComplianceScore(50.0, fallback_french_status, [
            {"name": "Coverage", "score": 50.0, "weight": 0.25, "status": "Needs Improvement"},
            {"name": "Quality", "score": 50.0, "weight": 0.20, "status": "Needs Improvement"},
            {"name": "Alignment", "score": 50.0, "weight": 0.20, "status": "Needs Improvement"},
            {"name": "Implementation", "score": 50.0, "weight": 0.15, "status": "Needs Improvement"},
            {"name": "French Compliance Level", "score": 20.0, "weight": 0.20, "status": "Non formalisé", "french_level": 1}
        ])
        
        return {
            "domain": domain,
            "coverage": {
                "coverage_percentage": 50.0,
                "topics_covered": 2,
                "total_reference_topics": 4,
                "coverage_depth": "Basic",
                "maturity_level": "Initial",
                "evidence_source": "Fallback analysis"
            },
            "gaps": [{
                "gap_id": f"GAP_{domain}_001",
                "title": "Analysis could not be completed - French compliance assessment needed",
                "description": f"Detailed French compliance analysis for {domain} domain could not be completed due to system limitations",
                "severity": "Medium",
                "risk_impact": "Medium",
                "domain": domain,
                "recommendation": "Perform manual French compliance assessment or retry automated analysis",
                "evidence": "System fallback with French framework"
            }],
            "alignment": {
                "alignment_score": 50.0,
                "strong_areas": ["Basic framework present"],
                "improvement_areas": ["French compliance assessment needed"],
                "consistency_level": "Medium",
                "evidence_source": "Fallback analysis"
            },
            "quantitative_scores": {
                "coverage_score": 50.0,
                "quality_score": 50.0,
                "alignment_score": 50.0,
                "implementation_score": 50.0
            },
            "strategic_insights": {
                "key_strengths": ["Policy framework exists"],
                "improvement_priorities": ["Complete French compliance analysis"],
                "strategic_focus": domain,
                "insight_source": "Fallback analysis"
            },
            "french_compliance_status": fallback_french_status,
            "score": fallback_score,
            "evidence_based": False,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _create_fallback_overall_score_french(self, domain_results: Dict[str, Any]) -> FrenchComplianceScore:
        """Create fallback overall score with French framework when assessment fails."""
        
        if domain_results:
            domain_scores = []
            for result in domain_results.values():
                if isinstance(result.get("score"), FrenchComplianceScore):
                    domain_scores.append(result["score"].score)
                else:
                    domain_scores.append(50.0)
            
            overall_score = sum(domain_scores) / len(domain_scores) if domain_scores else 50.0
        else:
            overall_score = 50.0
        
        fallback_french_status = {
            "policy_status": {"level": 1, "description": "Politique non formalisée"},
            "implementation_status": {"level": 1, "description": "Implémentation partielle de la politique"},
            "automation_status": {"level": 0, "description": "Non automatisé"},
            "reporting_status": {"level": 0, "description": "Non rapporté"},
            "overall_compliance_level": 1,
            "compliance_meaning": "Non formalisé",
            "compliance_level_description": "Non formalisé"
        }
        
        fallback_criteria = [
            {
                "name": "Overall Compliance Maturity",
                "score": overall_score,
                "weight": 0.8,
                "status": "Developing" if overall_score >= 60 else "Initial"
            },
            {
                "name": "French Compliance Level",
                "score": 20.0,
                "weight": 0.2,
                "status": "Non formalisé",
                "french_level": 1
            }
        ]
        
        compliance_score = FrenchComplianceScore(overall_score, fallback_french_status, fallback_criteria)
        compliance_score.recommendations = self._get_default_recommendations()
        compliance_score.recommendations_french = self._get_default_french_recommendations()
        
        return compliance_score