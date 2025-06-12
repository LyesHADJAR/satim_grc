GAP_ANALYSIS_PROMPT = """
You are a compliance expert analyzing policy gaps. Compare the company policy content with reference framework requirements and identify specific gaps.

DOMAIN: {domain}

COMPANY POLICY CONTENT:
{company_policy_content}

REFERENCE FRAMEWORK CONTENT:
{reference_framework_content}

Analyze and provide a JSON response with the following structure:

{{
  "gaps": [
    {{
      "type": "missing|insufficient|outdated",
      "severity": "critical|high|medium|low",
      "description": "Clear description of what is missing or insufficient",
      "reference_requirement": "Specific requirement from reference framework",
      "suggested_action": "Concrete action to address this gap"
    }}
  ]
}}

ANALYSIS GUIDELINES:
1. Focus on substantive gaps that impact compliance
2. Mark as "critical" if gap creates significant compliance risk
3. Mark as "high" if gap affects core security/compliance requirements
4. Mark as "medium" if gap affects best practices
5. Mark as "low" if gap is minor or cosmetic
6. Be specific in descriptions and suggested actions
7. Only include real gaps, not minor differences in wording

Return only valid JSON.
"""

OVERLAP_ANALYSIS_PROMPT = """
You are a compliance expert analyzing policy overlaps. Identify areas where company policies exceed or go beyond reference framework requirements.

DOMAIN: {domain}

COMPANY POLICY CONTENT:
{company_policy_content}

REFERENCE FRAMEWORK CONTENT:
{reference_framework_content}

Analyze and provide a JSON response with the following structure:

{{
  "overlaps": [
    {{
      "description": "Description of how company policy exceeds reference",
      "company_provision": "Specific company policy text that exceeds requirements",
      "reference_requirement": "Corresponding reference requirement",
      "value_assessment": "positive|neutral|excessive"
    }}
  ]
}}

ANALYSIS GUIDELINES:
1. "positive" - Company policy adds valuable security/compliance measures
2. "neutral" - Company policy is more detailed but not significantly better
3. "excessive" - Company policy may be unnecessarily restrictive or complex
4. Focus on meaningful differences, not minor wording variations
5. Consider business impact of additional requirements

Return only valid JSON.
"""