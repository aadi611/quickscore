"""
AI-powered startup evaluation service using OpenAI.
"""
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from app.services.llm_service import LLMService
from app.core.config import settings

logger = logging.getLogger(__name__)


class StartupEvaluator:
    """AI-powered startup evaluation using structured prompts and OpenAI."""
    
    def __init__(self):
        self.llm_service = LLMService()
        
        # Evaluation prompts with specific scoring criteria
        self.prompts = {
            "pitch_analysis": {
                "system": """You are a top-tier VC partner at a leading venture capital firm evaluating pre-seed startups. 
                
You have 15+ years of experience investing in successful startups and understand what makes companies successful at the pre-seed stage. You focus on team-market fit, problem clarity, and early traction signals rather than just financial projections.

Your analysis should be thorough but concise, providing actionable insights that help make investment decisions. Consider the unique challenges and opportunities of pre-seed companies.

Return your analysis as a JSON object with the exact structure requested.""",
                
                "template": """Analyze this pitch deck content and provide scoring (0-10 scale) with detailed reasoning:

PITCH CONTENT:
{content}

STARTUP CONTEXT:
- Industry: {industry}
- Stage: {stage}
- Description: {description}

Provide analysis in this exact JSON format:
{{
    "problem_solution_fit": {{
        "score": <0-10>,
        "explanation": "<2-sentence assessment of problem clarity and solution appropriateness>",
        "strengths": ["<strength1>", "<strength2>"],
        "concerns": ["<concern1>", "<concern2>"]
    }},
    "market_opportunity": {{
        "score": <0-10>,
        "explanation": "<2-sentence assessment of market size and timing>",
        "tam_estimate": "<estimated TAM if mentioned or assessable>",
        "market_timing": "<assessment of market timing and trends>"
    }},
    "business_model_clarity": {{
        "score": <0-10>,
        "explanation": "<2-sentence assessment of monetization strategy>",
        "revenue_streams": ["<stream1>", "<stream2>"],
        "scalability": "<assessment of business model scalability>"
    }},
    "team_strength": {{
        "score": <0-10>,
        "explanation": "<2-sentence assessment of team capability>",
        "domain_expertise": "<assessment of relevant experience>",
        "execution_indicators": ["<indicator1>", "<indicator2>"]
    }},
    "differentiation": {{
        "score": <0-10>,
        "explanation": "<2-sentence assessment of competitive advantages>",
        "key_differentiators": ["<diff1>", "<diff2>"],
        "competitive_moat": "<assessment of defensibility>"
    }},
    "overall_assessment": {{
        "investment_thesis": "<3-sentence investment thesis>",
        "key_risks": ["<risk1>", "<risk2>", "<risk3>"],
        "next_milestones": ["<milestone1>", "<milestone2>"],
        "funding_readiness": "<assessment of readiness for investment>"
    }}
}}"""
            },
            
            "founder_assessment": {
                "system": """You are an experienced venture capitalist specializing in founder evaluation for pre-seed investments. 

You understand that at the pre-seed stage, the team is often the most critical factor for success. You evaluate founders based on domain expertise, execution history, leadership potential, network quality, and coachability signals.

Focus on identifying founders who can execute in uncertain environments, adapt quickly, and build strong teams. Consider both professional background and entrepreneurial indicators.

Return your assessment as a structured JSON object.""",
                
                "template": """Evaluate these founder profiles for pre-seed investment potential:

FOUNDER DATA:
{founder_data}

STARTUP CONTEXT:
- Industry: {industry}
- Stage: {stage}
- Problem being solved: {problem}

Provide assessment in this exact JSON format:
{{
    "domain_expertise": {{
        "score": <0-10>,
        "explanation": "<assessment of relevant industry experience>",
        "years_in_domain": <estimated years>,
        "previous_roles": ["<role1>", "<role2>"],
        "technical_depth": "<assessment of technical knowledge>"
    }},
    "execution_track_record": {{
        "score": <0-10>,
        "explanation": "<assessment of past execution and achievements>",
        "previous_startups": <number>,
        "notable_achievements": ["<achievement1>", "<achievement2>"],
        "leadership_experience": "<assessment of team leadership>"
    }},
    "leadership_potential": {{
        "score": <0-10>,
        "explanation": "<assessment of leadership capabilities>",
        "team_building": "<indicators of ability to attract talent>",
        "vision_communication": "<ability to articulate vision>"
    }},
    "network_strength": {{
        "score": <0-10>,
        "explanation": "<assessment of professional network quality>",
        "industry_connections": "<quality of industry relationships>",
        "advisor_potential": "<potential to attract strong advisors>"
    }},
    "coachability_signals": {{
        "score": <0-10>,
        "explanation": "<indicators of openness to feedback and learning>",
        "learning_indicators": ["<indicator1>", "<indicator2>"],
        "adaptability": "<assessment of ability to pivot and adapt>"
    }},
    "overall_founder_assessment": {{
        "composite_score": <average of above scores>,
        "key_strengths": ["<strength1>", "<strength2>", "<strength3>"],
        "development_areas": ["<area1>", "<area2>"],
        "investment_confidence": "<high/medium/low with reasoning>",
        "founder_market_fit": "<assessment of fit between founders and market>"
    }}
}}"""
            },
            
            "market_analysis": {
                "system": """You are a market research expert and venture capital analyst specializing in early-stage market assessment.

You excel at analyzing market opportunities for pre-seed startups, focusing on Total Addressable Market (TAM), market timing, competitive landscape, and entry barriers. You understand how to assess markets that may be emerging or rapidly evolving.

Your analysis helps VCs understand the market context and opportunity size for investment decisions. Consider both current market state and future potential.

Provide structured market assessment as JSON.""",
                
                "template": """Analyze the market opportunity for this startup:

MARKET CONTEXT:
- Industry: {industry}
- Target Market: {target_market}
- Geographic Focus: {geography}
- Business Model: {business_model}

COMPETITIVE LANDSCAPE:
{competitors}

ADDITIONAL CONTEXT:
{additional_context}

Provide analysis in this exact JSON format:
{{
    "tam_analysis": {{
        "tam_estimate_usd": "<estimated TAM in USD, provide reasoning>",
        "sam_estimate_usd": "<estimated SAM in USD>",
        "som_estimate_usd": "<estimated SOM in USD for first 3 years>",
        "market_size_confidence": "<high/medium/low with explanation>",
        "growth_rate": "<estimated annual growth rate>"
    }},
    "market_timing": {{
        "score": <0-10>,
        "explanation": "<assessment of market timing and trends>",
        "market_maturity": "<emerging/growth/mature>",
        "technology_readiness": "<assessment of enabling technologies>",
        "regulatory_environment": "<impact of regulations on market>"
    }},
    "competitive_intensity": {{
        "score": <0-10 where 10 is most competitive>,
        "explanation": "<assessment of current competition level>",
        "major_competitors": ["<competitor1>", "<competitor2>", "<competitor3>"],
        "competitive_advantages": ["<advantage1>", "<advantage2>"],
        "differentiation_opportunity": "<assessment of differentiation potential>"
    }},
    "barriers_to_entry": {{
        "score": <0-10 where 10 is highest barriers>,
        "explanation": "<assessment of market entry difficulty>",
        "key_barriers": ["<barrier1>", "<barrier2>"],
        "startup_advantages": ["<advantage1>", "<advantage2>"],
        "capital_requirements": "<assessment of capital needed for market entry>"
    }},
    "growth_potential": {{
        "three_year_outlook": "<assessment of 3-year market potential>",
        "key_drivers": ["<driver1>", "<driver2>", "<driver3>"],
        "potential_risks": ["<risk1>", "<risk2>"],
        "market_expansion": "<potential for geographic or segment expansion>"
    }},
    "overall_market_assessment": {{
        "market_attractiveness": <0-10>,
        "investment_thesis": "<2-sentence market investment thesis>",
        "ideal_entry_strategy": "<recommended market entry approach>",
        "key_success_factors": ["<factor1>", "<factor2>", "<factor3>"]
    }}
}}"""
            }
        }
    
    async def evaluate_startup(
        self,
        pitch_deck_content: Optional[Dict] = None,
        founder_data: Optional[Dict] = None,
        market_data: Optional[Dict] = None,
        startup_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive startup evaluation using AI analysis.
        
        Args:
            pitch_deck_content: Extracted pitch deck content
            founder_data: Founder profile information
            market_data: Market context and competitive information
            startup_context: Basic startup information (name, industry, stage, etc.)
            
        Returns:
            Comprehensive evaluation results with scores and insights
        """
        start_time = datetime.now()
        
        try:
            # Prepare evaluation tasks
            evaluation_tasks = []
            
            # Pitch deck analysis
            if pitch_deck_content and startup_context:
                pitch_content = self._format_pitch_content(pitch_deck_content)
                pitch_prompt = self.prompts["pitch_analysis"]["template"].format(
                    content=pitch_content,
                    industry=startup_context.get("industry", "Unknown"),
                    stage=startup_context.get("stage", "pre_seed"),
                    description=startup_context.get("description", "No description provided")
                )
                
                evaluation_tasks.append({
                    "system": self.prompts["pitch_analysis"]["system"],
                    "user": pitch_prompt,
                    "response_format": "json",
                    "task_type": "pitch_analysis"
                })
            
            # Founder assessment
            if founder_data and startup_context:
                founder_prompt = self.prompts["founder_assessment"]["template"].format(
                    founder_data=json.dumps(founder_data, indent=2),
                    industry=startup_context.get("industry", "Unknown"),
                    stage=startup_context.get("stage", "pre_seed"),
                    problem=startup_context.get("problem", "Problem description not available")
                )
                
                evaluation_tasks.append({
                    "system": self.prompts["founder_assessment"]["system"],
                    "user": founder_prompt,
                    "response_format": "json",
                    "task_type": "founder_assessment"
                })
            
            # Market analysis
            if market_data and startup_context:
                market_prompt = self.prompts["market_analysis"]["template"].format(
                    industry=startup_context.get("industry", "Unknown"),
                    target_market=market_data.get("target_market", "Not specified"),
                    geography=market_data.get("geography", "Global"),
                    business_model=startup_context.get("business_model", "Not specified"),
                    competitors=json.dumps(market_data.get("competitors", []), indent=2),
                    additional_context=market_data.get("additional_context", "No additional context")
                )
                
                evaluation_tasks.append({
                    "system": self.prompts["market_analysis"]["system"],
                    "user": market_prompt,
                    "response_format": "json",
                    "task_type": "market_analysis"
                })
            
            if not evaluation_tasks:
                return {
                    "success": False,
                    "error": "No evaluation tasks could be created - insufficient data provided"
                }
            
            # Execute evaluations concurrently
            results = await self.llm_service.batch_completions(evaluation_tasks, max_concurrent=3)
            
            # Process and structure results
            evaluation_results = self._process_evaluation_results(results, evaluation_tasks)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "evaluation_results": evaluation_results,
                "processing_time": processing_time,
                "tasks_completed": len([r for r in results if r.get("success", False)]),
                "total_tasks": len(evaluation_tasks),
                "raw_llm_outputs": results  # For debugging/transparency
            }
            
        except Exception as e:
            logger.error(f"Error in startup evaluation: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
    
    def _format_pitch_content(self, pitch_deck_content: Dict) -> str:
        """Format pitch deck content for LLM analysis."""
        content_parts = []
        
        if "structured_content" in pitch_deck_content:
            structured = pitch_deck_content["structured_content"]
            
            # Add sections if available
            if "sections" in structured:
                for section_name, section_data in structured["sections"].items():
                    content_parts.append(f"\n{section_name.upper()}:")
                    if isinstance(section_data, dict) and "content" in section_data:
                        for item in section_data["content"]:
                            content_parts.append(f"- {item}")
                    else:
                        content_parts.append(f"- {section_data}")
            
            # Add key metrics if available
            if "key_metrics" in structured:
                content_parts.append("\nKEY METRICS:")
                for metric_name, values in structured["key_metrics"].items():
                    content_parts.append(f"- {metric_name}: {values}")
        
        # Fallback to raw content if structured content is not available
        if not content_parts and "raw_content" in pitch_deck_content:
            raw = pitch_deck_content["raw_content"]
            if "full_text" in raw:
                # Truncate if too long
                text = raw["full_text"]
                if len(text) > 8000:  # Limit for API
                    text = text[:8000] + "... [content truncated]"
                content_parts.append(text)
        
        return "\n".join(content_parts) if content_parts else "No content available for analysis"
    
    def _process_evaluation_results(
        self, 
        results: List[Dict], 
        tasks: List[Dict]
    ) -> Dict[str, Any]:
        """Process and structure evaluation results."""
        processed_results = {}
        
        for i, result in enumerate(results):
            task_type = tasks[i]["task_type"]
            
            if result.get("success", False):
                try:
                    # Parse JSON response
                    content = result["content"]
                    parsed_content = json.loads(content)
                    
                    processed_results[task_type] = {
                        "success": True,
                        "data": parsed_content,
                        "model_used": result.get("model_used"),
                        "tokens_used": result.get("tokens_used")
                    }
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON for {task_type}: {e}")
                    processed_results[task_type] = {
                        "success": False,
                        "error": f"JSON parsing error: {str(e)}",
                        "raw_content": result["content"]
                    }
            else:
                processed_results[task_type] = {
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
        
        return processed_results
    
    async def generate_insights(
        self, 
        evaluation_results: Dict[str, Any], 
        startup_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate actionable insights from evaluation results."""
        
        insights_prompt = f"""Based on the comprehensive evaluation results, generate actionable insights for this startup:

STARTUP: {startup_context.get('name', 'Unknown')}
INDUSTRY: {startup_context.get('industry', 'Unknown')}
STAGE: {startup_context.get('stage', 'pre_seed')}

EVALUATION RESULTS:
{json.dumps(evaluation_results, indent=2)}

Generate insights in this JSON format:
{{
    "key_strengths": ["<strength1>", "<strength2>", "<strength3>"],
    "critical_risks": ["<risk1>", "<risk2>", "<risk3>"],
    "immediate_priorities": ["<priority1>", "<priority2>", "<priority3>"],
    "funding_readiness": {{
        "score": <0-10>,
        "explanation": "<assessment of readiness for funding>",
        "blockers": ["<blocker1>", "<blocker2>"],
        "timeline_estimate": "<estimated time to funding readiness>"
    }},
    "strategic_recommendations": ["<rec1>", "<rec2>", "<rec3>"],
    "comparable_companies": ["<company1>", "<company2>", "<company3>"],
    "investor_fit": {{
        "ideal_investor_type": "<type of investor most suitable>",
        "investment_stage": "<recommended investment stage>",
        "value_add_needs": ["<need1>", "<need2>"]
    }}
}}"""
        
        insights_result = await self.llm_service.generate_completion(
            system_prompt="You are a strategic advisor helping startups understand their evaluation results and plan next steps.",
            user_prompt=insights_prompt,
            response_format="json",
            temperature=0.4
        )
        
        if insights_result.get("success"):
            try:
                return {
                    "success": True,
                    "insights": json.loads(insights_result["content"]),
                    "tokens_used": insights_result.get("tokens_used")
                }
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "error": "Failed to parse insights JSON",
                    "raw_content": insights_result["content"]
                }
        else:
            return {
                "success": False,
                "error": insights_result.get("error", "Failed to generate insights")
            }