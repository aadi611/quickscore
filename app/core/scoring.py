"""
Scoring engine for startup evaluation with weighted algorithms and confidence calculation.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
import statistics
from datetime import datetime

from app.core.config import SCORING_WEIGHTS, RECOMMENDATION_THRESHOLDS

logger = logging.getLogger(__name__)


class ScoringEngine:
    """
    Comprehensive scoring engine that combines AI evaluation results
    with weighted algorithms to generate final startup scores.
    """
    
    def __init__(self):
        self.scoring_weights = SCORING_WEIGHTS
        self.recommendation_thresholds = RECOMMENDATION_THRESHOLDS
        
        # Confidence factors and their weights
        self.confidence_factors = {
            "data_completeness": 0.30,
            "source_quality": 0.25,
            "consistency": 0.25,
            "evaluation_depth": 0.20
        }
    
    def calculate_composite_score(
        self,
        evaluation_results: Dict[str, Any],
        startup_stage: str = "pre_seed",
        additional_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Calculate the final composite score for a startup.
        
        Args:
            evaluation_results: Results from AI evaluation service
            startup_stage: Stage of the startup (affects weighting)
            additional_data: Additional scoring data (scraped metrics, etc.)
            
        Returns:
            Dictionary containing scores, recommendation, and confidence metrics
        """
        try:
            # Extract scores from AI evaluation results
            extracted_scores = self._extract_scores_from_evaluation(evaluation_results)
            
            # Apply stage-specific weights
            weights = self.scoring_weights.get(startup_stage, self.scoring_weights["pre_seed"])
            
            # Calculate weighted composite score
            composite_score = self._calculate_weighted_score(extracted_scores, weights)
            
            # Calculate confidence metrics
            confidence_metrics = self._calculate_confidence(evaluation_results, extracted_scores)
            
            # Adjust score based on confidence
            adjusted_score = self._apply_confidence_adjustment(composite_score, confidence_metrics)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(adjusted_score, confidence_metrics)
            
            # Calculate individual category scores for display
            category_scores = self._calculate_category_scores(extracted_scores)
            
            return {
                "success": True,
                "overall_score": round(adjusted_score, 1),
                "raw_composite_score": round(composite_score, 1),
                "category_scores": category_scores,
                "recommendation": recommendation["recommendation"],
                "confidence": recommendation["confidence"],
                "confidence_metrics": confidence_metrics,
                "scoring_breakdown": {
                    "weights_used": weights,
                    "stage": startup_stage,
                    "adjustment_applied": round(adjusted_score - composite_score, 1)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating composite score: {e}")
            return {
                "success": False,
                "error": str(e),
                "overall_score": 0,
                "recommendation": "no",
                "confidence": "low"
            }
    
    def _extract_scores_from_evaluation(self, evaluation_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical scores from AI evaluation results."""
        scores = {
            "team": 0.0,
            "market": 0.0,
            "product": 0.0,
            "traction": 0.0,
            "pitch_quality": 0.0
        }
        
        # Extract from pitch analysis
        if "pitch_analysis" in evaluation_results:
            pitch_data = evaluation_results["pitch_analysis"]
            if pitch_data.get("success") and "data" in pitch_data:
                pitch_scores = pitch_data["data"]
                
                # Map pitch analysis scores to our categories
                scores["product"] = self._safe_score_extract(pitch_scores, "problem_solution_fit.score", default=0)
                scores["market"] = self._safe_score_extract(pitch_scores, "market_opportunity.score", default=0)
                scores["pitch_quality"] = self._safe_score_extract(pitch_scores, "business_model_clarity.score", default=0)
                
                # Average differentiation and other factors for product score
                differentiation_score = self._safe_score_extract(pitch_scores, "differentiation.score", default=0)
                scores["product"] = (scores["product"] + differentiation_score) / 2
        
        # Extract from founder assessment
        if "founder_assessment" in evaluation_results:
            founder_data = evaluation_results["founder_assessment"]
            if founder_data.get("success") and "data" in founder_data:
                founder_scores = founder_data["data"]
                
                # Calculate team score from founder metrics
                domain_score = self._safe_score_extract(founder_scores, "domain_expertise.score", default=0)
                execution_score = self._safe_score_extract(founder_scores, "execution_track_record.score", default=0)
                leadership_score = self._safe_score_extract(founder_scores, "leadership_potential.score", default=0)
                network_score = self._safe_score_extract(founder_scores, "network_strength.score", default=0)
                coachability_score = self._safe_score_extract(founder_scores, "coachability_signals.score", default=0)
                
                # Weighted average for team score
                scores["team"] = (
                    domain_score * 0.30 +
                    execution_score * 0.25 +
                    leadership_score * 0.20 +
                    network_score * 0.15 +
                    coachability_score * 0.10
                )
        
        # Extract from market analysis
        if "market_analysis" in evaluation_results:
            market_data = evaluation_results["market_analysis"]
            if market_data.get("success") and "data" in market_data:
                market_scores = market_data["data"]
                
                # Update market score with more detailed analysis
                timing_score = self._safe_score_extract(market_scores, "market_timing.score", default=0)
                attractiveness_score = self._safe_score_extract(market_scores, "overall_market_assessment.market_attractiveness", default=0)
                
                # Adjust for competitive intensity (invert since high competition is bad)
                competitive_intensity = self._safe_score_extract(market_scores, "competitive_intensity.score", default=5)
                competition_adjusted = 10 - competitive_intensity
                
                scores["market"] = (timing_score + attractiveness_score + competition_adjusted) / 3
        
        # Set default traction score (would be enhanced with actual traction data)
        scores["traction"] = 5.0  # Neutral score for pre-seed
        
        return scores
    
    def _safe_score_extract(self, data: Dict, path: str, default: float = 0.0) -> float:
        """Safely extract score from nested dictionary using dot notation."""
        try:
            keys = path.split('.')
            value = data
            for key in keys:
                value = value[key]
            
            # Ensure it's a number
            return float(value) if value is not None else default
        except (KeyError, TypeError, ValueError):
            return default
    
    def _calculate_weighted_score(self, scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted composite score."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for category, score in scores.items():
            if category in weights and score > 0:  # Only include categories with actual scores
                weight = weights[category]
                weighted_sum += score * weight
                total_weight += weight
        
        # Normalize by actual weights used (in case some categories are missing)
        if total_weight > 0:
            return (weighted_sum / total_weight) * 10  # Scale to 0-100
        else:
            return 0.0
    
    def _calculate_confidence(
        self, 
        evaluation_results: Dict[str, Any], 
        extracted_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate confidence metrics for the evaluation."""
        
        # Data completeness: How much data was available for evaluation
        completed_evaluations = sum(1 for result in evaluation_results.values() 
                                  if isinstance(result, dict) and result.get("success", False))
        total_possible_evaluations = 3  # pitch, founder, market
        data_completeness = completed_evaluations / total_possible_evaluations
        
        # Source quality: Quality of the underlying data sources
        source_quality = self._assess_source_quality(evaluation_results)
        
        # Consistency: How consistent are the scores across categories
        consistency = self._assess_score_consistency(extracted_scores)
        
        # Evaluation depth: How detailed was each evaluation
        evaluation_depth = self._assess_evaluation_depth(evaluation_results)
        
        # Calculate overall confidence
        confidence_factors = {
            "data_completeness": data_completeness,
            "source_quality": source_quality,
            "consistency": consistency,
            "evaluation_depth": evaluation_depth
        }
        
        overall_confidence = sum(
            factor_value * self.confidence_factors[factor_name]
            for factor_name, factor_value in confidence_factors.items()
        )
        
        return {
            "overall_confidence": overall_confidence,
            "factors": confidence_factors,
            "confidence_level": self._get_confidence_level(overall_confidence)
        }
    
    def _assess_source_quality(self, evaluation_results: Dict[str, Any]) -> float:
        """Assess the quality of data sources used in evaluation."""
        quality_score = 0.0
        source_count = 0
        
        for evaluation_type, result in evaluation_results.items():
            if isinstance(result, dict) and result.get("success", False):
                source_count += 1
                
                # Check for token usage (indicates substantial processing)
                if result.get("tokens_used", {}).get("total_tokens", 0) > 1000:
                    quality_score += 0.8
                elif result.get("tokens_used", {}).get("total_tokens", 0) > 500:
                    quality_score += 0.6
                else:
                    quality_score += 0.4
        
        return quality_score / max(source_count, 1)
    
    def _assess_score_consistency(self, scores: Dict[str, float]) -> float:
        """Assess consistency of scores across categories."""
        valid_scores = [score for score in scores.values() if score > 0]
        
        if len(valid_scores) < 2:
            return 0.5  # Neutral score if not enough data
        
        # Calculate coefficient of variation (lower is more consistent)
        mean_score = statistics.mean(valid_scores)
        if mean_score == 0:
            return 0.5
        
        std_dev = statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0
        cv = std_dev / mean_score
        
        # Convert CV to consistency score (invert and normalize)
        consistency = max(0, 1 - (cv / 0.5))  # CV of 0.5 or higher = 0 consistency
        
        return consistency
    
    def _assess_evaluation_depth(self, evaluation_results: Dict[str, Any]) -> float:
        """Assess how detailed and thorough each evaluation was."""
        depth_scores = []
        
        for evaluation_type, result in evaluation_results.items():
            if isinstance(result, dict) and result.get("success", False):
                # Check for presence of detailed sections in the evaluation
                data = result.get("data", {})
                
                if evaluation_type == "pitch_analysis":
                    # Count detailed sections in pitch analysis
                    sections = ["problem_solution_fit", "market_opportunity", "business_model_clarity", 
                              "team_strength", "differentiation"]
                    present_sections = sum(1 for section in sections if section in data)
                    depth_scores.append(present_sections / len(sections))
                
                elif evaluation_type == "founder_assessment":
                    # Count detailed sections in founder assessment
                    sections = ["domain_expertise", "execution_track_record", "leadership_potential", 
                              "network_strength", "coachability_signals"]
                    present_sections = sum(1 for section in sections if section in data)
                    depth_scores.append(present_sections / len(sections))
                
                elif evaluation_type == "market_analysis":
                    # Count detailed sections in market analysis
                    sections = ["tam_analysis", "market_timing", "competitive_intensity", 
                              "barriers_to_entry", "growth_potential"]
                    present_sections = sum(1 for section in sections if section in data)
                    depth_scores.append(present_sections / len(sections))
        
        return statistics.mean(depth_scores) if depth_scores else 0.5
    
    def _apply_confidence_adjustment(self, composite_score: float, confidence_metrics: Dict) -> float:
        """Apply confidence-based adjustment to the composite score."""
        confidence = confidence_metrics["overall_confidence"]
        
        # Apply conservative adjustment for low confidence
        if confidence < 0.3:
            # Significantly reduce score for very low confidence
            adjustment_factor = 0.7
        elif confidence < 0.5:
            # Moderately reduce score for low confidence
            adjustment_factor = 0.85
        elif confidence > 0.8:
            # Slight boost for high confidence
            adjustment_factor = 1.05
        else:
            # No adjustment for medium confidence
            adjustment_factor = 1.0
        
        adjusted_score = composite_score * adjustment_factor
        
        # Ensure score stays within bounds
        return max(0, min(100, adjusted_score))
    
    def _generate_recommendation(
        self, 
        adjusted_score: float, 
        confidence_metrics: Dict
    ) -> Dict[str, str]:
        """Generate investment recommendation based on score and confidence."""
        confidence_level = confidence_metrics["confidence_level"]
        
        # Determine base recommendation from score
        if adjusted_score >= self.recommendation_thresholds["strong_yes"]:
            base_recommendation = "strong_yes"
        elif adjusted_score >= self.recommendation_thresholds["yes"]:
            base_recommendation = "yes"
        elif adjusted_score >= self.recommendation_thresholds["maybe"]:
            base_recommendation = "maybe"
        else:
            base_recommendation = "no"
        
        # Adjust recommendation based on confidence
        final_recommendation = base_recommendation
        if confidence_level == "low":
            # Downgrade recommendation for low confidence
            if base_recommendation == "strong_yes":
                final_recommendation = "yes"
            elif base_recommendation == "yes":
                final_recommendation = "maybe"
        elif confidence_level == "high" and base_recommendation == "yes" and adjusted_score > 70:
            # Potentially upgrade to strong yes for high confidence
            final_recommendation = "strong_yes"
        
        return {
            "recommendation": final_recommendation,
            "confidence": confidence_level
        }
    
    def _calculate_category_scores(self, extracted_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate display-friendly category scores (0-100 scale)."""
        return {
            category: round(score * 10, 1)  # Convert 0-10 to 0-100
            for category, score in extracted_scores.items()
        }
    
    def _get_confidence_level(self, confidence_score: float) -> str:
        """Convert numerical confidence to categorical level."""
        if confidence_score >= 0.75:
            return "high"
        elif confidence_score >= 0.5:
            return "medium"
        else:
            return "low"
    
    def generate_score_explanation(
        self, 
        scoring_results: Dict[str, Any], 
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate detailed explanation of how the score was calculated."""
        
        explanation = {
            "score_breakdown": {
                "overall_score": scoring_results["overall_score"],
                "category_contributions": {},
                "weighting_explanation": {}
            },
            "confidence_explanation": {
                "level": scoring_results["confidence"],
                "factors": scoring_results["confidence_metrics"]["factors"],
                "impact_on_score": scoring_results["scoring_breakdown"]["adjustment_applied"]
            },
            "recommendation_rationale": {
                "recommendation": scoring_results["recommendation"],
                "score_threshold": self._get_threshold_explanation(scoring_results["recommendation"]),
                "key_factors": []
            }
        }
        
        # Add category contributions
        weights = scoring_results["scoring_breakdown"]["weights_used"]
        category_scores = scoring_results["category_scores"]
        
        for category, score in category_scores.items():
            if category in weights:
                contribution = (score / 100) * weights[category] * 100
                explanation["score_breakdown"]["category_contributions"][category] = {
                    "score": score,
                    "weight": weights[category],
                    "contribution": round(contribution, 1)
                }
        
        # Add weighting explanation
        explanation["score_breakdown"]["weighting_explanation"] = {
            "stage": scoring_results["scoring_breakdown"]["stage"],
            "rationale": self._get_weighting_rationale(scoring_results["scoring_breakdown"]["stage"])
        }
        
        return explanation
    
    def _get_threshold_explanation(self, recommendation: str) -> str:
        """Get explanation of recommendation thresholds."""
        threshold_explanations = {
            "strong_yes": f"Score ≥ {self.recommendation_thresholds['strong_yes']} with high confidence",
            "yes": f"Score ≥ {self.recommendation_thresholds['yes']} with adequate confidence",
            "maybe": f"Score ≥ {self.recommendation_thresholds['maybe']} but with concerns or low confidence",
            "no": f"Score < {self.recommendation_thresholds['maybe']} or significant risk factors identified"
        }
        return threshold_explanations.get(recommendation, "Unknown recommendation")
    
    def _get_weighting_rationale(self, stage: str) -> str:
        """Get explanation for why specific weights are used for this stage."""
        stage_rationales = {
            "pre_seed": "Pre-seed weighting emphasizes team (40%) as the primary success factor, with market opportunity (25%) as secondary. Product and traction have lower weights due to early stage.",
            "seed": "Seed stage balances team (30%) with market (25%) and product (20%), while increasing traction weight (20%) as companies should show early validation."
        }
        return stage_rationales.get(stage, "Standard weighting applied")