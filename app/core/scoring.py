
"""
Scoring engine for QuickScore.

Combines AI evaluation results with stage-specific weighted scoring,
confidence assessment, and investment recommendation generation.

Input scores: 0-10
Output scores: 0-100
"""

import logging
import math

from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Any

from app.core.config import (
    SCORING_WEIGHTS,
    RECOMMENDATION_THRESHOLDS,
)

logger = logging.getLogger(__name__)


class ScoringEngine:
    """
    Calculate startup scores, confidence metrics, and recommendations.

    Responsibilities:
        1. Extract scores from AI evaluation results.
        2. Apply startup-stage-specific scoring weights.
        3. Calculate composite scores.
        4. Assess evaluation confidence.
        5. Adjust scores based on confidence.
        6. Generate investment recommendations.
        7. Explain scoring decisions.
    """

    SCORE_CATEGORIES = (
        "team",
        "market",
        "product",
        "traction",
        "pitch_quality",
    )

    EVALUATION_TYPES = (
        "pitch_analysis",
        "founder_assessment",
        "market_analysis",
    )

    CONFIDENCE_FACTORS = {
        "data_completeness": 0.30,
        "source_quality": 0.25,
        "consistency": 0.25,
        "evaluation_depth": 0.20,
    }

    EVALUATION_SECTIONS = {
        "pitch_analysis": (
            "problem_solution_fit",
            "market_opportunity",
            "business_model_clarity",
            "team_strength",
            "differentiation",
        ),
        "founder_assessment": (
            "domain_expertise",
            "execution_track_record",
            "leadership_potential",
            "network_strength",
            "coachability_signals",
        ),
        "market_analysis": (
            "tam_analysis",
            "market_timing",
            "competitive_intensity",
            "barriers_to_entry",
            "growth_potential",
        ),
    }

    CONFIDENCE_ADJUSTMENTS = {
        "low": 0.70,
        "medium": 1.00,
        "high": 1.05,
    }

    # ================================================================
    # INITIALIZATION
    # ================================================================

    def __init__(self) -> None:
        """Initialize the scoring engine."""

        self.scoring_weights = SCORING_WEIGHTS
        self.recommendation_thresholds = RECOMMENDATION_THRESHOLDS

        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate scoring weights and recommendation thresholds."""

        if not self.scoring_weights:
            raise ValueError("Scoring weights cannot be empty.")

        for stage, weights in self.scoring_weights.items():

            if not weights:
                raise ValueError(
                    f"Missing scoring weights for stage: {stage}"
                )

            if any(weight < 0 for weight in weights.values()):
                raise ValueError(
                    f"Negative scoring weight detected in {stage}."
                )

            if not math.isclose(
                sum(weights.values()),
                1.0,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"Scoring weights for {stage} must sum to 1.0."
                )

        required_thresholds = {
            "strong_yes",
            "yes",
            "maybe",
            "no",
        }

        if not required_thresholds.issubset(
            self.recommendation_thresholds
        ):
            raise ValueError(
                "Missing recommendation thresholds."
            )

        if any(
            not 0 <= threshold <= 100
            for threshold in self.recommendation_thresholds.values()
        ):
            raise ValueError(
                "Recommendation thresholds must be between 0 and 100."
            )

        thresholds = self.recommendation_thresholds

        if not (
            thresholds["strong_yes"]
            > thresholds["yes"]
            > thresholds["maybe"]
            >= thresholds["no"]
        ):
            raise ValueError(
                "Recommendation thresholds must be in descending order."
            )

    # ================================================================
    # MAIN SCORING METHOD
    # ================================================================

    def calculate_composite_score(
        self,
        evaluation_results: dict[str, Any],
        startup_stage: str = "pre_seed",
        additional_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Calculate the final composite score for a startup.

        Args:
            evaluation_results:
                Results from AI evaluation services.

            startup_stage:
                Startup stage. Supported stages are defined in
                SCORING_WEIGHTS.

            additional_data:
                Optional external scoring information, including
                actual traction metrics.

        Returns:
            Dictionary containing:
                - overall_score
                - raw_composite_score
                - category_scores
                - recommendation
                - confidence
                - confidence_metrics
                - scoring_breakdown
                - timestamp
        """

        try:
            if not isinstance(evaluation_results, dict):
                raise ValueError(
                    "Evaluation results must be a dictionary."
                )

            weights = self._get_stage_weights(startup_stage)

            # Step 1: Extract category scores.
            extracted_scores = self._extract_scores_from_evaluation(
                evaluation_results
            )

            # Step 2: Incorporate additional scoring data.
            additional_data = additional_data or {}

            if "traction" in additional_data:

                traction_score = self._safe_score_extract(
                    additional_data,
                    "traction",
                )

                if traction_score is not None:
                    extracted_scores["traction"] = traction_score

            # Step 3: Calculate weighted composite score.
            composite_score = self._calculate_weighted_score(
                extracted_scores,
                weights,
            )

            # Step 4: Calculate confidence.
            confidence_metrics = self._calculate_confidence(
                evaluation_results,
                extracted_scores,
            )

            # Step 5: Apply confidence adjustment.
            adjusted_score = self._apply_confidence_adjustment(
                composite_score,
                confidence_metrics,
            )

            # Step 6: Generate recommendation.
            recommendation = self._generate_recommendation(
                adjusted_score,
                confidence_metrics,
            )

            # Step 7: Calculate category scores.
            category_scores = self._calculate_category_scores(
                extracted_scores
            )

            missing_categories = [
                category
                for category, score in extracted_scores.items()
                if score is None
            ]

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
                    "adjustment_applied": round(
                        adjusted_score - composite_score,
                        1,
                    ),
                    "missing_categories": missing_categories,
                },
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

        except (ValueError, TypeError, KeyError):
            logger.exception(
                "Failed to calculate composite startup score."
            )

            return {
                "success": False,
                "error": "Unable to calculate startup score.",
                "overall_score": 0,
                "recommendation": "no",
                "confidence": "low",
            }

    # ================================================================
    # STAGE CONFIGURATION
    # ================================================================

    def _get_stage_weights(
        self,
        startup_stage: str,
    ) -> dict[str, float]:
        """Retrieve weights for a supported startup stage."""

        weights = self.scoring_weights.get(startup_stage)

        if weights is None:
            raise ValueError(
                f"Unsupported startup stage: {startup_stage}. "
                f"Supported stages: {list(self.scoring_weights)}"
            )

        return dict(weights)

    # ================================================================
    # SCORE EXTRACTION
    # ================================================================

    def _extract_scores_from_evaluation(
        self,
        evaluation_results: dict[str, Any],
    ) -> dict[str, float | None]:
        """
        Extract numerical scores from AI evaluation results.

        Missing scores are represented as None.
        A genuine score of zero remains valid.
        """

        scores: dict[str, float | None] = {
            category: None
            for category in self.SCORE_CATEGORIES
        }

        # ------------------------------------------------------------
        # Pitch Analysis
        # ------------------------------------------------------------

        pitch_data = self._get_evaluation_data(
            evaluation_results,
            "pitch_analysis",
        )

        if pitch_data is not None:

            problem_solution_score = self._safe_score_extract(
                pitch_data,
                "problem_solution_fit.score",
            )

            differentiation_score = self._safe_score_extract(
                pitch_data,
                "differentiation.score",
            )

            scores["product"] = self._average_available([
                problem_solution_score,
                differentiation_score,
            ])

            scores["market"] = self._safe_score_extract(
                pitch_data,
                "market_opportunity.score",
            )

            scores["pitch_quality"] = self._safe_score_extract(
                pitch_data,
                "business_model_clarity.score",
            )

        # ------------------------------------------------------------
        # Founder Assessment
        # ------------------------------------------------------------

        founder_data = self._get_evaluation_data(
            evaluation_results,
            "founder_assessment",
        )

        if founder_data is not None:

            founder_weights = {
                "domain_expertise": 0.30,
                "execution_track_record": 0.25,
                "leadership_potential": 0.20,
                "network_strength": 0.15,
                "coachability_signals": 0.10,
            }

            founder_scores = {
                name: self._safe_score_extract(
                    founder_data,
                    f"{name}.score",
                )
                for name in founder_weights
            }

            if any(
                score is not None
                for score in founder_scores.values()
            ):
                scores["team"] = self._calculate_weighted_score(
                    founder_scores,
                    founder_weights,
                )

        # ------------------------------------------------------------
        # Market Analysis
        # ------------------------------------------------------------

        market_data = self._get_evaluation_data(
            evaluation_results,
            "market_analysis",
        )

        if market_data is not None:

            timing_score = self._safe_score_extract(
                market_data,
                "market_timing.score",
            )

            attractiveness_score = self._safe_score_extract(
                market_data,
                "overall_market_assessment.market_attractiveness",
            )

            competitive_intensity = self._safe_score_extract(
                market_data,
                "competitive_intensity.score",
            )

            competition_adjusted = (
                10 - competitive_intensity
                if competitive_intensity is not None
                else None
            )

            # Market analysis takes precedence when available.
            scores["market"] = self._average_available([
                timing_score,
                attractiveness_score,
                competition_adjusted,
            ])

        return scores

    def _get_evaluation_data(
        self,
        evaluation_results: dict[str, Any],
        evaluation_type: str,
    ) -> dict[str, Any] | None:
        """Return data from a successful evaluation."""

        result = evaluation_results.get(evaluation_type)

        if not isinstance(result, dict):
            return None

        if not result.get("success"):
            return None

        data = result.get("data")

        if not isinstance(data, dict):
            return None

        return data

    def _safe_score_extract(
        self,
        data: dict[str, Any],
        path: str,
    ) -> float | None:
        """
        Safely extract numerical values using dot notation.

        Valid range: 0-10.

        Returns None for missing or invalid values.
        """

        try:
            value: Any = data

            for key in path.split("."):

                if not isinstance(value, dict):
                    return None

                value = value.get(key)

            if value is None or isinstance(value, bool):
                return None

            score = float(value)

            if not math.isfinite(score):
                return None

            if not 0 <= score <= 10:
                return None

            return score

        except (TypeError, ValueError):
            return None

    @staticmethod
    def _average_available(
        values: list[float | None],
    ) -> float | None:
        """Calculate the average of available scores."""

        valid_values = [
            value
            for value in values
            if value is not None
        ]

        if not valid_values:
            return None

        return mean(valid_values)

    # ================================================================
    # WEIGHTED SCORING
    # ================================================================

    def _calculate_weighted_score(
        self,
        scores: dict[str, float | None],
        weights: dict[str, float],
    ) -> float:
        """
        Calculate normalized weighted composite score.

        Missing categories are excluded.
        Remaining weights are normalized automatically.
        """

        weighted_sum = 0.0
        total_weight = 0.0

        for category, score in scores.items():

            weight = weights.get(category, 0.0)

            if score is None or weight <= 0:
                continue

            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        # Convert 0-10 into 0-100.
        return (weighted_sum / total_weight) * 10

    def _calculate_category_scores(
        self,
        extracted_scores: dict[str, float | None],
    ) -> dict[str, float | None]:
        """Convert individual category scores to a 0-100 scale."""

        return {
            category: (
                round(score * 10, 1)
                if score is not None
                else None
            )
            for category, score in extracted_scores.items()
        }

    # ================================================================
    # CONFIDENCE CALCULATION
    # ================================================================

    def _calculate_confidence(
        self,
        evaluation_results: dict[str, Any],
        extracted_scores: dict[str, float | None],
    ) -> dict[str, Any]:
        """
        Calculate confidence based on four independent factors.

        Factors:
            1. Data completeness.
            2. Source quality.
            3. Score consistency.
            4. Evaluation depth.
        """

        completed_evaluations = sum(
            self._get_evaluation_data(
                evaluation_results,
                evaluation_type,
            ) is not None
            for evaluation_type in self.EVALUATION_TYPES
        )

        data_completeness = (
            completed_evaluations / len(self.EVALUATION_TYPES)
        )

        source_quality = self._assess_source_quality(
            evaluation_results
        )

        consistency = self._assess_score_consistency(
            extracted_scores
        )

        evaluation_depth = self._assess_evaluation_depth(
            evaluation_results
        )

        factors = {
            "data_completeness": data_completeness,
            "source_quality": source_quality,
            "consistency": consistency,
            "evaluation_depth": evaluation_depth,
        }

        overall_confidence = sum(
            factors[name] * weight
            for name, weight in self.CONFIDENCE_FACTORS.items()
        )

        return {
            "overall_confidence": round(
                overall_confidence,
                3,
            ),
            "factors": {
                name: round(value, 3)
                for name, value in factors.items()
            },
            "confidence_level": self._get_confidence_level(
                overall_confidence
            ),
        }

    # ================================================================
    # SOURCE QUALITY
    # ================================================================

    def _assess_source_quality(
        self,
        evaluation_results: dict[str, Any],
    ) -> float:
        """
        Assess explicitly supplied source-quality metadata.

        Expected optional field:

            {
                "success": True,
                "source_quality": 0.85,
                "data": {}
            }

        Source quality must be between 0 and 1.

        If unavailable, use a neutral baseline of 0.5.
        """

        quality_scores = []

        for evaluation_type in self.EVALUATION_TYPES:

            result = evaluation_results.get(evaluation_type)

            if not isinstance(result, dict):
                continue

            if not result.get("success"):
                continue

            quality = result.get("source_quality")

            if quality is None:
                continue

            try:
                quality = float(quality)

                if (
                    math.isfinite(quality)
                    and 0 <= quality <= 1
                ):
                    quality_scores.append(quality)

            except (TypeError, ValueError):
                continue

        return (
            mean(quality_scores)
            if quality_scores
            else 0.5
        )

    # ================================================================
    # SCORE CONSISTENCY
    # ================================================================

    def _assess_score_consistency(
        self,
        scores: dict[str, float | None],
    ) -> float:
        """
        Assess consistency across available category scores.

        Lower score variation indicates higher consistency.

        Returns a normalized value between 0 and 1.
        """

        valid_scores = [
            score
            for score in scores.values()
            if score is not None
        ]

        if len(valid_scores) < 2:
            return 0.5

        average_score = mean(valid_scores)

        if average_score == 0:
            return 1.0

        standard_deviation = stdev(valid_scores)

        coefficient_of_variation = (
            standard_deviation / average_score
        )

        consistency = 1 - (
            coefficient_of_variation / 0.5
        )

        return max(
            0.0,
            min(1.0, consistency),
        )

    # ================================================================
    # EVALUATION DEPTH
    # ================================================================

    def _assess_evaluation_depth(
        self,
        evaluation_results: dict[str, Any],
    ) -> float:
        """Assess the completeness of individual AI evaluations."""

        depth_scores = []

        for evaluation_type, sections in self.EVALUATION_SECTIONS.items():

            data = self._get_evaluation_data(
                evaluation_results,
                evaluation_type,
            )

            if data is None:
                continue

            present_sections = sum(
                section in data
                for section in sections
            )

            depth_scores.append(
                present_sections / len(sections)
            )

        return (
            mean(depth_scores)
            if depth_scores
            else 0.5
        )

    # ================================================================
    # CONFIDENCE ADJUSTMENT
    # ================================================================

    def _apply_confidence_adjustment(
        self,
        composite_score: float,
        confidence_metrics: dict[str, Any],
    ) -> float:
        """Adjust the composite score according to confidence."""

        confidence = confidence_metrics["overall_confidence"]

        if confidence < 0.3:
            adjustment_factor = self.CONFIDENCE_ADJUSTMENTS["low"]

        elif confidence > 0.8:
            adjustment_factor = self.CONFIDENCE_ADJUSTMENTS["high"]

        else:
            adjustment_factor = self.CONFIDENCE_ADJUSTMENTS["medium"]

        adjusted_score = composite_score * adjustment_factor

        return round(
            max(0.0, min(100.0, adjusted_score)),
            2,
        )

    # ================================================================
    # RECOMMENDATION GENERATION
    # ================================================================

    def _generate_recommendation(
        self,
        adjusted_score: float,
        confidence_metrics: dict[str, Any],
    ) -> dict[str, str]:
        """Generate an investment recommendation."""

        thresholds = self.recommendation_thresholds

        if adjusted_score >= thresholds["strong_yes"]:
            recommendation = "strong_yes"

        elif adjusted_score >= thresholds["yes"]:
            recommendation = "yes"

        elif adjusted_score >= thresholds["maybe"]:
            recommendation = "maybe"

        else:
            recommendation = "no"

        confidence_level = confidence_metrics["confidence_level"]

        # Downgrade recommendations when confidence is low.
        if confidence_level == "low":

            downgrade_map = {
                "strong_yes": "yes",
                "yes": "maybe",
            }

            recommendation = downgrade_map.get(
                recommendation,
                recommendation,
            )

        # Upgrade only when confidence is high and score supports it.
        elif (
            confidence_level == "high"
            and recommendation == "yes"
            and adjusted_score > 70
        ):
            recommendation = "strong_yes"

        return {
            "recommendation": recommendation,
            "confidence": confidence_level,
        }

    @staticmethod
    def _get_confidence_level(
        confidence_score: float,
    ) -> str:
        """Convert numerical confidence to a categorical level."""

        if confidence_score >= 0.75:
            return "high"

        if confidence_score >= 0.5:
            return "medium"

        return "low"

    # ================================================================
    # SCORE EXPLANATION
    # ================================================================

    def generate_score_explanation(
        self,
        scoring_results: dict[str, Any],
        evaluation_results: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate a detailed explanation of the scoring calculation.

        Returns:
            Score breakdown, category contributions,
            confidence explanation, and recommendation rationale.
        """

        if not scoring_results.get("success"):
            raise ValueError(
                "Cannot explain unsuccessful scoring results."
            )

        breakdown = scoring_results["scoring_breakdown"]

        category_scores = scoring_results["category_scores"]

        weights = breakdown["weights_used"]

        active_weight = sum(
            weights.get(category, 0.0)
            for category, score in category_scores.items()
            if score is not None
        )

        contributions = {}

        for category, score in category_scores.items():

            if score is None or category not in weights:
                continue

            contribution = (
                score * weights[category] / active_weight
                if active_weight > 0
                else 0.0
            )

            contributions[category] = {
                "score": score,
                "weight": weights[category],
                "contribution": round(contribution, 2),
            }

        return {
            "score_breakdown": {
                "overall_score": scoring_results["overall_score"],
                "category_contributions": contributions,
                "weighting_explanation": {
                    "stage": breakdown["stage"],
                    "rationale": self._get_weighting_rationale(
                        breakdown["stage"]
                    ),
                },
            },
            "confidence_explanation": {
                "level": scoring_results["confidence"],
                "factors": scoring_results["confidence_metrics"]["factors"],
                "impact_on_score": breakdown["adjustment_applied"],
            },
            "recommendation_rationale": {
                "recommendation": scoring_results["recommendation"],
                "score_threshold": self._get_threshold_explanation(
                    scoring_results["recommendation"]
                ),
                "key_factors": [],
            },
        }

    # ================================================================
    # RECOMMENDATION EXPLANATIONS
    # ================================================================

    def _get_threshold_explanation(
        self,
        recommendation: str,
    ) -> str:
        """Explain the threshold for the selected recommendation."""

        thresholds = self.recommendation_thresholds

        explanations = {
            "strong_yes": (
                f"Score >= {thresholds['strong_yes']}."
            ),
            "yes": (
                f"Score >= {thresholds['yes']}."
            ),
            "maybe": (
                f"Score >= {thresholds['maybe']}."
            ),
            "no": (
                f"Score < {thresholds['maybe']}."
            ),
        }

        return explanations.get(
            recommendation,
            "Unknown recommendation.",
        )

    @staticmethod
    def _get_weighting_rationale(
        stage: str,
    ) -> str:
        """Explain why specific weights are used for each stage."""

        rationales = {
            "pre_seed": (
                "Pre-seed scoring emphasizes the founding team (40%) "
                "and market opportunity (25%). Product and traction "
                "have lower weights because early-stage startups "
                "typically have limited market validation."
            ),
            "seed": (
                "Seed scoring balances team (30%), market (25%), "
                "product (20%), and traction (20%). Greater emphasis "
                "is placed on early market validation and product "
                "development."
            ),
        }

        return rationales.get(
            stage,
            "Stage-specific scoring weights applied.",
        )
