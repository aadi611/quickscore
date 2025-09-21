"""
Service for finding comparable startups based on multiple similarity metrics.
"""
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from app.core.database import get_db
from app.models import Startup, Analysis
from app.services.embedding_similarity import EmbeddingSimilarityService
from app.services.feature_engineering import FeatureEngineeringService

logger = logging.getLogger(__name__)


@dataclass
class SimilarityWeights:
    """Weights for different similarity dimensions."""
    
    industry_weight: float = 0.25
    stage_weight: float = 0.20
    geography_weight: float = 0.15
    team_size_weight: float = 0.10
    business_model_weight: float = 0.15
    embedding_weight: float = 0.15


@dataclass
class ComparableStartup:
    """Represents a comparable startup with similarity metrics."""
    
    startup_id: str
    company_name: str
    industry: str
    stage: str
    location: str
    founded_year: Optional[int]
    
    # Similarity scores
    overall_similarity: float
    industry_similarity: float
    stage_similarity: float
    geography_similarity: float
    team_similarity: float
    business_model_similarity: float
    embedding_similarity: float
    
    # Additional metrics
    performance_score: Optional[float]
    funding_amount: Optional[float]
    employee_count: Optional[int]
    
    # Insights
    similarity_reasons: List[str]
    key_differences: List[str]


class ComparableStartupsFinder:
    """Advanced service for finding comparable startups."""
    
    def __init__(self):
        self.embedding_service = EmbeddingSimilarityService()
        self.feature_service = FeatureEngineeringService()
        self.weights = SimilarityWeights()
    
    async def find_comparable_startups(
        self,
        target_startup_id: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        weights: Optional[SimilarityWeights] = None,
        db: Optional[Session] = None
    ) -> List[ComparableStartup]:
        """Find the most comparable startups to a target startup."""
        
        if db is None:
            db = next(get_db())
        
        if weights:
            self.weights = weights
        
        logger.info(f"Finding comparable startups for {target_startup_id}")
        
        # Get target startup
        target_startup = db.query(Startup).filter(Startup.id == target_startup_id).first()
        if not target_startup:
            raise ValueError(f"Target startup {target_startup_id} not found")
        
        # Get candidate startups (excluding target)
        candidates_query = db.query(Startup).filter(Startup.id != target_startup_id)
        
        # Apply filters
        if filters:
            candidates_query = self._apply_filters(candidates_query, filters)
        
        candidate_startups = candidates_query.all()
        
        if not candidate_startups:
            logger.warning("No candidate startups found")
            return []
        
        # Calculate similarities for all candidates
        comparable_startups = await self._calculate_similarities(
            target_startup, 
            candidate_startups, 
            db
        )
        
        # Sort by overall similarity and return top K
        comparable_startups.sort(key=lambda x: x.overall_similarity, reverse=True)
        
        logger.info(f"Found {len(comparable_startups[:top_k])} comparable startups")
        
        return comparable_startups[:top_k]
    
    async def find_comparable_by_criteria(
        self,
        criteria: Dict[str, Any],
        top_k: int = 10,
        db: Optional[Session] = None
    ) -> List[ComparableStartup]:
        """Find comparable startups based on specific criteria."""
        
        if db is None:
            db = next(get_db())
        
        logger.info(f"Finding startups matching criteria: {criteria}")
        
        # Build query based on criteria
        query = db.query(Startup)
        
        # Industry matching
        if 'industry' in criteria:
            industries = criteria['industry'] if isinstance(criteria['industry'], list) else [criteria['industry']]
            query = query.filter(Startup.industry.in_(industries))
        
        # Stage matching
        if 'stage' in criteria:
            stages = criteria['stage'] if isinstance(criteria['stage'], list) else [criteria['stage']]
            query = query.filter(Startup.stage.in_(stages))
        
        # Geography matching
        if 'location' in criteria:
            locations = criteria['location'] if isinstance(criteria['location'], list) else [criteria['location']]
            query = query.filter(Startup.location.in_(locations))
        
        # Founded year range
        if 'founded_year_min' in criteria:
            query = query.filter(Startup.founded_year >= criteria['founded_year_min'])
        
        if 'founded_year_max' in criteria:
            query = query.filter(Startup.founded_year <= criteria['founded_year_max'])
        
        # Team size range
        if 'team_size_min' in criteria:
            query = query.filter(Startup.team_size >= criteria['team_size_min'])
        
        if 'team_size_max' in criteria:
            query = query.filter(Startup.team_size <= criteria['team_size_max'])
        
        # Funding range
        if 'funding_min' in criteria:
            query = query.filter(Startup.funding_amount >= criteria['funding_min'])
        
        if 'funding_max' in criteria:
            query = query.filter(Startup.funding_amount <= criteria['funding_max'])
        
        # Business model
        if 'business_model' in criteria:
            models = criteria['business_model'] if isinstance(criteria['business_model'], list) else [criteria['business_model']]
            query = query.filter(Startup.business_model.in_(models))
        
        matching_startups = query.limit(top_k * 2).all()  # Get more for ranking
        
        if not matching_startups:
            return []
        
        # Create comparable startup objects with basic similarity
        comparable_startups = []
        for startup in matching_startups:
            similarity_score = self._calculate_criteria_match_score(startup, criteria)
            
            comparable_startup = ComparableStartup(
                startup_id=startup.id,
                company_name=startup.company_name,
                industry=startup.industry,
                stage=startup.stage,
                location=startup.location,
                founded_year=startup.founded_year,
                overall_similarity=similarity_score,
                industry_similarity=1.0 if startup.industry in criteria.get('industry', []) else 0.0,
                stage_similarity=1.0 if startup.stage in criteria.get('stage', []) else 0.0,
                geography_similarity=1.0 if startup.location in criteria.get('location', []) else 0.0,
                team_similarity=0.5,  # Default
                business_model_similarity=1.0 if startup.business_model in criteria.get('business_model', []) else 0.0,
                embedding_similarity=0.5,  # Default
                performance_score=None,
                funding_amount=startup.funding_amount,
                employee_count=startup.team_size,
                similarity_reasons=self._generate_criteria_reasons(startup, criteria),
                key_differences=[]
            )
            
            comparable_startups.append(comparable_startup)
        
        # Sort and return top K
        comparable_startups.sort(key=lambda x: x.overall_similarity, reverse=True)
        return comparable_startups[:top_k]
    
    async def _calculate_similarities(
        self,
        target_startup: Startup,
        candidate_startups: List[Startup],
        db: Session
    ) -> List[ComparableStartup]:
        """Calculate comprehensive similarity scores for candidate startups."""
        
        comparable_startups = []
        
        # Get target startup features and embeddings
        target_features = await self._get_startup_features(target_startup, db)
        target_embedding = await self._get_startup_embedding(target_startup, db)
        
        for candidate in candidate_startups:
            try:
                # Calculate individual similarity dimensions
                similarities = await self._calculate_all_similarity_dimensions(
                    target_startup, 
                    candidate, 
                    target_features, 
                    target_embedding, 
                    db
                )
                
                # Calculate overall weighted similarity
                overall_similarity = (
                    similarities['industry'] * self.weights.industry_weight +
                    similarities['stage'] * self.weights.stage_weight +
                    similarities['geography'] * self.weights.geography_weight +
                    similarities['team'] * self.weights.team_size_weight +
                    similarities['business_model'] * self.weights.business_model_weight +
                    similarities['embedding'] * self.weights.embedding_weight
                )
                
                # Get performance score if available
                performance_score = await self._get_performance_score(candidate.id, db)
                
                # Generate insights
                similarity_reasons = self._generate_similarity_reasons(target_startup, candidate, similarities)
                key_differences = self._generate_key_differences(target_startup, candidate)
                
                comparable_startup = ComparableStartup(
                    startup_id=candidate.id,
                    company_name=candidate.company_name,
                    industry=candidate.industry,
                    stage=candidate.stage,
                    location=candidate.location,
                    founded_year=candidate.founded_year,
                    overall_similarity=overall_similarity,
                    industry_similarity=similarities['industry'],
                    stage_similarity=similarities['stage'],
                    geography_similarity=similarities['geography'],
                    team_similarity=similarities['team'],
                    business_model_similarity=similarities['business_model'],
                    embedding_similarity=similarities['embedding'],
                    performance_score=performance_score,
                    funding_amount=candidate.funding_amount,
                    employee_count=candidate.team_size,
                    similarity_reasons=similarity_reasons,
                    key_differences=key_differences
                )
                
                comparable_startups.append(comparable_startup)
                
            except Exception as e:
                logger.error(f"Error calculating similarity for startup {candidate.id}: {e}")
                continue
        
        return comparable_startups
    
    async def _calculate_all_similarity_dimensions(
        self,
        target: Startup,
        candidate: Startup,
        target_features: Dict[str, Any],
        target_embedding: Optional[np.ndarray],
        db: Session
    ) -> Dict[str, float]:
        """Calculate all similarity dimensions between target and candidate."""
        
        similarities = {}
        
        # Industry similarity
        similarities['industry'] = self._calculate_industry_similarity(target, candidate)
        
        # Stage similarity
        similarities['stage'] = self._calculate_stage_similarity(target, candidate)
        
        # Geography similarity
        similarities['geography'] = self._calculate_geography_similarity(target, candidate)
        
        # Team similarity
        similarities['team'] = self._calculate_team_similarity(target, candidate)
        
        # Business model similarity
        similarities['business_model'] = self._calculate_business_model_similarity(target, candidate)
        
        # Embedding similarity
        similarities['embedding'] = await self._calculate_embedding_similarity(
            target, candidate, target_embedding, db
        )
        
        return similarities
    
    def _calculate_industry_similarity(self, target: Startup, candidate: Startup) -> float:
        """Calculate industry similarity score."""
        if target.industry == candidate.industry:
            return 1.0
        
        # Industry hierarchy mapping for partial similarity
        industry_groups = {
            'fintech': ['financial_services', 'payments', 'banking', 'insurance'],
            'healthtech': ['healthcare', 'medtech', 'biotech', 'pharmaceuticals'],
            'edtech': ['education', 'e_learning', 'training'],
            'ecommerce': ['retail', 'marketplace', 'consumer_goods'],
            'enterprise': ['b2b_software', 'saas', 'productivity', 'automation'],
            'mobility': ['transportation', 'automotive', 'logistics'],
            'energy': ['clean_energy', 'renewable', 'sustainability']
        }
        
        # Find groups for both industries
        target_group = None
        candidate_group = None
        
        for group, industries in industry_groups.items():
            if target.industry.lower() in industries:
                target_group = group
            if candidate.industry.lower() in industries:
                candidate_group = group
        
        # Partial similarity if in same group
        if target_group and target_group == candidate_group:
            return 0.7
        
        return 0.0
    
    def _calculate_stage_similarity(self, target: Startup, candidate: Startup) -> float:
        """Calculate stage similarity score."""
        if target.stage == candidate.stage:
            return 1.0
        
        # Stage progression mapping
        stage_order = {
            'idea': 0,
            'prototype': 1,
            'mvp': 2,
            'early_traction': 3,
            'growth': 4,
            'scaling': 5,
            'mature': 6
        }
        
        target_idx = stage_order.get(target.stage.lower(), -1)
        candidate_idx = stage_order.get(candidate.stage.lower(), -1)
        
        if target_idx == -1 or candidate_idx == -1:
            return 0.0
        
        # Similarity decreases with stage distance
        distance = abs(target_idx - candidate_idx)
        
        if distance == 1:
            return 0.8
        elif distance == 2:
            return 0.6
        elif distance == 3:
            return 0.3
        else:
            return 0.1
    
    def _calculate_geography_similarity(self, target: Startup, candidate: Startup) -> float:
        """Calculate geography similarity score."""
        if target.location == candidate.location:
            return 1.0
        
        # Geography hierarchy (city -> state/region -> country)
        location_hierarchy = {
            'san_francisco': {'region': 'california', 'country': 'usa'},
            'palo_alto': {'region': 'california', 'country': 'usa'},
            'new_york': {'region': 'new_york', 'country': 'usa'},
            'london': {'region': 'england', 'country': 'uk'},
            'berlin': {'region': 'germany', 'country': 'germany'},
            'paris': {'region': 'france', 'country': 'france'},
            'toronto': {'region': 'ontario', 'country': 'canada'},
            'singapore': {'region': 'singapore', 'country': 'singapore'},
            'bangalore': {'region': 'karnataka', 'country': 'india'},
            'mumbai': {'region': 'maharashtra', 'country': 'india'}
        }
        
        target_info = location_hierarchy.get(target.location.lower(), {})
        candidate_info = location_hierarchy.get(candidate.location.lower(), {})
        
        # Same region
        if target_info.get('region') == candidate_info.get('region'):
            return 0.8
        
        # Same country
        if target_info.get('country') == candidate_info.get('country'):
            return 0.6
        
        return 0.2
    
    def _calculate_team_similarity(self, target: Startup, candidate: Startup) -> float:
        """Calculate team size similarity score."""
        if not target.team_size or not candidate.team_size:
            return 0.5  # Default if data missing
        
        # Team size buckets
        def get_team_bucket(size):
            if size <= 5:
                return 'micro'
            elif size <= 15:
                return 'small'
            elif size <= 50:
                return 'medium'
            elif size <= 200:
                return 'large'
            else:
                return 'enterprise'
        
        target_bucket = get_team_bucket(target.team_size)
        candidate_bucket = get_team_bucket(candidate.team_size)
        
        if target_bucket == candidate_bucket:
            return 1.0
        
        # Adjacent buckets have partial similarity
        bucket_order = ['micro', 'small', 'medium', 'large', 'enterprise']
        target_idx = bucket_order.index(target_bucket)
        candidate_idx = bucket_order.index(candidate_bucket)
        
        distance = abs(target_idx - candidate_idx)
        
        if distance == 1:
            return 0.7
        elif distance == 2:
            return 0.4
        else:
            return 0.1
    
    def _calculate_business_model_similarity(self, target: Startup, candidate: Startup) -> float:
        """Calculate business model similarity score."""
        if target.business_model == candidate.business_model:
            return 1.0
        
        # Business model compatibility matrix
        model_groups = {
            'subscription': ['saas', 'subscription', 'recurring'],
            'transaction': ['marketplace', 'transaction', 'commission'],
            'advertising': ['freemium', 'advertising', 'media'],
            'enterprise': ['b2b', 'enterprise', 'licensing'],
            'product': ['product_sales', 'retail', 'hardware']
        }
        
        target_group = None
        candidate_group = None
        
        for group, models in model_groups.items():
            if target.business_model.lower() in models:
                target_group = group
            if candidate.business_model.lower() in models:
                candidate_group = group
        
        if target_group and target_group == candidate_group:
            return 0.8
        
        return 0.2
    
    async def _calculate_embedding_similarity(
        self,
        target: Startup,
        candidate: Startup,
        target_embedding: Optional[np.ndarray],
        db: Session
    ) -> float:
        """Calculate embedding-based similarity score."""
        
        try:
            if target_embedding is None:
                return 0.5  # Default if no embedding
            
            # Get candidate embedding
            candidate_embedding = await self._get_startup_embedding(candidate, db)
            
            if candidate_embedding is None:
                return 0.5
            
            # Calculate cosine similarity
            similarity = np.dot(target_embedding, candidate_embedding) / (
                np.linalg.norm(target_embedding) * np.linalg.norm(candidate_embedding)
            )
            
            # Normalize to 0-1 range
            return (similarity + 1) / 2
            
        except Exception as e:
            logger.error(f"Error calculating embedding similarity: {e}")
            return 0.5
    
    async def _get_startup_features(self, startup: Startup, db: Session) -> Dict[str, Any]:
        """Get engineered features for a startup."""
        try:
            return await self.feature_service.extract_comprehensive_features(startup.id, db)
        except Exception as e:
            logger.error(f"Error getting features for startup {startup.id}: {e}")
            return {}
    
    async def _get_startup_embedding(self, startup: Startup, db: Session) -> Optional[np.ndarray]:
        """Get embedding for a startup."""
        try:
            # Create description for embedding
            description = f"{startup.company_name} is a {startup.stage} stage {startup.industry} startup"
            if startup.description:
                description += f" that {startup.description}"
            
            embedding = await self.embedding_service.get_embedding(description)
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding for startup {startup.id}: {e}")
            return None
    
    async def _get_performance_score(self, startup_id: str, db: Session) -> Optional[float]:
        """Get the latest performance score for a startup."""
        try:
            latest_analysis = db.query(Analysis).filter(
                Analysis.startup_id == startup_id
            ).order_by(Analysis.created_at.desc()).first()
            
            if latest_analysis and latest_analysis.final_score:
                return latest_analysis.final_score
            
            return None
        except Exception as e:
            logger.error(f"Error getting performance score for startup {startup_id}: {e}")
            return None
    
    def _generate_similarity_reasons(
        self, 
        target: Startup, 
        candidate: Startup, 
        similarities: Dict[str, float]
    ) -> List[str]:
        """Generate human-readable reasons for similarity."""
        reasons = []
        
        # High similarity reasons
        if similarities['industry'] >= 0.8:
            if target.industry == candidate.industry:
                reasons.append(f"Same industry ({target.industry})")
            else:
                reasons.append(f"Related industries ({target.industry} / {candidate.industry})")
        
        if similarities['stage'] >= 0.8:
            if target.stage == candidate.stage:
                reasons.append(f"Same development stage ({target.stage})")
            else:
                reasons.append(f"Similar development stages")
        
        if similarities['geography'] >= 0.8:
            if target.location == candidate.location:
                reasons.append(f"Same location ({target.location})")
            else:
                reasons.append(f"Same region")
        
        if similarities['team'] >= 0.8:
            reasons.append(f"Similar team size")
        
        if similarities['business_model'] >= 0.8:
            if target.business_model == candidate.business_model:
                reasons.append(f"Same business model ({target.business_model})")
            else:
                reasons.append(f"Compatible business models")
        
        if similarities['embedding'] >= 0.8:
            reasons.append("Similar business concepts and approach")
        
        return reasons
    
    def _generate_key_differences(self, target: Startup, candidate: Startup) -> List[str]:
        """Generate key differences between startups."""
        differences = []
        
        if target.industry != candidate.industry:
            differences.append(f"Industry: {target.industry} vs {candidate.industry}")
        
        if target.stage != candidate.stage:
            differences.append(f"Stage: {target.stage} vs {candidate.stage}")
        
        if target.location != candidate.location:
            differences.append(f"Location: {target.location} vs {candidate.location}")
        
        if target.business_model != candidate.business_model:
            differences.append(f"Business model: {target.business_model} vs {candidate.business_model}")
        
        if target.team_size and candidate.team_size:
            size_diff = abs(target.team_size - candidate.team_size)
            if size_diff > 10:
                differences.append(f"Team size: {target.team_size} vs {candidate.team_size}")
        
        if target.funding_amount and candidate.funding_amount:
            funding_ratio = max(target.funding_amount, candidate.funding_amount) / min(target.funding_amount, candidate.funding_amount)
            if funding_ratio > 2:
                differences.append(f"Funding: ${target.funding_amount:,.0f} vs ${candidate.funding_amount:,.0f}")
        
        return differences
    
    def _calculate_criteria_match_score(self, startup: Startup, criteria: Dict[str, Any]) -> float:
        """Calculate how well a startup matches given criteria."""
        score = 0.0
        total_criteria = 0
        
        # Industry match
        if 'industry' in criteria:
            total_criteria += 1
            industries = criteria['industry'] if isinstance(criteria['industry'], list) else [criteria['industry']]
            if startup.industry in industries:
                score += 1.0
        
        # Stage match
        if 'stage' in criteria:
            total_criteria += 1
            stages = criteria['stage'] if isinstance(criteria['stage'], list) else [criteria['stage']]
            if startup.stage in stages:
                score += 1.0
        
        # Location match
        if 'location' in criteria:
            total_criteria += 1
            locations = criteria['location'] if isinstance(criteria['location'], list) else [criteria['location']]
            if startup.location in locations:
                score += 1.0
        
        # Business model match
        if 'business_model' in criteria:
            total_criteria += 1
            models = criteria['business_model'] if isinstance(criteria['business_model'], list) else [criteria['business_model']]
            if startup.business_model in models:
                score += 1.0
        
        # Founded year range
        if 'founded_year_min' in criteria or 'founded_year_max' in criteria:
            total_criteria += 1
            min_year = criteria.get('founded_year_min', 1900)
            max_year = criteria.get('founded_year_max', 2100)
            if startup.founded_year and min_year <= startup.founded_year <= max_year:
                score += 1.0
        
        return score / total_criteria if total_criteria > 0 else 0.0
    
    def _generate_criteria_reasons(self, startup: Startup, criteria: Dict[str, Any]) -> List[str]:
        """Generate reasons why a startup matches criteria."""
        reasons = []
        
        if 'industry' in criteria:
            industries = criteria['industry'] if isinstance(criteria['industry'], list) else [criteria['industry']]
            if startup.industry in industries:
                reasons.append(f"Matches industry: {startup.industry}")
        
        if 'stage' in criteria:
            stages = criteria['stage'] if isinstance(criteria['stage'], list) else [criteria['stage']]
            if startup.stage in stages:
                reasons.append(f"Matches stage: {startup.stage}")
        
        if 'location' in criteria:
            locations = criteria['location'] if isinstance(criteria['location'], list) else [criteria['location']]
            if startup.location in locations:
                reasons.append(f"Matches location: {startup.location}")
        
        return reasons
    
    def _apply_filters(self, query, filters: Dict[str, Any]):
        """Apply filters to candidate query."""
        
        if 'min_funding' in filters:
            query = query.filter(Startup.funding_amount >= filters['min_funding'])
        
        if 'max_funding' in filters:
            query = query.filter(Startup.funding_amount <= filters['max_funding'])
        
        if 'min_team_size' in filters:
            query = query.filter(Startup.team_size >= filters['min_team_size'])
        
        if 'max_team_size' in filters:
            query = query.filter(Startup.team_size <= filters['max_team_size'])
        
        if 'industries' in filters:
            query = query.filter(Startup.industry.in_(filters['industries']))
        
        if 'stages' in filters:
            query = query.filter(Startup.stage.in_(filters['stages']))
        
        if 'locations' in filters:
            query = query.filter(Startup.location.in_(filters['locations']))
        
        if 'exclude_ids' in filters:
            query = query.filter(~Startup.id.in_(filters['exclude_ids']))
        
        return query


# Global comparable startups finder instance
comparable_finder = ComparableStartupsFinder()