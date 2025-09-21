"""
Embedding similarity engine for startup comparison and pattern matching.
"""
import logging
import asyncio
import pickle
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import json

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingSimilarityEngine:
    """Advanced embedding similarity system for startup analysis."""
    
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"  # Lightweight but effective
        self.model = None
        self.startup_embeddings = {}
        self.success_patterns = {}
        self.cache_dir = Path("embeddings_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Success indicators for pattern matching
        self.success_indicators = [
            "acquired by major company", "ipo", "unicorn status", "series c",
            "profitable", "rapid growth", "market leader", "viral adoption",
            "billion dollar valuation", "successful exit", "industry leader"
        ]
        
        # Market categories for similarity matching
        self.market_categories = [
            "fintech", "healthtech", "edtech", "proptech", "retailtech",
            "enterprise software", "consumer apps", "ai/ml", "blockchain",
            "cybersecurity", "automotive", "logistics", "media", "gaming"
        ]
    
    async def initialize_model(self):
        """Initialize the sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            try:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    None, lambda: SentenceTransformer(self.model_name)
                )
                logger.info("Sentence transformer model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load sentence transformer model: {e}")
                raise
    
    async def generate_startup_embedding(self, startup_data: Dict[str, Any]) -> np.ndarray:
        """Generate comprehensive embedding for startup."""
        await self.initialize_model()
        
        # Create comprehensive text representation
        text_representation = self._create_text_representation(startup_data)
        
        # Generate embedding
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, self.model.encode, text_representation
        )
        
        return embedding
    
    def _create_text_representation(self, startup_data: Dict[str, Any]) -> str:
        """Create comprehensive text representation for embedding."""
        text_parts = []
        
        # Basic information
        name = startup_data.get('name', '')
        description = startup_data.get('description', '')
        industry = startup_data.get('industry', '')
        stage = startup_data.get('stage', '')
        
        if name:
            text_parts.append(f"Company: {name}")
        if description:
            text_parts.append(f"Description: {description}")
        if industry:
            text_parts.append(f"Industry: {industry}")
        if stage:
            text_parts.append(f"Stage: {stage}")
        
        # Founder information
        founders = startup_data.get('founders', [])
        for founder in founders:
            if founder.get('background'):
                text_parts.append(f"Founder background: {founder['background']}")
            if founder.get('role'):
                text_parts.append(f"Founder role: {founder['role']}")
        
        # Analysis insights
        analysis = startup_data.get('analysis', {})
        
        # Product analysis
        product_analysis = analysis.get('product_analysis', {})
        if product_analysis.get('description'):
            text_parts.append(f"Product: {product_analysis['description']}")
        if product_analysis.get('strengths'):
            text_parts.append(f"Strengths: {' '.join(product_analysis['strengths'])}")
        
        # Market analysis
        market_analysis = analysis.get('market_analysis', {})
        if market_analysis.get('market_size'):
            text_parts.append(f"Market: {market_analysis['market_size']}")
        if market_analysis.get('competition'):
            text_parts.append(f"Competition: {market_analysis['competition']}")
        
        # Team analysis
        team_analysis = analysis.get('team_analysis', {})
        if team_analysis.get('strengths'):
            text_parts.append(f"Team strengths: {' '.join(team_analysis['strengths'])}")
        
        # Traction analysis
        traction_analysis = analysis.get('traction_analysis', {})
        if traction_analysis.get('metrics'):
            text_parts.append(f"Traction: {traction_analysis['metrics']}")
        
        # Website data
        website_data = startup_data.get('website_data', {})
        if website_data.get('product_features'):
            features = ' '.join(website_data['product_features'][:5])  # Limit features
            text_parts.append(f"Features: {features}")
        
        if website_data.get('tech_stack'):
            tech_stack = ' '.join(website_data['tech_stack'])
            text_parts.append(f"Technology: {tech_stack}")
        
        # GitHub data
        github_data = startup_data.get('github_data', {})
        if github_data.get('success'):
            repo_data = github_data.get('data', {})
            languages = repo_data.get('languages', {})
            if languages.get('all_languages'):
                langs = ' '.join(languages['all_languages'])
                text_parts.append(f"Programming languages: {langs}")
        
        return ' '.join(text_parts)
    
    async def find_similar_startups(
        self, 
        startup_data: Dict[str, Any], 
        comparison_database: List[Dict[str, Any]], 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Find most similar startups from comparison database."""
        await self.initialize_model()
        
        # Generate embedding for target startup
        target_embedding = await self.generate_startup_embedding(startup_data)
        
        # Generate embeddings for comparison startups (with caching)
        comparison_embeddings = []
        comparison_metadata = []
        
        for comp_startup in comparison_database:
            comp_id = self._generate_startup_id(comp_startup)
            
            # Check cache first
            cached_embedding = await self._load_cached_embedding(comp_id)
            if cached_embedding is not None:
                embedding = cached_embedding
            else:
                embedding = await self.generate_startup_embedding(comp_startup)
                await self._cache_embedding(comp_id, embedding)
            
            comparison_embeddings.append(embedding)
            comparison_metadata.append({
                'startup_data': comp_startup,
                'startup_id': comp_id
            })
        
        # Calculate similarities
        if not comparison_embeddings:
            return []
        
        comparison_matrix = np.array(comparison_embeddings)
        similarities = cosine_similarity([target_embedding], comparison_matrix)[0]
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity_score = float(similarities[idx])
            startup_info = comparison_metadata[idx]['startup_data']
            
            results.append({
                'similarity_score': similarity_score,
                'startup': startup_info,
                'match_reasons': self._analyze_similarity_reasons(
                    startup_data, startup_info, similarity_score
                )
            })
        
        return results
    
    def _generate_startup_id(self, startup_data: Dict[str, Any]) -> str:
        """Generate unique ID for startup for caching."""
        # Create hash from key startup attributes
        key_attrs = [
            startup_data.get('name', ''),
            startup_data.get('description', ''),
            startup_data.get('industry', ''),
            str(startup_data.get('founders', []))
        ]
        
        content = ''.join(key_attrs)
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _load_cached_embedding(self, startup_id: str) -> Optional[np.ndarray]:
        """Load cached embedding if available."""
        cache_file = self.cache_dir / f"{startup_id}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached embedding for {startup_id}: {e}")
        
        return None
    
    async def _cache_embedding(self, startup_id: str, embedding: np.ndarray):
        """Cache embedding for future use."""
        cache_file = self.cache_dir / f"{startup_id}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding for {startup_id}: {e}")
    
    def _analyze_similarity_reasons(
        self, 
        target_startup: Dict[str, Any], 
        similar_startup: Dict[str, Any], 
        similarity_score: float
    ) -> List[str]:
        """Analyze why two startups are similar."""
        reasons = []
        
        # Industry similarity
        target_industry = target_startup.get('industry', '').lower()
        similar_industry = similar_startup.get('industry', '').lower()
        
        if target_industry and similar_industry:
            if target_industry == similar_industry:
                reasons.append(f"Same industry: {target_industry}")
            elif any(cat in target_industry and cat in similar_industry 
                    for cat in self.market_categories):
                reasons.append("Similar market category")
        
        # Stage similarity
        target_stage = target_startup.get('stage', '').lower()
        similar_stage = similar_startup.get('stage', '').lower()
        
        if target_stage and similar_stage and target_stage == similar_stage:
            reasons.append(f"Same stage: {target_stage}")
        
        # Technology similarity
        target_tech = self._extract_tech_keywords(target_startup)
        similar_tech = self._extract_tech_keywords(similar_startup)
        
        common_tech = set(target_tech) & set(similar_tech)
        if common_tech:
            reasons.append(f"Similar technology: {', '.join(list(common_tech)[:3])}")
        
        # Business model similarity
        target_model = self._extract_business_model(target_startup)
        similar_model = self._extract_business_model(similar_startup)
        
        if target_model and similar_model and target_model == similar_model:
            reasons.append(f"Similar business model: {target_model}")
        
        # Team background similarity
        target_backgrounds = self._extract_founder_backgrounds(target_startup)
        similar_backgrounds = self._extract_founder_backgrounds(similar_startup)
        
        common_backgrounds = set(target_backgrounds) & set(similar_backgrounds)
        if common_backgrounds:
            reasons.append(f"Similar founder backgrounds: {', '.join(list(common_backgrounds)[:2])}")
        
        # High-level similarity score
        if similarity_score > 0.8:
            reasons.append("Very high overall similarity")
        elif similarity_score > 0.6:
            reasons.append("High overall similarity")
        elif similarity_score > 0.4:
            reasons.append("Moderate overall similarity")
        
        return reasons[:5]  # Limit to top 5 reasons
    
    def _extract_tech_keywords(self, startup_data: Dict[str, Any]) -> List[str]:
        """Extract technology keywords from startup data."""
        tech_keywords = []
        
        # From website tech stack
        website_data = startup_data.get('website_data', {})
        tech_stack = website_data.get('tech_stack', [])
        tech_keywords.extend([tech.lower() for tech in tech_stack])
        
        # From GitHub languages
        github_data = startup_data.get('github_data', {})
        if github_data.get('success'):
            languages = github_data.get('data', {}).get('languages', {})
            if languages.get('all_languages'):
                tech_keywords.extend([lang.lower() for lang in languages['all_languages']])
        
        # From description
        description = startup_data.get('description', '').lower()
        common_tech = [
            'ai', 'ml', 'blockchain', 'mobile', 'web', 'cloud',
            'saas', 'api', 'iot', 'analytics', 'automation'
        ]
        
        tech_keywords.extend([tech for tech in common_tech if tech in description])
        
        return list(set(tech_keywords))  # Remove duplicates
    
    def _extract_business_model(self, startup_data: Dict[str, Any]) -> Optional[str]:
        """Extract business model from startup data."""
        # Check pricing info
        website_data = startup_data.get('website_data', {})
        pricing_info = website_data.get('pricing_info', {})
        pricing_model = pricing_info.get('pricing_model')
        
        if pricing_model:
            return pricing_model
        
        # Infer from description
        description = startup_data.get('description', '').lower()
        
        if any(word in description for word in ['subscription', 'monthly', 'saas']):
            return 'subscription'
        elif any(word in description for word in ['marketplace', 'commission', 'transaction']):
            return 'marketplace'
        elif any(word in description for word in ['advertising', 'ads', 'sponsored']):
            return 'advertising'
        elif any(word in description for word in ['freemium', 'premium', 'upgrade']):
            return 'freemium'
        
        return None
    
    def _extract_founder_backgrounds(self, startup_data: Dict[str, Any]) -> List[str]:
        """Extract founder background keywords."""
        backgrounds = []
        
        founders = startup_data.get('founders', [])
        for founder in founders:
            background = founder.get('background', '').lower()
            
            # Extract company backgrounds
            if 'google' in background:
                backgrounds.append('google')
            elif 'facebook' in background or 'meta' in background:
                backgrounds.append('meta')
            elif 'amazon' in background:
                backgrounds.append('amazon')
            elif 'microsoft' in background:
                backgrounds.append('microsoft')
            elif 'apple' in background:
                backgrounds.append('apple')
            elif 'startup' in background or 'founder' in background:
                backgrounds.append('startup_experience')
            elif 'consulting' in background:
                backgrounds.append('consulting')
            elif 'finance' in background or 'investment' in background:
                backgrounds.append('finance')
            elif 'engineering' in background or 'technical' in background:
                backgrounds.append('technical')
        
        return list(set(backgrounds))
    
    async def identify_success_patterns(
        self, 
        successful_startups: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Identify patterns in successful startups."""
        await self.initialize_model()
        
        logger.info(f"Analyzing patterns in {len(successful_startups)} successful startups")
        
        # Generate embeddings for successful startups
        embeddings = []
        startup_features = []
        
        for startup in successful_startups:
            embedding = await self.generate_startup_embedding(startup)
            embeddings.append(embedding)
            
            # Extract categorical features
            features = {
                'industry': startup.get('industry', '').lower(),
                'stage_at_success': startup.get('stage', '').lower(),
                'tech_stack': self._extract_tech_keywords(startup),
                'business_model': self._extract_business_model(startup),
                'founder_backgrounds': self._extract_founder_backgrounds(startup)
            }
            startup_features.append(features)
        
        if not embeddings:
            return {'patterns': [], 'clusters': []}
        
        # Cluster successful startups to find patterns
        embeddings_array = np.array(embeddings)
        
        # Determine optimal number of clusters (max 10)
        n_clusters = min(10, max(2, len(embeddings) // 5))
        
        loop = asyncio.get_event_loop()
        kmeans = await loop.run_in_executor(
            None, lambda: KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings_array)
        )
        
        # Analyze clusters to identify patterns
        patterns = []
        clusters = []
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
            cluster_startups = [successful_startups[i] for i in cluster_indices]
            cluster_features = [startup_features[i] for i in cluster_indices]
            
            # Analyze common patterns in this cluster
            pattern = self._analyze_cluster_patterns(cluster_startups, cluster_features)
            pattern['cluster_id'] = cluster_id
            pattern['startup_count'] = len(cluster_startups)
            
            patterns.append(pattern)
            
            # Store cluster information
            clusters.append({
                'cluster_id': cluster_id,
                'center': kmeans.cluster_centers_[cluster_id].tolist(),
                'startup_count': len(cluster_startups),
                'representative_startups': [
                    startup.get('name', 'Unknown') for startup in cluster_startups[:3]
                ]
            })
        
        # Store patterns for future use
        self.success_patterns = {
            'patterns': patterns,
            'clusters': clusters,
            'model_centers': kmeans.cluster_centers_
        }
        
        return self.success_patterns
    
    def _analyze_cluster_patterns(
        self, 
        cluster_startups: List[Dict[str, Any]], 
        cluster_features: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze patterns within a cluster of successful startups."""
        pattern = {
            'common_industries': {},
            'common_stages': {},
            'common_technologies': {},
            'common_business_models': {},
            'common_founder_backgrounds': {},
            'pattern_description': ''
        }
        
        # Count frequencies
        for features in cluster_features:
            # Industries
            industry = features['industry']
            if industry:
                pattern['common_industries'][industry] = pattern['common_industries'].get(industry, 0) + 1
            
            # Stages
            stage = features['stage_at_success']
            if stage:
                pattern['common_stages'][stage] = pattern['common_stages'].get(stage, 0) + 1
            
            # Technologies
            for tech in features['tech_stack']:
                pattern['common_technologies'][tech] = pattern['common_technologies'].get(tech, 0) + 1
            
            # Business models
            model = features['business_model']
            if model:
                pattern['common_business_models'][model] = pattern['common_business_models'].get(model, 0) + 1
            
            # Founder backgrounds
            for bg in features['founder_backgrounds']:
                pattern['common_founder_backgrounds'][bg] = pattern['common_founder_backgrounds'].get(bg, 0) + 1
        
        # Generate pattern description
        description_parts = []
        
        # Most common industry
        if pattern['common_industries']:
            top_industry = max(pattern['common_industries'], key=pattern['common_industries'].get)
            description_parts.append(f"{top_industry} companies")
        
        # Most common technology
        if pattern['common_technologies']:
            top_tech = max(pattern['common_technologies'], key=pattern['common_technologies'].get)
            if pattern['common_technologies'][top_tech] > 1:
                description_parts.append(f"using {top_tech}")
        
        # Most common business model
        if pattern['common_business_models']:
            top_model = max(pattern['common_business_models'], key=pattern['common_business_models'].get)
            if pattern['common_business_models'][top_model] > 1:
                description_parts.append(f"with {top_model} model")
        
        # Most common founder background
        if pattern['common_founder_backgrounds']:
            top_bg = max(pattern['common_founder_backgrounds'], key=pattern['common_founder_backgrounds'].get)
            if pattern['common_founder_backgrounds'][top_bg] > 1:
                description_parts.append(f"founded by {top_bg} professionals")
        
        pattern['pattern_description'] = ', '.join(description_parts) if description_parts else 'Diverse pattern'
        
        return pattern
    
    async def match_to_success_patterns(
        self, 
        startup_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Match startup to identified success patterns."""
        if not self.success_patterns:
            logger.warning("No success patterns available. Run identify_success_patterns first.")
            return []
        
        await self.initialize_model()
        
        # Generate embedding for target startup
        startup_embedding = await self.generate_startup_embedding(startup_data)
        
        # Calculate similarity to each pattern cluster
        pattern_matches = []
        
        for i, cluster_center in enumerate(self.success_patterns['model_centers']):
            similarity = cosine_similarity([startup_embedding], [cluster_center])[0][0]
            
            # Get corresponding pattern
            pattern = self.success_patterns['patterns'][i]
            
            match = {
                'pattern_id': i,
                'similarity_score': float(similarity),
                'pattern_description': pattern['pattern_description'],
                'startup_count': pattern['startup_count'],
                'match_strength': self._calculate_match_strength(startup_data, pattern),
                'recommendations': self._generate_pattern_recommendations(startup_data, pattern)
            }
            
            pattern_matches.append(match)
        
        # Sort by similarity score
        pattern_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return pattern_matches[:5]  # Return top 5 matches
    
    def _calculate_match_strength(self, startup_data: Dict[str, Any], pattern: Dict[str, Any]) -> float:
        """Calculate how well startup matches pattern (0-100)."""
        match_score = 0.0
        total_weight = 0.0
        
        # Industry match
        startup_industry = startup_data.get('industry', '').lower()
        if startup_industry and startup_industry in pattern['common_industries']:
            frequency = pattern['common_industries'][startup_industry]
            weight = 25.0
            match_score += weight * (frequency / pattern.get('startup_count', 1))
            total_weight += weight
        
        # Technology match
        startup_tech = set(self._extract_tech_keywords(startup_data))
        pattern_tech = set(pattern['common_technologies'].keys())
        tech_overlap = len(startup_tech & pattern_tech)
        
        if tech_overlap > 0:
            weight = 20.0
            match_score += weight * (tech_overlap / max(len(pattern_tech), 1))
            total_weight += weight
        
        # Business model match
        startup_model = self._extract_business_model(startup_data)
        if startup_model and startup_model in pattern['common_business_models']:
            frequency = pattern['common_business_models'][startup_model]
            weight = 15.0
            match_score += weight * (frequency / pattern.get('startup_count', 1))
            total_weight += weight
        
        # Founder background match
        startup_backgrounds = set(self._extract_founder_backgrounds(startup_data))
        pattern_backgrounds = set(pattern['common_founder_backgrounds'].keys())
        bg_overlap = len(startup_backgrounds & pattern_backgrounds)
        
        if bg_overlap > 0:
            weight = 15.0
            match_score += weight * (bg_overlap / max(len(pattern_backgrounds), 1))
            total_weight += weight
        
        # Stage match
        startup_stage = startup_data.get('stage', '').lower()
        if startup_stage and startup_stage in pattern['common_stages']:
            frequency = pattern['common_stages'][startup_stage]
            weight = 10.0
            match_score += weight * (frequency / pattern.get('startup_count', 1))
            total_weight += weight
        
        return (match_score / max(total_weight, 1)) * 100.0 if total_weight > 0 else 0.0
    
    def _generate_pattern_recommendations(
        self, 
        startup_data: Dict[str, Any], 
        pattern: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on pattern matching."""
        recommendations = []
        
        # Industry recommendations
        startup_industry = startup_data.get('industry', '').lower()
        top_industries = sorted(pattern['common_industries'].items(), key=lambda x: x[1], reverse=True)
        
        if startup_industry not in [industry for industry, _ in top_industries[:3]]:
            if top_industries:
                top_industry = top_industries[0][0]
                recommendations.append(f"Consider positioning closer to {top_industry} market")
        
        # Technology recommendations
        startup_tech = set(self._extract_tech_keywords(startup_data))
        top_technologies = sorted(pattern['common_technologies'].items(), key=lambda x: x[1], reverse=True)
        
        missing_important_tech = [tech for tech, count in top_technologies[:3] 
                                if tech not in startup_tech and count > 1]
        
        if missing_important_tech:
            recommendations.append(f"Consider adopting {missing_important_tech[0]} technology")
        
        # Business model recommendations
        startup_model = self._extract_business_model(startup_data)
        top_models = sorted(pattern['common_business_models'].items(), key=lambda x: x[1], reverse=True)
        
        if startup_model not in [model for model, _ in top_models[:2]] and top_models:
            top_model = top_models[0][0]
            recommendations.append(f"Explore {top_model} business model")
        
        # Founder background recommendations
        startup_backgrounds = set(self._extract_founder_backgrounds(startup_data))
        top_backgrounds = sorted(pattern['common_founder_backgrounds'].items(), key=lambda x: x[1], reverse=True)
        
        valuable_backgrounds = [bg for bg, count in top_backgrounds[:3] 
                              if bg not in startup_backgrounds and count > 1]
        
        if valuable_backgrounds:
            recommendations.append(f"Consider adding advisor/team member with {valuable_backgrounds[0]} background")
        
        return recommendations[:4]  # Limit to 4 recommendations
    
    async def calculate_market_fit_score(self, startup_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate market fit score based on similarity to successful patterns."""
        pattern_matches = await self.match_to_success_patterns(startup_data)
        
        if not pattern_matches:
            return {
                'market_fit_score': 50.0,  # Neutral score
                'confidence': 0.0,
                'explanation': 'No success patterns available for comparison'
            }
        
        # Calculate weighted average of pattern matches
        total_score = 0.0
        total_weight = 0.0
        
        for match in pattern_matches:
            # Weight by both similarity and pattern startup count
            weight = match['similarity_score'] * math.log(match['startup_count'] + 1)
            total_score += match['match_strength'] * weight
            total_weight += weight
        
        market_fit_score = total_score / total_weight if total_weight > 0 else 50.0
        
        # Calculate confidence based on best pattern match
        best_match = pattern_matches[0]
        confidence = best_match['similarity_score'] * (best_match['startup_count'] / 10.0)
        confidence = min(confidence, 1.0)
        
        # Generate explanation
        explanation = f"Based on similarity to {len(pattern_matches)} success patterns. "
        explanation += f"Best match: {best_match['pattern_description']} "
        explanation += f"(similarity: {best_match['similarity_score']:.2f})"
        
        return {
            'market_fit_score': market_fit_score,
            'confidence': confidence,
            'explanation': explanation,
            'top_pattern_matches': pattern_matches[:3]
        }