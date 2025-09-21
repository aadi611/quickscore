"""
Advanced feature engineering for startup analysis and ML modeling.
"""
import logging
import re
import math
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class StartupFeatures:
    """Container for engineered startup features."""
    
    # Team Features
    team_size: int = 0
    founder_experience_score: float = 0.0
    technical_founder_present: bool = False
    previous_startup_experience: int = 0
    education_prestige_score: float = 0.0
    linkedin_network_strength: float = 0.0
    
    # Market Features
    market_size_score: float = 0.0
    market_growth_rate: float = 0.0
    competitive_landscape_score: float = 0.0
    market_timing_score: float = 0.0
    
    # Product Features
    product_complexity_score: float = 0.0
    technical_innovation_score: float = 0.0
    user_experience_score: float = 0.0
    scalability_score: float = 0.0
    
    # Traction Features
    user_growth_rate: float = 0.0
    revenue_growth_rate: float = 0.0
    engagement_metrics_score: float = 0.0
    customer_retention_score: float = 0.0
    
    # Business Model Features
    revenue_model_strength: float = 0.0
    unit_economics_score: float = 0.0
    moat_strength_score: float = 0.0
    
    # Technical Features
    github_activity_score: float = 0.0
    tech_stack_modernity: float = 0.0
    code_quality_score: float = 0.0
    
    # External Validation Features
    media_mention_score: float = 0.0
    investor_interest_score: float = 0.0
    customer_validation_score: float = 0.0
    
    # Risk Features
    regulatory_risk_score: float = 0.0
    technical_risk_score: float = 0.0
    market_risk_score: float = 0.0


class AdvancedFeatureEngine:
    """Advanced feature engineering pipeline for startup analysis."""
    
    def __init__(self):
        self.sentence_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_weights = self._initialize_feature_weights()
        
        # University rankings for education prestige scoring
        self.top_universities = {
            'stanford', 'harvard', 'mit', 'berkeley', 'caltech', 'princeton',
            'yale', 'chicago', 'penn', 'columbia', 'cornell', 'dartmouth',
            'brown', 'cmu', 'northwestern', 'duke', 'johns hopkins'
        }
        
        # Technical roles for founder analysis
        self.technical_roles = {
            'cto', 'chief technology officer', 'vp engineering', 'head of engineering',
            'principal engineer', 'staff engineer', 'architect', 'developer',
            'programmer', 'software engineer', 'data scientist', 'ml engineer'
        }
        
        # Competitive markets for risk assessment
        self.high_competition_markets = {
            'e-commerce', 'social media', 'food delivery', 'ride sharing',
            'messaging', 'productivity', 'note-taking', 'project management'
        }
    
    def _initialize_feature_weights(self) -> Dict[str, float]:
        """Initialize feature importance weights based on startup success factors."""
        return {
            'team_features': 0.35,
            'market_features': 0.25,
            'product_features': 0.20,
            'traction_features': 0.15,
            'business_model_features': 0.05
        }
    
    async def extract_startup_features(self, startup_data: Dict[str, Any]) -> StartupFeatures:
        """Extract comprehensive features from startup data."""
        logger.info(f"Extracting features for startup: {startup_data.get('name', 'Unknown')}")
        
        features = StartupFeatures()
        
        # Extract team features
        await self._extract_team_features(startup_data, features)
        
        # Extract market features
        await self._extract_market_features(startup_data, features)
        
        # Extract product features
        await self._extract_product_features(startup_data, features)
        
        # Extract traction features
        await self._extract_traction_features(startup_data, features)
        
        # Extract business model features
        await self._extract_business_model_features(startup_data, features)
        
        # Extract technical features
        await self._extract_technical_features(startup_data, features)
        
        # Extract external validation features
        await self._extract_external_validation_features(startup_data, features)
        
        # Extract risk features
        await self._extract_risk_features(startup_data, features)
        
        return features
    
    async def _extract_team_features(self, data: Dict[str, Any], features: StartupFeatures):
        """Extract team-related features."""
        founders = data.get('founders', [])
        linkedin_data = data.get('linkedin_profiles', [])
        website_team = data.get('website_data', {}).get('team_members', [])
        
        # Team size
        features.team_size = len(founders) + len(website_team)
        
        # Founder experience analysis
        total_experience = 0
        startup_experience = 0
        has_technical_founder = False
        education_scores = []
        network_scores = []
        
        for founder in founders:
            # Experience scoring
            experience_years = self._calculate_experience_years(founder.get('background', ''))
            total_experience += experience_years
            
            # Startup experience
            if self._has_startup_experience(founder.get('background', '')):
                startup_experience += 1
            
            # Technical founder check
            if self._is_technical_founder(founder.get('role', ''), founder.get('background', '')):
                has_technical_founder = True
            
            # Education prestige
            education_score = self._calculate_education_prestige(founder.get('education', []))
            education_scores.append(education_score)
        
        # LinkedIn network analysis
        for profile in linkedin_data:
            if profile.get('success'):
                network_score = self._calculate_network_strength(profile.get('data', {}))
                network_scores.append(network_score)
        
        # Set team features
        features.founder_experience_score = total_experience / max(len(founders), 1)
        features.technical_founder_present = has_technical_founder
        features.previous_startup_experience = startup_experience
        features.education_prestige_score = np.mean(education_scores) if education_scores else 0.0
        features.linkedin_network_strength = np.mean(network_scores) if network_scores else 0.0
    
    def _calculate_experience_years(self, background: str) -> float:
        """Calculate years of experience from background text."""
        # Look for experience patterns
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*at',
            r'(\d+)\+?\s*years?\s*in'
        ]
        
        max_years = 0
        for pattern in experience_patterns:
            matches = re.findall(pattern, background.lower())
            for match in matches:
                years = int(match)
                max_years = max(max_years, years)
        
        return min(max_years, 25)  # Cap at 25 years
    
    def _has_startup_experience(self, background: str) -> bool:
        """Check if founder has previous startup experience."""
        startup_indicators = [
            'founder', 'co-founder', 'startup', 'entrepreneur',
            'founded', 'started', 'launched', 'built'
        ]
        
        background_lower = background.lower()
        return any(indicator in background_lower for indicator in startup_indicators)
    
    def _is_technical_founder(self, role: str, background: str) -> bool:
        """Check if founder has technical background."""
        combined_text = f"{role} {background}".lower()
        
        return any(tech_role in combined_text for tech_role in self.technical_roles)
    
    def _calculate_education_prestige(self, education: List[Dict]) -> float:
        """Calculate education prestige score (0-100)."""
        if not education:
            return 0.0
        
        max_score = 0.0
        for edu in education:
            school = edu.get('school', '').lower()
            degree = edu.get('degree', '').lower()
            
            # Check for top universities
            for top_uni in self.top_universities:
                if top_uni in school:
                    score = 100.0
                    
                    # Bonus for advanced degrees
                    if any(adv_degree in degree for adv_degree in ['phd', 'mba', 'master']):
                        score += 10.0
                    
                    max_score = max(max_score, score)
                    break
            else:
                # Score for other universities
                if any(uni_indicator in school for uni_indicator in ['university', 'college', 'institute']):
                    score = 50.0
                    if any(adv_degree in degree for adv_degree in ['phd', 'mba', 'master']):
                        score += 15.0
                    max_score = max(max_score, score)
        
        return min(max_score, 100.0)
    
    def _calculate_network_strength(self, linkedin_data: Dict) -> float:
        """Calculate LinkedIn network strength score (0-100)."""
        connections = linkedin_data.get('connections', '0')
        
        # Parse connections count
        try:
            if '+' in connections:
                conn_count = int(connections.replace('+', ''))
            else:
                conn_count = int(connections)
        except:
            conn_count = 0
        
        # Score based on connections
        if conn_count >= 500:
            return 100.0
        elif conn_count >= 200:
            return 80.0
        elif conn_count >= 100:
            return 60.0
        elif conn_count >= 50:
            return 40.0
        elif conn_count > 0:
            return 20.0
        else:
            return 0.0
    
    async def _extract_market_features(self, data: Dict[str, Any], features: StartupFeatures):
        """Extract market-related features."""
        analysis = data.get('analysis', {})
        market_analysis = analysis.get('market_analysis', {})
        
        # Market size scoring
        market_size_text = market_analysis.get('market_size', '').lower()
        features.market_size_score = self._score_market_size(market_size_text)
        
        # Market growth rate
        features.market_growth_rate = self._extract_growth_rate(market_size_text)
        
        # Competitive landscape
        competition_text = market_analysis.get('competition', '').lower()
        features.competitive_landscape_score = self._score_competition(competition_text)
        
        # Market timing
        timing_indicators = self._analyze_market_timing(data)
        features.market_timing_score = timing_indicators
    
    def _score_market_size(self, market_text: str) -> float:
        """Score market size based on text analysis (0-100)."""
        # Look for market size indicators
        size_patterns = [
            (r'\$(\d+(?:\.\d+)?)\s*(?:billion|b)', 1000),  # Billions
            (r'\$(\d+(?:\.\d+)?)\s*(?:million|m)', 1),     # Millions
            (r'(\d+(?:\.\d+)?)\s*billion', 1000),          # Billions without $
            (r'(\d+(?:\.\d+)?)\s*million', 1)              # Millions without $
        ]
        
        max_size = 0
        for pattern, multiplier in size_patterns:
            matches = re.findall(pattern, market_text)
            for match in matches:
                size = float(match) * multiplier
                max_size = max(max_size, size)
        
        # Score based on market size in millions
        if max_size >= 10000:  # $10B+
            return 100.0
        elif max_size >= 5000:   # $5B+
            return 90.0
        elif max_size >= 1000:   # $1B+
            return 80.0
        elif max_size >= 500:    # $500M+
            return 70.0
        elif max_size >= 100:    # $100M+
            return 60.0
        elif max_size >= 50:     # $50M+
            return 40.0
        elif max_size > 0:       # Any size mentioned
            return 30.0
        else:
            return 0.0
    
    def _extract_growth_rate(self, market_text: str) -> float:
        """Extract market growth rate percentage."""
        growth_patterns = [
            r'(\d+(?:\.\d+)?)%\s*(?:growth|growing|cagr)',
            r'growing\s*(?:at\s*)?(\d+(?:\.\d+)?)%',
            r'(\d+(?:\.\d+)?)%\s*annual'
        ]
        
        max_growth = 0.0
        for pattern in growth_patterns:
            matches = re.findall(pattern, market_text)
            for match in matches:
                growth = float(match)
                max_growth = max(max_growth, growth)
        
        return min(max_growth, 100.0)  # Cap at 100%
    
    def _score_competition(self, competition_text: str) -> float:
        """Score competitive landscape (0-100, higher = less competitive)."""
        # Negative indicators (high competition)
        negative_indicators = [
            'saturated', 'crowded', 'many competitors', 'intense competition',
            'commoditized', 'price war', 'highly competitive'
        ]
        
        # Positive indicators (low competition)
        positive_indicators = [
            'blue ocean', 'first mover', 'unique', 'differentiated',
            'niche', 'underserved', 'untapped', 'emerging'
        ]
        
        negative_count = sum(1 for indicator in negative_indicators if indicator in competition_text)
        positive_count = sum(1 for indicator in positive_indicators if indicator in competition_text)
        
        # Calculate score
        base_score = 50.0
        base_score -= negative_count * 15  # Penalty for competition
        base_score += positive_count * 20  # Bonus for differentiation
        
        return max(0.0, min(100.0, base_score))
    
    def _analyze_market_timing(self, data: Dict[str, Any]) -> float:
        """Analyze market timing factors (0-100)."""
        # Check for timing indicators in description
        description = data.get('description', '').lower()
        
        # Positive timing indicators
        positive_timing = [
            'covid', 'pandemic', 'remote work', 'digital transformation',
            'ai boom', 'climate change', 'sustainability', 'regulation',
            'emerging', 'trend', 'growing demand'
        ]
        
        # Count timing indicators
        timing_score = 0.0
        for indicator in positive_timing:
            if indicator in description:
                timing_score += 15.0
        
        # Check company age for timing
        founded_year = self._extract_founded_year(data)
        if founded_year:
            current_year = datetime.now().year
            age = current_year - founded_year
            
            # Optimal timing is 1-3 years
            if 1 <= age <= 3:
                timing_score += 30.0
            elif age <= 5:
                timing_score += 20.0
            elif age <= 7:
                timing_score += 10.0
        
        return min(timing_score, 100.0)
    
    def _extract_founded_year(self, data: Dict[str, Any]) -> Optional[int]:
        """Extract founding year from various data sources."""
        # Check website data
        website_data = data.get('website_data', {})
        company_info = website_data.get('company_info', {})
        founded = company_info.get('founded')
        
        if founded:
            try:
                return int(founded)
            except:
                pass
        
        # Check GitHub creation date
        github_data = data.get('github_data', {})
        if github_data.get('success'):
            repo_stats = github_data.get('data', {}).get('repo_stats', {})
            created_at = repo_stats.get('created_at')
            
            if created_at:
                try:
                    return datetime.fromisoformat(created_at.replace('Z', '+00:00')).year
                except:
                    pass
        
        return None
    
    async def _extract_product_features(self, data: Dict[str, Any], features: StartupFeatures):
        """Extract product-related features."""
        analysis = data.get('analysis', {})
        product_analysis = analysis.get('product_analysis', {})
        website_data = data.get('website_data', {})
        github_data = data.get('github_data', {})
        
        # Product complexity
        features.product_complexity_score = self._score_product_complexity(
            product_analysis, website_data, github_data
        )
        
        # Technical innovation
        features.technical_innovation_score = self._score_technical_innovation(
            product_analysis, website_data.get('tech_stack', [])
        )
        
        # Scalability
        features.scalability_score = self._score_scalability(product_analysis, github_data)
    
    def _score_product_complexity(self, product_analysis: Dict, website_data: Dict, github_data: Dict) -> float:
        """Score product complexity (0-100)."""
        complexity_score = 0.0
        
        # Check for complex features in product description
        complex_features = [
            'ai', 'machine learning', 'blockchain', 'iot', 'api',
            'real-time', 'analytics', 'automation', 'integration'
        ]
        
        product_text = product_analysis.get('description', '').lower()
        complexity_score += sum(10 for feature in complex_features if feature in product_text)
        
        # Check tech stack complexity
        tech_stack = website_data.get('tech_stack', [])
        modern_tech_count = sum(1 for tech in tech_stack if tech.lower() in [
            'react', 'node.js', 'python', 'kubernetes', 'docker',
            'postgresql', 'mongodb', 'redis', 'elasticsearch'
        ])
        complexity_score += modern_tech_count * 5
        
        # Check GitHub metrics for code complexity
        if github_data.get('success'):
            repo_stats = github_data.get('data', {}).get('repo_stats', {})
            languages = github_data.get('data', {}).get('languages', {})
            
            # Multiple languages indicate complexity
            complexity_score += len(languages.get('all_languages', [])) * 3
            
            # Repository size
            size_kb = repo_stats.get('size_kb', 0)
            if size_kb > 10000:  # >10MB
                complexity_score += 20
            elif size_kb > 1000:  # >1MB
                complexity_score += 10
        
        return min(complexity_score, 100.0)
    
    def _score_technical_innovation(self, product_analysis: Dict, tech_stack: List[str]) -> float:
        """Score technical innovation level (0-100)."""
        innovation_score = 0.0
        
        # Innovation keywords
        innovation_keywords = [
            'ai', 'artificial intelligence', 'machine learning', 'deep learning',
            'blockchain', 'cryptocurrency', 'quantum', 'augmented reality',
            'virtual reality', 'iot', 'edge computing', 'serverless'
        ]
        
        product_text = product_analysis.get('description', '').lower()
        innovation_score += sum(15 for keyword in innovation_keywords if keyword in product_text)
        
        # Modern tech stack bonus
        modern_tech = [
            'react', 'vue', 'angular', 'node.js', 'python', 'go', 'rust',
            'kubernetes', 'docker', 'tensorflow', 'pytorch'
        ]
        
        tech_stack_lower = [tech.lower() for tech in tech_stack]
        innovation_score += sum(5 for tech in modern_tech if tech in tech_stack_lower)
        
        return min(innovation_score, 100.0)
    
    def _score_scalability(self, product_analysis: Dict, github_data: Dict) -> float:
        """Score product scalability (0-100)."""
        scalability_score = 0.0
        
        # Scalability indicators
        scalability_keywords = [
            'cloud', 'aws', 'azure', 'gcp', 'microservices', 'api',
            'kubernetes', 'docker', 'serverless', 'distributed',
            'horizontal scaling', 'load balancing'
        ]
        
        product_text = product_analysis.get('description', '').lower()
        scalability_score += sum(10 for keyword in scalability_keywords if keyword in product_text)
        
        # GitHub activity indicates active development
        if github_data.get('success'):
            activity = github_data.get('data', {}).get('activity', {})
            commits_per_week = activity.get('commits_per_week', 0)
            
            if commits_per_week > 10:
                scalability_score += 30
            elif commits_per_week > 5:
                scalability_score += 20
            elif commits_per_week > 0:
                scalability_score += 10
        
        return min(scalability_score, 100.0)
    
    async def _extract_traction_features(self, data: Dict[str, Any], features: StartupFeatures):
        """Extract traction-related features."""
        analysis = data.get('analysis', {})
        traction_analysis = analysis.get('traction_analysis', {})
        
        # Parse traction metrics
        traction_text = traction_analysis.get('metrics', '').lower()
        
        # User growth rate
        features.user_growth_rate = self._extract_metric_growth(traction_text, 'user')
        
        # Revenue growth rate
        features.revenue_growth_rate = self._extract_metric_growth(traction_text, 'revenue')
        
        # Engagement score
        features.engagement_metrics_score = self._score_engagement_metrics(traction_text)
        
        # Customer validation
        features.customer_validation_score = self._score_customer_validation(data)
    
    def _extract_metric_growth(self, traction_text: str, metric_type: str) -> float:
        """Extract growth rate for specific metric."""
        patterns = [
            rf'{metric_type}.*?(\d+(?:\.\d+)?)%.*?growth',
            rf'(\d+(?:\.\d+)?)%.*?{metric_type}.*?growth',
            rf'{metric_type}.*?growing.*?(\d+(?:\.\d+)?)%'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, traction_text)
            if matches:
                return float(matches[0])
        
        return 0.0
    
    def _score_engagement_metrics(self, traction_text: str) -> float:
        """Score engagement metrics (0-100)."""
        engagement_score = 0.0
        
        # Look for engagement indicators
        engagement_indicators = [
            'daily active users', 'monthly active users', 'retention rate',
            'engagement rate', 'time spent', 'repeat customers',
            'referral rate', 'viral coefficient'
        ]
        
        for indicator in engagement_indicators:
            if indicator in traction_text:
                engagement_score += 15.0
        
        # Look for specific metrics
        if re.search(r'\d+%.*retention', traction_text):
            engagement_score += 20.0
        
        if re.search(r'\d+.*daily.*active', traction_text):
            engagement_score += 25.0
        
        return min(engagement_score, 100.0)
    
    def _score_customer_validation(self, data: Dict[str, Any]) -> float:
        """Score customer validation level (0-100)."""
        validation_score = 0.0
        
        # Check website for customer logos
        website_data = data.get('website_data', {})
        customer_logos = website_data.get('customer_logos', [])
        validation_score += min(len(customer_logos) * 10, 50)
        
        # Check for customer testimonials or case studies
        website_features = website_data.get('product_features', [])
        testimonial_indicators = ['testimonial', 'case study', 'customer story', 'success story']
        
        for feature in website_features:
            if any(indicator in feature.lower() for indicator in testimonial_indicators):
                validation_score += 15.0
                break
        
        # Check pricing (indicates paying customers)
        pricing_info = website_data.get('pricing_info', {})
        if pricing_info.get('plans'):
            validation_score += 20.0
        
        return min(validation_score, 100.0)
    
    async def _extract_business_model_features(self, data: Dict[str, Any], features: StartupFeatures):
        """Extract business model features."""
        website_data = data.get('website_data', {})
        pricing_info = website_data.get('pricing_info', {})
        
        # Revenue model strength
        features.revenue_model_strength = self._score_revenue_model(pricing_info)
        
        # Unit economics (basic estimation)
        features.unit_economics_score = self._estimate_unit_economics(data)
        
        # Competitive moat
        features.moat_strength_score = self._score_competitive_moat(data)
    
    def _score_revenue_model(self, pricing_info: Dict) -> float:
        """Score revenue model strength (0-100)."""
        score = 0.0
        
        pricing_model = pricing_info.get('pricing_model', '')
        
        # Score based on pricing model
        if pricing_model == 'subscription':
            score += 40.0  # Recurring revenue is valuable
        elif pricing_model == 'usage-based':
            score += 35.0  # Scalable with usage
        elif pricing_model == 'one-time':
            score += 20.0  # Less predictable
        
        # Multiple pricing plans indicate sophistication
        plans = pricing_info.get('plans', [])
        if len(plans) > 1:
            score += 30.0
        elif len(plans) == 1:
            score += 15.0
        
        # Free tier can drive adoption
        if pricing_info.get('free_tier'):
            score += 15.0
        
        return min(score, 100.0)
    
    def _estimate_unit_economics(self, data: Dict[str, Any]) -> float:
        """Estimate unit economics health (0-100)."""
        # This is a simplified estimation based on available data
        score = 50.0  # Neutral baseline
        
        # Check for premium pricing indicators
        website_data = data.get('website_data', {})
        pricing_info = website_data.get('pricing_info', {})
        
        plans = pricing_info.get('plans', [])
        for plan in plans:
            # Look for high-value pricing
            price_match = re.search(r'\$(\d+)', plan)
            if price_match:
                price = int(price_match.group(1))
                if price > 100:  # Monthly plan >$100
                    score += 20.0
                elif price > 50:
                    score += 10.0
        
        return min(score, 100.0)
    
    def _score_competitive_moat(self, data: Dict[str, Any]) -> float:
        """Score competitive moat strength (0-100)."""
        moat_score = 0.0
        
        analysis = data.get('analysis', {})
        product_analysis = analysis.get('product_analysis', {})
        
        # Check for moat indicators
        moat_indicators = [
            'patent', 'proprietary', 'exclusive', 'network effect',
            'data advantage', 'switching cost', 'regulatory barrier',
            'brand recognition', 'first mover advantage'
        ]
        
        product_text = product_analysis.get('description', '').lower()
        moat_score += sum(15 for indicator in moat_indicators if indicator in product_text)
        
        # Technical complexity as moat
        github_data = data.get('github_data', {})
        if github_data.get('success'):
            languages = github_data.get('data', {}).get('languages', {})
            if len(languages.get('all_languages', [])) > 3:
                moat_score += 20.0  # Complex tech stack
        
        return min(moat_score, 100.0)
    
    async def _extract_technical_features(self, data: Dict[str, Any], features: StartupFeatures):
        """Extract technical features from GitHub and tech stack."""
        github_data = data.get('github_data', {})
        website_data = data.get('website_data', {})
        
        if github_data.get('success'):
            github_metrics = github_data.get('data', {})
            
            # GitHub activity score
            features.github_activity_score = github_metrics.get('health_score', 0.0)
            
            # Code quality estimation
            features.code_quality_score = self._estimate_code_quality(github_metrics)
        
        # Tech stack modernity
        tech_stack = website_data.get('tech_stack', [])
        features.tech_stack_modernity = self._score_tech_stack_modernity(tech_stack)
    
    def _estimate_code_quality(self, github_metrics: Dict) -> float:
        """Estimate code quality from GitHub metrics (0-100)."""
        score = 0.0
        
        repo_stats = github_metrics.get('repo_stats', {})
        activity = github_metrics.get('activity', {})
        contributors = github_metrics.get('contributors', {})
        
        # Multiple contributors indicate code reviews
        contributor_count = contributors.get('total_contributors', 0)
        if contributor_count > 5:
            score += 30.0
        elif contributor_count > 2:
            score += 20.0
        elif contributor_count > 1:
            score += 10.0
        
        # Regular commits indicate maintenance
        commits_per_week = activity.get('commits_per_week', 0)
        if commits_per_week > 5:
            score += 25.0
        elif commits_per_week > 2:
            score += 15.0
        elif commits_per_week > 0:
            score += 5.0
        
        # Low open issues relative to stars
        stars = repo_stats.get('stars', 0)
        open_issues = repo_stats.get('open_issues', 0)
        
        if stars > 0:
            issue_ratio = open_issues / max(stars, 1)
            if issue_ratio < 0.1:
                score += 25.0
            elif issue_ratio < 0.2:
                score += 15.0
            elif issue_ratio < 0.5:
                score += 5.0
        
        return min(score, 100.0)
    
    def _score_tech_stack_modernity(self, tech_stack: List[str]) -> float:
        """Score technology stack modernity (0-100)."""
        if not tech_stack:
            return 0.0
        
        modern_technologies = {
            'react': 20, 'vue': 18, 'angular': 15, 'svelte': 20,
            'node.js': 18, 'python': 15, 'go': 20, 'rust': 25, 'typescript': 20,
            'kubernetes': 25, 'docker': 20, 'serverless': 25,
            'postgresql': 15, 'mongodb': 15, 'redis': 15, 'elasticsearch': 18,
            'graphql': 20, 'api': 10, 'microservices': 25
        }
        
        score = 0.0
        tech_stack_lower = [tech.lower() for tech in tech_stack]
        
        for tech in tech_stack_lower:
            if tech in modern_technologies:
                score += modern_technologies[tech]
        
        return min(score, 100.0)
    
    async def _extract_external_validation_features(self, data: Dict[str, Any], features: StartupFeatures):
        """Extract external validation features."""
        # Media mentions (would need news API integration)
        features.media_mention_score = 0.0  # Placeholder
        
        # Investor interest (based on funding info if available)
        features.investor_interest_score = self._score_investor_interest(data)
        
        # Customer validation from website
        website_data = data.get('website_data', {})
        features.customer_validation_score = self._score_customer_validation(data)
    
    def _score_investor_interest(self, data: Dict[str, Any]) -> float:
        """Score investor interest level (0-100)."""
        # This would integrate with funding databases in production
        # For now, use basic indicators
        
        description = data.get('description', '').lower()
        funding_indicators = [
            'funded', 'investment', 'series a', 'series b', 'seed funding',
            'venture capital', 'angel investor', 'raised', 'funding round'
        ]
        
        score = 0.0
        for indicator in funding_indicators:
            if indicator in description:
                score += 25.0
        
        return min(score, 100.0)
    
    async def _extract_risk_features(self, data: Dict[str, Any], features: StartupFeatures):
        """Extract risk assessment features."""
        analysis = data.get('analysis', {})
        market_analysis = analysis.get('market_analysis', {})
        
        # Regulatory risk
        features.regulatory_risk_score = self._assess_regulatory_risk(data)
        
        # Technical risk
        features.technical_risk_score = self._assess_technical_risk(data)
        
        # Market risk
        features.market_risk_score = self._assess_market_risk(market_analysis)
    
    def _assess_regulatory_risk(self, data: Dict[str, Any]) -> float:
        """Assess regulatory risk (0-100, higher = more risk)."""
        description = data.get('description', '').lower()
        
        high_risk_sectors = [
            'fintech', 'healthcare', 'biotech', 'cryptocurrency',
            'gambling', 'cannabis', 'insurance', 'banking'
        ]
        
        risk_score = 0.0
        for sector in high_risk_sectors:
            if sector in description:
                risk_score += 30.0
        
        return min(risk_score, 100.0)
    
    def _assess_technical_risk(self, data: Dict[str, Any]) -> float:
        """Assess technical execution risk (0-100, higher = more risk)."""
        risk_score = 0.0
        
        # High technical complexity increases risk
        github_data = data.get('github_data', {})
        if github_data.get('success'):
            health_score = github_data.get('data', {}).get('health_score', 50.0)
            # Invert health score for risk (low health = high risk)
            risk_score += (100.0 - health_score) * 0.5
        
        # Check for risky technology choices
        website_data = data.get('website_data', {})
        tech_stack = website_data.get('tech_stack', [])
        
        risky_tech = ['blockchain', 'cryptocurrency', 'quantum', 'experimental']
        tech_stack_lower = [tech.lower() for tech in tech_stack]
        
        for tech in risky_tech:
            if any(tech in stack_tech for stack_tech in tech_stack_lower):
                risk_score += 20.0
        
        return min(risk_score, 100.0)
    
    def _assess_market_risk(self, market_analysis: Dict) -> float:
        """Assess market risk (0-100, higher = more risk)."""
        risk_score = 0.0
        
        # High competition increases risk
        competition_text = market_analysis.get('competition', '').lower()
        
        high_risk_indicators = [
            'saturated market', 'intense competition', 'price wars',
            'commoditized', 'many competitors', 'declining market'
        ]
        
        for indicator in high_risk_indicators:
            if indicator in competition_text:
                risk_score += 20.0
        
        return min(risk_score, 100.0)
    
    def features_to_dict(self, features: StartupFeatures) -> Dict[str, Any]:
        """Convert features object to dictionary for ML processing."""
        return {
            'team_size': features.team_size,
            'founder_experience_score': features.founder_experience_score,
            'technical_founder_present': int(features.technical_founder_present),
            'previous_startup_experience': features.previous_startup_experience,
            'education_prestige_score': features.education_prestige_score,
            'linkedin_network_strength': features.linkedin_network_strength,
            'market_size_score': features.market_size_score,
            'market_growth_rate': features.market_growth_rate,
            'competitive_landscape_score': features.competitive_landscape_score,
            'market_timing_score': features.market_timing_score,
            'product_complexity_score': features.product_complexity_score,
            'technical_innovation_score': features.technical_innovation_score,
            'user_experience_score': features.user_experience_score,
            'scalability_score': features.scalability_score,
            'user_growth_rate': features.user_growth_rate,
            'revenue_growth_rate': features.revenue_growth_rate,
            'engagement_metrics_score': features.engagement_metrics_score,
            'customer_retention_score': features.customer_retention_score,
            'revenue_model_strength': features.revenue_model_strength,
            'unit_economics_score': features.unit_economics_score,
            'moat_strength_score': features.moat_strength_score,
            'github_activity_score': features.github_activity_score,
            'tech_stack_modernity': features.tech_stack_modernity,
            'code_quality_score': features.code_quality_score,
            'media_mention_score': features.media_mention_score,
            'investor_interest_score': features.investor_interest_score,
            'customer_validation_score': features.customer_validation_score,
            'regulatory_risk_score': features.regulatory_risk_score,
            'technical_risk_score': features.technical_risk_score,
            'market_risk_score': features.market_risk_score
        }
    
    async def calculate_composite_scores(self, features: StartupFeatures) -> Dict[str, float]:
        """Calculate composite scores for major categories."""
        feature_dict = self.features_to_dict(features)
        
        composite_scores = {
            'team_score': np.mean([
                feature_dict['founder_experience_score'],
                feature_dict['technical_founder_present'] * 100,
                feature_dict['education_prestige_score'],
                feature_dict['linkedin_network_strength']
            ]),
            'market_score': np.mean([
                feature_dict['market_size_score'],
                feature_dict['competitive_landscape_score'],
                feature_dict['market_timing_score']
            ]),
            'product_score': np.mean([
                feature_dict['product_complexity_score'],
                feature_dict['technical_innovation_score'],
                feature_dict['scalability_score']
            ]),
            'traction_score': np.mean([
                feature_dict['engagement_metrics_score'],
                feature_dict['customer_validation_score']
            ]),
            'technical_score': np.mean([
                feature_dict['github_activity_score'],
                feature_dict['tech_stack_modernity'],
                feature_dict['code_quality_score']
            ]),
            'risk_score': 100 - np.mean([  # Invert risk scores
                feature_dict['regulatory_risk_score'],
                feature_dict['technical_risk_score'],
                feature_dict['market_risk_score']
            ])
        }
        
        return composite_scores