"""
Enhanced Celery worker with advanced async tasks for startup analysis.
"""
import logging
import traceback
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

from celery import Celery
from celery.signals import worker_ready
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models import Startup, Analysis, BatchAnalysis
from app.services.document_processor import DocumentProcessor
from app.services.ai_evaluator import AIEvaluator
from app.services.web_scraper import IntelligentScraper
from app.services.feature_engineering import AdvancedFeatureEngine
from app.services.embedding_similarity import EmbeddingSimilarityEngine
from app.services.ml_prediction import MLPredictionEngine
from app.core.scoring import ScoringEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "quickscore_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=['app.worker']
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=1800,  # 30 minutes
    task_soft_time_limit=1500,  # 25 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_routes={
        'app.worker.analyze_startup_comprehensive': {'queue': 'analysis'},
        'app.worker.scrape_startup_data': {'queue': 'scraping'},
        'app.worker.train_ml_models': {'queue': 'ml_training'},
        'app.worker.batch_analyze_startups': {'queue': 'batch'},
        'app.worker.find_comparable_startups': {'queue': 'similarity'},
    }
)

# Initialize services
document_processor = DocumentProcessor()
ai_evaluator = AIEvaluator()
web_scraper = IntelligentScraper()
feature_engine = AdvancedFeatureEngine()
embedding_engine = EmbeddingSimilarityEngine()
ml_engine = MLPredictionEngine()
scoring_engine = ScoringEngine()


def run_async_task(coro):
    """Helper to run async functions in Celery tasks."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    except Exception as e:
        logger.error(f"Error in async task: {e}")
        raise
    finally:
        loop.close()


@celery_app.task(bind=True, max_retries=3)
def analyze_startup_comprehensive(self, startup_id: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Comprehensive startup analysis with all Day 2 features.
    Enhanced version of the original analysis task.
    """
    if options is None:
        options = {}
    
    logger.info(f"Starting comprehensive analysis for startup {startup_id}")
    
    try:
        # Get database session
        db = next(get_db())
        startup = db.query(Startup).filter(Startup.id == startup_id).first()
        
        if not startup:
            raise ValueError(f"Startup with ID {startup_id} not found")
        
        # Prepare startup data
        startup_data = {
            'id': str(startup.id),
            'name': startup.name,
            'description': startup.description,
            'industry': startup.industry,
            'stage': startup.stage,
            'founders': [
                {
                    'id': str(founder.id),
                    'name': founder.name,
                    'role': founder.role,
                    'background': founder.background,
                    'linkedin_url': founder.linkedin_url
                }
                for founder in startup.founders
            ],
            'documents': []
        }
        
        # Phase 1: Document Processing
        logger.info("Phase 1: Processing documents")
        if startup.documents:
            for doc in startup.documents:
                doc_content = run_async_task(
                    document_processor.process_document(doc.file_path, doc.document_type)
                )
                startup_data['documents'].append({
                    'type': doc.document_type,
                    'content': doc_content
                })
        
        # Phase 2: Web Scraping (New in Day 2)
        logger.info("Phase 2: Web scraping")
        scraping_results = {}
        
        if options.get('scrape_linkedin', True):
            # Scrape LinkedIn profiles
            linkedin_profiles = []
            for founder in startup.founders:
                if founder.linkedin_url:
                    linkedin_data = run_async_task(
                        web_scraper.scrape_linkedin_profile(founder.linkedin_url)
                    )
                    linkedin_profiles.append(linkedin_data)
            startup_data['linkedin_profiles'] = linkedin_profiles
        
        if options.get('scrape_website', True) and startup.website_url:
            # Scrape company website
            website_data = run_async_task(
                web_scraper.scrape_company_website(startup.website_url)
            )
            startup_data['website_data'] = website_data.get('data', {})
        
        if options.get('scrape_github', True) and startup.github_url:
            # Scrape GitHub repository
            github_data = run_async_task(
                web_scraper.extract_github_metrics(startup.github_url)
            )
            startup_data['github_data'] = github_data
        
        # Phase 3: AI Analysis (Enhanced)
        logger.info("Phase 3: AI analysis")
        analysis_results = {}
        
        # Original AI evaluations
        if startup_data.get('documents'):
            pitch_analysis = run_async_task(
                ai_evaluator.analyze_pitch_deck(startup_data['documents'])
            )
            analysis_results['pitch_analysis'] = pitch_analysis
        
        team_analysis = run_async_task(
            ai_evaluator.evaluate_team(startup_data['founders'])
        )
        analysis_results['team_analysis'] = team_analysis
        
        market_analysis = run_async_task(
            ai_evaluator.analyze_market_opportunity(startup_data)
        )
        analysis_results['market_analysis'] = market_analysis
        
        product_analysis = run_async_task(
            ai_evaluator.evaluate_product_viability(startup_data)
        )
        analysis_results['product_analysis'] = product_analysis
        
        traction_analysis = run_async_task(
            ai_evaluator.assess_traction_metrics(startup_data)
        )
        analysis_results['traction_analysis'] = traction_analysis
        
        startup_data['analysis'] = analysis_results
        
        # Phase 4: Feature Engineering (New in Day 2)
        logger.info("Phase 4: Feature engineering")
        features = run_async_task(
            feature_engine.extract_startup_features(startup_data)
        )
        feature_dict = feature_engine.features_to_dict(features)
        startup_data['engineered_features'] = feature_dict
        
        # Calculate composite scores
        composite_scores = run_async_task(
            feature_engine.calculate_composite_scores(features)
        )
        startup_data['composite_scores'] = composite_scores
        
        # Phase 5: ML Predictions (New in Day 2)
        logger.info("Phase 5: ML predictions")
        if options.get('ml_predictions', True):
            ml_predictions = run_async_task(
                ml_engine.predict_startup_success(startup_data)
            )
            startup_data['ml_predictions'] = ml_predictions
            
            # Get prediction explanations
            ml_explanations = run_async_task(
                ml_engine.get_prediction_explanation(startup_data, ml_predictions)
            )
            startup_data['ml_explanations'] = ml_explanations
        
        # Phase 6: Similarity Analysis (New in Day 2)
        logger.info("Phase 6: Similarity analysis")
        if options.get('similarity_analysis', True):
            # Generate startup embedding
            startup_embedding = run_async_task(
                embedding_engine.generate_startup_embedding(startup_data)
            )
            startup_data['embedding'] = startup_embedding.tolist()
            
            # Calculate market fit score
            market_fit = run_async_task(
                embedding_engine.calculate_market_fit_score(startup_data)
            )
            startup_data['market_fit'] = market_fit
        
        # Phase 7: Enhanced Scoring
        logger.info("Phase 7: Enhanced scoring")
        
        # Original scoring
        final_score = scoring_engine.calculate_composite_score(
            analysis_results, startup.stage
        )
        
        # Enhanced scoring with ML and features
        enhanced_score = {
            'original_score': final_score,
            'composite_scores': composite_scores,
            'ml_success_probability': startup_data.get('ml_predictions', {}).get('overall_success_score', 50.0),
            'market_fit_score': startup_data.get('market_fit', {}).get('market_fit_score', 50.0)
        }
        
        # Calculate final enhanced score
        weights = {
            'original_score': 0.4,
            'ml_success_probability': 0.3,
            'market_fit_score': 0.2,
            'composite_average': 0.1
        }
        
        composite_average = sum(composite_scores.values()) / len(composite_scores) if composite_scores else 50.0
        
        final_enhanced_score = (
            enhanced_score['original_score'] * weights['original_score'] +
            enhanced_score['ml_success_probability'] * weights['ml_success_probability'] +
            enhanced_score['market_fit_score'] * weights['market_fit_score'] +
            composite_average * weights['composite_average']
        )
        
        enhanced_score['final_score'] = final_enhanced_score
        startup_data['enhanced_scoring'] = enhanced_score
        
        # Phase 8: Save Results
        logger.info("Phase 8: Saving results")
        
        # Create or update analysis record
        analysis = Analysis(
            startup_id=startup.id,
            team_score=team_analysis.get('score', 0),
            market_score=market_analysis.get('score', 0),
            product_score=product_analysis.get('score', 0),
            traction_score=traction_analysis.get('score', 0),
            overall_score=final_enhanced_score,
            insights=analysis_results,
            status='completed',
            ml_predictions=startup_data.get('ml_predictions'),
            engineered_features=feature_dict,
            similarity_data=startup_data.get('market_fit'),
            created_at=datetime.utcnow()
        )
        
        db.add(analysis)
        db.commit()
        
        logger.info(f"Comprehensive analysis completed for startup {startup_id}")
        
        return {
            'status': 'completed',
            'startup_id': startup_id,
            'analysis_id': str(analysis.id),
            'final_score': final_enhanced_score,
            'phase_results': {
                'documents_processed': len(startup_data.get('documents', [])),
                'linkedin_profiles_scraped': len(startup_data.get('linkedin_profiles', [])),
                'website_scraped': bool(startup_data.get('website_data')),
                'github_analyzed': bool(startup_data.get('github_data')),
                'features_extracted': len(feature_dict),
                'ml_predictions_generated': bool(startup_data.get('ml_predictions')),
                'similarity_analyzed': bool(startup_data.get('market_fit'))
            },
            'enhanced_insights': {
                'composite_scores': composite_scores,
                'ml_success_probability': startup_data.get('ml_predictions', {}).get('overall_success_score'),
                'market_fit_score': startup_data.get('market_fit', {}).get('market_fit_score'),
                'top_strengths': _extract_top_strengths(startup_data),
                'improvement_areas': _extract_improvement_areas(startup_data)
            }
        }
        
    except Exception as e:
        logger.error(f"Comprehensive analysis failed for startup {startup_id}: {e}")
        logger.error(traceback.format_exc())
        
        # Update analysis with error status
        try:
            db = next(get_db())
            analysis = Analysis(
                startup_id=startup_id,
                status='failed',
                insights={'error': str(e), 'traceback': traceback.format_exc()},
                created_at=datetime.utcnow()
            )
            db.add(analysis)
            db.commit()
        except:
            pass
        
        # Retry task
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying analysis for startup {startup_id} (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60 * (self.request.retries + 1))
        
        return {
            'status': 'failed',
            'startup_id': startup_id,
            'error': str(e)
        }


@celery_app.task(bind=True, max_retries=2)
def scrape_startup_data(self, startup_id: str, scraping_options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dedicated task for web scraping startup data.
    """
    logger.info(f"Starting web scraping for startup {startup_id}")
    
    try:
        db = next(get_db())
        startup = db.query(Startup).filter(Startup.id == startup_id).first()
        
        if not startup:
            raise ValueError(f"Startup with ID {startup_id} not found")
        
        scraping_results = {}
        
        # LinkedIn scraping
        if scraping_options.get('linkedin', True):
            linkedin_results = []
            for founder in startup.founders:
                if founder.linkedin_url:
                    result = run_async_task(
                        web_scraper.scrape_linkedin_profile(founder.linkedin_url)
                    )
                    linkedin_results.append(result)
            scraping_results['linkedin'] = linkedin_results
        
        # Website scraping
        if scraping_options.get('website', True) and startup.website_url:
            website_result = run_async_task(
                web_scraper.scrape_company_website(startup.website_url)
            )
            scraping_results['website'] = website_result
        
        # GitHub scraping
        if scraping_options.get('github', True) and startup.github_url:
            github_result = run_async_task(
                web_scraper.extract_github_metrics(startup.github_url)
            )
            scraping_results['github'] = github_result
        
        logger.info(f"Web scraping completed for startup {startup_id}")
        
        return {
            'status': 'completed',
            'startup_id': startup_id,
            'scraping_results': scraping_results,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Web scraping failed for startup {startup_id}: {e}")
        
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=60 * (self.request.retries + 1))
        
        return {
            'status': 'failed',
            'startup_id': startup_id,
            'error': str(e)
        }


@celery_app.task(bind=True)
def train_ml_models(self, training_data_query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task for training ML models on historical startup data.
    """
    logger.info("Starting ML model training")
    
    try:
        # Get training data from database
        db = next(get_db())
        
        # Build query based on criteria
        query = db.query(Startup).join(Analysis)
        
        if training_data_query.get('min_score'):
            query = query.filter(Analysis.overall_score >= training_data_query['min_score'])
        
        if training_data_query.get('industries'):
            query = query.filter(Startup.industry.in_(training_data_query['industries']))
        
        if training_data_query.get('stages'):
            query = query.filter(Startup.stage.in_(training_data_query['stages']))
        
        startups = query.limit(training_data_query.get('limit', 1000)).all()
        
        # Prepare training data
        training_data = []
        for startup in startups:
            startup_data = {
                'id': str(startup.id),
                'name': startup.name,
                'description': startup.description,
                'industry': startup.industry,
                'stage': startup.stage,
                'analysis': startup.analyses[0].insights if startup.analyses else {},
                'success_metrics': _determine_success_metrics(startup)
            }
            training_data.append(startup_data)
        
        # Train models
        target_metrics = training_data_query.get('target_metrics', ['overall_success', 'funding_success'])
        
        training_results = run_async_task(
            ml_engine.train_success_prediction_models(training_data, target_metrics)
        )
        
        logger.info("ML model training completed")
        
        return {
            'status': 'completed',
            'training_data_size': len(training_data),
            'target_metrics': target_metrics,
            'training_results': training_results,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"ML model training failed: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }


@celery_app.task(bind=True)
def batch_analyze_startups(self, startup_ids: List[str], analysis_options: Dict[str, Any]) -> str:
    """
    Task for batch analysis of multiple startups.
    """
    batch_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Starting batch analysis {batch_id} for {len(startup_ids)} startups")
    
    try:
        db = next(get_db())
        
        # Create batch analysis record
        batch_analysis = BatchAnalysis(
            batch_id=batch_id,
            startup_ids=startup_ids,
            status='in_progress',
            total_count=len(startup_ids),
            completed_count=0,
            failed_count=0,
            created_at=datetime.utcnow()
        )
        db.add(batch_analysis)
        db.commit()
        
        # Process each startup
        results = []
        for startup_id in startup_ids:
            try:
                # Trigger individual analysis
                analysis_task = analyze_startup_comprehensive.delay(startup_id, analysis_options)
                result = analysis_task.get(timeout=1800)  # 30 minutes timeout
                
                results.append({
                    'startup_id': startup_id,
                    'status': 'completed',
                    'result': result
                })
                
                batch_analysis.completed_count += 1
                
            except Exception as e:
                logger.error(f"Analysis failed for startup {startup_id} in batch {batch_id}: {e}")
                results.append({
                    'startup_id': startup_id,
                    'status': 'failed',
                    'error': str(e)
                })
                
                batch_analysis.failed_count += 1
            
            db.commit()
        
        # Update batch status
        batch_analysis.status = 'completed'
        batch_analysis.results = results
        batch_analysis.completed_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Batch analysis {batch_id} completed")
        
        return batch_id
        
    except Exception as e:
        logger.error(f"Batch analysis {batch_id} failed: {e}")
        
        # Update batch with error
        try:
            db = next(get_db())
            batch_analysis = db.query(BatchAnalysis).filter(BatchAnalysis.batch_id == batch_id).first()
            if batch_analysis:
                batch_analysis.status = 'failed'
                batch_analysis.error_message = str(e)
                db.commit()
        except:
            pass
        
        return batch_id


@celery_app.task(bind=True)
def find_comparable_startups(self, startup_id: str, comparison_options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task for finding and analyzing comparable startups.
    """
    logger.info(f"Finding comparable startups for {startup_id}")
    
    try:
        db = next(get_db())
        startup = db.query(Startup).filter(Startup.id == startup_id).first()
        
        if not startup:
            raise ValueError(f"Startup with ID {startup_id} not found")
        
        # Prepare startup data
        startup_data = {
            'name': startup.name,
            'description': startup.description,
            'industry': startup.industry,
            'stage': startup.stage,
            'analysis': startup.analyses[0].insights if startup.analyses else {}
        }
        
        # Get comparison database
        comparison_query = db.query(Startup).filter(Startup.id != startup_id)
        
        if comparison_options.get('same_industry'):
            comparison_query = comparison_query.filter(Startup.industry == startup.industry)
        
        if comparison_options.get('same_stage'):
            comparison_query = comparison_query.filter(Startup.stage == startup.stage)
        
        comparison_startups = comparison_query.limit(500).all()
        
        # Prepare comparison data
        comparison_data = []
        for comp_startup in comparison_startups:
            comp_data = {
                'id': str(comp_startup.id),
                'name': comp_startup.name,
                'description': comp_startup.description,
                'industry': comp_startup.industry,
                'stage': comp_startup.stage,
                'analysis': comp_startup.analyses[0].insights if comp_startup.analyses else {}
            }
            comparison_data.append(comp_data)
        
        # Find similar startups
        similar_startups = run_async_task(
            embedding_engine.find_similar_startups(
                startup_data, 
                comparison_data, 
                top_k=comparison_options.get('top_k', 10)
            )
        )
        
        logger.info(f"Found {len(similar_startups)} comparable startups for {startup_id}")
        
        return {
            'status': 'completed',
            'startup_id': startup_id,
            'similar_startups': similar_startups,
            'comparison_pool_size': len(comparison_data),
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Finding comparable startups failed for {startup_id}: {e}")
        return {
            'status': 'failed',
            'startup_id': startup_id,
            'error': str(e)
        }


@celery_app.task
def identify_success_patterns(training_data_criteria: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task for identifying success patterns from historical data.
    """
    logger.info("Identifying success patterns from historical data")
    
    try:
        db = next(get_db())
        
        # Get successful startups
        query = db.query(Startup).join(Analysis).filter(
            Analysis.overall_score >= training_data_criteria.get('min_success_score', 80)
        )
        
        if training_data_criteria.get('industries'):
            query = query.filter(Startup.industry.in_(training_data_criteria['industries']))
        
        successful_startups = query.limit(training_data_criteria.get('limit', 200)).all()
        
        # Prepare data
        success_data = []
        for startup in successful_startups:
            startup_data = {
                'name': startup.name,
                'description': startup.description,
                'industry': startup.industry,
                'stage': startup.stage,
                'analysis': startup.analyses[0].insights if startup.analyses else {}
            }
            success_data.append(startup_data)
        
        # Identify patterns
        patterns = run_async_task(
            embedding_engine.identify_success_patterns(success_data)
        )
        
        logger.info(f"Identified {len(patterns.get('patterns', []))} success patterns")
        
        return {
            'status': 'completed',
            'patterns': patterns,
            'successful_startups_analyzed': len(success_data),
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Success pattern identification failed: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }


def _extract_top_strengths(startup_data: Dict[str, Any]) -> List[str]:
    """Extract top strengths from analysis data."""
    strengths = []
    
    # From composite scores
    composite_scores = startup_data.get('composite_scores', {})
    for category, score in composite_scores.items():
        if score > 75:
            strengths.append(f"Strong {category.replace('_', ' ')}")
    
    # From ML predictions
    ml_predictions = startup_data.get('ml_predictions', {})
    if ml_predictions.get('overall_success_score', 0) > 75:
        strengths.append("High ML-predicted success probability")
    
    # From feature analysis
    features = startup_data.get('engineered_features', {})
    if features.get('founder_experience_score', 0) > 80:
        strengths.append("Experienced founding team")
    
    if features.get('market_size_score', 0) > 80:
        strengths.append("Large market opportunity")
    
    return strengths[:5]


def _extract_improvement_areas(startup_data: Dict[str, Any]) -> List[str]:
    """Extract areas for improvement from analysis data."""
    improvements = []
    
    # From composite scores
    composite_scores = startup_data.get('composite_scores', {})
    for category, score in composite_scores.items():
        if score < 40:
            improvements.append(f"Improve {category.replace('_', ' ')}")
    
    # From feature analysis
    features = startup_data.get('engineered_features', {})
    if features.get('customer_validation_score', 0) < 40:
        improvements.append("Strengthen customer validation")
    
    if features.get('github_activity_score', 0) < 40:
        improvements.append("Increase development activity")
    
    return improvements[:5]


def _determine_success_metrics(startup) -> Dict[str, Any]:
    """Determine success metrics for a startup (for ML training)."""
    # This would integrate with external data sources in production
    # For now, use basic heuristics
    
    metrics = {
        'funding_success': False,
        'growth_success': False,
        'exit_success': False,
        'overall_success': False
    }
    
    # Use analysis score as proxy for success
    if startup.analyses:
        latest_analysis = startup.analyses[0]
        score = latest_analysis.overall_score
        
        if score > 80:
            metrics['overall_success'] = True
        if score > 75:
            metrics['funding_success'] = True
        if score > 70:
            metrics['growth_success'] = True
    
    return metrics


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handle worker ready signal."""
    logger.info("Celery worker is ready and enhanced with Day 2 features")


if __name__ == '__main__':
    celery_app.start()