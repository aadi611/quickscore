"""
Celery worker configuration and task definitions.
"""
import logging
import asyncio
from typing import Dict, Any, Optional
from celery import Celery
from celery.signals import worker_ready
import json

from app.core.config import settings
from app.models.database import SessionLocal
from app.models.crud import startup_crud, analysis_crud, founder_crud
from app.services.document_processor import DocumentProcessor
from app.services.ai_evaluator import StartupEvaluator
from app.services.web_scraper import IntelligentScraper
from app.core.scoring import ScoringEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "quickscore_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.worker"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_routes={
        "app.worker.process_analysis_task": {"queue": "analysis"},
        "app.worker.batch_analysis_task": {"queue": "batch"},
        "app.worker.scrape_data_task": {"queue": "scraping"},
        "app.worker.update_market_data_task": {"queue": "maintenance"}
    },
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=50
)


@worker_ready.connect
def worker_ready_handler(sender, **kwargs):
    """Handle worker ready signal."""
    logger.info("Celery worker is ready and waiting for tasks")


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_analysis_task(
    self, 
    startup_id: str, 
    analysis_id: str, 
    analysis_config: Dict[str, Any]
):
    """
    Background task to process complete startup analysis.
    
    This task orchestrates the entire analysis pipeline:
    1. Extract documents
    2. Scrape web data
    3. Generate features
    4. Run LLM evaluation
    5. Calculate scores
    6. Find comparables
    7. Generate report
    """
    db = SessionLocal()
    
    try:
        logger.info(f"Starting analysis task for startup {startup_id}")
        
        # Update analysis status
        analysis_crud.update_status(db, analysis_id, "processing")
        
        # Initialize services
        document_processor = DocumentProcessor()
        ai_evaluator = StartupEvaluator()
        scoring_engine = ScoringEngine()
        scraper = IntelligentScraper() if 'IntelligentScraper' in globals() else None
        
        # Get startup data
        startup = startup_crud.get(db, startup_id)
        if not startup:
            raise Exception(f"Startup {startup_id} not found")
        
        # Step 1: Process documents if provided
        processed_documents = {}
        if "documents" in analysis_config:
            for doc_type, doc_path in analysis_config["documents"].items():
                try:
                    with open(doc_path, 'rb') as f:
                        doc_content = f.read()
                    
                    result = asyncio.run(document_processor.process_document(
                        file_content=doc_content,
                        filename=doc_path,
                        document_type=doc_type
                    ))
                    processed_documents[doc_type] = result
                    logger.info(f"Processed {doc_type} document")
                except Exception as e:
                    logger.warning(f"Failed to process {doc_type}: {e}")
        
        # Step 2: Scrape web data if URLs provided
        scraped_data = {}
        if scraper and "urls" in analysis_config:
            try:
                # This would be implemented when scraper is available
                logger.info("Web scraping would be performed here")
            except Exception as e:
                logger.warning(f"Web scraping failed: {e}")
        
        # Step 3: Prepare data for AI evaluation
        startup_context = {
            "name": startup.name,
            "industry": startup.industry,
            "stage": startup.stage,
            "description": startup.description,
            "website": str(startup.website) if startup.website else None
        }
        
        founder_data = analysis_config.get("founder_data", {})
        market_data = {
            "target_market": startup.industry,
            "geography": analysis_config.get("geography", "Global"),
            "competitors": analysis_config.get("competitors", []),
            "additional_context": startup.description or ""
        }
        
        # Step 4: Run AI evaluation
        logger.info("Running AI evaluation")
        evaluation_results = asyncio.run(ai_evaluator.evaluate_startup(
            pitch_deck_content=processed_documents.get("pitch_deck"),
            founder_data=founder_data if founder_data else None,
            market_data=market_data,
            startup_context=startup_context
        ))
        
        if not evaluation_results.get("success", False):
            raise Exception(f"AI evaluation failed: {evaluation_results.get('error')}")
        
        # Step 5: Calculate scores
        logger.info("Calculating composite scores")
        scoring_results = scoring_engine.calculate_composite_score(
            evaluation_results["evaluation_results"],
            startup_stage=startup.stage
        )
        
        if not scoring_results.get("success", False):
            raise Exception(f"Scoring failed: {scoring_results.get('error')}")
        
        # Step 6: Generate insights
        insights_result = asyncio.run(ai_evaluator.generate_insights(
            evaluation_results["evaluation_results"],
            startup_context
        ))
        
        # Step 7: Prepare final results
        final_results = {
            "overall_score": scoring_results["overall_score"],
            "team_score": scoring_results["category_scores"].get("team", 0),
            "market_score": scoring_results["category_scores"].get("market", 0),
            "product_score": scoring_results["category_scores"].get("product", 0),
            "traction_score": scoring_results["category_scores"].get("traction", 0),
            "pitch_quality_score": scoring_results["category_scores"].get("pitch_quality", 0),
            "recommendation": scoring_results["recommendation"],
            "confidence": scoring_results["confidence"],
            "insights": insights_result.get("insights", {}),
            "raw_llm_outputs": evaluation_results.get("raw_llm_outputs", {}),
            "processing_time": evaluation_results.get("processing_time", 0)
        }
        
        # Step 8: Update analysis with results
        analysis = analysis_crud.complete_analysis(db, analysis_id, final_results)
        
        logger.info(f"Analysis completed for startup {startup_id}")
        
        return {
            "success": True,
            "analysis_id": analysis_id,
            "startup_id": startup_id,
            "overall_score": analysis.overall_score,
            "recommendation": analysis.recommendation
        }
        
    except Exception as e:
        logger.error(f"Analysis task failed for startup {startup_id}: {e}")
        
        # Update analysis with error
        try:
            analysis_crud.update_status(db, analysis_id, "failed")
        except Exception as db_error:
            logger.error(f"Failed to update analysis status: {db_error}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying analysis task (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60 * (2 ** self.request.retries))
        
        raise e
        
    finally:
        db.close()


@celery_app.task(bind=True)
def batch_analysis_task(self, batch_id: str, startup_data_list: list):
    """
    Process multiple startups in batch.
    """
    db = SessionLocal()
    
    try:
        logger.info(f"Starting batch analysis {batch_id} for {len(startup_data_list)} startups")
        
        completed = 0
        failed = 0
        results = []
        
        for startup_data in startup_data_list:
            try:
                # Create startup if it doesn't exist
                startup = startup_crud.get_by_name(db, startup_data["name"])
                if not startup:
                    from app.models.schemas import StartupCreate
                    startup_create = StartupCreate(**startup_data)
                    startup = startup_crud.create(db, startup_create)
                
                # Create analysis
                analysis = analysis_crud.create(db, startup.id)
                
                # Process analysis (simplified for batch)
                analysis_config = {
                    "founder_data": startup_data.get("founder_data", {}),
                    "geography": startup_data.get("geography", "Global"),
                    "competitors": startup_data.get("competitors", [])
                }
                
                # Trigger individual analysis task
                result = process_analysis_task.delay(
                    str(startup.id), 
                    str(analysis.id), 
                    analysis_config
                )
                
                results.append({
                    "startup_id": startup.id,
                    "analysis_id": analysis.id,
                    "task_id": result.id
                })
                
                completed += 1
                
            except Exception as e:
                logger.error(f"Failed to process startup {startup_data.get('name', 'Unknown')}: {e}")
                failed += 1
        
        # Update batch status
        from app.models.crud import batch_crud
        batch_crud.update_progress(db, batch_id, completed, failed)
        
        logger.info(f"Batch analysis {batch_id} completed: {completed} successful, {failed} failed")
        
        return {
            "success": True,
            "batch_id": batch_id,
            "completed": completed,
            "failed": failed,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch analysis {batch_id} failed: {e}")
        raise e
        
    finally:
        db.close()


@celery_app.task
def scrape_data_task(startup_id: str, urls: Dict[str, str]):
    """
    Background task for web scraping.
    """
    db = SessionLocal()
    
    try:
        logger.info(f"Starting data scraping for startup {startup_id}")
        
        # This would use the IntelligentScraper when implemented
        scraped_data = {}
        
        for data_type, url in urls.items():
            try:
                # Placeholder for actual scraping
                scraped_data[data_type] = {"url": url, "scraped": True}
                logger.info(f"Scraped {data_type} from {url}")
            except Exception as e:
                logger.warning(f"Failed to scrape {data_type} from {url}: {e}")
        
        return {
            "success": True,
            "startup_id": startup_id,
            "scraped_data": scraped_data
        }
        
    except Exception as e:
        logger.error(f"Scraping task failed for startup {startup_id}: {e}")
        raise e
        
    finally:
        db.close()


@celery_app.task
def update_market_data_task():
    """
    Periodic task to update market data and comparable startups.
    """
    try:
        logger.info("Starting market data update task")
        
        # This would update market data, refresh comparable startups, etc.
        # Placeholder implementation
        
        updated_items = 0
        # Update logic would go here
        
        logger.info(f"Market data update completed: {updated_items} items updated")
        
        return {
            "success": True,
            "updated_items": updated_items
        }
        
    except Exception as e:
        logger.error(f"Market data update failed: {e}")
        raise e


# Periodic task configuration
celery_app.conf.beat_schedule = {
    'update-market-data': {
        'task': 'app.worker.update_market_data_task',
        'schedule': 86400.0,  # 24 hours
        'options': {'queue': 'maintenance'}
    },
}


if __name__ == "__main__":
    # For running worker directly
    celery_app.start()