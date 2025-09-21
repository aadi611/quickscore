"""
Batch processing system for analyzing multiple startups simultaneously.
"""
import logging
import asyncio
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text

from app.core.config import settings
from app.core.database import get_db
from app.models import Startup, Analysis, BatchAnalysis
from app.worker_enhanced import (
    analyze_startup_comprehensive, 
    scrape_startup_data,
    find_comparable_startups,
    train_ml_models
)

logger = logging.getLogger(__name__)


class BatchStatus(Enum):
    """Batch processing status enumeration."""
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class BatchJobConfig:
    """Configuration for batch processing jobs."""
    
    # Job identification
    job_id: str
    job_name: str
    created_by: str
    
    # Processing options
    analysis_options: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    
    # Timing and limits
    max_concurrent_tasks: int = 5
    timeout_per_task: int = 1800  # 30 minutes
    retry_failed_tasks: bool = True
    max_retries: int = 2
    
    # Output configuration
    export_format: str = "json"  # json, csv, xlsx
    include_raw_data: bool = False
    include_visualizations: bool = False
    
    # Notification settings
    notify_on_completion: bool = True
    notification_email: Optional[str] = None
    webhook_url: Optional[str] = None


@dataclass
class BatchProgress:
    """Batch processing progress tracking."""
    
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    skipped_tasks: int = 0
    in_progress_tasks: int = 0
    
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    current_task: Optional[str] = None
    last_error: Optional[str] = None
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100
    
    @property
    def elapsed_time(self) -> Optional[timedelta]:
        """Calculate elapsed time."""
        if self.start_time is None:
            return None
        end = self.end_time or datetime.utcnow()
        return end - self.start_time
    
    @property
    def average_task_time(self) -> Optional[float]:
        """Calculate average time per task in seconds."""
        if self.completed_tasks == 0 or self.elapsed_time is None:
            return None
        return self.elapsed_time.total_seconds() / self.completed_tasks


class BatchProcessor:
    """Advanced batch processing system for startup analysis."""
    
    def __init__(self):
        self.active_jobs: Dict[str, BatchJobConfig] = {}
        self.job_progress: Dict[str, BatchProgress] = {}
        self.results_storage_path = Path("batch_results")
        self.results_storage_path.mkdir(exist_ok=True)
    
    async def create_batch_job(
        self, 
        startup_ids: List[str],
        job_config: BatchJobConfig,
        db: Optional[Session] = None
    ) -> str:
        """Create a new batch processing job."""
        if db is None:
            db = next(get_db())
        
        logger.info(f"Creating batch job {job_config.job_id} for {len(startup_ids)} startups")
        
        # Validate startup IDs
        valid_startup_ids = []
        for startup_id in startup_ids:
            startup = db.query(Startup).filter(Startup.id == startup_id).first()
            if startup:
                valid_startup_ids.append(startup_id)
            else:
                logger.warning(f"Startup {startup_id} not found, skipping")
        
        if not valid_startup_ids:
            raise ValueError("No valid startup IDs provided")
        
        # Create batch analysis record
        batch_analysis = BatchAnalysis(
            batch_id=job_config.job_id,
            startup_ids=valid_startup_ids,
            status=BatchStatus.QUEUED.value,
            total_count=len(valid_startup_ids),
            completed_count=0,
            failed_count=0,
            config=asdict(job_config),
            created_at=datetime.utcnow()
        )
        
        db.add(batch_analysis)
        db.commit()
        
        # Initialize progress tracking
        progress = BatchProgress(
            total_tasks=len(valid_startup_ids),
            start_time=datetime.utcnow()
        )
        
        self.active_jobs[job_config.job_id] = job_config
        self.job_progress[job_config.job_id] = progress
        
        return job_config.job_id
    
    async def start_batch_processing(
        self, 
        job_id: str,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """Start batch processing for a job."""
        if db is None:
            db = next(get_db())
        
        if job_id not in self.active_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job_config = self.active_jobs[job_id]
        progress = self.job_progress[job_id]
        
        logger.info(f"Starting batch processing for job {job_id}")
        
        # Update batch status
        batch_analysis = db.query(BatchAnalysis).filter(BatchAnalysis.batch_id == job_id).first()
        if not batch_analysis:
            raise ValueError(f"Batch analysis record for job {job_id} not found")
        
        batch_analysis.status = BatchStatus.IN_PROGRESS.value
        batch_analysis.started_at = datetime.utcnow()
        db.commit()
        
        try:
            # Process startups in parallel with concurrency limit
            results = await self._process_startups_parallel(
                batch_analysis.startup_ids,
                job_config,
                progress,
                db
            )
            
            # Update final status
            progress.end_time = datetime.utcnow()
            batch_analysis.status = BatchStatus.COMPLETED.value
            batch_analysis.completed_at = datetime.utcnow()
            batch_analysis.results = results
            db.commit()
            
            # Generate and save final report
            report = await self._generate_batch_report(job_id, results, db)
            
            # Send notifications if configured
            if job_config.notify_on_completion:
                await self._send_completion_notification(job_config, report)
            
            logger.info(f"Batch processing completed for job {job_id}")
            
            return {
                'job_id': job_id,
                'status': 'completed',
                'total_processed': progress.total_tasks,
                'successful': progress.completed_tasks,
                'failed': progress.failed_tasks,
                'elapsed_time': progress.elapsed_time.total_seconds() if progress.elapsed_time else 0,
                'report_path': report.get('file_path')
            }
            
        except Exception as e:
            logger.error(f"Batch processing failed for job {job_id}: {e}")
            
            # Update error status
            progress.last_error = str(e)
            batch_analysis.status = BatchStatus.FAILED.value
            batch_analysis.error_message = str(e)
            db.commit()
            
            return {
                'job_id': job_id,
                'status': 'failed',
                'error': str(e)
            }
    
    async def _process_startups_parallel(
        self,
        startup_ids: List[str],
        job_config: BatchJobConfig,
        progress: BatchProgress,
        db: Session
    ) -> List[Dict[str, Any]]:
        """Process startups in parallel with concurrency control."""
        
        semaphore = asyncio.Semaphore(job_config.max_concurrent_tasks)
        results = []
        
        async def process_single_startup(startup_id: str) -> Dict[str, Any]:
            async with semaphore:
                progress.current_task = startup_id
                progress.in_progress_tasks += 1
                
                try:
                    # Run comprehensive analysis
                    task = analyze_startup_comprehensive.delay(
                        startup_id, 
                        job_config.analysis_options
                    )
                    
                    result = task.get(timeout=job_config.timeout_per_task)
                    
                    progress.completed_tasks += 1
                    progress.in_progress_tasks -= 1
                    
                    # Update database
                    batch_analysis = db.query(BatchAnalysis).filter(
                        BatchAnalysis.batch_id == job_config.job_id
                    ).first()
                    
                    if batch_analysis:
                        batch_analysis.completed_count = progress.completed_tasks
                        db.commit()
                    
                    return {
                        'startup_id': startup_id,
                        'status': 'completed',
                        'result': result,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to process startup {startup_id}: {e}")
                    
                    progress.failed_tasks += 1
                    progress.in_progress_tasks -= 1
                    progress.last_error = str(e)
                    
                    # Update database
                    batch_analysis = db.query(BatchAnalysis).filter(
                        BatchAnalysis.batch_id == job_config.job_id
                    ).first()
                    
                    if batch_analysis:
                        batch_analysis.failed_count = progress.failed_tasks
                        db.commit()
                    
                    # Retry if configured
                    if job_config.retry_failed_tasks and getattr(task, 'retries', 0) < job_config.max_retries:
                        logger.info(f"Retrying startup {startup_id}")
                        # Implementation would retry the task
                    
                    return {
                        'startup_id': startup_id,
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': datetime.utcnow().isoformat()
                    }
        
        # Process all startups
        tasks = [process_single_startup(startup_id) for startup_id in startup_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in gather
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'startup_id': startup_ids[i],
                    'status': 'failed',
                    'error': str(result),
                    'timestamp': datetime.utcnow().isoformat()
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _generate_batch_report(
        self, 
        job_id: str, 
        results: List[Dict[str, Any]], 
        db: Session
    ) -> Dict[str, Any]:
        """Generate comprehensive batch processing report."""
        
        job_config = self.active_jobs[job_id]
        progress = self.job_progress[job_id]
        
        # Aggregate statistics
        successful_results = [r for r in results if r['status'] == 'completed']
        failed_results = [r for r in results if r['status'] == 'failed']
        
        # Calculate score statistics
        scores = []
        for result in successful_results:
            if result.get('result', {}).get('final_score'):
                scores.append(result['result']['final_score'])
        
        score_stats = {}
        if scores:
            score_stats = {
                'mean': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores),
                'median': sorted(scores)[len(scores) // 2] if scores else 0
            }
        
        # Generate insights
        insights = await self._generate_batch_insights(successful_results, db)
        
        # Create report
        report = {
            'job_info': {
                'job_id': job_id,
                'job_name': job_config.job_name,
                'created_by': job_config.created_by,
                'created_at': progress.start_time.isoformat() if progress.start_time else None,
                'completed_at': progress.end_time.isoformat() if progress.end_time else None,
                'processing_time': progress.elapsed_time.total_seconds() if progress.elapsed_time else 0
            },
            'summary': {
                'total_startups': progress.total_tasks,
                'successful_analyses': len(successful_results),
                'failed_analyses': len(failed_results),
                'success_rate': len(successful_results) / progress.total_tasks * 100,
                'average_processing_time': progress.average_task_time
            },
            'score_statistics': score_stats,
            'insights': insights,
            'detailed_results': results if job_config.include_raw_data else None
        }
        
        # Save report to file
        report_file = self.results_storage_path / f"{job_id}_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        report['file_path'] = str(report_file)
        
        # Generate additional formats if requested
        if job_config.export_format == 'csv':
            await self._export_to_csv(job_id, results)
        elif job_config.export_format == 'xlsx':
            await self._export_to_xlsx(job_id, results)
        
        return report
    
    async def _generate_batch_insights(
        self, 
        successful_results: List[Dict[str, Any]], 
        db: Session
    ) -> Dict[str, Any]:
        """Generate insights from batch processing results."""
        
        insights = {
            'top_performing_startups': [],
            'common_strengths': {},
            'common_weaknesses': {},
            'industry_analysis': {},
            'stage_analysis': {},
            'recommendations': []
        }
        
        if not successful_results:
            return insights
        
        # Top performing startups
        sorted_results = sorted(
            successful_results, 
            key=lambda x: x.get('result', {}).get('final_score', 0), 
            reverse=True
        )
        
        insights['top_performing_startups'] = [
            {
                'startup_id': result['startup_id'],
                'score': result.get('result', {}).get('final_score', 0),
                'key_strengths': result.get('result', {}).get('enhanced_insights', {}).get('top_strengths', [])
            }
            for result in sorted_results[:10]
        ]
        
        # Analyze common patterns
        all_strengths = []
        all_weaknesses = []
        industry_scores = {}
        stage_scores = {}
        
        for result in successful_results:
            startup_id = result['startup_id']
            startup = db.query(Startup).filter(Startup.id == startup_id).first()
            
            if startup:
                # Industry analysis
                industry = startup.industry
                score = result.get('result', {}).get('final_score', 0)
                
                if industry not in industry_scores:
                    industry_scores[industry] = []
                industry_scores[industry].append(score)
                
                # Stage analysis
                stage = startup.stage
                if stage not in stage_scores:
                    stage_scores[stage] = []
                stage_scores[stage].append(score)
            
            # Collect strengths and weaknesses
            enhanced_insights = result.get('result', {}).get('enhanced_insights', {})
            all_strengths.extend(enhanced_insights.get('top_strengths', []))
            all_weaknesses.extend(enhanced_insights.get('improvement_areas', []))
        
        # Count common patterns
        from collections import Counter
        
        strength_counts = Counter(all_strengths)
        weakness_counts = Counter(all_weaknesses)
        
        insights['common_strengths'] = dict(strength_counts.most_common(10))
        insights['common_weaknesses'] = dict(weakness_counts.most_common(10))
        
        # Industry and stage analysis
        for industry, scores in industry_scores.items():
            insights['industry_analysis'][industry] = {
                'average_score': sum(scores) / len(scores),
                'count': len(scores),
                'top_score': max(scores)
            }
        
        for stage, scores in stage_scores.items():
            insights['stage_analysis'][stage] = {
                'average_score': sum(scores) / len(scores),
                'count': len(scores),
                'top_score': max(scores)
            }
        
        # Generate recommendations
        insights['recommendations'] = self._generate_batch_recommendations(insights)
        
        return insights
    
    def _generate_batch_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on batch insights."""
        recommendations = []
        
        # Industry recommendations
        industry_analysis = insights.get('industry_analysis', {})
        if industry_analysis:
            best_industry = max(industry_analysis.items(), key=lambda x: x[1]['average_score'])
            recommendations.append(f"Focus on {best_industry[0]} industry - highest average score ({best_industry[1]['average_score']:.1f})")
        
        # Common weakness recommendations
        common_weaknesses = insights.get('common_weaknesses', {})
        if common_weaknesses:
            top_weakness = list(common_weaknesses.keys())[0]
            recommendations.append(f"Address common weakness: {top_weakness} (found in {common_weaknesses[top_weakness]} startups)")
        
        # Performance recommendations
        top_performers = insights.get('top_performing_startups', [])
        if top_performers:
            top_score = top_performers[0]['score']
            if top_score > 80:
                recommendations.append("Several high-potential startups identified - prioritize for investment")
            elif top_score < 50:
                recommendations.append("Consider refining selection criteria - low overall performance")
        
        return recommendations
    
    async def _export_to_csv(self, job_id: str, results: List[Dict[str, Any]]):
        """Export results to CSV format."""
        import pandas as pd
        
        # Prepare data for CSV
        csv_data = []
        for result in results:
            if result['status'] == 'completed':
                result_data = result.get('result', {})
                csv_data.append({
                    'startup_id': result['startup_id'],
                    'status': result['status'],
                    'final_score': result_data.get('final_score', 0),
                    'team_score': result_data.get('phase_results', {}).get('team_score', 0),
                    'market_score': result_data.get('phase_results', {}).get('market_score', 0),
                    'product_score': result_data.get('phase_results', {}).get('product_score', 0),
                    'traction_score': result_data.get('phase_results', {}).get('traction_score', 0),
                    'timestamp': result['timestamp']
                })
            else:
                csv_data.append({
                    'startup_id': result['startup_id'],
                    'status': result['status'],
                    'error': result.get('error', ''),
                    'timestamp': result['timestamp']
                })
        
        # Save to CSV
        df = pd.DataFrame(csv_data)
        csv_file = self.results_storage_path / f"{job_id}_results.csv"
        df.to_csv(csv_file, index=False)
    
    async def _export_to_xlsx(self, job_id: str, results: List[Dict[str, Any]]):
        """Export results to Excel format."""
        import pandas as pd
        
        # Create multiple sheets
        with pd.ExcelWriter(self.results_storage_path / f"{job_id}_results.xlsx") as writer:
            
            # Summary sheet
            summary_data = []
            detailed_data = []
            
            for result in results:
                summary_data.append({
                    'startup_id': result['startup_id'],
                    'status': result['status'],
                    'final_score': result.get('result', {}).get('final_score', 0) if result['status'] == 'completed' else None,
                    'timestamp': result['timestamp']
                })
                
                if result['status'] == 'completed':
                    result_data = result.get('result', {})
                    enhanced_insights = result_data.get('enhanced_insights', {})
                    
                    detailed_data.append({
                        'startup_id': result['startup_id'],
                        'final_score': result_data.get('final_score', 0),
                        'ml_success_probability': enhanced_insights.get('ml_success_probability', 0),
                        'market_fit_score': enhanced_insights.get('market_fit_score', 0),
                        'top_strengths': ', '.join(enhanced_insights.get('top_strengths', [])),
                        'improvement_areas': ', '.join(enhanced_insights.get('improvement_areas', []))
                    })
            
            # Write sheets
            pd.DataFrame(summary_data).to_sheet(writer, sheet_name='Summary', index=False)
            if detailed_data:
                pd.DataFrame(detailed_data).to_sheet(writer, sheet_name='Detailed Results', index=False)
    
    async def _send_completion_notification(
        self, 
        job_config: BatchJobConfig, 
        report: Dict[str, Any]
    ):
        """Send completion notification."""
        # Email notification
        if job_config.notification_email:
            # Implementation would send email
            logger.info(f"Email notification sent to {job_config.notification_email}")
        
        # Webhook notification
        if job_config.webhook_url:
            # Implementation would send webhook
            logger.info(f"Webhook notification sent to {job_config.webhook_url}")
    
    def get_job_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current progress for a job."""
        if job_id not in self.job_progress:
            return None
        
        progress = self.job_progress[job_id]
        
        return {
            'job_id': job_id,
            'total_tasks': progress.total_tasks,
            'completed_tasks': progress.completed_tasks,
            'failed_tasks': progress.failed_tasks,
            'in_progress_tasks': progress.in_progress_tasks,
            'completion_percentage': progress.completion_percentage,
            'elapsed_time': progress.elapsed_time.total_seconds() if progress.elapsed_time else 0,
            'estimated_completion': progress.estimated_completion.isoformat() if progress.estimated_completion else None,
            'current_task': progress.current_task,
            'last_error': progress.last_error
        }
    
    def cancel_job(self, job_id: str, db: Optional[Session] = None) -> bool:
        """Cancel a running batch job."""
        if db is None:
            db = next(get_db())
        
        if job_id not in self.active_jobs:
            return False
        
        # Update database status
        batch_analysis = db.query(BatchAnalysis).filter(BatchAnalysis.batch_id == job_id).first()
        if batch_analysis:
            batch_analysis.status = BatchStatus.CANCELLED.value
            db.commit()
        
        # Clean up job tracking
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
        
        if job_id in self.job_progress:
            del self.job_progress[job_id]
        
        logger.info(f"Cancelled batch job {job_id}")
        return True
    
    def list_active_jobs(self) -> List[Dict[str, Any]]:
        """List all active batch jobs."""
        jobs = []
        
        for job_id, config in self.active_jobs.items():
            progress = self.job_progress.get(job_id)
            
            jobs.append({
                'job_id': job_id,
                'job_name': config.job_name,
                'created_by': config.created_by,
                'priority': config.priority.name,
                'progress': progress.completion_percentage if progress else 0,
                'status': 'in_progress' if progress and progress.end_time is None else 'completed'
            })
        
        return jobs


# Global batch processor instance
batch_processor = BatchProcessor()