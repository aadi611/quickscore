"""
Analysis API endpoints for startup evaluation.
"""
import logging
import asyncio
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.api.deps import (
    get_database, 
    get_document_processor, 
    get_ai_evaluator, 
    get_scoring_engine,
    rate_limit_check
)
from app.models.crud import startup_crud, analysis_crud, founder_crud
from app.models.schemas import APIResponse, AnalysisRequest
from app.services.document_processor import DocumentProcessor
from app.services.ai_evaluator import StartupEvaluator
from app.core.scoring import ScoringEngine

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/startups/{startup_id}/analyze", response_model=dict)
async def trigger_analysis(
    startup_id: UUID,
    founder_linkedin: Optional[str] = Form(None),
    additional_links: Optional[str] = Form(None),  # JSON string of additional URLs
    pitch_deck: Optional[UploadFile] = File(None),
    financial_doc: Optional[UploadFile] = File(None),
    db: Session = Depends(get_database),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    ai_evaluator: StartupEvaluator = Depends(get_ai_evaluator),
    scoring_engine: ScoringEngine = Depends(get_scoring_engine),
    _: None = Depends(rate_limit_check)
):
    """
    Trigger comprehensive analysis for a startup.
    
    This endpoint accepts documents and founder information to perform
    a complete AI-powered analysis of the startup. The analysis includes:
    - Document processing (pitch deck, financials)
    - Founder assessment
    - Market analysis
    - Composite scoring and recommendations
    """
    try:
        # Verify startup exists
        startup = startup_crud.get(db, startup_id)
        if not startup:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Startup not found"
            )
        
        # Create analysis record
        analysis = analysis_crud.create(db, startup_id)
        
        # Process uploaded documents
        processed_documents = {}
        
        if pitch_deck:
            logger.info(f"Processing pitch deck for startup {startup_id}")
            # Validate file
            if pitch_deck.size > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Pitch deck file too large (max 10MB)"
                )
            
            # Read and process document
            pitch_content = await pitch_deck.read()
            pitch_result = await document_processor.process_document(
                file_content=pitch_content,
                filename=pitch_deck.filename,
                document_type="pitch_deck"
            )
            processed_documents["pitch_deck"] = pitch_result
        
        if financial_doc:
            logger.info(f"Processing financial document for startup {startup_id}")
            # Validate file
            if financial_doc.size > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Financial document file too large (max 10MB)"
                )
            
            # Read and process document
            financial_content = await financial_doc.read()
            financial_result = await document_processor.process_document(
                file_content=financial_content,
                filename=financial_doc.filename,
                document_type="financial"
            )
            processed_documents["financial"] = financial_result
        
        # Prepare founder data
        founder_data = {}
        if founder_linkedin:
            # Create or update founder record
            founder_data = {
                "name": "Founder",  # Would extract from LinkedIn
                "linkedin_url": founder_linkedin,
                "profile_data": {"linkedin_url": founder_linkedin}
            }
        
        # Prepare startup context for AI evaluation
        startup_context = {
            "name": startup.name,
            "industry": startup.industry,
            "stage": startup.stage,
            "description": startup.description,
            "website": str(startup.website) if startup.website else None
        }
        
        # Prepare market data (basic for now)
        market_data = {
            "target_market": startup.industry,
            "geography": "Global",  # Default
            "competitors": [],  # Would be enhanced with web scraping
            "additional_context": startup.description or ""
        }
        
        # Perform AI evaluation
        logger.info(f"Starting AI evaluation for startup {startup_id}")
        evaluation_results = await ai_evaluator.evaluate_startup(
            pitch_deck_content=processed_documents.get("pitch_deck"),
            founder_data=founder_data if founder_data else None,
            market_data=market_data,
            startup_context=startup_context
        )
        
        if not evaluation_results.get("success", False):
            # Update analysis with error
            analysis_crud.update_status(db, analysis.id, "failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"AI evaluation failed: {evaluation_results.get('error', 'Unknown error')}"
            )
        
        # Calculate scores
        logger.info(f"Calculating scores for startup {startup_id}")
        scoring_results = scoring_engine.calculate_composite_score(
            evaluation_results["evaluation_results"],
            startup_stage=startup.stage
        )
        
        if not scoring_results.get("success", False):
            # Update analysis with error
            analysis_crud.update_status(db, analysis.id, "failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Scoring failed: {scoring_results.get('error', 'Unknown error')}"
            )
        
        # Generate insights
        insights_result = await ai_evaluator.generate_insights(
            evaluation_results["evaluation_results"],
            startup_context
        )
        
        # Prepare final results
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
        
        # Update analysis with results
        analysis = analysis_crud.complete_analysis(db, analysis.id, final_results)
        
        # Create founder record if provided
        if founder_data:
            founder_crud.create(db, startup_id, founder_data)
        
        logger.info(f"Analysis completed for startup {startup_id}")
        
        return {
            "success": True,
            "analysis_id": analysis.id,
            "status": "completed",
            "results": {
                "overall_score": analysis.overall_score,
                "recommendation": analysis.recommendation,
                "confidence": analysis.confidence,
                "processing_time": analysis.processing_time
            },
            "message": "Analysis completed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analysis for startup {startup_id}: {e}")
        # Update analysis status to failed if it was created
        if 'analysis' in locals():
            analysis_crud.update_status(db, analysis.id, "failed")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@router.get("/analyses/{analysis_id}", response_model=dict)
async def get_analysis(
    analysis_id: UUID,
    db: Session = Depends(get_database)
):
    """
    Get detailed analysis results.
    
    Retrieve comprehensive results for a specific analysis including
    scores, recommendations, insights, and detailed breakdowns.
    """
    try:
        analysis = analysis_crud.get(db, analysis_id)
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        # Get startup information
        startup = startup_crud.get(db, analysis.startup_id)
        
        # Prepare response
        response_data = {
            "id": analysis.id,
            "startup_id": analysis.startup_id,
            "startup_name": startup.name if startup else "Unknown",
            "status": analysis.status,
            "overall_score": analysis.overall_score,
            "scores": {
                "team": analysis.team_score,
                "market": analysis.market_score,
                "product": analysis.product_score,
                "traction": analysis.traction_score,
                "pitch_quality": analysis.pitch_quality_score
            },
            "recommendation": analysis.recommendation,
            "confidence": analysis.confidence,
            "insights": analysis.insights or {},
            "created_at": analysis.created_at,
            "completed_at": analysis.completed_at,
            "processing_time": analysis.processing_time,
            "error_message": analysis.error_message
        }
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis {analysis_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analysis"
        )


@router.get("/analyses", response_model=List[dict])
async def list_analyses(
    limit: int = 50,
    status_filter: Optional[str] = None,
    db: Session = Depends(get_database)
):
    """
    List recent analyses.
    
    Get a list of recent analyses with optional status filtering.
    """
    try:
        # This would be enhanced with proper pagination and filtering
        # For now, return a basic implementation
        
        query = db.query(analysis_crud.model)  # Would use proper query building
        
        if status_filter:
            query = query.filter(analysis_crud.model.status == status_filter)
        
        analyses = query.order_by(analysis_crud.model.created_at.desc()).limit(limit).all()
        
        # Format response
        analyses_data = []
        for analysis in analyses:
            startup = startup_crud.get(db, analysis.startup_id)
            analyses_data.append({
                "id": analysis.id,
                "startup_id": analysis.startup_id,
                "startup_name": startup.name if startup else "Unknown",
                "status": analysis.status,
                "overall_score": analysis.overall_score,
                "recommendation": analysis.recommendation,
                "created_at": analysis.created_at,
                "completed_at": analysis.completed_at
            })
        
        return analyses_data
        
    except Exception as e:
        logger.error(f"Error listing analyses: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analyses"
        )


@router.get("/comparables/{startup_id}", response_model=dict)
async def find_comparable_startups(
    startup_id: UUID,
    limit: int = 10,
    min_similarity: float = 0.7,
    db: Session = Depends(get_database)
):
    """
    Find comparable startups.
    
    Find startups similar to the given startup based on industry,
    stage, and business model characteristics.
    """
    try:
        # Verify startup exists
        startup = startup_crud.get(db, startup_id)
        if not startup:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Startup not found"
            )
        
        # This would use the ML similarity search
        # For now, return a placeholder response
        
        comparable_startups = [
            {
                "name": "Similar Startup 1",
                "industry": startup.industry,
                "stage": "seed",
                "similarity_score": 0.85,
                "total_funding": 2000000,
                "description": "AI-powered solution in similar domain"
            },
            {
                "name": "Similar Startup 2", 
                "industry": startup.industry,
                "stage": "pre_seed",
                "similarity_score": 0.78,
                "total_funding": 500000,
                "description": "Early stage company with similar approach"
            }
        ]
        
        return {
            "startup_id": startup_id,
            "startup_name": startup.name,
            "comparables": comparable_startups,
            "search_params": {
                "industry": startup.industry,
                "stage": startup.stage,
                "limit": limit,
                "min_similarity": min_similarity
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding comparables for startup {startup_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to find comparable startups"
        )


@router.post("/reports/generate", response_model=dict)
async def generate_investment_report(
    analysis_id: UUID,
    format: str = "pdf",
    template: str = "executive",
    db: Session = Depends(get_database)
):
    """
    Generate investment report.
    
    Generate a formatted investment report based on analysis results.
    Supports PDF, Excel, and JSON formats with different templates.
    """
    try:
        # Verify analysis exists
        analysis = analysis_crud.get(db, analysis_id)
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        if analysis.status != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Analysis is not completed yet"
            )
        
        # Get startup information
        startup = startup_crud.get(db, analysis.startup_id)
        
        # This would generate actual reports
        # For now, return a placeholder response
        
        report_data = {
            "report_id": f"report_{analysis_id}",
            "analysis_id": analysis_id,
            "startup_name": startup.name if startup else "Unknown",
            "format": format,
            "template": template,
            "download_url": f"/reports/download/report_{analysis_id}.{format}",
            "generated_at": "2024-01-01T00:00:00Z",
            "expires_at": "2024-01-08T00:00:00Z"
        }
        
        return report_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report for analysis {analysis_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate report"
        )