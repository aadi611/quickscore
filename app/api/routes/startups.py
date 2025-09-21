"""
Startup management API endpoints.
"""
import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.api.deps import get_database, rate_limit_check
from app.models.crud import startup_crud, analysis_crud, founder_crud
from app.models.schemas import (
    Startup, StartupCreate, StartupUpdate, 
    APIResponse, ErrorResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=Startup, status_code=status.HTTP_201_CREATED)
async def create_startup(
    startup_data: StartupCreate,
    db: Session = Depends(get_database),
    _: None = Depends(rate_limit_check)
):
    """
    Create a new startup entry.
    
    Creates a new startup in the database with the provided information.
    This is typically the first step before triggering an analysis.
    """
    try:
        # Check if startup with this name already exists
        existing_startup = startup_crud.get_by_name(db, startup_data.name)
        if existing_startup:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Startup with name '{startup_data.name}' already exists"
            )
        
        # Create new startup
        startup = startup_crud.create(db, startup_data)
        logger.info(f"Created startup: {startup.id} - {startup.name}")
        
        return startup
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating startup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create startup"
        )


@router.get("/", response_model=List[Startup])
async def list_startups(
    skip: int = Query(0, ge=0, description="Number of startups to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of startups to return"),
    industry: Optional[str] = Query(None, description="Filter by industry"),
    stage: Optional[str] = Query(None, description="Filter by startup stage"),
    db: Session = Depends(get_database)
):
    """
    List startups with optional filtering.
    
    Retrieve a paginated list of startups with optional filtering by industry and stage.
    """
    try:
        startups = startup_crud.get_multi(
            db, 
            skip=skip, 
            limit=limit, 
            industry=industry, 
            stage=stage
        )
        
        logger.info(f"Retrieved {len(startups)} startups")
        return startups
        
    except Exception as e:
        logger.error(f"Error listing startups: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve startups"
        )


@router.get("/{startup_id}", response_model=Startup)
async def get_startup(
    startup_id: UUID,
    db: Session = Depends(get_database)
):
    """
    Get a specific startup by ID.
    
    Retrieve detailed information about a specific startup including
    its basic information and metadata.
    """
    try:
        startup = startup_crud.get(db, startup_id)
        if not startup:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Startup not found"
            )
        
        return startup
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting startup {startup_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve startup"
        )


@router.put("/{startup_id}", response_model=Startup)
async def update_startup(
    startup_id: UUID,
    startup_update: StartupUpdate,
    db: Session = Depends(get_database),
    _: None = Depends(rate_limit_check)
):
    """
    Update an existing startup.
    
    Update the information for an existing startup. Only provided fields
    will be updated; omitted fields will remain unchanged.
    """
    try:
        startup = startup_crud.get(db, startup_id)
        if not startup:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Startup not found"
            )
        
        # Check for name conflicts if name is being updated
        if startup_update.name and startup_update.name != startup.name:
            existing_startup = startup_crud.get_by_name(db, startup_update.name)
            if existing_startup:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Startup with name '{startup_update.name}' already exists"
                )
        
        updated_startup = startup_crud.update(db, startup, startup_update)
        logger.info(f"Updated startup: {startup_id}")
        
        return updated_startup
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating startup {startup_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update startup"
        )


@router.delete("/{startup_id}", response_model=APIResponse)
async def delete_startup(
    startup_id: UUID,
    db: Session = Depends(get_database),
    _: None = Depends(rate_limit_check)
):
    """
    Delete a startup.
    
    Permanently delete a startup and all associated data including
    analyses, documents, and founder information.
    """
    try:
        startup = startup_crud.get(db, startup_id)
        if not startup:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Startup not found"
            )
        
        # Delete startup (cascade will handle related records)
        success = startup_crud.delete(db, startup_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete startup"
            )
        
        logger.info(f"Deleted startup: {startup_id}")
        
        return APIResponse(
            success=True,
            message="Startup deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting startup {startup_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete startup"
        )


@router.get("/{startup_id}/analyses", response_model=List[dict])
async def get_startup_analyses(
    startup_id: UUID,
    db: Session = Depends(get_database)
):
    """
    Get all analyses for a startup.
    
    Retrieve all analyses that have been performed for a specific startup,
    ordered by creation date (most recent first).
    """
    try:
        # Verify startup exists
        startup = startup_crud.get(db, startup_id)
        if not startup:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Startup not found"
            )
        
        # Get analyses
        analyses = analysis_crud.get_by_startup(db, startup_id)
        
        # Convert to response format
        analyses_data = []
        for analysis in analyses:
            analysis_data = {
                "id": analysis.id,
                "status": analysis.status,
                "overall_score": analysis.overall_score,
                "recommendation": analysis.recommendation,
                "confidence": analysis.confidence,
                "created_at": analysis.created_at,
                "completed_at": analysis.completed_at,
                "processing_time": analysis.processing_time
            }
            analyses_data.append(analysis_data)
        
        return analyses_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analyses for startup {startup_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analyses"
        )


@router.get("/{startup_id}/founders", response_model=List[dict])
async def get_startup_founders(
    startup_id: UUID,
    db: Session = Depends(get_database)
):
    """
    Get all founders for a startup.
    
    Retrieve information about all founders associated with a startup.
    """
    try:
        # Verify startup exists
        startup = startup_crud.get(db, startup_id)
        if not startup:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Startup not found"
            )
        
        # Get founders
        founders = founder_crud.get_by_startup(db, startup_id)
        
        # Convert to response format
        founders_data = []
        for founder in founders:
            founder_data = {
                "id": founder.id,
                "name": founder.name,
                "linkedin_url": founder.linkedin_url,
                "title": founder.title,
                "experience_years": founder.experience_years,
                "previous_exits": founder.previous_exits,
                "domain_expert": founder.domain_expert,
                "score": founder.score,
                "created_at": founder.created_at
            }
            founders_data.append(founder_data)
        
        return founders_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting founders for startup {startup_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve founders"
        )