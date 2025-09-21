"""
CRUD operations for database models.
"""
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from app.models import Startup, Analysis, Founder, Document, ComparableStartup, BatchAnalysis
from app.models.schemas import StartupCreate, StartupUpdate, AnalysisRequest


class CRUDStartup:
    """CRUD operations for Startup model."""
    
    def get(self, db: Session, startup_id: UUID) -> Optional[Startup]:
        return db.query(Startup).filter(Startup.id == startup_id).first()
    
    def get_by_name(self, db: Session, name: str) -> Optional[Startup]:
        return db.query(Startup).filter(Startup.name == name).first()
    
    def get_multi(
        self, 
        db: Session, 
        skip: int = 0, 
        limit: int = 100,
        industry: Optional[str] = None,
        stage: Optional[str] = None
    ) -> List[Startup]:
        query = db.query(Startup)
        
        if industry:
            query = query.filter(Startup.industry == industry)
        if stage:
            query = query.filter(Startup.stage == stage)
            
        return query.offset(skip).limit(limit).all()
    
    def create(self, db: Session, startup_in: StartupCreate) -> Startup:
        startup = Startup(**startup_in.dict())
        db.add(startup)
        db.commit()
        db.refresh(startup)
        return startup
    
    def update(
        self, 
        db: Session, 
        startup: Startup, 
        startup_update: StartupUpdate
    ) -> Startup:
        update_data = startup_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(startup, field, value)
        
        db.commit()
        db.refresh(startup)
        return startup
    
    def delete(self, db: Session, startup_id: UUID) -> bool:
        startup = self.get(db, startup_id)
        if startup:
            db.delete(startup)
            db.commit()
            return True
        return False


class CRUDAnalysis:
    """CRUD operations for Analysis model."""
    
    def get(self, db: Session, analysis_id: UUID) -> Optional[Analysis]:
        return db.query(Analysis).filter(Analysis.id == analysis_id).first()
    
    def get_by_startup(self, db: Session, startup_id: UUID) -> List[Analysis]:
        return db.query(Analysis).filter(
            Analysis.startup_id == startup_id
        ).order_by(desc(Analysis.created_at)).all()
    
    def get_latest_by_startup(self, db: Session, startup_id: UUID) -> Optional[Analysis]:
        return db.query(Analysis).filter(
            Analysis.startup_id == startup_id
        ).order_by(desc(Analysis.created_at)).first()
    
    def create(self, db: Session, startup_id: UUID) -> Analysis:
        analysis = Analysis(startup_id=startup_id, status="processing")
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        return analysis
    
    def update_status(self, db: Session, analysis_id: UUID, status: str) -> Optional[Analysis]:
        analysis = self.get(db, analysis_id)
        if analysis:
            analysis.status = status
            db.commit()
            db.refresh(analysis)
        return analysis
    
    def update_scores(
        self, 
        db: Session, 
        analysis_id: UUID, 
        scores: Dict[str, float]
    ) -> Optional[Analysis]:
        analysis = self.get(db, analysis_id)
        if analysis:
            for score_type, value in scores.items():
                setattr(analysis, score_type, value)
            db.commit()
            db.refresh(analysis)
        return analysis
    
    def complete_analysis(
        self,
        db: Session,
        analysis_id: UUID,
        results: Dict[str, Any]
    ) -> Optional[Analysis]:
        analysis = self.get(db, analysis_id)
        if analysis:
            analysis.status = "completed"
            analysis.overall_score = results.get("overall_score")
            analysis.team_score = results.get("team_score")
            analysis.market_score = results.get("market_score")
            analysis.product_score = results.get("product_score")
            analysis.traction_score = results.get("traction_score")
            analysis.pitch_quality_score = results.get("pitch_quality_score")
            analysis.recommendation = results.get("recommendation")
            analysis.confidence = results.get("confidence")
            analysis.insights = results.get("insights")
            analysis.raw_llm_outputs = results.get("raw_llm_outputs")
            analysis.processing_time = results.get("processing_time")
            
            from datetime import datetime
            analysis.completed_at = datetime.utcnow()
            
            db.commit()
            db.refresh(analysis)
        return analysis


class CRUDFounder:
    """CRUD operations for Founder model."""
    
    def get(self, db: Session, founder_id: UUID) -> Optional[Founder]:
        return db.query(Founder).filter(Founder.id == founder_id).first()
    
    def get_by_startup(self, db: Session, startup_id: UUID) -> List[Founder]:
        return db.query(Founder).filter(Founder.startup_id == startup_id).all()
    
    def create(
        self, 
        db: Session, 
        startup_id: UUID, 
        founder_data: Dict[str, Any]
    ) -> Founder:
        founder = Founder(startup_id=startup_id, **founder_data)
        db.add(founder)
        db.commit()
        db.refresh(founder)
        return founder
    
    def update_profile_data(
        self,
        db: Session,
        founder_id: UUID,
        profile_data: Dict[str, Any]
    ) -> Optional[Founder]:
        founder = self.get(db, founder_id)
        if founder:
            founder.profile_data = profile_data
            
            # Extract computed metrics from profile data
            founder.experience_years = self._calculate_experience_years(profile_data)
            founder.domain_expert = self._is_domain_expert(profile_data)
            founder.technical_background = self._has_technical_background(profile_data)
            founder.business_background = self._has_business_background(profile_data)
            
            db.commit()
            db.refresh(founder)
        return founder
    
    def _calculate_experience_years(self, profile_data: Dict) -> int:
        """Calculate years of experience from profile data."""
        # Implementation would analyze work experience
        # This is a simplified version
        return profile_data.get("total_experience_years", 0)
    
    def _is_domain_expert(self, profile_data: Dict) -> bool:
        """Determine if founder is a domain expert."""
        # Implementation would analyze industry experience
        return profile_data.get("domain_expert", False)
    
    def _has_technical_background(self, profile_data: Dict) -> bool:
        """Check if founder has technical background."""
        # Implementation would analyze job titles and skills
        return profile_data.get("technical_background", False)
    
    def _has_business_background(self, profile_data: Dict) -> bool:
        """Check if founder has business background."""
        # Implementation would analyze job titles and education
        return profile_data.get("business_background", False)


class CRUDComparable:
    """CRUD operations for ComparableStartup model."""
    
    def find_similar(
        self,
        db: Session,
        industry: str,
        description: str,
        limit: int = 10,
        min_similarity: float = 0.7
    ) -> List[ComparableStartup]:
        """Find similar startups based on industry and description."""
        # Basic implementation - would be enhanced with embedding similarity
        query = db.query(ComparableStartup).filter(
            ComparableStartup.industry == industry
        )
        
        return query.limit(limit).all()
    
    def create_or_update(
        self,
        db: Session,
        startup_data: Dict[str, Any]
    ) -> ComparableStartup:
        """Create or update comparable startup data."""
        existing = db.query(ComparableStartup).filter(
            ComparableStartup.name == startup_data["name"]
        ).first()
        
        if existing:
            for key, value in startup_data.items():
                setattr(existing, key, value)
            comparable = existing
        else:
            comparable = ComparableStartup(**startup_data)
            db.add(comparable)
        
        db.commit()
        db.refresh(comparable)
        return comparable


class CRUDBatchAnalysis:
    """CRUD operations for BatchAnalysis model."""
    
    def get(self, db: Session, batch_id: UUID) -> Optional[BatchAnalysis]:
        return db.query(BatchAnalysis).filter(BatchAnalysis.id == batch_id).first()
    
    def create(
        self,
        db: Session,
        name: str,
        total_startups: int,
        analysis_depth: str = "standard"
    ) -> BatchAnalysis:
        batch = BatchAnalysis(
            name=name,
            total_startups=total_startups,
            analysis_depth=analysis_depth
        )
        db.add(batch)
        db.commit()
        db.refresh(batch)
        return batch
    
    def update_progress(
        self,
        db: Session,
        batch_id: UUID,
        completed: int,
        failed: int
    ) -> Optional[BatchAnalysis]:
        batch = self.get(db, batch_id)
        if batch:
            batch.completed_startups = completed
            batch.failed_startups = failed
            
            if completed + failed >= batch.total_startups:
                batch.status = "completed"
                from datetime import datetime
                batch.completed_at = datetime.utcnow()
            
            db.commit()
            db.refresh(batch)
        return batch


# Initialize CRUD instances
startup_crud = CRUDStartup()
analysis_crud = CRUDAnalysis()
founder_crud = CRUDFounder()
comparable_crud = CRUDComparable()
batch_crud = CRUDBatchAnalysis()