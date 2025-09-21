"""
SQLAlchemy models for QuickScore application.
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.database import Base


class Startup(Base):
    """Startup entity model."""
    __tablename__ = "startups"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String(255), nullable=False, index=True)
    website = Column(String(255), nullable=True)
    linkedin_url = Column(String(500), nullable=True)
    industry = Column(String(100), nullable=False, index=True)
    stage = Column(String(50), nullable=False, default="pre_seed", index=True)
    description = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    analyses = relationship("Analysis", back_populates="startup", cascade="all, delete-orphan")
    founders = relationship("Founder", back_populates="startup", cascade="all, delete-orphan")


class Analysis(Base):
    """Analysis results model."""
    __tablename__ = "analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    startup_id = Column(UUID(as_uuid=True), ForeignKey("startups.id"), nullable=False)
    
    # Analysis status
    status = Column(String(20), nullable=False, default="processing", index=True)  # processing, completed, failed
    
    # Scores (0-100)
    overall_score = Column(Float, nullable=True)
    team_score = Column(Float, nullable=True)
    market_score = Column(Float, nullable=True)
    product_score = Column(Float, nullable=True)
    traction_score = Column(Float, nullable=True)
    pitch_quality_score = Column(Float, nullable=True)
    
    # Recommendations
    recommendation = Column(String(50), nullable=True)  # strong_yes, yes, maybe, no
    confidence = Column(String(20), nullable=True)  # high, medium, low
    
    # Structured insights
    insights = Column(JSON, nullable=True)  # {strengths: [], risks: [], next_steps: []}
    
    # Raw data and processing
    raw_llm_outputs = Column(JSON, nullable=True)
    extracted_data = Column(JSON, nullable=True)
    processing_time = Column(Float, nullable=True)  # seconds
    error_message = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    startup = relationship("Startup", back_populates="analyses")


class Founder(Base):
    """Founder profile model."""
    __tablename__ = "founders"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    startup_id = Column(UUID(as_uuid=True), ForeignKey("startups.id"), nullable=False)
    
    # Basic info
    name = Column(String(255), nullable=False)
    linkedin_url = Column(String(500), nullable=True)
    title = Column(String(255), nullable=True)
    
    # Extracted profile data
    profile_data = Column(JSON, nullable=True)  # LinkedIn scraped data
    
    # Computed metrics
    experience_years = Column(Integer, nullable=True)
    previous_exits = Column(Integer, nullable=False, default=0)
    previous_startups = Column(Integer, nullable=False, default=0)
    domain_expert = Column(Boolean, nullable=False, default=False)
    technical_background = Column(Boolean, nullable=False, default=False)
    business_background = Column(Boolean, nullable=False, default=False)
    
    # Scores
    score = Column(Float, nullable=True)  # Overall founder score 0-100
    network_strength_score = Column(Float, nullable=True)
    leadership_score = Column(Float, nullable=True)
    execution_score = Column(Float, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    startup = relationship("Startup", back_populates="founders")


class Document(Base):
    """Document storage and processing model."""
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    startup_id = Column(UUID(as_uuid=True), ForeignKey("startups.id"), nullable=False)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey("analyses.id"), nullable=True)
    
    # File info
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # pitch_deck, financial, other
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    
    # Processing status
    processed = Column(Boolean, nullable=False, default=False)
    extracted_content = Column(JSON, nullable=True)
    processing_error = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)


class ComparableStartup(Base):
    """Model for storing startup comparables data."""
    __tablename__ = "comparable_startups"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Basic startup info
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    industry = Column(String(100), nullable=False, index=True)
    stage = Column(String(50), nullable=False, index=True)
    
    # Funding data
    total_funding = Column(Float, nullable=True)  # USD
    last_round_amount = Column(Float, nullable=True)
    last_round_date = Column(DateTime, nullable=True)
    valuation = Column(Float, nullable=True)
    
    # Company metrics
    employee_count = Column(Integer, nullable=True)
    founded_year = Column(Integer, nullable=True)
    headquarters = Column(String(255), nullable=True)
    
    # Investors
    investors = Column(JSON, nullable=True)  # List of investor names
    
    # External data
    crunchbase_url = Column(String(500), nullable=True)
    website = Column(String(500), nullable=True)
    
    # Embedding for similarity search
    embedding = Column(JSON, nullable=True)  # Vector embedding for similarity
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class BatchAnalysis(Base):
    """Model for tracking batch analysis jobs."""
    __tablename__ = "batch_analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Batch info
    name = Column(String(255), nullable=False)
    total_startups = Column(Integer, nullable=False)
    completed_startups = Column(Integer, nullable=False, default=0)
    failed_startups = Column(Integer, nullable=False, default=0)
    
    # Status
    status = Column(String(20), nullable=False, default="processing")  # processing, completed, failed
    
    # Configuration
    analysis_depth = Column(String(20), nullable=False, default="standard")  # quick, standard, deep
    
    # Results
    results_file_path = Column(String(500), nullable=True)
    summary_stats = Column(JSON, nullable=True)
    
    # Celery task info
    celery_task_id = Column(String(255), nullable=True, index=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)