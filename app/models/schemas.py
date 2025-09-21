"""
Pydantic schemas for request/response validation.
"""
from datetime import datetime
from typing import Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


# Base schemas
class StartupBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    website: Optional[HttpUrl] = None
    linkedin_url: Optional[HttpUrl] = None
    industry: str = Field(..., min_length=1, max_length=100)
    stage: str = Field(default="pre_seed", max_length=50)
    description: Optional[str] = None


class StartupCreate(StartupBase):
    pass


class StartupUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    website: Optional[HttpUrl] = None
    linkedin_url: Optional[HttpUrl] = None
    industry: Optional[str] = Field(None, min_length=1, max_length=100)
    stage: Optional[str] = Field(None, max_length=50)
    description: Optional[str] = None


class Startup(StartupBase):
    id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Analysis schemas
class AnalysisScores(BaseModel):
    overall_score: Optional[float] = Field(None, ge=0, le=100)
    team_score: Optional[float] = Field(None, ge=0, le=100)
    market_score: Optional[float] = Field(None, ge=0, le=100)
    product_score: Optional[float] = Field(None, ge=0, le=100)
    traction_score: Optional[float] = Field(None, ge=0, le=100)
    pitch_quality_score: Optional[float] = Field(None, ge=0, le=100)


class AnalysisInsights(BaseModel):
    strengths: List[str] = []
    risks: List[str] = []
    next_steps: List[str] = []
    key_observations: List[str] = []


class AnalysisRequest(BaseModel):
    founder_linkedin: Optional[HttpUrl] = None
    additional_links: Optional[List[HttpUrl]] = []
    analysis_depth: str = Field(default="standard", pattern="^(quick|standard|deep)$")


class Analysis(BaseModel):
    id: UUID
    startup_id: UUID
    status: str
    recommendation: Optional[str] = None
    confidence: Optional[str] = None
    insights: Optional[AnalysisInsights] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AnalysisWithScores(Analysis):
    scores: AnalysisScores


# Founder schemas
class FounderBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    linkedin_url: Optional[HttpUrl] = None
    title: Optional[str] = Field(None, max_length=255)


class FounderCreate(FounderBase):
    startup_id: UUID


class Founder(FounderBase):
    id: UUID
    startup_id: UUID
    experience_years: Optional[int] = None
    previous_exits: int = 0
    previous_startups: int = 0
    domain_expert: bool = False
    technical_background: bool = False
    business_background: bool = False
    score: Optional[float] = Field(None, ge=0, le=100)
    created_at: datetime

    class Config:
        from_attributes = True


# Comparable startups schemas
class ComparableStartup(BaseModel):
    name: str
    description: Optional[str] = None
    industry: str
    stage: str
    total_funding: Optional[float] = None
    last_round_amount: Optional[float] = None
    valuation: Optional[float] = None
    employee_count: Optional[int] = None
    similarity_score: Optional[float] = Field(None, ge=0, le=1)
    investors: Optional[List[str]] = []


class ComparablesResponse(BaseModel):
    startup_id: UUID
    comparables: List[ComparableStartup]
    search_params: Dict[str, Union[str, float, int]]


# Batch analysis schemas
class BatchAnalysisRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    analysis_depth: str = Field(default="standard", pattern="^(quick|standard|deep)$")


class BatchAnalysisStatus(BaseModel):
    id: UUID
    name: str
    total_startups: int
    completed_startups: int
    failed_startups: int
    status: str
    progress_percentage: float
    estimated_completion: Optional[datetime] = None
    created_at: datetime


# Report generation schemas
class ReportRequest(BaseModel):
    analysis_id: UUID
    format: str = Field(..., pattern="^(pdf|excel|json)$")
    template: str = Field(default="executive", pattern="^(executive|detailed|pitch)$")
    include_comparables: bool = True
    recipient_email: Optional[str] = None


class ReportResponse(BaseModel):
    report_id: UUID
    download_url: str
    format: str
    created_at: datetime
    expires_at: datetime


# API Response schemas
class APIResponse(BaseModel):
    success: bool = True
    message: str = "Success"
    data: Optional[Dict] = None


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    error_code: Optional[str] = None
    details: Optional[Dict] = None


# File upload schemas
class DocumentUpload(BaseModel):
    filename: str
    file_type: str = Field(..., pattern="^(pitch_deck|financial|other)$")
    file_size: int = Field(..., gt=0)


# Health check schema
class HealthCheck(BaseModel):
    status: str = "healthy"
    timestamp: datetime
    version: str
    database: str = "connected"
    redis: str = "connected"
    celery: str = "connected"