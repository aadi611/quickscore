"""
Core configuration settings for QuickScore application.
"""
import os
import numpy as np
import pandas as pd
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "QuickScore"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Security
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: str
    TEST_DATABASE_URL: str = ""
    
    # Supabase Configuration
    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Upstash Redis (optional - for hosted Redis)
    UPSTASH_REDIS_REST_URL: str = ""
    UPSTASH_REDIS_REST_TOKEN: str = ""
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # LLM Provider Configuration
    LLM_PROVIDER: str = "groq"  # Options: 'openai', 'groq'
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4-1106-preview"
    
    # Groq Configuration  
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "meta-llama/llama-4-maverick-17b-128e-instruct"
    
    # External APIs
    GITHUB_TOKEN: str = ""
    
    # ML Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    MAX_WORKERS: int = 4
    
    # File Upload
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "docx", "xlsx", "csv"]
    
    # Scraping
    SCRAPER_DELAY_MIN: int = 1
    SCRAPER_DELAY_MAX: int = 3
    SCRAPER_TIMEOUT: int = 30
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Scoring weights for different startup stages
SCORING_WEIGHTS = {
    "pre_seed": {
        "team": 0.40,
        "market": 0.25,
        "product": 0.15,
        "traction": 0.10,
        "pitch_quality": 0.10
    },
    "seed": {
        "team": 0.30,
        "market": 0.25,
        "product": 0.20,
        "traction": 0.20,
        "pitch_quality": 0.05
    }
}

# Recommendation thresholds
RECOMMENDATION_THRESHOLDS = {
    "strong_yes": 75,
    "yes": 60,
    "maybe": 40,
    "no": 0
}

# Industry mappings for better analysis
INDUSTRY_CATEGORIES = {
    "saas": ["software", "saas", "platform", "cloud"],
    "fintech": ["finance", "fintech", "payments", "banking"],
    "healthtech": ["health", "medical", "biotech", "pharma"],
    "edtech": ["education", "learning", "training"],
    "ecommerce": ["ecommerce", "retail", "marketplace"],
    "enterprise": ["enterprise", "b2b", "automation"],
    "consumer": ["consumer", "b2c", "mobile app"],
    "deeptech": ["ai", "ml", "blockchain", "iot", "robotics"]
}
