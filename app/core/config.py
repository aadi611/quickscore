
"""
Core configuration settings for QuickScore.

Centralized, validated, environment-driven application configuration.
Compatible with Pydantic Settings v2.
"""

from functools import lru_cache
from typing import Literal, Mapping
from types import MappingProxyType

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        frozen=True,
        validate_default=True,
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "QuickScore"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------

    SECRET_KEY: str = Field(
        min_length=32,
        description="Application secret key. Use a cryptographically secure value.",
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, gt=0)

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    DATABASE_URL: str = Field(min_length=1)
    TEST_DATABASE_URL: str = ""

    # ------------------------------------------------------------------
    # Supabase
    # ------------------------------------------------------------------

    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------

    REDIS_URL: str = "redis://localhost:6379/0"

    UPSTASH_REDIS_REST_URL: str = ""
    UPSTASH_REDIS_REST_TOKEN: str = ""

    # ------------------------------------------------------------------
    # Celery
    # ------------------------------------------------------------------

    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------

    CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
        ]
    )

    # ------------------------------------------------------------------
    # LLM Providers
    # ------------------------------------------------------------------

    LLM_PROVIDER: Literal["openai", "groq"] = "groq"

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4-1106-preview"

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "meta-llama/llama-4-maverick-17b-128e-instruct"

    # ------------------------------------------------------------------
    # External APIs
    # ------------------------------------------------------------------

    GITHUB_TOKEN: str = ""

    # ------------------------------------------------------------------
    # Machine Learning
    # ------------------------------------------------------------------

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    MAX_WORKERS: int = Field(default=4, gt=0)

    # ------------------------------------------------------------------
    # File Upload
    # ------------------------------------------------------------------

    MAX_FILE_SIZE_MB: int = Field(default=10, gt=0)

    ALLOWED_EXTENSIONS: list[str] = Field(
        default_factory=lambda: ["pdf", "docx", "xlsx", "csv"]
    )

    # ------------------------------------------------------------------
    # Web Scraping
    # ------------------------------------------------------------------

    SCRAPER_DELAY_MIN: int = Field(default=1, ge=0)
    SCRAPER_DELAY_MAX: int = Field(default=3, ge=0)
    SCRAPER_TIMEOUT: int = Field(default=30, gt=0)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    LOG_LEVEL: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    LOG_FORMAT: Literal["json", "text"] = "json"

    # ------------------------------------------------------------------
    # Rate Limiting
    # ------------------------------------------------------------------

    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, gt=0)

    # ------------------------------------------------------------------
    # Field Validation
    # ------------------------------------------------------------------

    @field_validator("CORS_ORIGINS")
    @classmethod
    def validate_cors_origins(cls, origins: list[str]) -> list[str]:
        """Remove whitespace and reject empty origins."""
        normalized = [origin.strip().rstrip("/") for origin in origins]

        if any(not origin for origin in normalized):
            raise ValueError("CORS origins cannot contain empty values.")

        if "*" in normalized:
            raise ValueError(
                "Wildcard CORS origins are not allowed with credentials."
            )

        return normalized

    @field_validator("ALLOWED_EXTENSIONS")
    @classmethod
    def normalize_extensions(cls, extensions: list[str]) -> list[str]:
        """Normalize file extensions."""
        normalized = list(dict.fromkeys(
            ext.strip().lower().lstrip(".")
            for ext in extensions
        ))

        if not normalized or any(not ext for ext in normalized):
            raise ValueError("At least one valid file extension is required.")

        return normalized

    @model_validator(mode="after")
    def validate_scraper_delays(self) -> "Settings":
        """Ensure scraper delay boundaries are valid."""
        if self.SCRAPER_DELAY_MIN > self.SCRAPER_DELAY_MAX:
            raise ValueError(
                "SCRAPER_DELAY_MIN cannot exceed SCRAPER_DELAY_MAX."
            )

        return self


# ----------------------------------------------------------------------
# Settings Factory
# ----------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached application settings instance.

    Settings are initialized once and reused throughout the application.
    """
    return Settings()


# Backward-compatible global settings instance.
settings = get_settings()


# ----------------------------------------------------------------------
# Startup Scoring Configuration
# ----------------------------------------------------------------------

SCORING_WEIGHTS: Mapping[str, Mapping[str, float]] = MappingProxyType({
    "pre_seed": MappingProxyType({
        "team": 0.40,
        "market": 0.25,
        "product": 0.15,
        "traction": 0.10,
        "pitch_quality": 0.10,
    }),
    "seed": MappingProxyType({
        "team": 0.30,
        "market": 0.25,
        "product": 0.20,
        "traction": 0.20,
        "pitch_quality": 0.05,
    }),
})


# ----------------------------------------------------------------------
# Recommendation Thresholds
# ----------------------------------------------------------------------

RECOMMENDATION_THRESHOLDS: Mapping[str, int] = MappingProxyType({
    "strong_yes": 75,
    "yes": 60,
    "maybe": 40,
    "no": 0,
})


# ----------------------------------------------------------------------
# Industry Classification
# ----------------------------------------------------------------------

INDUSTRY_CATEGORIES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "saas": ("software", "saas", "platform", "cloud"),
    "fintech": ("finance", "fintech", "payments", "banking"),
    "healthtech": ("health", "medical", "biotech", "pharma"),
    "edtech": ("education", "learning", "training"),
    "ecommerce": ("ecommerce", "retail", "marketplace"),
    "enterprise": ("enterprise", "b2b", "automation"),
    "consumer": ("consumer", "b2c", "mobile app"),
    "deeptech": ("ai", "ml", "blockchain", "iot", "robotics"),
})
