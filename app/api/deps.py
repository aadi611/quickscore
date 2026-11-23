
"""
Application dependencies for FastAPI endpoints.

Provides dependency providers for database sessions, document processing,
AI evaluation, scoring, authentication, and rate limiting.
"""

from collections.abc import Generator
from typing import Annotated, TypedDict

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.scoring import ScoringEngine
from app.models.database import get_db
from app.services.ai_evaluator import StartupEvaluator
from app.services.document_processor import DocumentProcessor


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_database() -> Generator[Session, None, None]:
    """
    Provide a database session for the duration of a request.

    The underlying get_db dependency is responsible for session cleanup.
    """
    yield from get_db()


# ---------------------------------------------------------------------------
# Application Services
# ---------------------------------------------------------------------------

def get_document_processor() -> DocumentProcessor:
    """Provide a document processor instance."""
    return DocumentProcessor()


def get_ai_evaluator() -> StartupEvaluator:
    """Provide an AI evaluator instance."""
    return StartupEvaluator()


def get_scoring_engine() -> ScoringEngine:
    """Provide a scoring engine instance."""
    return ScoringEngine()


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class CurrentUser(TypedDict):
    """Authenticated user information."""

    user_id: str
    is_authenticated: bool


async def get_current_user() -> CurrentUser:
    """
    Resolve the currently authenticated user.

    TODO:
        Replace this placeholder with JWT/OAuth2 authentication.
    """
    return {
        "user_id": "anonymous",
        "is_authenticated": False,
    }


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------

async def rate_limit_check() -> None:
    """
    Validate request rate limits.

    TODO:
        Implement Redis-backed rate limiting.
        Consider using a sliding-window algorithm.
    """
    return None


# ---------------------------------------------------------------------------
# Dependency Type Aliases
# ---------------------------------------------------------------------------

DatabaseSession = Annotated[Session, Depends(get_database)]

DocumentProcessorDependency = Annotated[
    DocumentProcessor,
    Depends(get_document_processor),
]

AIEvaluatorDependency = Annotated[
    StartupEvaluator,
    Depends(get_ai_evaluator),
]

ScoringEngineDependency = Annotated[
    ScoringEngine,
    Depends(get_scoring_engine),
]

CurrentUserDependency = Annotated[
    CurrentUser,
    Depends(get_current_user),
]

RateLimitDependency = Annotated[
    None,
    Depends(rate_limit_check),
]
