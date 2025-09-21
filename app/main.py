"""
FastAPI main application module for QuickScore.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.core.config import settings
from app.api.routes import startups, analysis, demo
from app.models.database import init_database, is_using_supabase

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting QuickScore application...")
    
    # Initialize database (either SQLAlchemy or Supabase)
    db_init_success = await init_database()
    if db_init_success:
        db_type = "Supabase REST API" if is_using_supabase() else "SQLAlchemy PostgreSQL"
        logger.info(f"✅ Database initialized successfully using {db_type}")
    else:
        logger.warning("⚠️ Database initialization had issues, but continuing...")
    
    yield
    
    # Shutdown
    logger.info("Shutting down QuickScore application...")


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-powered pre-seed startup analyzer and investment decision platform",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error": str(exc) if settings.DEBUG else "An unexpected error occurred"
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "database": "Supabase REST API" if is_using_supabase() else "SQLAlchemy",
        "timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} API",
        "version": settings.VERSION,
        "docs_url": "/docs",
        "openapi_url": f"{settings.API_V1_STR}/openapi.json",
        "database": "Supabase REST API" if is_using_supabase() else "SQLAlchemy"
    }


# Include routers
app.include_router(
    startups.router,
    prefix=f"{settings.API_V1_STR}/startups",
    tags=["startups"]
)

app.include_router(
    analysis.router,
    prefix=f"{settings.API_V1_STR}",
    tags=["analysis"]
)

app.include_router(
    demo.router,
    prefix=f"{settings.API_V1_STR}/demo",
    tags=["demo"]
)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )