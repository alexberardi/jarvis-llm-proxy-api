"""LLM Proxy API - Main Entry Point.

OpenAI-compatible API that proxies requests to the model service.
This module sets up the FastAPI application and includes all route routers.
"""

import multiprocessing

# Fix for vLLM CUDA multiprocessing issue - set spawn method early
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # Already set, ignore
    pass

import logging

from dotenv import load_dotenv

# Load .env before any config/route imports that read env vars at module level
load_dotenv()

from fastapi import FastAPI

# Config setup
from config.logging_config import (
    setup_console_logging,
    setup_remote_logging,
    print_startup_info,
)
from config.debug_config import setup_debugpy
from config.service_config import (
    init as init_service_config,
    shutdown as shutdown_service_config,
)

# Route modules
from api.chat_routes import router as chat_router
from api.queue_routes import router as queue_router
from api.model_routes import router as model_router
from api.health_routes import router as health_router
from api.training_routes import router as training_router
from api.adapter_routes import router as adapter_router
from api.pipeline_routes import router as pipeline_router
from api.settings_routes import router as settings_router

# Initialize logging
logger = setup_console_logging()

# Set up debug mode
setup_debugpy()

# Print startup info
print_startup_info()

# Create FastAPI application
app = FastAPI()

# Include all route routers
app.include_router(chat_router)
app.include_router(queue_router)
app.include_router(model_router)
app.include_router(health_router)
app.include_router(training_router)
app.include_router(adapter_router)
app.include_router(pipeline_router)
app.include_router(settings_router)


@app.on_event("startup")
async def startup_event():
    """Initialize service config, remote logging, and pre-warm DB on startup."""
    # Service discovery first (auth URL, logs URL, etc.)
    try:
        from db.session import engine as db_engine
        init_service_config(db_engine=db_engine)
    except Exception as e:
        logger.warning(f"Service config init failed (non-fatal): {e}")
        # Still try without DB caching
        try:
            init_service_config()
        except Exception as e:
            pass

    setup_remote_logging()

    # Pre-warm database connection to avoid first-request failures
    try:
        from sqlalchemy import text

        from db.session import SessionLocal
        from db.models import TrainingJob  # noqa: F401 - import to trigger table mapping

        db = SessionLocal()
        # Simple query to establish connection and warm the pool
        db.execute(text("SELECT 1"))
        db.close()
        logger.info("Database connection pre-warmed successfully")
    except Exception as e:
        logger.warning(f"Database pre-warm failed (non-fatal): {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up service config and logging handlers on shutdown."""
    shutdown_service_config()
    for handler in logger.handlers:
        if hasattr(handler, "close"):
            handler.close()
