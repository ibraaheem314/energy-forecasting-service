"""Main FastAPI application for energy forecasting service."""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict, Any
from datetime import datetime, timedelta

from .schemas import ForecastRequest, ForecastResponse, HealthResponse
from ..services.models import ModelService
from ..services.features import FeatureService
from ..config import API_HOST, API_PORT

app = FastAPI(
    title="Energy Forecasting Service",
    description="API for energy consumption and production forecasting",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection
def get_model_service() -> ModelService:
    return ModelService()

def get_feature_service() -> FeatureService:
    return FeatureService()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {"message": "Energy Forecasting Service API", "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Add actual health checks here (database, model availability, etc.)
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            version="1.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/forecast", response_model=ForecastResponse)
async def create_forecast(
    request: ForecastRequest,
    model_service: ModelService = Depends(get_model_service),
    feature_service: FeatureService = Depends(get_feature_service)
):
    """Generate energy forecast."""
    try:
        # Prepare features
        features = await feature_service.prepare_features(
            start_time=request.start_time,
            end_time=request.end_time,
            location=request.location
        )
        
        # Generate forecast
        forecast = await model_service.predict(features)
        
        return ForecastResponse(
            forecast_id=f"forecast_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            location=request.location,
            start_time=request.start_time,
            end_time=request.end_time,
            predictions=forecast,
            created_at=datetime.utcnow()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@app.get("/forecasts/{forecast_id}")
async def get_forecast(forecast_id: str):
    """Retrieve a specific forecast by ID."""
    # Implementation would retrieve from database/cache
    raise HTTPException(status_code=501, detail="Not implemented yet")

@app.get("/models/status")
async def get_model_status(model_service: ModelService = Depends(get_model_service)):
    """Get current model status and metadata."""
    try:
        status = await model_service.get_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
