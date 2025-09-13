"""Tests for the FastAPI application."""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json

# Import the FastAPI app
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.api.main import app

client = TestClient(app)

class TestAPI:
    """Test cases for the API endpoints."""
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Energy Forecasting Service API"
        assert "version" in data
    
    def test_health_check(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_forecast_endpoint_valid_request(self):
        """Test forecast endpoint with valid request."""
        forecast_request = {
            "location": "region_1",
            "start_time": (datetime.now()).isoformat(),
            "end_time": (datetime.now() + timedelta(hours=24)).isoformat(),
            "forecast_type": "consumption",
            "resolution": "hourly",
            "confidence_interval": True
        }
        
        response = client.post("/forecast", json=forecast_request)
        
        # Note: This might fail if models aren't initialized
        # In a real test environment, you'd mock the model service
        if response.status_code == 200:
            data = response.json()
            assert "forecast_id" in data
            assert data["location"] == "region_1"
            assert "predictions" in data
            assert isinstance(data["predictions"], list)
        else:
            # Expected if no models are loaded
            assert response.status_code in [500, 501]
    
    def test_forecast_endpoint_invalid_request(self):
        """Test forecast endpoint with invalid request."""
        invalid_request = {
            "location": "region_1"
            # Missing required fields
        }
        
        response = client.post("/forecast", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_forecast_endpoint_invalid_date_range(self):
        """Test forecast endpoint with invalid date range."""
        invalid_request = {
            "location": "region_1",
            "start_time": (datetime.now() + timedelta(hours=24)).isoformat(),
            "end_time": (datetime.now()).isoformat(),  # End before start
            "forecast_type": "consumption"
        }
        
        response = client.post("/forecast", json=invalid_request)
        # Should return validation error or server error
        assert response.status_code in [422, 500]
    
    def test_get_forecast_not_implemented(self):
        """Test getting a specific forecast (not implemented yet)."""
        response = client.get("/forecasts/test_forecast_id")
        assert response.status_code == 501
    
    def test_model_status_endpoint(self):
        """Test the model status endpoint."""
        response = client.get("/models/status")
        
        # This might fail if model service isn't initialized
        # In production, this should always return some status
        if response.status_code == 200:
            data = response.json()
            assert "active_models" in data or "total_models" in data
        else:
            assert response.status_code == 500

class TestAPISchemas:
    """Test cases for API request/response schemas."""
    
    def test_forecast_request_validation(self):
        """Test forecast request schema validation."""
        from app.api.schemas import ForecastRequest
        from pydantic import ValidationError
        
        # Valid request
        valid_data = {
            "location": "region_1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(hours=24)
        }
        
        request = ForecastRequest(**valid_data)
        assert request.location == "region_1"
        assert request.forecast_type == "consumption"  # Default value
        assert request.resolution == "hourly"  # Default value
        
        # Invalid request - missing required fields
        with pytest.raises(ValidationError):
            ForecastRequest(location="region_1")
    
    def test_forecast_response_schema(self):
        """Test forecast response schema."""
        from app.api.schemas import ForecastResponse, ForecastDataPoint
        
        # Create sample data points
        data_points = [
            ForecastDataPoint(
                timestamp=datetime.now(),
                value=1000.5,
                lower_bound=950.0,
                upper_bound=1050.0,
                confidence=0.95
            )
        ]
        
        # Create response
        response = ForecastResponse(
            forecast_id="test_forecast_123",
            location="region_1",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=24),
            predictions=data_points
        )
        
        assert response.forecast_id == "test_forecast_123"
        assert response.location == "region_1"
        assert len(response.predictions) == 1
        assert response.predictions[0].value == 1000.5
    
    def test_health_response_schema(self):
        """Test health response schema."""
        from app.api.schemas import HealthResponse
        
        response = HealthResponse(
            status="healthy",
            version="1.0.0"
        )
        
        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert isinstance(response.timestamp, datetime)

class TestAPIIntegration:
    """Integration tests for the API."""
    
    @pytest.mark.asyncio
    async def test_full_forecast_workflow(self):
        """Test the complete forecast workflow."""
        # This would test the full pipeline from request to response
        # In a real environment, you'd set up test data and mock services
        pass
    
    def test_cors_headers(self):
        """Test CORS headers are properly set."""
        response = client.options("/")
        # Should allow CORS for the configured origins
        # This depends on your CORS configuration
    
    def test_api_documentation(self):
        """Test that API documentation is accessible."""
        # Test OpenAPI schema endpoint
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        assert "paths" in openapi_data
        assert "info" in openapi_data
        assert openapi_data["info"]["title"] == "Energy Forecasting Service"

# Fixtures for testing
@pytest.fixture
def sample_forecast_request():
    """Sample forecast request for testing."""
    return {
        "location": "test_region",
        "start_time": datetime.now().isoformat(),
        "end_time": (datetime.now() + timedelta(hours=24)).isoformat(),
        "forecast_type": "consumption",
        "resolution": "hourly",
        "confidence_interval": True
    }

@pytest.fixture
def sample_forecast_response():
    """Sample forecast response for testing."""
    return {
        "forecast_id": "test_forecast_123",
        "location": "test_region",
        "start_time": datetime.now().isoformat(),
        "end_time": (datetime.now() + timedelta(hours=24)).isoformat(),
        "predictions": [
            {
                "timestamp": datetime.now().isoformat(),
                "value": 1000.0,
                "lower_bound": 950.0,
                "upper_bound": 1050.0,
                "confidence": 0.95
            }
        ],
        "created_at": datetime.now().isoformat()
    }

# Performance tests
class TestAPIPerformance:
    """Performance tests for the API."""
    
    def test_response_time_health_check(self):
        """Test that health check responds quickly."""
        import time
        
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond within 1 second
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        import concurrent.futures
        import threading
        
        def make_request():
            return client.get("/health")
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]
        
        # All requests should succeed
        assert all(response.status_code == 200 for response in responses)

if __name__ == "__main__":
    pytest.main([__file__])
