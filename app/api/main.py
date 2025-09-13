from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta
from app.services.loader import load_timeseries
from app.services.models import load_production_model, predict_with_model

app = FastAPI(title="Energy Forecasting API", version="1.0.0")

class ForecastRequest(BaseModel):
    horizon: int = Field(168, ge=1, le=24*14, description="Horizon en heures (par défaut 7 jours)")
    city: str = Field("Paris")
    with_intervals: bool = Field(True)

class ForecastResponse(BaseModel):
    timestamps: List[str]
    yhat: List[float]
    yhat_lower: Optional[List[float]] = None
    yhat_upper: Optional[List[float]] = None
    model_name: str
    model_version: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    # Charger les données récentes
    df = load_timeseries(city=req.city)
    if df is None or df.empty:
        raise HTTPException(503, "Aucune donnée disponible")

    # Charger le modèle “production”
    model = load_production_model()
    if model is None:
        raise HTTPException(503, "Aucun modèle en production")

    # Prédire
    preds = predict_with_model(model, df, horizon=req.horizon, with_intervals=req.with_intervals)

    # Construire timeline future
    last_ts = df.index.max()
    future_index = [ (last_ts + timedelta(hours=i+1)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(req.horizon) ]

    return ForecastResponse(
        timestamps=future_index,
        yhat=[float(x) for x in preds["yhat"]],
        yhat_lower=[float(x) for x in preds["yhat_lower"]] if "yhat_lower" in preds else None,
        yhat_upper=[float(x) for x in preds["yhat_upper"]] if "yhat_upper" in preds else None,
        model_name=str(getattr(model, "name", "unknown")),
        model_version=str(getattr(model, "version", "0")),
    )

def main():
    import os, uvicorn
    uvicorn.run(
        "app.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,
    )
