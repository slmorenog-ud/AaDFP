"""
AI Service API using FastAPI.
Exposes endpoints for predictions, model info, and fairness metrics.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import os
from pathlib import Path

# Import pipeline
from src.pipeline import HCTPipeline

# Initialize FastAPI app
app = FastAPI(
    title="HCT Survival Prediction API",
    description="API for post-HCT survival predictions with equity focus",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[HCTPipeline] = None


# ==========================================
# Pydantic Models
# ==========================================

class PatientData(BaseModel):
    """Patient data for prediction."""
    id: str = Field(default="unknown", description="Patient identifier")
    age_at_hct: Optional[float] = Field(None, description="Age at transplant")
    donor_age: Optional[float] = Field(None, description="Donor age")
    year_hct: Optional[int] = Field(None, description="Year of transplant")
    race_group: Optional[str] = Field(None, description="Race/ethnicity")
    dri_score: Optional[str] = Field(None, description="Disease risk index")
    conditioning_intensity: Optional[str] = Field(None, description="Conditioning intensity")
    graft_type: Optional[str] = Field(None, description="Graft type")
    donor_related: Optional[str] = Field(None, description="Donor relationship")
    comorbidity_score: Optional[float] = Field(None, description="Comorbidity score")
    karnofsky_score: Optional[float] = Field(None, description="Karnofsky score")
    # Add more fields as needed
    
    class Config:
        extra = "allow"  # Allow additional fields


class PredictionResponse(BaseModel):
    """Response for single prediction."""
    patient_id: str
    event_probability: float
    risk_category: str
    confidence_lower: Optional[float]
    confidence_upper: Optional[float]
    reliability_score: float
    top_risk_factors: List[Dict[str, Any]]
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    patients: List[PatientData]


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]


class TrainRequest(BaseModel):
    """Request for training."""
    data_path: str
    model_type: str = "gbm"
    n_features: int = 25


class ModelInfo(BaseModel):
    """Model information."""
    is_trained: bool
    model_type: Optional[str]
    n_features: Optional[int]
    training_date: Optional[str]
    performance_metrics: Optional[Dict[str, float]]


class FairnessMetrics(BaseModel):
    """Fairness metrics response."""
    stratified_c_index: float
    c_index_disparity: float
    fairness_passed: bool
    group_metrics: Dict[str, Dict[str, float]]
    recommendations: List[str]


# ==========================================
# Startup/Shutdown Events
# ==========================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup if available."""
    global pipeline
    pipeline = HCTPipeline()
    
    model_path = Path("models/trained_pipeline.pkl")
    if model_path.exists():
        try:
            pipeline.load(str(model_path))
            print("Loaded pre-trained model from disk")
        except Exception as e:
            print(f"Could not load model: {e}")


# ==========================================
# API Endpoints
# ==========================================

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "HCT Survival Prediction API",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": pipeline.is_trained if pipeline else False
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(patient: PatientData):
    """
    Generate prediction for a single patient.
    
    Provides event probability, risk category, confidence intervals,
    and top risk factors based on SHAP analysis.
    """
    global pipeline
    
    if pipeline is None or not pipeline.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained")
    
    try:
        patient_dict = patient.dict()
        result = pipeline.predict(patient_dict)
        
        return PredictionResponse(
            patient_id=result.patient_id,
            event_probability=result.event_probability,
            risk_category=result.risk_category,
            confidence_lower=result.confidence_lower,
            confidence_upper=result.confidence_upper,
            reliability_score=result.reliability_score,
            top_risk_factors=result.top_risk_factors,
            timestamp=result.timestamp
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Generate predictions for multiple patients.
    
    Returns predictions for all patients along with summary statistics.
    """
    global pipeline
    
    if pipeline is None or not pipeline.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained")
    
    try:
        predictions = []
        for patient in request.patients:
            result = pipeline.predict(patient.dict(), include_explanation=False, include_confidence=False)
            predictions.append(PredictionResponse(
                patient_id=result.patient_id,
                event_probability=result.event_probability,
                risk_category=result.risk_category,
                confidence_lower=result.confidence_lower,
                confidence_upper=result.confidence_upper,
                reliability_score=result.reliability_score,
                top_risk_factors=result.top_risk_factors,
                timestamp=result.timestamp
            ))
        
        summary = {
            "total_predictions": len(predictions),
            "risk_distribution": {
                "low": sum(1 for p in predictions if p.risk_category == "Low"),
                "medium": sum(1 for p in predictions if p.risk_category == "Medium"),
                "high": sum(1 for p in predictions if p.risk_category == "High")
            },
            "average_probability": sum(p.event_probability for p in predictions) / len(predictions) if predictions else 0
        }
        
        return BatchPredictionResponse(predictions=predictions, summary=summary)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Get information about the current model.
    
    Returns training status, model type, and performance metrics.
    """
    global pipeline
    
    if pipeline is None:
        return ModelInfo(is_trained=False)
    
    info = ModelInfo(
        is_trained=pipeline.is_trained,
        model_type=None,
        n_features=None,
        training_date=None,
        performance_metrics=None
    )
    
    if pipeline.is_trained and pipeline.training_info:
        info.model_type = pipeline.training_info.get('stages', {}).get('modeling', {}).get('model_type')
        info.n_features = len(pipeline.feature_selector.selected_features)
        info.training_date = pipeline.training_info.get('timestamp')
        info.performance_metrics = pipeline.training_info.get('stages', {}).get('modeling', {}).get('cv_metrics', {})
    
    return info


@app.get("/fairness-metrics", response_model=FairnessMetrics, tags=["Fairness"])
async def get_fairness_metrics():
    """
    Get current fairness metrics.
    
    Returns stratified C-index, disparity metrics, and recommendations.
    """
    global pipeline
    
    if pipeline is None or not pipeline.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained")
    
    try:
        report = pipeline.get_fairness_report()
        c_index = report.get('c_index', {})
        
        return FairnessMetrics(
            stratified_c_index=c_index.get('stratified_c_index', 0),
            c_index_disparity=c_index.get('c_index_disparity', 0),
            fairness_passed=c_index.get('fairness_passed', False),
            group_metrics={
                'c_index_by_group': c_index.get('group_c_indices', {}),
                'accuracy_by_group': report.get('accuracy_by_group', {})
            },
            recommendations=[]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", tags=["Model"])
async def train_model(request: TrainRequest):
    """
    Train a new model.
    
    **Warning**: This will replace the current model.
    
    Args:
        data_path: Path to training data CSV
        model_type: 'gbm', 'rf', or 'ensemble'
        n_features: Number of features to select
    """
    global pipeline
    
    if not os.path.exists(request.data_path):
        raise HTTPException(status_code=400, detail=f"Data file not found: {request.data_path}")
    
    try:
        pipeline = HCTPipeline()
        results = pipeline.train(
            request.data_path,
            model_type=request.model_type,
            n_features=request.n_features
        )
        
        # Save trained model
        os.makedirs("models", exist_ok=True)
        pipeline.save("models/trained_pipeline.pkl")
        
        return {
            "status": "success",
            "message": "Model trained successfully",
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# Run with: uvicorn api:app --reload
# ==========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
