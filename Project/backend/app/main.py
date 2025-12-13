"""
Backend Service API using FastAPI.
Complete implementation with PostgreSQL database and expanded clinical data.
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import hashlib
import secrets
import httpx
import os
import redis
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import text

# Import database models
from .models import (
    get_db, User, Session, Patient, Prediction, AuditLog,
    SessionLocal, engine, Base
)

# Initialize database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="HCT Prediction Backend",
    description="Backend service for HCT Survival Prediction System with full clinical data support",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Redis connection for caching
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
except:
    redis_client = None

# AI Service URL
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://ai_service:8000")


# ==========================================
# Pydantic Models - Expanded for Full Data
# ==========================================

class UserLogin(BaseModel):
    email: str
    password: str


class UserRegister(BaseModel):
    email: str
    password: str
    name: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


class PatientCreate(BaseModel):
    """
    Complete patient data model with all 60+ clinical variables.
    Matches the full data dictionary for HCT predictions.
    """
    name: str = Field(..., description="Patient name")
    
    # Basic Demographics
    age_at_hct: Optional[float] = Field(None, ge=0, le=120, description="Age at transplant")
    year_hct: Optional[int] = Field(None, ge=1980, le=2030, description="Year of HCT")
    race_group: Optional[str] = Field(None, description="Race/ethnicity group")
    ethnicity: Optional[str] = Field(None, description="Ethnicity")
    
    # Donor Information
    donor_age: Optional[float] = Field(None, ge=0, le=120, description="Donor age")
    donor_related: Optional[str] = Field(None, description="Related vs. unrelated donor")
    sex_match: Optional[str] = Field(None, description="Donor/recipient sex match")
    
    # Disease Information
    prim_disease_hct: Optional[str] = Field(None, description="Primary disease for HCT")
    dri_score: Optional[str] = Field(None, description="Disease risk index")
    cyto_score: Optional[str] = Field(None, description="Cytogenetic score")
    cyto_score_detail: Optional[str] = Field(None, description="Detailed cytogenetics for DRI")
    mrd_hct: Optional[str] = Field(None, description="MRD at time of HCT")
    
    # Transplant Details
    graft_type: Optional[str] = Field(None, description="Graft type")
    prod_type: Optional[str] = Field(None, description="Product type")
    conditioning_intensity: Optional[str] = Field(None, description="Conditioning intensity")
    tbi_status: Optional[str] = Field(None, description="TBI status")
    in_vivo_tcd: Optional[str] = Field(None, description="In-vivo T-cell depletion")
    gvhd_proph: Optional[str] = Field(None, description="GVHD prophylaxis")
    rituximab: Optional[str] = Field(None, description="Rituximab given in conditioning")
    melphalan_dose: Optional[str] = Field(None, description="Melphalan dose")
    
    # HLA Matching - High Resolution
    hla_match_a_high: Optional[float] = Field(None, description="HLA-A high resolution match")
    hla_match_b_high: Optional[float] = Field(None, description="HLA-B high resolution match")
    hla_match_c_high: Optional[float] = Field(None, description="HLA-C high resolution match")
    hla_match_drb1_high: Optional[float] = Field(None, description="HLA-DRB1 high resolution match")
    hla_match_dqb1_high: Optional[float] = Field(None, description="HLA-DQB1 high resolution match")
    hla_high_res_6: Optional[float] = Field(None, description="6-locus high resolution match")
    hla_high_res_8: Optional[float] = Field(None, description="8-locus high resolution match")
    hla_high_res_10: Optional[float] = Field(None, description="10-locus high resolution match")
    
    # HLA Matching - Low Resolution
    hla_match_a_low: Optional[float] = Field(None, description="HLA-A low resolution match")
    hla_match_b_low: Optional[float] = Field(None, description="HLA-B low resolution match")
    hla_match_c_low: Optional[float] = Field(None, description="HLA-C low resolution match")
    hla_match_drb1_low: Optional[float] = Field(None, description="HLA-DRB1 low resolution match")
    hla_match_dqb1_low: Optional[float] = Field(None, description="HLA-DQB1 low resolution match")
    hla_low_res_6: Optional[float] = Field(None, description="6-locus low resolution match")
    hla_low_res_8: Optional[float] = Field(None, description="8-locus low resolution match")
    hla_low_res_10: Optional[float] = Field(None, description="10-locus low resolution match")
    hla_nmdp_6: Optional[float] = Field(None, description="NMDP 6-locus match")
    
    # T-Cell Epitope Matching
    tce_match: Optional[str] = Field(None, description="T-cell epitope matching")
    tce_imm_match: Optional[str] = Field(None, description="TCE immunogenicity match")
    tce_div_match: Optional[str] = Field(None, description="TCE diversity match")
    
    # CMV Status
    cmv_status: Optional[str] = Field(None, description="Donor/recipient CMV serostatus")
    
    # Performance Scores
    karnofsky_score: Optional[float] = Field(None, ge=0, le=100, description="Karnofsky performance score")
    comorbidity_score: Optional[float] = Field(None, ge=0, le=10, description="Sorror comorbidity score")
    
    # Comorbidities
    cardiac: Optional[str] = Field(None, description="Cardiac condition")
    arrhythmia: Optional[str] = Field(None, description="Arrhythmia")
    diabetes: Optional[str] = Field(None, description="Diabetes")
    hepatic_mild: Optional[str] = Field(None, description="Mild hepatic condition")
    hepatic_severe: Optional[str] = Field(None, description="Severe hepatic condition")
    obesity: Optional[str] = Field(None, description="Obesity")
    peptic_ulcer: Optional[str] = Field(None, description="Peptic ulcer")
    prior_tumor: Optional[str] = Field(None, description="Prior solid tumor")
    psych_disturb: Optional[str] = Field(None, description="Psychiatric disturbance")
    pulm_moderate: Optional[str] = Field(None, description="Moderate pulmonary condition")
    pulm_severe: Optional[str] = Field(None, description="Severe pulmonary condition")
    renal_issue: Optional[str] = Field(None, description="Renal condition")
    rheum_issue: Optional[str] = Field(None, description="Rheumatologic condition")
    vent_hist: Optional[str] = Field(None, description="History of mechanical ventilation")

    class Config:
        extra = "allow"


class PatientResponse(BaseModel):
    id: str
    name: str
    created_at: str
    clinical_data: Dict[str, Any]


class PredictionRequest(BaseModel):
    patient_id: str


class PredictionResponse(BaseModel):
    id: str
    patient_id: str
    event_probability: float
    risk_category: str
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    reliability_score: float
    top_risk_factors: Optional[List[Dict[str, Any]]] = None
    created_at: str


# ==========================================
# Auth Helpers
# ==========================================

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def create_token() -> str:
    return secrets.token_urlsafe(32)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: DBSession = Depends(get_db)
) -> Optional[User]:
    if credentials is None:
        return None
    
    token = credentials.credentials
    
    # Check session in database
    session = db.query(Session).filter(Session.token == token).first()
    
    if session is None:
        return None
    
    if session.expires_at < datetime.now(session.expires_at.tzinfo):
        db.delete(session)
        db.commit()
        return None
    
    return session.user


def require_auth(user: User = Depends(get_current_user)) -> User:
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def log_audit(db: DBSession, user_id: Optional[str], action: str, entity_type: str, entity_id: Optional[str] = None, details: Optional[dict] = None):
    """Log an audit entry"""
    try:
        audit = AuditLog(
            user_id=user_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            details=details
        )
        db.add(audit)
        db.commit()
    except Exception as e:
        print(f"Audit log error: {e}")


# ==========================================
# API Endpoints - Health
# ==========================================

@app.get("/", tags=["Health"])
async def root():
    return {"status": "healthy", "service": "HCT Prediction Backend", "version": "2.0.0"}


@app.get("/health", tags=["Health"])
async def health(db: DBSession = Depends(get_db)):
    # Check database connection
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except:
        db_status = "disconnected"
    
    # Check Redis connection
    try:
        if redis_client:
            redis_client.ping()
            redis_status = "connected"
        else:
            redis_status = "not configured"
    except:
        redis_status = "disconnected"
    
    # Check AI Service connection
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{AI_SERVICE_URL}/health", timeout=5.0)
            ai_status = "connected" if response.status_code == 200 else "error"
    except:
        ai_status = "disconnected"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": db_status,
            "redis": redis_status,
            "ai_service": ai_status
        }
    }


# ==========================================
# Auth Endpoints
# ==========================================

@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(data: UserLogin, db: DBSession = Depends(get_db)):
    """Login and get access token."""
    user = db.query(User).filter(User.email == data.email).first()
    
    if user is None or user.password_hash != hash_password(data.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account is disabled")
    
    # Create session
    token = create_token()
    session = Session(
        user_id=user.id,
        token=token,
        expires_at=datetime.now() + timedelta(hours=24)
    )
    db.add(session)
    db.commit()
    
    log_audit(db, str(user.id), "login", "user", str(user.id))
    
    return TokenResponse(
        access_token=token,
        user={"id": str(user.id), "email": user.email, "name": user.name, "role": user.role}
    )


@app.post("/auth/register", response_model=TokenResponse, tags=["Auth"])
async def register(data: UserRegister, db: DBSession = Depends(get_db)):
    """Register new user."""
    existing = db.query(User).filter(User.email == data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(
        email=data.email,
        password_hash=hash_password(data.password),
        name=data.name,
        role="user"
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Create session
    token = create_token()
    session = Session(
        user_id=user.id,
        token=token,
        expires_at=datetime.now() + timedelta(hours=24)
    )
    db.add(session)
    db.commit()
    
    log_audit(db, str(user.id), "register", "user", str(user.id))
    
    return TokenResponse(
        access_token=token,
        user={"id": str(user.id), "email": user.email, "name": user.name, "role": user.role}
    )


@app.post("/auth/logout", tags=["Auth"])
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: DBSession = Depends(get_db)
):
    """Logout and invalidate token."""
    if credentials:
        session = db.query(Session).filter(Session.token == credentials.credentials).first()
        if session:
            db.delete(session)
            db.commit()
    return {"status": "logged out"}


@app.get("/auth/me", tags=["Auth"])
async def get_me(user: User = Depends(require_auth)):
    """Get current user info."""
    return {"id": str(user.id), "email": user.email, "name": user.name, "role": user.role}


# ==========================================
# Patient Endpoints
# ==========================================

@app.get("/patients", response_model=List[PatientResponse], tags=["Patients"])
async def list_patients(user: User = Depends(require_auth), db: DBSession = Depends(get_db)):
    """List all patients for current user."""
    if user.role == "admin":
        patients = db.query(Patient).all()
    else:
        patients = db.query(Patient).filter(Patient.user_id == user.id).all()
    
    return [
        PatientResponse(
            id=str(p.id),
            name=p.name,
            created_at=p.created_at.isoformat(),
            clinical_data=p.to_dict()
        )
        for p in patients
    ]


@app.post("/patients", response_model=PatientResponse, tags=["Patients"])
async def create_patient(
    data: PatientCreate,
    user: User = Depends(require_auth),
    db: DBSession = Depends(get_db)
):
    """Create a new patient record with full clinical data."""
    patient_data = data.model_dump()
    name = patient_data.pop("name")
    
    patient = Patient(
        user_id=user.id,
        name=name,
        **patient_data
    )
    
    db.add(patient)
    db.commit()
    db.refresh(patient)
    
    log_audit(db, str(user.id), "create", "patient", str(patient.id), {"name": name})
    
    return PatientResponse(
        id=str(patient.id),
        name=patient.name,
        created_at=patient.created_at.isoformat(),
        clinical_data=patient.to_dict()
    )


@app.get("/patients/{patient_id}", response_model=PatientResponse, tags=["Patients"])
async def get_patient(
    patient_id: str,
    user: User = Depends(require_auth),
    db: DBSession = Depends(get_db)
):
    """Get patient by ID."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    if str(patient.user_id) != str(user.id) and user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    return PatientResponse(
        id=str(patient.id),
        name=patient.name,
        created_at=patient.created_at.isoformat(),
        clinical_data=patient.to_dict()
    )


@app.put("/patients/{patient_id}", response_model=PatientResponse, tags=["Patients"])
async def update_patient(
    patient_id: str,
    data: PatientCreate,
    user: User = Depends(require_auth),
    db: DBSession = Depends(get_db)
):
    """Update patient record."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    if str(patient.user_id) != str(user.id) and user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Update fields
    patient_data = data.model_dump(exclude_unset=True)
    for field, value in patient_data.items():
        if hasattr(patient, field):
            setattr(patient, field, value)
    
    db.commit()
    db.refresh(patient)
    
    log_audit(db, str(user.id), "update", "patient", str(patient.id))
    
    return PatientResponse(
        id=str(patient.id),
        name=patient.name,
        created_at=patient.created_at.isoformat(),
        clinical_data=patient.to_dict()
    )


@app.delete("/patients/{patient_id}", tags=["Patients"])
async def delete_patient(
    patient_id: str,
    user: User = Depends(require_auth),
    db: DBSession = Depends(get_db)
):
    """Delete patient record."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    if str(patient.user_id) != str(user.id) and user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    log_audit(db, str(user.id), "delete", "patient", str(patient.id), {"name": patient.name})
    
    db.delete(patient)
    db.commit()
    
    return {"status": "deleted"}


# ==========================================
# Prediction Endpoints
# ==========================================

@app.post("/predictions", response_model=PredictionResponse, tags=["Predictions"])
async def create_prediction(
    data: PredictionRequest,
    user: User = Depends(require_auth),
    db: DBSession = Depends(get_db)
):
    """
    Create a new prediction for a patient.
    Sends full clinical data to AI service.
    """
    patient = db.query(Patient).filter(Patient.id == data.patient_id).first()
    
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    if str(patient.user_id) != str(user.id) and user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Prepare full clinical data for AI service
    ai_payload = patient.to_dict()
    
    # Call AI service with full data
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{AI_SERVICE_URL}/predict",
                json=ai_payload,
                timeout=30.0
            )
            response.raise_for_status()
            ai_result = response.json()
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="AI service timeout")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                raise HTTPException(status_code=503, detail="AI model not trained")
            raise HTTPException(status_code=502, detail=f"AI service error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"AI service connection error: {str(e)}")
    
    # Store prediction in database
    prediction = Prediction(
        patient_id=patient.id,
        user_id=user.id,
        event_probability=ai_result["event_probability"],
        risk_category=ai_result["risk_category"],
        confidence_lower=ai_result.get("confidence_lower"),
        confidence_upper=ai_result.get("confidence_upper"),
        reliability_score=ai_result.get("reliability_score", 0.0),
        top_risk_factors=ai_result.get("top_risk_factors"),
        model_version=ai_result.get("model_version", "1.0.0"),
        features_used=list(ai_payload.keys())
    )
    
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    
    log_audit(db, str(user.id), "create", "prediction", str(prediction.id), {
        "patient_id": str(patient.id),
        "risk_category": prediction.risk_category
    })
    
    return PredictionResponse(
        id=str(prediction.id),
        patient_id=str(prediction.patient_id),
        event_probability=prediction.event_probability,
        risk_category=prediction.risk_category,
        confidence_lower=prediction.confidence_lower,
        confidence_upper=prediction.confidence_upper,
        reliability_score=prediction.reliability_score,
        top_risk_factors=prediction.top_risk_factors,
        created_at=prediction.created_at.isoformat()
    )


@app.get("/predictions", response_model=List[PredictionResponse], tags=["Predictions"])
async def list_predictions(user: User = Depends(require_auth), db: DBSession = Depends(get_db)):
    """List all predictions for current user."""
    if user.role == "admin":
        predictions = db.query(Prediction).order_by(Prediction.created_at.desc()).all()
    else:
        predictions = db.query(Prediction).filter(Prediction.user_id == user.id).order_by(Prediction.created_at.desc()).all()
    
    return [
        PredictionResponse(
            id=str(p.id),
            patient_id=str(p.patient_id),
            event_probability=p.event_probability,
            risk_category=p.risk_category,
            confidence_lower=p.confidence_lower,
            confidence_upper=p.confidence_upper,
            reliability_score=p.reliability_score,
            top_risk_factors=p.top_risk_factors,
            created_at=p.created_at.isoformat()
        )
        for p in predictions
    ]


@app.get("/predictions/{prediction_id}", response_model=PredictionResponse, tags=["Predictions"])
async def get_prediction(
    prediction_id: str,
    user: User = Depends(require_auth),
    db: DBSession = Depends(get_db)
):
    """Get prediction by ID."""
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    if str(prediction.user_id) != str(user.id) and user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    return PredictionResponse(
        id=str(prediction.id),
        patient_id=str(prediction.patient_id),
        event_probability=prediction.event_probability,
        risk_category=prediction.risk_category,
        confidence_lower=prediction.confidence_lower,
        confidence_upper=prediction.confidence_upper,
        reliability_score=prediction.reliability_score,
        top_risk_factors=prediction.top_risk_factors,
        created_at=prediction.created_at.isoformat()
    )


@app.delete("/predictions/{prediction_id}", tags=["Predictions"])
async def delete_prediction(
    prediction_id: str,
    user: User = Depends(require_auth),
    db: DBSession = Depends(get_db)
):
    """Delete prediction record."""
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    if str(prediction.user_id) != str(user.id) and user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    log_audit(db, str(user.id), "delete", "prediction", str(prediction.id), {})
    
    db.delete(prediction)
    db.commit()
    
    return {"status": "deleted"}


# ==========================================
# Dashboard Endpoint
# ==========================================

@app.get("/dashboard", tags=["Dashboard"])
async def get_dashboard(user: User = Depends(require_auth), db: DBSession = Depends(get_db)):
    """Get dashboard summary data."""
    if user.role == "admin":
        total_patients = db.query(Patient).count()
        predictions = db.query(Prediction).all()
    else:
        total_patients = db.query(Patient).filter(Patient.user_id == user.id).count()
        predictions = db.query(Prediction).filter(Prediction.user_id == user.id).all()
    
    risk_distribution = {
        "Low": sum(1 for p in predictions if p.risk_category == "Low"),
        "Medium": sum(1 for p in predictions if p.risk_category == "Medium"),
        "High": sum(1 for p in predictions if p.risk_category == "High")
    }
    
    # Get recent predictions with patient info
    recent_predictions = []
    for p in sorted(predictions, key=lambda x: x.created_at, reverse=True)[:5]:
        patient = db.query(Patient).filter(Patient.id == p.patient_id).first()
        recent_predictions.append({
            "id": str(p.id),
            "patient_id": str(p.patient_id),
            "patient_name": patient.name if patient else "Unknown",
            "event_probability": p.event_probability,
            "risk_category": p.risk_category,
            "created_at": p.created_at.isoformat()
        })
    
    return {
        "total_patients": total_patients,
        "total_predictions": len(predictions),
        "risk_distribution": risk_distribution,
        "recent_predictions": recent_predictions
    }


# ==========================================
# AI Service Endpoints (Proxy)
# ==========================================

@app.get("/ai/model-info", tags=["AI"])
async def get_model_info(user: User = Depends(require_auth)):
    """Get information about the current AI model."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{AI_SERVICE_URL}/model-info", timeout=10.0)
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"AI service error: {str(e)}")


@app.get("/ai/fairness-metrics", tags=["AI"])
async def get_fairness_metrics(user: User = Depends(require_auth)):
    """Get fairness metrics from the AI model."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{AI_SERVICE_URL}/fairness-metrics", timeout=10.0)
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"AI service error: {str(e)}")


@app.post("/ai/train", tags=["AI"])
async def train_model(user: User = Depends(require_auth), db: DBSession = Depends(get_db)):
    """Trigger model training (admin only)."""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{AI_SERVICE_URL}/train",
                json={"data_path": "/app/data/raw/train.csv", "model_type": "gbm", "n_features": 25},
                timeout=300.0  # 5 minutes for training
            )
            result = response.json()
            
            log_audit(db, str(user.id), "train", "model", None, result)
            
            return result
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"AI service error: {str(e)}")


# ==========================================
# Statistics Endpoint
# ==========================================

@app.get("/stats", tags=["Statistics"])
async def get_stats(user: User = Depends(require_auth), db: DBSession = Depends(get_db)):
    """Get system statistics (admin only)."""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return {
        "total_users": db.query(User).count(),
        "total_patients": db.query(Patient).count(),
        "total_predictions": db.query(Prediction).count(),
        "predictions_by_risk": {
            "Low": db.query(Prediction).filter(Prediction.risk_category == "Low").count(),
            "Medium": db.query(Prediction).filter(Prediction.risk_category == "Medium").count(),
            "High": db.query(Prediction).filter(Prediction.risk_category == "High").count()
        }
    }


# ==========================================
# Run with: uvicorn main:app --reload --port 8001
# ==========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
