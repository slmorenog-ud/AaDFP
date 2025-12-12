"""
Backend Service API using FastAPI.
Provides REST API for the frontend and orchestrates calls to AI Service.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import hashlib
import secrets

# Initialize FastAPI app
app = FastAPI(
    title="HCT Prediction Backend",
    description="Backend service for HCT Survival Prediction System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)


# ==========================================
# In-memory "Database" (replace with real DB)
# ==========================================

# Users store
users_db: Dict[str, Dict] = {
    "admin@example.com": {
        "id": "1",
        "email": "admin@example.com",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin",
        "name": "Admin User"
    }
}

# Sessions store
sessions: Dict[str, Dict] = {}

# Patients store
patients_db: Dict[str, Dict] = {}

# Predictions store
predictions_db: Dict[str, Dict] = {}


# ==========================================
# Pydantic Models
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
    name: str = Field(..., description="Patient name")
    age_at_hct: float = Field(..., description="Age at transplant")
    donor_age: Optional[float] = None
    year_hct: int = Field(default=2024)
    race_group: str = Field(default="White")
    dri_score: str = Field(default="Intermediate")
    conditioning_intensity: str = Field(default="MAC")
    graft_type: str = Field(default="Peripheral blood")
    donor_related: str = Field(default="Unrelated")
    comorbidity_score: float = Field(default=0)
    karnofsky_score: float = Field(default=90)
    
    @field_validator('year_hct')
    @classmethod
    def validate_year_hct(cls, v):
        current_year = datetime.now().year
        if v > current_year:
            raise ValueError(f'Year cannot be greater than current year ({current_year})')
        if v < 1980:
            raise ValueError('Year must be 1980 or later')
        return v
    
    @field_validator('age_at_hct')
    @classmethod
    def validate_age_at_hct(cls, v):
        if v < 0 or v > 120:
            raise ValueError('Age at HCT must be between 0 and 120')
        return v
    
    @field_validator('donor_age')
    @classmethod
    def validate_donor_age(cls, v):
        if v is not None and (v < 0 or v > 120):
            raise ValueError('Donor age must be between 0 and 120')
        return v
    
    @field_validator('comorbidity_score')
    @classmethod
    def validate_comorbidity_score(cls, v):
        if v < 0 or v > 10:
            raise ValueError('Comorbidity score must be between 0 and 10')
        return v
    
    @field_validator('karnofsky_score')
    @classmethod
    def validate_karnofsky_score(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Karnofsky score must be between 0 and 100')
        return v
    
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
    reliability_score: float
    created_at: str


# ==========================================
# Auth Helpers
# ==========================================

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def create_token() -> str:
    return secrets.token_urlsafe(32)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict]:
    if credentials is None:
        return None
    
    token = credentials.credentials
    session = sessions.get(token)
    
    if session is None:
        return None
    
    if session["expires"] < datetime.now():
        del sessions[token]
        return None
    
    return session["user"]


def require_auth(user: Dict = Depends(get_current_user)) -> Dict:
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# ==========================================
# API Endpoints
# ==========================================

@app.get("/", tags=["Health"])
async def root():
    return {"status": "healthy", "service": "HCT Prediction Backend"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ==========================================
# Auth Endpoints
# ==========================================

@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(data: UserLogin):
    """Login and get access token."""
    user = users_db.get(data.email)
    
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if user["password_hash"] != hash_password(data.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token()
    sessions[token] = {
        "user": {k: v for k, v in user.items() if k != "password_hash"},
        "expires": datetime.now() + timedelta(hours=24)
    }
    
    return TokenResponse(
        access_token=token,
        user={k: v for k, v in user.items() if k != "password_hash"}
    )


@app.post("/auth/register", response_model=TokenResponse, tags=["Auth"])
async def register(data: UserRegister):
    """Register new user."""
    if data.email in users_db:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_id = str(len(users_db) + 1)
    users_db[data.email] = {
        "id": user_id,
        "email": data.email,
        "password_hash": hash_password(data.password),
        "role": "user",
        "name": data.name
    }
    
    user = users_db[data.email]
    token = create_token()
    sessions[token] = {
        "user": {k: v for k, v in user.items() if k != "password_hash"},
        "expires": datetime.now() + timedelta(hours=24)
    }
    
    return TokenResponse(
        access_token=token,
        user={k: v for k, v in user.items() if k != "password_hash"}
    )


@app.post("/auth/logout", tags=["Auth"])
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Logout and invalidate token."""
    if credentials and credentials.credentials in sessions:
        del sessions[credentials.credentials]
    return {"status": "logged out"}


@app.get("/auth/me", tags=["Auth"])
async def get_me(user: Dict = Depends(require_auth)):
    """Get current user info."""
    return user


# ==========================================
# Patient Endpoints
# ==========================================

@app.get("/patients", response_model=List[PatientResponse], tags=["Patients"])
async def list_patients(user: Dict = Depends(require_auth)):
    """List all patients for current user."""
    user_patients = [
        PatientResponse(
            id=p["id"],
            name=p["name"],
            created_at=p["created_at"],
            clinical_data=p["clinical_data"]
        )
        for p in patients_db.values()
        if p["user_id"] == user["id"] or user["role"] == "admin"
    ]
    return user_patients


@app.post("/patients", response_model=PatientResponse, tags=["Patients"])
async def create_patient(data: PatientCreate, user: Dict = Depends(require_auth)):
    """Create a new patient record."""
    patient_id = f"patient_{len(patients_db) + 1}"
    
    clinical_data = data.dict()
    name = clinical_data.pop("name")
    
    patient = {
        "id": patient_id,
        "user_id": user["id"],
        "name": name,
        "clinical_data": clinical_data,
        "created_at": datetime.now().isoformat()
    }
    
    patients_db[patient_id] = patient
    
    return PatientResponse(
        id=patient["id"],
        name=patient["name"],
        created_at=patient["created_at"],
        clinical_data=patient["clinical_data"]
    )


@app.get("/patients/{patient_id}", response_model=PatientResponse, tags=["Patients"])
async def get_patient(patient_id: str, user: Dict = Depends(require_auth)):
    """Get patient by ID."""
    patient = patients_db.get(patient_id)
    
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    if patient["user_id"] != user["id"] and user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    return PatientResponse(
        id=patient["id"],
        name=patient["name"],
        created_at=patient["created_at"],
        clinical_data=patient["clinical_data"]
    )


# ==========================================
# Prediction Endpoints
# ==========================================

@app.post("/predictions", response_model=PredictionResponse, tags=["Predictions"])
async def create_prediction(data: PredictionRequest, user: Dict = Depends(require_auth)):
    """
    Create a prediction for a patient.
    
    In production, this would call the AI Service.
    For now, returns a mock prediction.
    """
    patient = patients_db.get(data.patient_id)
    
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    if patient["user_id"] != user["id"] and user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    # TODO: Call AI Service for real prediction
    # For now, generate mock prediction based on clinical data
    clinical = patient["clinical_data"]
    
    # Simple mock risk calculation
    age = clinical.get("age_at_hct", 50)
    comorbidity = clinical.get("comorbidity_score", 0)
    karnofsky = clinical.get("karnofsky_score", 90)
    
    # Higher age and comorbidity increase risk, higher Karnofsky decreases it
    base_risk = 0.3
    age_factor = (age - 40) / 100 if age > 40 else 0
    comorbidity_factor = comorbidity * 0.1
    karnofsky_factor = (100 - karnofsky) / 200
    
    probability = min(1.0, max(0.0, base_risk + age_factor + comorbidity_factor + karnofsky_factor))
    
    if probability < 0.3:
        risk_category = "Low"
    elif probability < 0.7:
        risk_category = "Medium"
    else:
        risk_category = "High"
    
    prediction_id = f"pred_{len(predictions_db) + 1}"
    
    prediction = {
        "id": prediction_id,
        "patient_id": data.patient_id,
        "user_id": user["id"],
        "event_probability": probability,
        "risk_category": risk_category,
        "reliability_score": 0.75,
        "created_at": datetime.now().isoformat()
    }
    
    predictions_db[prediction_id] = prediction
    
    return PredictionResponse(
        id=prediction["id"],
        patient_id=prediction["patient_id"],
        event_probability=prediction["event_probability"],
        risk_category=prediction["risk_category"],
        reliability_score=prediction["reliability_score"],
        created_at=prediction["created_at"]
    )


@app.get("/predictions", response_model=List[PredictionResponse], tags=["Predictions"])
async def list_predictions(user: Dict = Depends(require_auth)):
    """List all predictions for current user."""
    user_predictions = [
        PredictionResponse(
            id=p["id"],
            patient_id=p["patient_id"],
            event_probability=p["event_probability"],
            risk_category=p["risk_category"],
            reliability_score=p["reliability_score"],
            created_at=p["created_at"]
        )
        for p in predictions_db.values()
        if p["user_id"] == user["id"] or user["role"] == "admin"
    ]
    return user_predictions


@app.get("/predictions/{prediction_id}", response_model=PredictionResponse, tags=["Predictions"])
async def get_prediction(prediction_id: str, user: Dict = Depends(require_auth)):
    """Get prediction by ID."""
    prediction = predictions_db.get(prediction_id)
    
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    if prediction["user_id"] != user["id"] and user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    return PredictionResponse(
        id=prediction["id"],
        patient_id=prediction["patient_id"],
        event_probability=prediction["event_probability"],
        risk_category=prediction["risk_category"],
        reliability_score=prediction["reliability_score"],
        created_at=prediction["created_at"]
    )


# ==========================================
# Dashboard Endpoint
# ==========================================

@app.get("/dashboard", tags=["Dashboard"])
async def get_dashboard(user: Dict = Depends(require_auth)):
    """Get dashboard summary data."""
    user_patients = [p for p in patients_db.values() if p["user_id"] == user["id"] or user["role"] == "admin"]
    user_predictions = [p for p in predictions_db.values() if p["user_id"] == user["id"] or user["role"] == "admin"]
    
    risk_distribution = {
        "Low": sum(1 for p in user_predictions if p["risk_category"] == "Low"),
        "Medium": sum(1 for p in user_predictions if p["risk_category"] == "Medium"),
        "High": sum(1 for p in user_predictions if p["risk_category"] == "High")
    }
    
    return {
        "total_patients": len(user_patients),
        "total_predictions": len(user_predictions),
        "risk_distribution": risk_distribution,
        "recent_predictions": sorted(user_predictions, key=lambda x: x["created_at"], reverse=True)[:5]
    }


# ==========================================
# Run with: uvicorn main:app --reload --port 8001
# ==========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
