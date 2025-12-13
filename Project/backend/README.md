# Backend Service - Technical Documentation

## FastAPI REST API for HCT Prediction System

This service provides the main REST API connecting the frontend with the AI service and databases.

---

## 🏗️ Architecture

```mermaid
flowchart TB
    subgraph Backend["Backend Service"]
        Router["FastAPI Router"]
        
        subgraph Modules["Business Modules"]
            Auth["Auth Module"]
            Patients["Patients CRUD"]
            Predictions["Predictions Service"]
        end
        
        subgraph Storage["Data Storage"]
            PG[("PostgreSQL<br/>users | patients<br/>predictions | audit_logs")]
            Redis[("Redis<br/>Sessions")]
        end
    end
    
    FE["🖥️ Frontend<br/>(React)"] -->|HTTP| Router
    Router --> Auth
    Router --> Patients
    Router --> Predictions
    
    Auth --> PG
    Auth --> Redis
    Patients --> PG
    Predictions --> PG
    Predictions -->|HTTP| AI["🤖 AI Service<br/>(Port 8000)"]
    
    style Router fill:#009688,stroke:#333
    style PG fill:#336791,stroke:#333
    style Redis fill:#dc382d,stroke:#333
```

---

## 🔌 API Endpoints

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "ai_service": "available"
}
```

---

### Authentication

#### Register User
```http
POST /auth/register
Content-Type: application/json

{
  "email": "doctor@hospital.com",
  "password": "SecurePass123!",
  "full_name": "Dr. John Smith",
  "role": "doctor"
}
```

**Response:**
```json
{
  "id": "uuid",
  "email": "doctor@hospital.com",
  "full_name": "Dr. John Smith",
  "role": "doctor",
  "created_at": "2025-12-12T..."
}
```

#### Login
```http
POST /auth/login
Content-Type: application/x-www-form-urlencoded

username=doctor@hospital.com&password=SecurePass123!
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### Get Current User
```http
GET /auth/me
Authorization: Bearer {token}
```

---

### Patients

#### Create Patient
```http
POST /patients
Authorization: Bearer {token}
Content-Type: application/json

{
  "name": "María García López",
  "age_at_hct": 45,
  "karnofsky_score": 80,
  "comorbidity_score": 2,
  "sex": "female",
  "race_group": "Hispanic",
  ...
}
```

**Response:**
```json
{
  "id": "uuid",
  "name": "María García López",
  ...
  "created_at": "2025-12-12T..."
}
```

#### List Patients
```http
GET /patients?page=1&limit=20&search=García
Authorization: Bearer {token}
```

**Response:**
```json
{
  "patients": [...],
  "total": 150,
  "page": 1,
  "pages": 8
}
```

#### Get Patient by ID
```http
GET /patients/{patient_id}
Authorization: Bearer {token}
```

#### Update Patient
```http
PUT /patients/{patient_id}
Authorization: Bearer {token}
Content-Type: application/json

{
  "karnofsky_score": 70,
  ...
}
```

#### Delete Patient
```http
DELETE /patients/{patient_id}
Authorization: Bearer {token}
```

---

### Predictions

#### Generate Prediction
```http
POST /predictions
Authorization: Bearer {token}
Content-Type: application/json

{
  "patient_id": "patient-uuid"
}
```

**Response:**
```json
{
  "id": "prediction-uuid",
  "patient_id": "patient-uuid",
  "event_probability": 0.35,
  "risk_category": "Medium",
  "confidence_level": "high",
  "confidence_interval": {
    "lower": 0.30,
    "upper": 0.40
  },
  "reliability_score": 0.85,
  "top_risk_factors": [
    {"factor": "comorbidity_score", "importance": 0.25},
    {"factor": "age_at_hct", "importance": 0.18}
  ],
  "created_at": "2025-12-12T...",
  "created_by": "user-uuid"
}
```

#### List Predictions
```http
GET /predictions?page=1&limit=20
Authorization: Bearer {token}
```

#### Get Patient Predictions
```http
GET /patients/{patient_id}/predictions
Authorization: Bearer {token}
```

---

### Audit Log

#### List Audit Events
```http
GET /audit?page=1&limit=50&action=prediction
Authorization: Bearer {token}  # Admin only
```

**Response:**
```json
{
  "events": [
    {
      "id": "uuid",
      "user_id": "user-uuid",
      "action": "prediction_created",
      "resource_type": "prediction",
      "resource_id": "prediction-uuid",
      "details": {...},
      "ip_address": "192.168.1.1",
      "created_at": "2025-12-12T..."
    }
  ],
  "total": 500
}
```

---

## 📊 Database Models

### User

```python
class User(Base):
    __tablename__ = "users"
    
    id: UUID (PK)
    email: str (unique, indexed)
    password_hash: str
    full_name: str
    role: str  # 'admin', 'doctor', 'researcher'
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_login: datetime
    
    # Relationships
    predictions: List[Prediction]
    audit_logs: List[AuditLog]
```

---

### Patient

```python
class Patient(Base):
    __tablename__ = "patients"
    
    id: UUID (PK)
    name: str  # Required
    
    # Basic Demographics (Tab 1)
    age_at_hct: int  # Required
    sex: str  # 'male', 'female'
    race_group: str
    ethnicity: str
    
    # Clinical Scores (Tab 2)
    karnofsky_score: int  # Required (10-100)
    comorbidity_score: int  # Required (0-10)
    
    # Disease Information (Tab 3)
    prim_disease_hct: str
    disease_status_at_hct: str
    dri_score: str
    time_dx_hct: int
    tbi_status: str
    
    # Donor Information (Tab 4)
    donor_age: int
    donor_related: str  # 'related', 'unrelated'
    donor_sex: str
    hla_match_c_high: str
    hla_high_res_8: int
    graft_type: str  # 'Bone Marrow', 'PBSC', 'Cord Blood'
    
    # Conditioning (Tab 5)
    conditioning_intensity: str
    gvhd_prophylaxis: str
    
    # Comorbidities (Tab 6) - 14 fields
    hepatic_mild: bool
    hepatic_moderate_severe: bool
    cardiac: bool
    pulmonary_moderate: bool
    pulmonary_severe: bool
    renal: bool
    prior_tumor: bool
    rheumatologic: bool
    peptic_ulcer: bool
    diabetes: bool
    cerebrovascular: bool
    psychiatric: bool
    infection: bool
    obesity: bool
    
    # Other Variables (Tab 7)
    year_hct: int
    prod_type: str
    cyto_score: str
    
    # Metadata
    created_at: datetime
    updated_at: datetime
    created_by: UUID (FK -> users.id)
```

---

### Prediction

```python
class Prediction(Base):
    __tablename__ = "predictions"
    
    id: UUID (PK)
    patient_id: UUID (FK -> patients.id, indexed)
    
    # Results
    event_probability: float
    risk_category: str  # 'Low', 'Medium', 'High'
    confidence_level: str  # 'high', 'borderline (near X)'
    confidence_lower: float
    confidence_upper: float
    reliability_score: float
    
    # Risk factors
    top_risk_factors: JSON  # [{"factor": str, "importance": float}]
    
    # Model info
    model_version: str
    feature_count: int
    
    # Metadata
    created_at: datetime
    created_by: UUID (FK -> users.id)
    
    # Relationships
    patient: Patient
    creator: User
```

---

### AuditLog

```python
class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id: UUID (PK)
    user_id: UUID (FK -> users.id, nullable)
    
    action: str  # e.g., 'user_login', 'prediction_created'
    resource_type: str  # 'user', 'patient', 'prediction'
    resource_id: UUID
    
    details: JSON  # Additional action context
    ip_address: str
    user_agent: str
    
    created_at: datetime (indexed)
```

---

### Session

```python
class Session(Base):
    __tablename__ = "sessions"
    
    id: UUID (PK)
    user_id: UUID (FK -> users.id)
    
    token_hash: str (indexed)
    expires_at: datetime
    
    ip_address: str
    user_agent: str
    
    created_at: datetime
    last_activity: datetime
```

---

## 🔐 Authentication Flow

```
┌─────────┐                          ┌─────────┐
│ Client  │                          │ Backend │
└────┬────┘                          └────┬────┘
     │                                     │
     │  POST /auth/login                   │
     │  {email, password}                  │
     ├────────────────────────────────────►│
     │                                     │
     │                              ┌──────┴──────┐
     │                              │ Validate    │
     │                              │ credentials │
     │                              └──────┬──────┘
     │                                     │
     │                              ┌──────┴──────┐
     │                              │ Create JWT  │
     │                              │ + Session   │
     │                              └──────┬──────┘
     │                                     │
     │  {access_token, expires_in}         │
     │◄────────────────────────────────────┤
     │                                     │
     │  GET /patients                      │
     │  Authorization: Bearer {token}      │
     ├────────────────────────────────────►│
     │                                     │
     │                              ┌──────┴──────┐
     │                              │ Validate    │
     │                              │ JWT token   │
     │                              └──────┬──────┘
     │                                     │
     │  {patients: [...]}                  │
     │◄────────────────────────────────────┤
     │                                     │
```

---

## 📝 Audit Logging

All significant actions are automatically logged:

| Action | When |
|--------|------|
| `user_registered` | New user signup |
| `user_login` | Successful login |
| `user_login_failed` | Failed login attempt |
| `patient_created` | New patient added |
| `patient_updated` | Patient data modified |
| `patient_deleted` | Patient removed |
| `prediction_created` | New prediction generated |
| `prediction_viewed` | Prediction accessed |

---

## ⚙️ Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@postgres:5432/hct_db

# Redis
REDIS_URL=redis://redis:6379/0

# AI Service
AI_SERVICE_URL=http://ai_service:8000

# Security
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=60
ALGORITHM=HS256

# CORS
ALLOWED_ORIGINS=http://localhost:5173,http://frontend:80
```

### Dependencies (`requirements.txt`)

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
asyncpg==0.29.0
psycopg2-binary==2.9.9
redis==5.0.1
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
pydantic==2.5.2
pydantic-settings==2.1.0
httpx==0.25.2
python-multipart==0.0.6
```

---

## 🏃 Running

### With Docker

```bash
docker-compose up backend
```

### Local Development

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

---

## 🧪 Testing

### Health Check

```bash
curl http://localhost:8001/health
```

### API Documentation

- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

---

## 🔒 Security Features

1. **JWT Authentication**: Secure token-based auth
2. **Password Hashing**: Bcrypt with salt
3. **Rate Limiting**: Protection against brute force
4. **CORS**: Configurable allowed origins
5. **Input Validation**: Pydantic schemas
6. **Audit Logging**: Complete action history
7. **Session Management**: Redis-backed sessions

---

**Version**: 1.0.0  
**Last updated**: December 2025
