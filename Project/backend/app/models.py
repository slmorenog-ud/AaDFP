"""
Database Models using SQLAlchemy
Complete patient data model with all 60+ clinical variables
"""

from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean, DateTime, Text, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func
import uuid
import os

# Database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://hct_user:hct_secure_password_2024@localhost:5432/hct_prediction_db")

# Create engine and session
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    role = Column(String(50), default="user")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    patients = relationship("Patient", back_populates="user", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    """Session model for token management"""
    __tablename__ = "sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token = Column(String(255), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="sessions")


class Patient(Base):
    """
    Patient model with complete clinical data.
    Based on full data dictionary with 60+ variables for HCT prediction.
    """
    __tablename__ = "patients"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # ==========================================
    # Basic Demographics
    # ==========================================
    age_at_hct = Column(Float)
    year_hct = Column(Integer)
    race_group = Column(String(100), index=True)
    ethnicity = Column(String(100))
    
    # ==========================================
    # Donor Information
    # ==========================================
    donor_age = Column(Float)
    donor_related = Column(String(100))
    sex_match = Column(String(20))
    
    # ==========================================
    # Disease Information
    # ==========================================
    prim_disease_hct = Column(String(100))
    dri_score = Column(String(100), index=True)
    cyto_score = Column(String(100))
    cyto_score_detail = Column(String(100))
    mrd_hct = Column(String(50))
    
    # ==========================================
    # Transplant Details
    # ==========================================
    graft_type = Column(String(100))
    prod_type = Column(String(20))
    conditioning_intensity = Column(String(100))
    tbi_status = Column(String(100))
    in_vivo_tcd = Column(String(50))
    gvhd_proph = Column(String(200))
    rituximab = Column(String(50))
    melphalan_dose = Column(String(100))
    
    # ==========================================
    # HLA Matching - High Resolution
    # ==========================================
    hla_match_a_high = Column(Float)
    hla_match_b_high = Column(Float)
    hla_match_c_high = Column(Float)
    hla_match_drb1_high = Column(Float)
    hla_match_dqb1_high = Column(Float)
    hla_high_res_6 = Column(Float)
    hla_high_res_8 = Column(Float)
    hla_high_res_10 = Column(Float)
    
    # ==========================================
    # HLA Matching - Low Resolution
    # ==========================================
    hla_match_a_low = Column(Float)
    hla_match_b_low = Column(Float)
    hla_match_c_low = Column(Float)
    hla_match_drb1_low = Column(Float)
    hla_match_dqb1_low = Column(Float)
    hla_low_res_6 = Column(Float)
    hla_low_res_8 = Column(Float)
    hla_low_res_10 = Column(Float)
    hla_nmdp_6 = Column(Float)
    
    # ==========================================
    # T-Cell Epitope Matching
    # ==========================================
    tce_match = Column(String(100))
    tce_imm_match = Column(String(50))
    tce_div_match = Column(String(100))
    
    # ==========================================
    # CMV Status
    # ==========================================
    cmv_status = Column(String(20))
    
    # ==========================================
    # Performance Scores
    # ==========================================
    karnofsky_score = Column(Float)
    comorbidity_score = Column(Float)
    
    # ==========================================
    # Comorbidities
    # ==========================================
    cardiac = Column(String(50))
    arrhythmia = Column(String(50))
    diabetes = Column(String(50))
    hepatic_mild = Column(String(50))
    hepatic_severe = Column(String(50))
    obesity = Column(String(50))
    peptic_ulcer = Column(String(50))
    prior_tumor = Column(String(50))
    psych_disturb = Column(String(50))
    pulm_moderate = Column(String(50))
    pulm_severe = Column(String(50))
    renal_issue = Column(String(50))
    rheum_issue = Column(String(50))
    vent_hist = Column(String(50))
    
    # ==========================================
    # Outcome Variables (for training data)
    # ==========================================
    efs = Column(String(50))
    efs_time = Column(Float)
    
    # Relationships
    user = relationship("User", back_populates="patients")
    predictions = relationship("Prediction", back_populates="patient", cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert patient to dictionary for AI service"""
        return {
            "id": str(self.id),
            "age_at_hct": self.age_at_hct,
            "year_hct": self.year_hct,
            "race_group": self.race_group,
            "ethnicity": self.ethnicity,
            "donor_age": self.donor_age,
            "donor_related": self.donor_related,
            "sex_match": self.sex_match,
            "prim_disease_hct": self.prim_disease_hct,
            "dri_score": self.dri_score,
            "cyto_score": self.cyto_score,
            "cyto_score_detail": self.cyto_score_detail,
            "mrd_hct": self.mrd_hct,
            "graft_type": self.graft_type,
            "prod_type": self.prod_type,
            "conditioning_intensity": self.conditioning_intensity,
            "tbi_status": self.tbi_status,
            "in_vivo_tcd": self.in_vivo_tcd,
            "gvhd_proph": self.gvhd_proph,
            "rituximab": self.rituximab,
            "melphalan_dose": self.melphalan_dose,
            "hla_match_a_high": self.hla_match_a_high,
            "hla_match_b_high": self.hla_match_b_high,
            "hla_match_c_high": self.hla_match_c_high,
            "hla_match_drb1_high": self.hla_match_drb1_high,
            "hla_match_dqb1_high": self.hla_match_dqb1_high,
            "hla_high_res_6": self.hla_high_res_6,
            "hla_high_res_8": self.hla_high_res_8,
            "hla_high_res_10": self.hla_high_res_10,
            "hla_match_a_low": self.hla_match_a_low,
            "hla_match_b_low": self.hla_match_b_low,
            "hla_match_c_low": self.hla_match_c_low,
            "hla_match_drb1_low": self.hla_match_drb1_low,
            "hla_match_dqb1_low": self.hla_match_dqb1_low,
            "hla_low_res_6": self.hla_low_res_6,
            "hla_low_res_8": self.hla_low_res_8,
            "hla_low_res_10": self.hla_low_res_10,
            "hla_nmdp_6": self.hla_nmdp_6,
            "tce_match": self.tce_match,
            "tce_imm_match": self.tce_imm_match,
            "tce_div_match": self.tce_div_match,
            "cmv_status": self.cmv_status,
            "karnofsky_score": self.karnofsky_score,
            "comorbidity_score": self.comorbidity_score,
            "cardiac": self.cardiac,
            "arrhythmia": self.arrhythmia,
            "diabetes": self.diabetes,
            "hepatic_mild": self.hepatic_mild,
            "hepatic_severe": self.hepatic_severe,
            "obesity": self.obesity,
            "peptic_ulcer": self.peptic_ulcer,
            "prior_tumor": self.prior_tumor,
            "psych_disturb": self.psych_disturb,
            "pulm_moderate": self.pulm_moderate,
            "pulm_severe": self.pulm_severe,
            "renal_issue": self.renal_issue,
            "rheum_issue": self.rheum_issue,
            "vent_hist": self.vent_hist,
        }


class Prediction(Base):
    """Prediction model with full result storage"""
    __tablename__ = "predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Prediction Results
    event_probability = Column(Float, nullable=False)
    risk_category = Column(String(50), nullable=False, index=True)
    confidence_lower = Column(Float)
    confidence_upper = Column(Float)
    reliability_score = Column(Float)
    
    # Model Information
    model_version = Column(String(100))
    features_used = Column(JSONB)
    
    # Explanation Data
    top_risk_factors = Column(JSONB)
    shap_values = Column(JSONB)
    
    # Relationships
    patient = relationship("Patient", back_populates="predictions")
    user = relationship("User", back_populates="predictions")


class ModelRegistry(Base):
    """Model registry for tracking trained models"""
    __tablename__ = "model_registry"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version = Column(String(100), unique=True, nullable=False)
    model_type = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=False, index=True)
    
    # Performance Metrics
    cv_accuracy = Column(Float)
    cv_auc = Column(Float)
    stratified_c_index = Column(Float)
    c_index_disparity = Column(Float)
    fairness_passed = Column(Boolean)
    
    # Training Details
    n_features = Column(Integer)
    features_selected = Column(JSONB)
    training_samples = Column(Integer)
    training_config = Column(JSONB)


class AuditLog(Base):
    """Audit log for tracking user actions"""
    __tablename__ = "audit_log"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False, index=True)
    entity_type = Column(String(100), nullable=False)
    entity_id = Column(UUID(as_uuid=True))
    details = Column(JSONB)
    ip_address = Column(INET)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)


# Create all tables
def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
