-- HCT Prediction Database Schema
-- Optimized for the full data dictionary with 60+ clinical variables

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==========================================
-- Users Table
-- ==========================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create index for email lookups
CREATE INDEX idx_users_email ON users(email);

-- ==========================================
-- Sessions Table (for token management)
-- ==========================================
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sessions_token ON sessions(token);
CREATE INDEX idx_sessions_user_id ON sessions(user_id);

-- ==========================================
-- Patients Table - Complete Clinical Data
-- Based on full data dictionary (60+ variables)
-- ==========================================
CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Basic Demographics
    age_at_hct DECIMAL(5,2),
    year_hct INTEGER,
    race_group VARCHAR(100),
    ethnicity VARCHAR(100),
    
    -- Donor Information
    donor_age DECIMAL(5,2),
    donor_related VARCHAR(100),
    sex_match VARCHAR(20),
    
    -- Disease Information
    prim_disease_hct VARCHAR(100),
    dri_score VARCHAR(100),
    cyto_score VARCHAR(100),
    cyto_score_detail VARCHAR(100),
    mrd_hct VARCHAR(50),
    
    -- Transplant Details
    graft_type VARCHAR(100),
    prod_type VARCHAR(20),
    conditioning_intensity VARCHAR(100),
    tbi_status VARCHAR(100),
    in_vivo_tcd VARCHAR(50),
    gvhd_proph VARCHAR(200),
    rituximab VARCHAR(50),
    melphalan_dose VARCHAR(100),
    
    -- HLA Matching - High Resolution
    hla_match_a_high DECIMAL(5,2),
    hla_match_b_high DECIMAL(5,2),
    hla_match_c_high DECIMAL(5,2),
    hla_match_drb1_high DECIMAL(5,2),
    hla_match_dqb1_high DECIMAL(5,2),
    hla_high_res_6 DECIMAL(5,2),
    hla_high_res_8 DECIMAL(5,2),
    hla_high_res_10 DECIMAL(5,2),
    
    -- HLA Matching - Low Resolution
    hla_match_a_low DECIMAL(5,2),
    hla_match_b_low DECIMAL(5,2),
    hla_match_c_low DECIMAL(5,2),
    hla_match_drb1_low DECIMAL(5,2),
    hla_match_dqb1_low DECIMAL(5,2),
    hla_low_res_6 DECIMAL(5,2),
    hla_low_res_8 DECIMAL(5,2),
    hla_low_res_10 DECIMAL(5,2),
    hla_nmdp_6 DECIMAL(5,2),
    
    -- T-Cell Epitope Matching
    tce_match VARCHAR(100),
    tce_imm_match VARCHAR(50),
    tce_div_match VARCHAR(100),
    
    -- CMV Status
    cmv_status VARCHAR(20),
    
    -- Performance Scores
    karnofsky_score DECIMAL(5,2),
    comorbidity_score DECIMAL(5,2),
    
    -- Comorbidities
    cardiac VARCHAR(50),
    arrhythmia VARCHAR(50),
    diabetes VARCHAR(50),
    hepatic_mild VARCHAR(50),
    hepatic_severe VARCHAR(50),
    obesity VARCHAR(50),
    peptic_ulcer VARCHAR(50),
    prior_tumor VARCHAR(50),
    psych_disturb VARCHAR(50),
    pulm_moderate VARCHAR(50),
    pulm_severe VARCHAR(50),
    renal_issue VARCHAR(50),
    rheum_issue VARCHAR(50),
    vent_hist VARCHAR(50),
    
    -- Outcome Variables (for training data reference, typically not used for new predictions)
    efs VARCHAR(50),
    efs_time DECIMAL(10,2)
);

-- Create indexes for common queries
CREATE INDEX idx_patients_user_id ON patients(user_id);
CREATE INDEX idx_patients_race_group ON patients(race_group);
CREATE INDEX idx_patients_dri_score ON patients(dri_score);
CREATE INDEX idx_patients_created_at ON patients(created_at);

-- ==========================================
-- Predictions Table
-- ==========================================
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Prediction Results
    event_probability DECIMAL(5,4) NOT NULL,
    risk_category VARCHAR(50) NOT NULL,
    confidence_lower DECIMAL(5,4),
    confidence_upper DECIMAL(5,4),
    reliability_score DECIMAL(5,4),
    
    -- Model Information
    model_version VARCHAR(100),
    features_used JSONB,
    
    -- Explanation Data
    top_risk_factors JSONB,
    shap_values JSONB
);

CREATE INDEX idx_predictions_patient_id ON predictions(patient_id);
CREATE INDEX idx_predictions_user_id ON predictions(user_id);
CREATE INDEX idx_predictions_created_at ON predictions(created_at);
CREATE INDEX idx_predictions_risk_category ON predictions(risk_category);

-- ==========================================
-- Model Registry Table
-- ==========================================
CREATE TABLE IF NOT EXISTS model_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version VARCHAR(100) UNIQUE NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE,
    
    -- Performance Metrics
    cv_accuracy DECIMAL(5,4),
    cv_auc DECIMAL(5,4),
    stratified_c_index DECIMAL(5,4),
    c_index_disparity DECIMAL(5,4),
    fairness_passed BOOLEAN,
    
    -- Training Details
    n_features INTEGER,
    features_selected JSONB,
    training_samples INTEGER,
    training_config JSONB
);

CREATE INDEX idx_model_registry_is_active ON model_registry(is_active);

-- ==========================================
-- Audit Log Table
-- ==========================================
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    entity_id UUID,
    details JSONB,
    ip_address INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX idx_audit_log_created_at ON audit_log(created_at);
CREATE INDEX idx_audit_log_action ON audit_log(action);

-- ==========================================
-- Insert Default Admin User
-- Password: admin123 (hashed with SHA256)
-- ==========================================
INSERT INTO users (email, password_hash, name, role) 
VALUES (
    'admin@example.com', 
    '240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9',
    'Admin User',
    'admin'
) ON CONFLICT (email) DO NOTHING;

-- ==========================================
-- Create Updated At Trigger Function
-- ==========================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to tables
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_patients_updated_at 
    BEFORE UPDATE ON patients 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
