# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2025-12-12

### Added

#### Core System
- Complete Docker Compose orchestration with 5 services
- PostgreSQL 15 database with full patient schema
- Redis 7 for session management
- Health checks for all services

#### AI Service (Port 8000)
- 7-module ML pipeline architecture (M1-M7)
- GBM model with 45 selected features
- Forced inclusion of 14 comorbidity features
- Clinical adjustments for age, Karnofsky, DRI, comorbidities
- Confidence level indicators for borderline cases
- 5-fold stratified cross-validation
- Model persistence with pickle

#### Backend API (Port 8001)
- FastAPI REST API
- JWT authentication
- SQLAlchemy ORM with 60+ patient fields
- Patient CRUD operations
- Prediction generation and history
- Audit logging for all actions
- Session management with Redis

#### Frontend (Port 80)
- React + Vite single-page application
- 7-tab patient form
- Real-time input validation
- Paste content sanitization
- Double-click submission prevention
- Risk category visualization (Low/Medium/High)
- Confidence level display
- Top risk factors visualization

#### Documentation
- Technical README for each service
- User Guide with user stories
- Architecture diagrams (Mermaid)
- API endpoint documentation

### Security
- Password hashing with bcrypt
- JWT token authentication
- Input sanitization
- Audit trail logging

---

## [0.1.0] - 2025-12-10

### Added
- Initial project structure
- Basic ML pipeline prototype
- Docker configuration drafts

---

## Future Roadmap

### [1.1.0] - Planned
- [ ] EMR integration (HL7 FHIR)
- [ ] Batch prediction upload (CSV)
- [ ] PDF report generation
- [ ] Email notifications

### [1.2.0] - Planned
- [ ] Hospital SSO integration
- [ ] Multi-language support
- [ ] Mobile responsive design
- [ ] Advanced analytics dashboard

### [2.0.0] - Planned
- [ ] Survival curve visualization
- [ ] Time-to-event predictions
- [ ] Model retraining UI
- [ ] A/B testing framework
