# Contributing Guide

Thank you for your interest in contributing to the HCT Prediction System!

---

## 📋 Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing Requirements](#testing-requirements)
6. [Pull Request Process](#pull-request-process)

---

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct. Please be respectful and inclusive in all interactions.

---

## Getting Started

### Prerequisites

- Docker & Docker Compose
- Git
- Python 3.11+ (for local development)
- Node.js 18+ (for frontend development)

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/AFP.git
cd AFP/Project
```

### Set Up Development Environment

```bash
# Copy environment template
cp .env.example .env

# Start all services
docker-compose up --build

# Or for development with hot-reload:
docker-compose -f docker-compose.dev.yml up
```

---

## Development Workflow

### Branch Naming

| Type | Format | Example |
|------|--------|---------|
| Feature | `feature/description` | `feature/add-pdf-export` |
| Bug Fix | `fix/description` | `fix/validation-error` |
| Documentation | `docs/description` | `docs/api-examples` |
| Refactor | `refactor/description` | `refactor/pipeline-modules` |

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**
```bash
feat(ai): add confidence interval to predictions
fix(frontend): resolve paste validation in name field
docs(api): add authentication examples
```

---

## Coding Standards

### Python (Backend & AI Service)

```python
# Follow PEP 8
# Use type hints
def predict(patient_data: dict) -> PredictionResult:
    """
    Generate a survival prediction for a patient.
    
    Args:
        patient_data: Dictionary with patient variables
        
    Returns:
        PredictionResult with probability and risk category
    """
    pass

# Use meaningful variable names
event_probability = model.predict_proba(features)[0, 1]

# Constants in UPPER_CASE
RISK_THRESHOLD_LOW = 0.28
RISK_THRESHOLD_HIGH = 0.55
```

### JavaScript/React (Frontend)

```javascript
// Use functional components with hooks
const PatientForm = ({ onSubmit }) => {
  const [formData, setFormData] = useState({});
  
  // Descriptive function names
  const handleFieldChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };
  
  return (
    // JSX here
  );
};

// Use PropTypes or TypeScript for type checking
PatientForm.propTypes = {
  onSubmit: PropTypes.func.isRequired,
};
```

### SQL/Database

```sql
-- Use snake_case for columns
-- Add comments for complex queries
-- Include indexes for frequently queried columns

CREATE TABLE patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    age_at_hct INTEGER NOT NULL CHECK (age_at_hct >= 0 AND age_at_hct <= 120),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_patients_created_at ON patients(created_at);
```

---

## Testing Requirements

### AI Service

```bash
cd ai_service

# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Test specific module
pytest tests/test_pipeline.py -v
```

**Required Coverage:** Minimum 80%

### Backend

```bash
cd backend

# Run tests
pytest app/tests/ -v

# Test API endpoints
pytest app/tests/test_api.py -v
```

### Frontend

```bash
cd frontend

# Run tests
npm test

# Run with coverage
npm test -- --coverage
```

### Integration Tests

```bash
# From project root
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

---

## Pull Request Process

### Before Submitting

1. **Update your branch:**
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Run all tests:**
   ```bash
   # Backend
   pytest
   
   # Frontend
   npm test
   ```

3. **Check linting:**
   ```bash
   # Python
   flake8 .
   black --check .
   
   # JavaScript
   npm run lint
   ```

4. **Update documentation** if needed

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

### Review Process

1. Create PR against `main` branch
2. Wait for automated checks to pass
3. Request review from maintainers
4. Address feedback
5. Squash and merge when approved

---

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Contact maintainers for sensitive issues

---

Thank you for contributing! 🎉
