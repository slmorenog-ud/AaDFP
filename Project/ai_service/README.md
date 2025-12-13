# AI Service - Technical Documentation

## HCT Survival Prediction Service

This service implements the Machine Learning pipeline for post-hematopoietic cell transplant survival prediction.

---

## 📐 7-Module Architecture

```mermaid
flowchart LR
    subgraph Training["Training Flow"]
        direction LR
        M1["M1<br/>Preprocessing"]
        M2["M2<br/>Equity"]
        M3["M3<br/>Features"]
        M4["M4<br/>Model"]
    end
    
    subgraph Inference["Inference Flow"]
        direction LR
        M5["M5<br/>Calibration"]
        M6["M6<br/>Uncertainty"]
        M7["M7<br/>Output"]
    end
    
    M1 --> M2 --> M3 --> M4
    M4 --> M5 --> M6 --> M7
    
    style M1 fill:#e3f2fd,stroke:#1976d2
    style M2 fill:#e8f5e9,stroke:#388e3c
    style M3 fill:#fff3e0,stroke:#f57c00
    style M4 fill:#fce4ec,stroke:#c2185b
    style M5 fill:#f3e5f5,stroke:#7b1fa2
    style M6 fill:#e0f7fa,stroke:#0097a7
    style M7 fill:#fff8e1,stroke:#ffa000
```

---

## 🔌 API Endpoints

### Health Check
```
GET /health
Response: { "status": "healthy", "model_loaded": true }
```

### Model Information
```
GET /model/info
Response: {
  "model_loaded": true,
  "model_type": "gbm",
  "n_features": 45,
  "training_date": "2025-12-12T...",
  "performance": { "auc": 0.74, "accuracy": 0.70 }
}
```

### Generate Prediction
```
POST /predict
Body: { "id": "patient-uuid", "age_at_hct": 45, ... }
Response: {
  "event_probability": 0.35,
  "risk_category": "Medium",
  "confidence_level": "high",
  "confidence_lower": 0.30,
  "confidence_upper": 0.40,
  "reliability_score": 0.85,
  "top_risk_factors": [...]
}
```

### Train Model
```
POST /train
Body: { "model_type": "gbm", "n_features": 45 }
Response: { "status": "success", "metrics": {...} }
```

### Fairness Metrics
```
GET /fairness
Response: {
  "overall_auc": 0.74,
  "group_metrics": {...},
  "disparity": 0.05
}
```

---

## 📦 Detailed Modules

### M1: Preprocessing (`m1_preprocessing.py`)

```python
class DataPreprocessor:
    def load_data(filepath: str) -> pd.DataFrame
    def validate_data(df) -> ValidationReport
    def fit_transform(df) -> Tuple[X, y_event, y_time]
    def transform(df) -> pd.DataFrame
```

**Features:**
- CSV loading with encoding handling
- Required column validation (efs, efs_time, race_group)
- Missing value detection > 30%
- Imputation: median for numeric, mode for categorical
- Categorical encoding (LabelEncoder)
- Optional normalization (StandardScaler)

---

### M2: Equity Analysis (`m2_equity.py`)

```python
class EquityAnalyzer:
    def analyze_representation(df, group_col) -> Dict
    def calculate_equity_weights(y, groups) -> np.array
    def compute_disparity_metrics(y_true, y_pred, groups) -> Dict
```

**Features:**
- Representation calculation by demographic group
- Sample weights for balanced training
- Disparity metrics: max difference in accuracy/recall

---

### M3: Feature Selection (`m3_features.py`)

```python
class FeatureSelector:
    FORCED_COMORBIDITY_FEATURES = [14 comorbidities]
    ADDITIONAL_CLINICAL_FEATURES = [13 clinical variables]
    
    def select_features(X, y, n_features=45) -> List[str]
    def get_feature_importance(X, y) -> pd.DataFrame
```

**Selection Strategy:**
1. Force inclusion of 14 comorbidities
2. Force inclusion of 13 key clinical variables
3. Complete to n_features with statistical importance

---

### M4: Predictive Modeling (`m4_models.py`)

```python
class PredictiveModel:
    MODEL_CONFIGS = {
        'gbm': { n_estimators: 100, max_depth: 5, ... },
        'rf': { n_estimators: 100, max_depth: 10, ... }
    }
    
    def train(X, y, sample_weight=None) -> ModelMetrics
    def predict_proba(X) -> np.array
    def cross_validate(X, y, cv=5) -> CVResults
```

**Models:**
- **GBM** (default): Gradient Boosting, best for tabular data
- **RF**: Random Forest, more robust to overfitting

---

### M5: Fairness Calibration (`m5_calibration.py`)

```python
class FairnessCalibrator:
    def fit(y_true, y_pred, groups)
    def calibrate(y_pred, groups) -> np.array
    def evaluate_calibration() -> Dict
```

**Techniques:**
- Isotonic calibration by group
- Threshold adjustment for equity

---

### M6: Uncertainty Quantification (`m6_uncertainty.py`)

```python
class UncertaintyQuantifier:
    def estimate_confidence(model, X, n_bootstrap=100) -> Tuple[lower, upper]
    def calculate_reliability_score(X, X_train) -> float
```

**Metrics:**
- Bootstrap confidence intervals
- Reliability score based on distance to training data

---

### M7: Output Generation (`m7_outputs.py`)

```python
class OutputGenerator:
    THRESHOLDS = { 'LOW': 0.28, 'MEDIUM': 0.55 }
    BORDERLINE_MARGIN = 0.05  # 5%
    
    def generate_prediction(patient_data) -> PredictionResult
    def get_risk_category(probability) -> str
    def get_confidence_level(probability) -> str
```

**Risk Categories:**
| Probability | Category |
|-------------|----------|
| < 0.28 | Low |
| 0.28 - 0.55 | Medium |
| > 0.55 | High |

**Confidence Levels:**
- `high`: Far from thresholds (> 5%)
- `borderline (near X)`: Close to threshold (≤ 5%)

---

## 🔄 Complete Pipeline (`pipeline.py`)

```python
class HCTPipeline:
    def train(data_path, model_type='gbm', n_features=45) -> Dict
    def predict(patient_data: Dict) -> PredictionResult
    def save(path: str)
    def load(path: str)
```

### Training Flow

1. **M1**: Load and validate data
2. **M2**: Calculate equity weights
3. **M3**: Select features (45)
4. **M4**: Train model with CV
5. **M5**: Calibrate for fairness
6. **M6**: Configure uncertainty estimation
7. **M7**: Configure output generator
8. Save model (.pkl)

### Prediction Flow

1. Receive patient data
2. Preprocess with saved transformers
3. Generate base prediction
4. Apply clinical adjustments
5. Estimate uncertainty
6. Generate structured output

---

## 📊 Post-Prediction Clinical Adjustments

```python
def _apply_clinical_adjustments(probability, patient_data) -> float:
    """Adjusts probability based on known clinical factors."""
    
    # Extreme age
    if age < 18: prob += 0.03
    if age > 60: prob += 0.05
    if age > 70: prob += 0.10
    
    # Comorbidity score
    if score >= 5: prob += 0.15
    elif score >= 3: prob += 0.05
    
    # Karnofsky score
    if karnofsky <= 50: prob += 0.15
    elif karnofsky <= 70: prob += 0.05
    elif karnofsky >= 90: prob -= 0.05
    
    # DRI score
    if dri == 'Very high': prob += 0.10
    elif dri == 'High': prob += 0.05
    
    # Individual comorbidities
    active_comorbs = count_active_comorbidities()
    if active_comorbs >= 4: prob += 0.10
    elif active_comorbs >= 2: prob += 0.05
    
    return clip(prob, 0.01, 0.99)
```

---

## 💾 Model Persistence

The model is saved in `models/pipeline.pkl`:

```python
# File structure
{
    'preprocessor': DataPreprocessor,
    'feature_selector': FeatureSelector,
    'model': PredictiveModel,
    'calibrator': FairnessCalibrator,
    'uncertainty': UncertaintyQuantifier,
    'output_gen': OutputGenerator,
    'training_info': {
        'timestamp': '2025-12-12T...',
        'n_features': 45,
        'model_type': 'gbm',
        'metrics': {...}
    }
}
```

---

## 🧪 Testing

### Quick Test Script

```bash
python test_predictions.py
```

### Included Test Cases

| Case | Description | Expected |
|------|-------------|----------|
| 1 | Young, healthy, MSD | Low |
| 2 | Middle adult, comorbidities | Medium |
| 3 | Elderly, multiple comorbidities | High |
| 4 | Pediatric, healthy | Low |
| 5 | Adult, many comorbidities | High |
| 6 | Cord blood, moderate risk | Medium |

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.74 |
| Accuracy | 70% |
| Brier Score | 0.18 |
| Max Group Disparity | 5% |

---

## ⚙️ Configuration

### Environment Variables

```bash
DATABASE_URL=postgresql://user:pass@host:5432/db
PYTHONUNBUFFERED=1
```

### Dependencies (`requirements.txt`)

```
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
scipy==1.11.4
pydantic==2.5.2
```

---

## 🔒 Security Considerations

1. **No sensitive data**: Model does not store patient data
2. **Input validation**: All endpoints validate data
3. **Timeouts**: 30 seconds for predictions
4. **Health checks**: Continuous availability verification

---

**Version**: 1.0.0  
**Last updated**: December 2025
