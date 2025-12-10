"""
AI Service package initialization.
7-Module ML Pipeline for HCT Survival Prediction.
"""

# M1: Data Preprocessing
from .m1_preprocessing import DataPreprocessor

# M2: Equity Analysis
from .m2_equity import EquityAnalyzer

# M3: Feature Selection
from .m3_features import FeatureSelector

# M4: Predictive Modeling
from .m4_models import PredictiveModel, EnsembleModel

# M5: Fairness Calibration
from .m5_calibration import FairnessCalibrator

# M6: Uncertainty Quantification
from .m6_uncertainty import UncertaintyQuantifier

# M7: System Outputs
from .m7_outputs import OutputGenerator, PredictionResult

# Pipeline Orchestrator
from .pipeline import HCTPipeline, train_pipeline

__all__ = [
    # M1
    'DataPreprocessor',
    # M2
    'EquityAnalyzer',
    # M3
    'FeatureSelector',
    # M4
    'PredictiveModel',
    'EnsembleModel',
    # M5
    'FairnessCalibrator',
    # M6
    'UncertaintyQuantifier',
    # M7
    'OutputGenerator',
    'PredictionResult',
    # Pipeline
    'HCTPipeline',
    'train_pipeline'
]
