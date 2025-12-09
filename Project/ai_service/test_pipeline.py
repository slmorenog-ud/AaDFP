    """
Test script for HCT Survival Prediction Pipeline.
Validates all 7 modules and end-to-end functionality.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from pathlib import Path


def test_preprocessing():
    """Test M1: Data Preprocessing Module."""
    print("\n" + "="*60)
    print("Testing M1: Data Preprocessing")
    print("="*60)
    
    from src.preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    
    # Load data
    data_path = Path("data/raw/train.csv")
    if not data_path.exists():
        print(f"  ⚠ Data file not found: {data_path}")
        return None, None, None
    
    df = preprocessor.load_data(str(data_path))
    print(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Validate
    report = preprocessor.validate_data(df)
    print(f"  ✓ Validation: {len(report.issues)} issues found")
    
    # Preprocess
    X, y_event, y_time = preprocessor.fit_transform(df)
    print(f"  ✓ Preprocessed features shape: {X.shape}")
    print(f"  ✓ Event rate: {y_event.mean():.2%}")
    
    return df, preprocessor, (X, y_event)


def test_equity_analysis(df):
    """Test M2: Equity Analysis Module."""
    print("\n" + "="*60)
    print("Testing M2: Equity Analysis")
    print("="*60)
    
    from src.equity import EquityAnalyzer
    
    analyzer = EquityAnalyzer(group_col='race_group')
    
    # Stratified analysis
    analysis = analyzer.stratified_analysis(df, 'efs')
    print(f"  ✓ Groups analyzed: {len(analysis.group_counts)}")
    for group, count in list(analysis.group_counts.items())[:3]:
        print(f"    - {group}: {count} patients")
    
    # Bias detection
    bias = analyzer.detect_bias(df, 'efs')
    print(f"  ✓ Bias detected: {bias.bias_detected}")
    print(f"  ✓ Max disparity: {bias.max_disparity:.4f}")
    
    # Reweighting
    weights = analyzer.calculate_reweights(df)
    print(f"  ✓ Sample weights calculated: min={weights.min():.2f}, max={weights.max():.2f}")
    
    return analyzer, weights


def test_feature_selection(X, y):
    """Test M3: Feature Selection Module."""
    print("\n" + "="*60)
    print("Testing M3: Feature Selection")
    print("="*60)
    
    from src.features import FeatureSelector
    
    selector = FeatureSelector()
    
    # Select features
    selected = selector.select_features(X, y, n_features=20, method='combined')
    print(f"  ✓ Selected {len(selected)} features")
    print(f"  ✓ Top 5 features: {selected[:5]}")
    
    # Feature report
    report = selector.get_feature_report()
    print(f"  ✓ Clinical features included: {len(report['clinical_features_included'])}")
    
    return selector, selected


def test_modeling(X, y, selected_features, weights=None):
    """Test M4: Predictive Modeling Module."""
    print("\n" + "="*60)
    print("Testing M4: Predictive Modeling")
    print("="*60)
    
    from src.models import PredictiveModel
    from sklearn.model_selection import train_test_split
    
    model = PredictiveModel()
    
    # Select features
    X_selected = X[selected_features]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Cross-validation
    cv_results = model.cross_validate(X_train, y_train, n_splits=3, model_type='gbm')
    print(f"  ✓ CV Accuracy: {cv_results.mean_accuracy:.4f} (+/- {cv_results.std_accuracy:.4f})")
    print(f"  ✓ CV AUC: {cv_results.mean_auc:.4f} (+/- {cv_results.std_auc:.4f})")
    
    # Train model
    if weights is not None:
        train_weights = weights[:len(X_train)]
        model.train(X_train, y_train, 'gbm', sample_weights=train_weights)
    else:
        model.train(X_train, y_train, 'gbm')
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Metrics
    metrics = model.calculate_metrics(y_test.values, y_pred, y_proba)
    print(f"  ✓ Test Accuracy: {metrics.accuracy:.4f}")
    print(f"  ✓ Test AUC-ROC: {metrics.auc_roc:.4f}")
    print(f"  ✓ Test F1: {metrics.f1:.4f}")
    
    # Feature importance
    importance = model.get_feature_importance()
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  ✓ Top 5 important features:")
    for feat, imp in top_features:
        print(f"    - {feat}: {imp:.4f}")
    
    return model, (X_test, y_test, y_proba)


def test_calibration(y_true, y_proba, groups):
    """Test M5: Fairness Calibration Module."""
    print("\n" + "="*60)
    print("Testing M5: Fairness Calibration")
    print("="*60)
    
    from src.calibration import FairnessCalibrator
    
    calibrator = FairnessCalibrator()
    
    # Calibrate probabilities
    result = calibrator.calibrate_probabilities(y_true, y_proba, groups)
    print(f"  ✓ Calibration improvement: {result.improvement:.4f}")
    
    # Stratified C-index
    c_index = calibrator.stratified_c_index(y_true, result.calibrated_probas, groups)
    print(f"  ✓ Stratified C-index: {c_index['stratified_c_index']:.4f}")
    print(f"  ✓ C-index disparity: {c_index['c_index_disparity']:.4f}")
    print(f"  ✓ Fairness passed: {c_index['fairness_passed']}")
    
    return calibrator


def test_uncertainty(model, X):
    """Test M6: Uncertainty Quantification Module."""
    print("\n" + "="*60)
    print("Testing M6: Uncertainty Quantification")
    print("="*60)
    
    from src.uncertainty import UncertaintyQuantifier
    
    uq = UncertaintyQuantifier()
    
    # Get predictions
    y_proba = model.predict_proba(X)
    
    # Risk stratification
    risk = uq.risk_stratification(y_proba)
    print(f"  ✓ Risk distribution:")
    print(f"    - Low: {(risk == 0).sum()}")
    print(f"    - Medium: {(risk == 1).sum()}")
    print(f"    - High: {(risk == 2).sum()}")
    
    # Reliability assessment
    reliability = uq.assess_reliability(y_proba)
    print(f"  ✓ Average reliability: {reliability.average_reliability:.4f}")
    print(f"  ✓ High confidence predictions: {reliability.high_confidence_mask.sum()}")
    
    # Chaos analysis
    chaos = uq.chaos_analysis(model.model, X)
    print(f"  ✓ Stability rating: {chaos['stability_rating']}")
    
    return uq


def test_outputs():
    """Test M7: System Outputs Module."""
    print("\n" + "="*60)
    print("Testing M7: System Outputs")
    print("="*60)
    
    from src.outputs import OutputGenerator, PredictionResult
    
    output_gen = OutputGenerator()
    
    # Generate sample prediction report
    result = output_gen.generate_prediction_report(
        patient_id="test_patient_1",
        probability=0.65,
        confidence_interval=(0.55, 0.75),
        reliability=0.8
    )
    
    print(f"  ✓ Prediction generated:")
    print(f"    - Patient ID: {result.patient_id}")
    print(f"    - Event probability: {result.event_probability:.1%}")
    print(f"    - Risk category: {result.risk_category}")
    print(f"    - Reliability: {result.reliability_score:.2f}")
    
    # Clinical summary
    summary = output_gen.format_clinical_summary(result)
    print(f"  ✓ Clinical summary generated ({len(summary)} characters)")
    
    return output_gen


def test_full_pipeline():
    """Test complete pipeline integration."""
    print("\n" + "="*60)
    print("Testing Full Pipeline Integration")
    print("="*60)
    
    from src.pipeline import HCTPipeline
    
    data_path = Path("data/raw/train.csv")
    if not data_path.exists():
        print(f"  ⚠ Skipping pipeline test - data file not found")
        return
    
    pipeline = HCTPipeline()
    
    # Train pipeline
    print("  Training pipeline...")
    results = pipeline.train(str(data_path), model_type='gbm', n_features=20)
    
    print(f"\n  ✓ Pipeline trained successfully!")
    print(f"  ✓ Training timestamp: {results['timestamp']}")
    
    # Check stages
    for stage, data in results['stages'].items():
        print(f"  ✓ {stage}: completed")
    
    # Get fairness report
    fairness = pipeline.get_fairness_report()
    c_idx = fairness.get('c_index', {}).get('stratified_c_index', 0)
    print(f"\n  ✓ Final stratified C-index: {c_idx:.4f}")
    
    return pipeline


def run_all_tests():
    """Run all module tests."""
    print("\n" + "#"*60)
    print("#  HCT Survival Prediction - Test Suite")
    print("#"*60)
    
    try:
        # M1: Preprocessing
        result = test_preprocessing()
        if result[0] is None:
            print("\n⚠ Skipping remaining tests - no data available")
            return
        
        df, preprocessor, (X, y) = result
        
        # M2: Equity Analysis
        analyzer, weights = test_equity_analysis(df)
        
        # M3: Feature Selection
        selector, selected = test_feature_selection(X, y)
        
        # M4: Modeling
        model, test_data = test_modeling(X, y, selected, weights)
        X_test, y_test, y_proba = test_data
        
        # M5: Calibration
        groups = df.iloc[X_test.index]['race_group'].values if 'race_group' in df.columns else np.array(['Unknown'] * len(y_test))
        calibrator = test_calibration(y_test.values, y_proba, groups)
        
        # M6: Uncertainty
        uq = test_uncertainty(model, X_test)
        
        # M7: Outputs
        output_gen = test_outputs()
        
        # Full Pipeline
        pipeline = test_full_pipeline()
        
        print("\n" + "#"*60)
        print("#  All Tests Completed Successfully! ✓")
        print("#"*60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
