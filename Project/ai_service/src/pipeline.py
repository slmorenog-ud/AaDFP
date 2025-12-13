"""
HCT Survival Prediction Pipeline
Orchestrates all 7 modules for end-to-end prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pickle
import json
from datetime import datetime

# Import all modules (M1-M7)
from .m1_preprocessing import DataPreprocessor
from .m2_equity import EquityAnalyzer
from .m3_features import FeatureSelector
from .m4_models import PredictiveModel, EnsembleModel
from .m5_calibration import FairnessCalibrator
from .m6_uncertainty import UncertaintyQuantifier
from .m7_outputs import OutputGenerator, PredictionResult


class HCTPipeline:
    """
    Complete HCT Survival Prediction Pipeline.
    
    Orchestrates the 7-module architecture:
    M1: Data Preprocessing
    M2: Equity Analysis
    M3: Feature Selection
    M4: Predictive Modeling
    M5: Fairness Calibration
    M6: Uncertainty Quantification
    M7: System Outputs
    """
    
    def __init__(self, group_col: str = 'race_group'):
        self.preprocessor = DataPreprocessor()
        self.equity_analyzer = EquityAnalyzer(group_col=group_col)
        self.feature_selector = FeatureSelector()
        self.model = PredictiveModel()
        self.calibrator = FairnessCalibrator()
        self.uncertainty = UncertaintyQuantifier()
        self.output_gen = OutputGenerator()
        
        self.group_col = group_col
        self.is_trained = False
        self.training_info: Dict = {}
        
        # Store processed data for reference
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.groups_train: Optional[np.ndarray] = None
    
    def train(
        self,
        data_path: str,
        model_type: str = 'gbm',
        n_features: int = 25,
        use_equity_weights: bool = True
    ) -> Dict:
        """
        Train the complete pipeline.
        
        Args:
            data_path: Path to training data CSV
            model_type: 'gbm', 'rf', or 'ensemble'
            n_features: Number of features to select
            use_equity_weights: Whether to use equity-aware sample weights
        
        Returns:
            Dictionary with training results
        """
        print("=" * 60)
        print("HCT Survival Prediction - Training Pipeline")
        print("=" * 60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_path': data_path,
            'stages': {}
        }
        
        # ==========================================
        # M1: Data Preprocessing
        # ==========================================
        print("\n[M1] Data Preprocessing...")
        
        df = self.preprocessor.load_data(data_path)
        df_original = df.copy()
        
        validation = self.preprocessor.validate_data(df)
        results['stages']['preprocessing'] = {
            'total_rows': validation.total_rows,
            'total_columns': validation.total_columns,
            'issues': validation.issues
        }
        
        X, y_event, y_time = self.preprocessor.fit_transform(df)
        
        print(f"   - Loaded {len(X)} samples with {len(X.columns)} features")
        
        # ==========================================
        # M2: Equity Analysis
        # ==========================================
        print("\n[M2] Equity Analysis...")
        
        # Store groups for later use
        if self.group_col in df_original.columns:
            self.groups_train = df_original[self.group_col].values
            
            stratified = self.equity_analyzer.stratified_analysis(df_original, 'efs')
            bias_report = self.equity_analyzer.detect_bias(df_original, 'efs')
            
            results['stages']['equity_analysis'] = {
                'group_counts': stratified.group_counts,
                'group_event_rates': stratified.group_event_rates,
                'bias_detected': bias_report.bias_detected,
                'max_disparity': bias_report.max_disparity
            }
            
            print(f"   - Analyzed {len(stratified.group_counts)} demographic groups")
            print(f"   - Bias detected: {bias_report.bias_detected}")
            
            # Calculate sample weights
            if use_equity_weights:
                sample_weights = self.equity_analyzer.calculate_reweights(df_original)
            else:
                sample_weights = None
        else:
            self.groups_train = None
            sample_weights = None
            results['stages']['equity_analysis'] = {'error': f'Group column {self.group_col} not found'}
        
        # ==========================================
        # M3: Feature Selection
        # ==========================================
        print("\n[M3] Feature Selection...")
        
        selected_features = self.feature_selector.select_features(
            X, y_event, 
            df_original=df_original,
            n_features=n_features,
            method='combined',
            group_col=self.group_col
        )
        
        feature_report = self.feature_selector.get_feature_report()
        results['stages']['feature_selection'] = {
            'n_selected': len(selected_features),
            'selected_features': selected_features[:10],  # Top 10
            'clinical_features_included': len(feature_report['clinical_features_included']),
            'availability_concerns': len(feature_report['availability_concerns'])
        }
        
        # Filter to selected features
        X_selected = X[selected_features]
        
        print(f"   - Selected {len(selected_features)} features")
        
        # ==========================================
        # M4: Predictive Modeling
        # ==========================================
        print("\n[M4] Predictive Modeling...")
        
        # Cross-validation first
        cv_results = self.model.cross_validate(
            X_selected, y_event,
            groups=self.groups_train,
            n_splits=5,
            model_type=model_type if model_type != 'ensemble' else 'gbm'
        )
        
        print(f"   - CV Accuracy: {cv_results.mean_accuracy:.4f} (+/- {cv_results.std_accuracy:.4f})")
        print(f"   - CV AUC: {cv_results.mean_auc:.4f} (+/- {cv_results.std_auc:.4f})")
        
        # Train final model
        if model_type == 'ensemble':
            self.model = EnsembleModel()
            self.model.train(X_selected, y_event, sample_weights=sample_weights)
        else:
            self.model.train(X_selected, y_event, model_type, sample_weights=sample_weights)
        
        # Get predictions on training data for calibration
        y_proba = self.model.predict_proba(X_selected)
        y_pred = self.model.predict(X_selected)
        
        from sklearn.metrics import accuracy_score, roc_auc_score
        train_metrics = {
            'accuracy': float(accuracy_score(y_event, y_pred)),
            'auc_roc': float(roc_auc_score(y_event, y_proba))
        }
        
        results['stages']['modeling'] = {
            'model_type': model_type,
            'cv_accuracy': cv_results.mean_accuracy,
            'cv_auc': cv_results.mean_auc,
            'train_metrics': train_metrics,
            'feature_importance': self.model.get_feature_importance() if hasattr(self.model, 'get_feature_importance') else {}
        }
        
        # ==========================================
        # M5: Fairness Calibration
        # ==========================================
        print("\n[M5] Fairness Calibration...")
        
        if self.groups_train is not None:
            # Calibrate probabilities
            calibration_result = self.calibrator.calibrate_probabilities(
                y_event.values, y_proba, self.groups_train
            )
            
            # Calculate stratified C-index
            c_index_results = self.calibrator.stratified_c_index(
                y_event.values, calibration_result.calibrated_probas, self.groups_train
            )
            
            results['stages']['fairness'] = {
                'stratified_c_index': c_index_results['stratified_c_index'],
                'c_index_disparity': c_index_results['c_index_disparity'],
                'fairness_passed': c_index_results['fairness_passed'],
                'calibration_improvement': calibration_result.improvement
            }
            
            print(f"   - Stratified C-index: {c_index_results['stratified_c_index']:.4f}")
            print(f"   - Disparity: {c_index_results['c_index_disparity']:.4f}")
            print(f"   - Fairness passed: {c_index_results['fairness_passed']}")
        else:
            results['stages']['fairness'] = {'error': 'No group information available'}
        
        # ==========================================
        # M6: Uncertainty Quantification
        # ==========================================
        print("\n[M6] Uncertainty Quantification...")
        
        # Chaos analysis
        chaos_results = self.uncertainty.chaos_analysis(
            self.model.model if hasattr(self.model, 'model') else self.model.models[0].model,
            X_selected
        )
        
        results['stages']['uncertainty'] = {
            'stability_rating': chaos_results['stability_rating'],
            'max_prediction_change': max(chaos_results['prediction_changes']),
            'noise_levels_tested': chaos_results['noise_levels']
        }
        
        print(f"   - Stability rating: {chaos_results['stability_rating']}")
        
        # Store training data
        self.X_train = X_selected
        self.y_train = y_event
        self.is_trained = True
        self.training_info = results
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        
        return results
    
    def _apply_clinical_adjustments(self, base_proba: float, patient_data: Dict) -> float:
        """
        Apply clinical risk adjustments based on evidence-based factors
        that may not be fully captured by the ML model.
        
        This implements a multiplicative adjustment model based on:
        - Comorbidity burden
        - Age extremes
        - Disease risk index
        - Donor match quality
        - Performance status
        
        Returns adjusted probability capped at [0.05, 0.95]
        """
        adjustment = 1.0
        
        # 1. Age adjustments (U-shaped risk curve)
        age = patient_data.get('age_at_hct', 40)
        if age is not None:
            try:
                age = float(age)
                if age < 18:  # Pediatric - generally better outcomes
                    adjustment *= 0.85
                elif age > 60:  # Elderly - increased risk
                    adjustment *= 1.0 + (age - 60) * 0.02  # 2% increase per year over 60
            except (ValueError, TypeError):
                pass
        
        # 2. Comorbidity adjustments
        comorbidity_score = patient_data.get('comorbidity_score', 0)
        if comorbidity_score is not None:
            try:
                comorbidity_score = float(comorbidity_score)
                if comorbidity_score >= 5:
                    adjustment *= 1.4  # 40% increase for high comorbidity
                elif comorbidity_score >= 3:
                    adjustment *= 1.2  # 20% increase for moderate
            except (ValueError, TypeError):
                pass
        
        # Count individual comorbidities
        comorbidity_fields = ['cardiac', 'arrhythmia', 'diabetes', 'hepatic_mild', 
                             'hepatic_severe', 'obesity', 'pulm_moderate', 'pulm_severe',
                             'renal_issue', 'psych_disturb']
        active_comorbidities = sum(1 for c in comorbidity_fields 
                                   if str(patient_data.get(c, '')).upper() == 'Y')
        if active_comorbidities >= 4:
            adjustment *= 1.3  # Additional 30% for multiple comorbidities
        elif active_comorbidities >= 2:
            adjustment *= 1.15
        
        # 3. Performance status (Karnofsky)
        karnofsky = patient_data.get('karnofsky_score', 90)
        if karnofsky is not None:
            try:
                karnofsky = float(karnofsky)
                if karnofsky <= 50:
                    adjustment *= 1.5  # 50% increase for poor performance
                elif karnofsky <= 70:
                    adjustment *= 1.25
                elif karnofsky >= 90:
                    adjustment *= 0.9  # 10% decrease for good performance
            except (ValueError, TypeError):
                pass
        
        # 4. Disease risk adjustments
        dri_score = str(patient_data.get('dri_score', '')).lower()
        if 'very high' in dri_score or 'high' in dri_score:
            adjustment *= 1.25
        elif 'low' in dri_score:
            adjustment *= 0.85
        
        # 5. Donor match quality
        hla_match = patient_data.get('hla_high_res_8', 8)
        if hla_match is not None:
            try:
                hla_match = float(hla_match)
                if hla_match <= 5:
                    adjustment *= 1.3  # Mismatched
                elif hla_match >= 8:
                    adjustment *= 0.9  # Well matched
            except (ValueError, TypeError):
                pass
        
        # Apply adjustment
        adjusted_proba = base_proba * adjustment
        
        # Cap between 0.05 and 0.95
        return max(0.05, min(0.95, adjusted_proba))

    def predict(
        self,
        patient_data: Dict,
        include_explanation: bool = True,
        include_confidence: bool = True
    ) -> PredictionResult:
        """
        Generate prediction for a single patient.
        
        Args:
            patient_data: Dictionary with patient features
            include_explanation: Whether to include SHAP explanations
            include_confidence: Whether to include confidence intervals
        
        Returns:
            PredictionResult object
        """
        if not self.is_trained:
            raise ValueError("Pipeline not trained. Call train() first.")
        
        # Clean patient data - remove None values and convert to proper types
        clean_data = {}
        for key, value in patient_data.items():
            if value is not None and value != '' and str(value).lower() != 'none':
                clean_data[key] = value
        
        # Convert to DataFrame
        df = pd.DataFrame([clean_data])
        
        # Preprocess
        X = self.preprocessor.transform(df)
        
        # Ensure all required features are present with defaults
        for feat in self.feature_selector.selected_features:
            if feat not in X.columns:
                X[feat] = 0
        
        # Select only the features used by the model in correct order
        X = X[self.feature_selector.selected_features]
        
        # Predict
        proba = self.model.predict_proba(X)[0]
        
        # Apply clinical risk adjustments based on known high-impact factors
        # that may not be fully captured by the model
        proba = self._apply_clinical_adjustments(proba, clean_data)
        
        # Calibrate if group info available
        if 'race_group' in patient_data:
            group = np.array([patient_data['race_group']])
            proba = self.calibrator.transform_probas(np.array([proba]), group)[0]
        
        # Confidence interval (simplified for single prediction)
        ci = None
        if include_confidence and hasattr(self, 'X_train'):
            try:
                ci_result = self.uncertainty.bootstrap_confidence_intervals(
                    type(self.model.model if hasattr(self.model, 'model') else self.model.models[0].model),
                    self.X_train, self.y_train, X,
                    n_bootstrap=20
                )
                ci = (ci_result.lower_bound[0], ci_result.upper_bound[0])
            except:
                ci = None
        
        # SHAP explanation
        shap_vals = None
        if include_explanation:
            try:
                self.output_gen.setup_explainer(
                    self.model.model if hasattr(self.model, 'model') else self.model.models[0].model,
                    self.X_train
                )
                shap_vals = self.output_gen.generate_shap_explanation(X)
            except:
                shap_vals = None
        
        # Reliability 
        reliability = self.uncertainty.assess_reliability(np.array([proba]))
        
        # Generate result
        result = self.output_gen.generate_prediction_report(
            patient_id=patient_data.get('id', 'unknown'),
            probability=proba,
            confidence_interval=ci,
            reliability=reliability.reliability_scores[0],
            shap_values=shap_vals
        )
        
        return result
    
    def batch_predict(
        self,
        data_path: str,
        output_path: Optional[str] = None
    ) -> Tuple[List[PredictionResult], Dict]:
        """
        Generate predictions for multiple patients.
        
        Args:
            data_path: Path to data CSV
            output_path: Optional path to save predictions
        
        Returns:
            Tuple of (predictions list, summary dict)
        """
        if not self.is_trained:
            raise ValueError("Pipeline not trained. Call train() first.")
        
        # Load and preprocess
        df = self.preprocessor.load_data(data_path)
        df_original = df.copy()
        X = self.preprocessor.transform(df)
        
        # Select features
        X = X[[c for c in self.feature_selector.selected_features if c in X.columns]]
        for feat in self.feature_selector.selected_features:
            if feat not in X.columns:
                X[feat] = 0
        X = X[self.feature_selector.selected_features]
        
        # Predict
        probas = self.model.predict_proba(X)
        
        # Calibrate
        if self.group_col in df_original.columns:
            groups = df_original[self.group_col].values
            probas = self.calibrator.transform_probas(probas, groups)
        
        # Reliability
        reliability_result = self.uncertainty.assess_reliability(probas)
        
        # Generate patient IDs
        if 'ID' in df_original.columns:
            patient_ids = df_original['ID'].astype(str).tolist()
        else:
            patient_ids = [f"patient_{i}" for i in range(len(df_original))]
        
        # Generate predictions
        predictions = self.output_gen.generate_batch_predictions(
            patient_ids=patient_ids,
            probabilities=probas,
            reliability_scores=reliability_result.reliability_scores
        )
        
        # Summary
        summary = {
            'total_predictions': len(predictions),
            'risk_distribution': {
                'low': sum(1 for p in predictions if p.risk_category == 'Low'),
                'medium': sum(1 for p in predictions if p.risk_category == 'Medium'),
                'high': sum(1 for p in predictions if p.risk_category == 'High')
            },
            'average_probability': float(probas.mean()),
            'average_reliability': float(reliability_result.average_reliability)
        }
        
        # Save if path provided
        if output_path:
            self.output_gen.export_predictions_csv(predictions, output_path)
            print(f"Predictions saved to: {output_path}")
        
        return predictions, summary
    
    def get_fairness_report(self) -> Dict:
        """Get fairness metrics for trained model."""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        y_proba = self.model.predict_proba(self.X_train)
        y_pred = (y_proba >= 0.5).astype(int)
        
        return self.calibrator.generate_fairness_report(
            self.y_train.values, y_proba, self.groups_train
        )
    
    def save(self, filepath: str) -> None:
        """Save trained pipeline to disk."""
        if not self.is_trained:
            raise ValueError("Pipeline not trained")
        
        pipeline_data = {
            'preprocessor': self.preprocessor,
            'equity_analyzer': self.equity_analyzer,
            'feature_selector': self.feature_selector,
            'model': self.model,
            'calibrator': self.calibrator,
            'training_info': self.training_info,
            'group_col': self.group_col
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
    
    def load(self, filepath: str) -> 'HCTPipeline':
        """Load trained pipeline from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.preprocessor = data['preprocessor']
        self.equity_analyzer = data['equity_analyzer']
        self.feature_selector = data['feature_selector']
        self.model = data['model']
        self.calibrator = data['calibrator']
        self.training_info = data['training_info']
        self.group_col = data['group_col']
        self.is_trained = True
        
        return self


# Convenience function for quick training
def train_pipeline(
    data_path: str,
    model_type: str = 'gbm',
    n_features: int = 25
) -> Tuple[HCTPipeline, Dict]:
    """
    Quick function to train a pipeline.
    
    Args:
        data_path: Path to training data
        model_type: Model type to use
        n_features: Number of features
    
    Returns:
        Tuple of (trained pipeline, training results)
    """
    pipeline = HCTPipeline()
    results = pipeline.train(data_path, model_type, n_features)
    return pipeline, results
