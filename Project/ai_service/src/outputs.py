"""
Module M7: System Outputs
Generates predictions, explanations, and fairness dashboards.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
from datetime import datetime

# SHAP is optional for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


@dataclass
class PredictionResult:
    """Individual patient prediction result."""
    patient_id: str
    event_probability: float
    risk_category: str
    confidence_lower: Optional[float]
    confidence_upper: Optional[float]
    reliability_score: float
    top_risk_factors: List[Dict[str, float]]
    timestamp: str


@dataclass
class FairnessDashboard:
    """Fairness metrics dashboard."""
    stratified_c_index: float
    group_metrics: Dict[str, Dict[str, float]]
    disparity_metrics: Dict[str, float]
    fairness_passed: bool
    recommendations: List[str]


class OutputGenerator:
    """
    M7: System Outputs
    
    Generates comprehensive outputs for clinical decision support
    and equity monitoring.
    """
    
    def __init__(self):
        self.shap_explainer = None
        self.feature_names: List[str] = []
    
    def setup_explainer(self, model, X_background: pd.DataFrame) -> None:
        """
        Setup SHAP explainer for model interpretation.
        
        Args:
            model: Trained model
            X_background: Background data for SHAP
        """
        if not SHAP_AVAILABLE:
            print("Warning: SHAP not installed. Interpretability features limited.")
            return
        
        self.feature_names = X_background.columns.tolist()
        
        # Use TreeExplainer for tree-based models
        try:
            self.shap_explainer = shap.TreeExplainer(model)
        except:
            # Fallback to KernelExplainer
            self.shap_explainer = shap.KernelExplainer(
                model.predict_proba,
                shap.sample(X_background, 100)
            )
    
    def generate_shap_explanation(
        self,
        X_sample: pd.DataFrame,
        top_n: int = 10
    ) -> Dict:
        """
        Generate SHAP-based explanation for predictions.
        
        Args:
            X_sample: Sample to explain
            top_n: Number of top features to include
        
        Returns:
            Dictionary with SHAP explanations
        """
        if self.shap_explainer is None:
            return {'error': 'SHAP explainer not configured'}
        
        shap_values = self.shap_explainer.shap_values(X_sample)
        
        # For binary classification, use positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        explanations = []
        for i in range(len(X_sample)):
            sample_shap = shap_values[i]
            sample_values = X_sample.iloc[i]
            
            # Get feature contributions
            contributions = list(zip(self.feature_names, sample_shap, sample_values))
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_features = []
            for feat, shap_val, feat_val in contributions[:top_n]:
                top_features.append({
                    'feature': feat,
                    'shap_value': float(shap_val),
                    'feature_value': float(feat_val) if isinstance(feat_val, (int, float)) else str(feat_val),
                    'direction': 'increases risk' if shap_val > 0 else 'decreases risk'
                })
            
            explanations.append({
                'top_features': top_features,
                'base_value': float(self.shap_explainer.expected_value[1]) if hasattr(self.shap_explainer.expected_value, '__len__') else float(self.shap_explainer.expected_value)
            })
        
        return {'explanations': explanations}
    
    def generate_prediction_report(
        self,
        patient_id: str,
        probability: float,
        confidence_interval: Optional[tuple] = None,
        reliability: float = 0.5,
        shap_values: Optional[Dict] = None
    ) -> PredictionResult:
        """
        Generate comprehensive prediction report for a patient.
        
        Args:
            patient_id: Patient identifier
            probability: Event probability
            confidence_interval: Optional (lower, upper) bounds
            reliability: Reliability score
            shap_values: Optional SHAP explanation
        
        Returns:
            PredictionResult object
        """
        # Determine risk category
        if probability < 0.3:
            risk_category = 'Low'
        elif probability < 0.7:
            risk_category = 'Medium'
        else:
            risk_category = 'High'
        
        # Extract top risk factors from SHAP
        top_risk_factors = []
        if shap_values and 'explanations' in shap_values and len(shap_values['explanations']) > 0:
            top_risk_factors = shap_values['explanations'][0].get('top_features', [])[:5]
        
        return PredictionResult(
            patient_id=patient_id,
            event_probability=float(probability),
            risk_category=risk_category,
            confidence_lower=float(confidence_interval[0]) if confidence_interval else None,
            confidence_upper=float(confidence_interval[1]) if confidence_interval else None,
            reliability_score=float(reliability),
            top_risk_factors=top_risk_factors,
            timestamp=datetime.now().isoformat()
        )
    
    def generate_fairness_dashboard(
        self,
        fairness_report: Dict
    ) -> FairnessDashboard:
        """
        Generate fairness dashboard from calibration report.
        
        Args:
            fairness_report: Report from FairnessCalibrator
        
        Returns:
            FairnessDashboard object
        """
        c_index_data = fairness_report.get('c_index', {})
        
        recommendations = []
        if not c_index_data.get('fairness_passed', False):
            recommendations.append("Apply probability calibration by demographic group")
            recommendations.append("Consider reweighting training samples")
            recommendations.append("Review feature selection for potential bias sources")
        
        if fairness_report.get('tpr_disparity', 0) > 0.10:
            recommendations.append("True positive rate disparity exceeds threshold - review model for underserved groups")
        
        return FairnessDashboard(
            stratified_c_index=c_index_data.get('stratified_c_index', 0),
            group_metrics={
                'c_index_by_group': c_index_data.get('group_c_indices', {}),
                'accuracy_by_group': fairness_report.get('accuracy_by_group', {}),
                'tpr_by_group': fairness_report.get('tpr_by_group', {})
            },
            disparity_metrics={
                'c_index_disparity': c_index_data.get('c_index_disparity', 0),
                'accuracy_disparity': fairness_report.get('accuracy_disparity', 0),
                'tpr_disparity': fairness_report.get('tpr_disparity', 0)
            },
            fairness_passed=c_index_data.get('fairness_passed', False),
            recommendations=recommendations
        )
    
    def generate_batch_predictions(
        self,
        patient_ids: List[str],
        probabilities: np.ndarray,
        confidence_intervals: Optional[np.ndarray] = None,
        reliability_scores: Optional[np.ndarray] = None,
        shap_explanations: Optional[Dict] = None
    ) -> List[PredictionResult]:
        """
        Generate predictions for multiple patients.
        
        Args:
            patient_ids: List of patient identifiers
            probabilities: Array of event probabilities
            confidence_intervals: Optional array of (lower, upper) bounds
            reliability_scores: Optional reliability scores
            shap_explanations: Optional SHAP explanations dict
        
        Returns:
            List of PredictionResult objects
        """
        results = []
        
        for i, (pid, prob) in enumerate(zip(patient_ids, probabilities)):
            ci = None
            if confidence_intervals is not None:
                ci = (confidence_intervals[0][i], confidence_intervals[1][i])
            
            reliability = 0.5
            if reliability_scores is not None:
                reliability = reliability_scores[i]
            
            shap_vals = None
            if shap_explanations and 'explanations' in shap_explanations:
                if i < len(shap_explanations['explanations']):
                    shap_vals = {'explanations': [shap_explanations['explanations'][i]]}
            
            result = self.generate_prediction_report(
                patient_id=str(pid),
                probability=prob,
                confidence_interval=ci,
                reliability=reliability,
                shap_values=shap_vals
            )
            results.append(result)
        
        return results
    
    def export_predictions_csv(
        self,
        predictions: List[PredictionResult],
        filepath: str
    ) -> None:
        """
        Export predictions to CSV file.
        
        Args:
            predictions: List of prediction results
            filepath: Output file path
        """
        data = []
        for pred in predictions:
            row = {
                'patient_id': pred.patient_id,
                'event_probability': pred.event_probability,
                'risk_category': pred.risk_category,
                'confidence_lower': pred.confidence_lower,
                'confidence_upper': pred.confidence_upper,
                'reliability_score': pred.reliability_score,
                'timestamp': pred.timestamp
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def export_predictions_json(
        self,
        predictions: List[PredictionResult],
        filepath: str
    ) -> None:
        """
        Export predictions to JSON file with full details.
        
        Args:
            predictions: List of prediction results
            filepath: Output file path
        """
        data = [asdict(pred) for pred in predictions]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def generate_quality_report(
        self,
        model_metrics: Dict,
        fairness_dashboard: FairnessDashboard,
        uncertainty_report: Dict
    ) -> Dict:
        """
        Generate comprehensive quality assurance report.
        
        Args:
            model_metrics: Model performance metrics
            fairness_dashboard: Fairness metrics
            uncertainty_report: Uncertainty analysis
        
        Returns:
            Dictionary with quality report
        """
        # Determine overall quality score
        quality_factors = []
        
        # Model accuracy factor
        accuracy = model_metrics.get('accuracy', 0)
        quality_factors.append(accuracy)
        
        # Fairness factor
        fairness_score = 1.0 if fairness_dashboard.fairness_passed else 0.5
        quality_factors.append(fairness_score)
        
        # Reliability factor
        avg_reliability = uncertainty_report.get('reliability', {}).get('average', 0.5)
        quality_factors.append(avg_reliability)
        
        overall_quality = np.mean(quality_factors)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_quality_score': float(overall_quality),
            'model_performance': {
                'accuracy': model_metrics.get('accuracy', 0),
                'auc_roc': model_metrics.get('auc_roc', 0),
                'precision': model_metrics.get('precision', 0),
                'recall': model_metrics.get('recall', 0)
            },
            'fairness_assessment': {
                'stratified_c_index': fairness_dashboard.stratified_c_index,
                'fairness_passed': fairness_dashboard.fairness_passed,
                'disparity_metrics': fairness_dashboard.disparity_metrics
            },
            'uncertainty_assessment': {
                'average_reliability': avg_reliability,
                'risk_distribution': uncertainty_report.get('risk_distribution', {})
            },
            'recommendations': fairness_dashboard.recommendations,
            'quality_grade': 'A' if overall_quality >= 0.8 else ('B' if overall_quality >= 0.6 else 'C')
        }
    
    def format_clinical_summary(
        self,
        prediction: PredictionResult
    ) -> str:
        """
        Format prediction as clinical-friendly summary.
        
        Args:
            prediction: Prediction result
        
        Returns:
            Formatted summary string
        """
        summary = f"""
=== HCT Survival Prediction Report ===
Patient ID: {prediction.patient_id}
Report Generated: {prediction.timestamp}

PREDICTION SUMMARY
------------------
Event Probability: {prediction.event_probability:.1%}
Risk Category: {prediction.risk_category}
Reliability Score: {prediction.reliability_score:.2f}
"""
        
        if prediction.confidence_lower is not None:
            summary += f"\n95% Confidence Interval: [{prediction.confidence_lower:.1%}, {prediction.confidence_upper:.1%}]"
        
        if prediction.top_risk_factors:
            summary += "\n\nTOP RISK FACTORS\n----------------"
            for i, factor in enumerate(prediction.top_risk_factors, 1):
                direction = factor.get('direction', 'unknown')
                summary += f"\n{i}. {factor['feature']}: {direction}"
        
        summary += "\n\n=== End of Report ==="
        
        return summary
