"""
Module M6: Uncertainty Quantification
Provides confidence intervals and prediction reliability assessment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import GradientBoostingClassifier
from dataclasses import dataclass


@dataclass
class ConfidenceInterval:
    """Confidence interval for predictions."""
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    point_estimate: np.ndarray
    confidence_level: float


@dataclass
class PredictionReliability:
    """Reliability assessment for predictions."""
    reliability_scores: np.ndarray
    high_confidence_mask: np.ndarray
    low_confidence_mask: np.ndarray
    average_reliability: float


class UncertaintyQuantifier:
    """
    M6: Uncertainty Quantification Module
    
    Manages chaos and sensitivity through confidence intervals
    and reliability assessment.
    """
    
    def __init__(self):
        self.bootstrap_models: List = []
        self.n_bootstrap: int = 0
    
    def bootstrap_confidence_intervals(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_predict: pd.DataFrame,
        n_bootstrap: int = 50,
        confidence: float = 0.95,
        sample_weights: Optional[np.ndarray] = None
    ) -> ConfidenceInterval:
        """
        Generate confidence intervals using bootstrap sampling.
        
        Args:
            model: Sklearn model class (not instance)
            X_train: Training features
            y_train: Training labels
            X_predict: Features to predict
            n_bootstrap: Number of bootstrap iterations
            confidence: Confidence level (0-1)
            sample_weights: Optional sample weights
        
        Returns:
            ConfidenceInterval with bounds
        """
        n_samples = len(X_train)
        bootstrap_predictions = []
        
        self.bootstrap_models = []
        self.n_bootstrap = n_bootstrap
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_train.iloc[indices]
            y_boot = y_train.iloc[indices]
            
            # Get weights for bootstrap sample
            if sample_weights is not None:
                weights_boot = sample_weights[indices]
            else:
                weights_boot = None
            
            # Train model
            boot_model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=4,
                random_state=i
            )
            
            if weights_boot is not None:
                boot_model.fit(X_boot, y_boot, sample_weight=weights_boot)
            else:
                boot_model.fit(X_boot, y_boot)
            
            self.bootstrap_models.append(boot_model)
            
            # Predict probabilities
            proba = boot_model.predict_proba(X_predict)[:, 1]
            bootstrap_predictions.append(proba)
        
        # Stack predictions
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # Calculate confidence intervals
        alpha = 1 - confidence
        lower_bound = np.percentile(bootstrap_predictions, alpha/2 * 100, axis=0)
        upper_bound = np.percentile(bootstrap_predictions, (1 - alpha/2) * 100, axis=0)
        point_estimate = np.mean(bootstrap_predictions, axis=0)
        
        return ConfidenceInterval(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            point_estimate=point_estimate,
            confidence_level=confidence
        )
    
    def prediction_variance(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate prediction variance from bootstrap ensemble.
        
        Args:
            X: Features to predict
        
        Returns:
            Array of prediction variances
        """
        if not self.bootstrap_models:
            raise ValueError("Bootstrap models not fitted. Call bootstrap_confidence_intervals first.")
        
        predictions = []
        for model in self.bootstrap_models:
            proba = model.predict_proba(X)[:, 1]
            predictions.append(proba)
        
        predictions = np.array(predictions)
        variance = np.var(predictions, axis=0)
        
        return variance
    
    def identify_low_confidence(
        self,
        y_proba: np.ndarray,
        variance: Optional[np.ndarray] = None,
        threshold_proba: float = 0.3,
        threshold_variance: float = 0.05
    ) -> np.ndarray:
        """
        Identify predictions with low confidence.
        
        Low confidence is defined as:
        - Probability close to 0.5 (uncertain)
        - High prediction variance
        
        Args:
            y_proba: Probability predictions
            variance: Optional prediction variances
            threshold_proba: Distance from 0.5 to be considered uncertain
            threshold_variance: Variance threshold
        
        Returns:
            Boolean mask of low-confidence predictions
        """
        # Predictions close to 0.5 are uncertain
        uncertainty_distance = np.abs(y_proba - 0.5)
        proba_uncertain = uncertainty_distance < threshold_proba
        
        if variance is not None:
            high_variance = variance > threshold_variance
            low_confidence = proba_uncertain | high_variance
        else:
            low_confidence = proba_uncertain
        
        return low_confidence
    
    def risk_stratification(
        self,
        y_proba: np.ndarray,
        thresholds: Tuple[float, float] = (0.3, 0.7)
    ) -> np.ndarray:
        """
        Stratify patients into risk categories.
        
        Args:
            y_proba: Probability predictions (P(event))
            thresholds: Low-high thresholds for risk categories
        
        Returns:
            Array of risk categories (0=low, 1=medium, 2=high)
        """
        low_thresh, high_thresh = thresholds
        
        risk_category = np.ones(len(y_proba), dtype=int)  # Default: medium
        risk_category[y_proba < low_thresh] = 0  # Low risk
        risk_category[y_proba >= high_thresh] = 2  # High risk
        
        return risk_category
    
    def risk_stratification_with_bounds(
        self,
        confidence_interval: ConfidenceInterval,
        thresholds: Tuple[float, float] = (0.3, 0.7)
    ) -> Dict[str, np.ndarray]:
        """
        Stratify risk with uncertainty bounds.
        
        Args:
            confidence_interval: Confidence intervals for predictions
            thresholds: Risk thresholds
        
        Returns:
            Dictionary with point estimate and bound-based stratification
        """
        point_risk = self.risk_stratification(
            confidence_interval.point_estimate, thresholds
        )
        lower_risk = self.risk_stratification(
            confidence_interval.lower_bound, thresholds
        )
        upper_risk = self.risk_stratification(
            confidence_interval.upper_bound, thresholds
        )
        
        # Identify unstable stratifications (where bounds give different categories)
        unstable = (lower_risk != upper_risk)
        
        return {
            'point_estimate_risk': point_risk,
            'lower_bound_risk': lower_risk,
            'upper_bound_risk': upper_risk,
            'unstable_stratification': unstable,
            'stability_rate': 1 - unstable.mean()
        }
    
    def assess_reliability(
        self,
        y_proba: np.ndarray,
        variance: Optional[np.ndarray] = None
    ) -> PredictionReliability:
        """
        Comprehensive reliability assessment of predictions.
        
        Reliability is based on:
        - Distance from 0.5 (more certain = more reliable)
        - Prediction variance (if available)
        
        Args:
            y_proba: Probability predictions
            variance: Optional prediction variances
        
        Returns:
            PredictionReliability assessment
        """
        # Base reliability from probability certainty
        certainty = np.abs(y_proba - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1
        
        if variance is not None:
            # Combine with variance (low variance = more reliable)
            max_var = max(variance.max(), 0.01)  # Prevent division by zero
            variance_reliability = 1 - (variance / max_var)
            reliability_scores = (certainty + variance_reliability) / 2
        else:
            reliability_scores = certainty
        
        # Define thresholds
        high_confidence_mask = reliability_scores >= 0.6
        low_confidence_mask = reliability_scores < 0.3
        
        return PredictionReliability(
            reliability_scores=reliability_scores,
            high_confidence_mask=high_confidence_mask,
            low_confidence_mask=low_confidence_mask,
            average_reliability=float(reliability_scores.mean())
        )
    
    def sensitivity_analysis(
        self,
        model,
        X: pd.DataFrame,
        feature_to_perturb: str,
        perturbation_range: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Analyze sensitivity of predictions to feature perturbations.
        
        Args:
            model: Trained model
            X: Original features
            feature_to_perturb: Feature name to perturb
            perturbation_range: Values to perturb by
        
        Returns:
            Dictionary with sensitivity analysis results
        """
        baseline_proba = model.predict_proba(X)[:, 1]
        sensitivities = []
        
        for perturbation in perturbation_range:
            X_perturbed = X.copy()
            X_perturbed[feature_to_perturb] = X[feature_to_perturb] + perturbation
            
            perturbed_proba = model.predict_proba(X_perturbed)[:, 1]
            delta = perturbed_proba - baseline_proba
            sensitivities.append(delta)
        
        sensitivities = np.array(sensitivities)
        
        return {
            'perturbation_values': perturbation_range,
            'mean_sensitivity': sensitivities.mean(axis=1),
            'std_sensitivity': sensitivities.std(axis=1),
            'max_sensitivity': np.abs(sensitivities).max(axis=1),
            'feature': feature_to_perturb
        }
    
    def chaos_analysis(
        self,
        model,
        X: pd.DataFrame,
        noise_levels: List[float] = [0.01, 0.05, 0.10, 0.15]
    ) -> Dict:
        """
        Analyze model stability under random noise (chaos/butterfly effect).
        
        Args:
            model: Trained model
            X: Features
            noise_levels: List of noise standard deviations to test
        
        Returns:
            Dictionary with chaos analysis results
        """
        baseline_proba = model.predict_proba(X)[:, 1]
        baseline_pred = (baseline_proba >= 0.5).astype(int)
        
        results = {
            'noise_levels': noise_levels,
            'accuracy_impact': [],
            'prediction_changes': [],
            'probability_shifts': []
        }
        
        for noise_level in noise_levels:
            # Add Gaussian noise
            noise = np.random.normal(0, noise_level, X.shape)
            X_noisy = X + noise
            
            # Get predictions
            noisy_proba = model.predict_proba(X_noisy)[:, 1]
            noisy_pred = (noisy_proba >= 0.5).astype(int)
            
            # Calculate impacts
            prediction_change_rate = (baseline_pred != noisy_pred).mean()
            proba_shift = np.abs(baseline_proba - noisy_proba).mean()
            
            results['prediction_changes'].append(float(prediction_change_rate))
            results['probability_shifts'].append(float(proba_shift))
        
        # Stability assessment
        max_change = max(results['prediction_changes'])
        results['stability_rating'] = 'High' if max_change < 0.05 else ('Medium' if max_change < 0.15 else 'Low')
        
        return results
    
    def generate_uncertainty_report(
        self,
        y_proba: np.ndarray,
        confidence_interval: Optional[ConfidenceInterval] = None,
        variance: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Generate comprehensive uncertainty report.
        
        Args:
            y_proba: Probability predictions
            confidence_interval: Optional confidence intervals
            variance: Optional prediction variances
        
        Returns:
            Dictionary with uncertainty analysis
        """
        report = {
            'n_predictions': len(y_proba),
            'mean_probability': float(y_proba.mean()),
            'std_probability': float(y_proba.std()),
        }
        
        # Risk stratification
        risk_cats = self.risk_stratification(y_proba)
        report['risk_distribution'] = {
            'low': int((risk_cats == 0).sum()),
            'medium': int((risk_cats == 1).sum()),
            'high': int((risk_cats == 2).sum())
        }
        
        # Reliability assessment
        reliability = self.assess_reliability(y_proba, variance)
        report['reliability'] = {
            'average': reliability.average_reliability,
            'high_confidence_count': int(reliability.high_confidence_mask.sum()),
            'low_confidence_count': int(reliability.low_confidence_mask.sum())
        }
        
        # Confidence intervals
        if confidence_interval is not None:
            ci_width = confidence_interval.upper_bound - confidence_interval.lower_bound
            report['confidence_intervals'] = {
                'mean_width': float(ci_width.mean()),
                'max_width': float(ci_width.max()),
                'confidence_level': confidence_interval.confidence_level
            }
        
        return report
