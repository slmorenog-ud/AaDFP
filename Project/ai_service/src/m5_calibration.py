"""
Module M5: Fairness Calibration
Adjusts model predictions for equitable performance across demographic groups.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Results from calibration process."""
    original_probas: np.ndarray
    calibrated_probas: np.ndarray
    improvement: float
    group_improvements: Dict[str, float]


@dataclass
class ThresholdResult:
    """Results from threshold optimization."""
    default_threshold: float
    optimized_thresholds: Dict[str, float]
    disparity_reduction: float


class FairnessCalibrator:
    """
    M5: Fairness Calibration Module
    
    Adjusts model predictions to maintain comparable accuracy
    across different patient populations.
    """
    
    def __init__(self):
        self.calibrators: Dict[str, IsotonicRegression] = {}
        self.group_thresholds: Dict[str, float] = {}
        self.overall_threshold: float = 0.5
    
    def calibrate_probabilities(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        groups: np.ndarray,
        method: str = 'isotonic'
    ) -> CalibrationResult:
        """
        Calibrate probabilities for each demographic group.
        
        Args:
            y_true: True labels
            y_proba: Raw probability predictions
            groups: Demographic group labels
            method: 'isotonic' or 'platt'
        
        Returns:
            CalibrationResult with calibrated probabilities
        """
        calibrated = y_proba.copy()
        group_improvements = {}
        
        unique_groups = np.unique(groups)
        
        for group in unique_groups:
            mask = groups == group
            y_t = y_true[mask]
            y_p = y_proba[mask]
            
            if len(y_t) < 30:
                continue
            
            # Fit calibrator for this group
            if method == 'isotonic':
                calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
                calibrator.fit(y_p, y_t)
                y_calibrated = calibrator.predict(y_p)
            else:
                # Platt scaling (sigmoid)
                from sklearn.linear_model import LogisticRegression
                calibrator = LogisticRegression()
                calibrator.fit(y_p.reshape(-1, 1), y_t)
                y_calibrated = calibrator.predict_proba(y_p.reshape(-1, 1))[:, 1]
            
            self.calibrators[str(group)] = calibrator
            calibrated[mask] = y_calibrated
            
            # Calculate improvement
            original_auc = roc_auc_score(y_t, y_p)
            calibrated_auc = roc_auc_score(y_t, y_calibrated)
            group_improvements[str(group)] = calibrated_auc - original_auc
        
        overall_improvement = np.mean(list(group_improvements.values())) if group_improvements else 0
        
        return CalibrationResult(
            original_probas=y_proba,
            calibrated_probas=calibrated,
            improvement=float(overall_improvement),
            group_improvements=group_improvements
        )
    
    def transform_probas(
        self,
        y_proba: np.ndarray,
        groups: np.ndarray
    ) -> np.ndarray:
        """
        Apply fitted calibrators to new data.
        
        Args:
            y_proba: Raw probability predictions
            groups: Demographic group labels
        
        Returns:
            Calibrated probabilities
        """
        calibrated = y_proba.copy()
        
        for group, calibrator in self.calibrators.items():
            mask = groups == group
            if mask.any():
                calibrated[mask] = calibrator.predict(y_proba[mask])
        
        return calibrated
    
    def optimize_thresholds(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        groups: np.ndarray,
        metric: str = 'f1',
        fairness_constraint: float = 0.10
    ) -> ThresholdResult:
        """
        Optimize classification thresholds for fairness.
        
        Args:
            y_true: True labels
            y_proba: Probability predictions
            groups: Demographic group labels
            metric: Optimization metric ('f1', 'accuracy', 'balanced')
            fairness_constraint: Maximum allowed disparity
        
        Returns:
            ThresholdResult with optimized thresholds
        """
        unique_groups = np.unique(groups)
        optimized_thresholds = {}
        
        # Find optimal threshold for each group
        for group in unique_groups:
            mask = groups == group
            y_t = y_true[mask]
            y_p = y_proba[mask]
            
            if len(y_t) < 30:
                optimized_thresholds[str(group)] = 0.5
                continue
            
            best_threshold = 0.5
            best_score = 0
            
            for threshold in np.arange(0.3, 0.7, 0.02):
                y_pred = (y_p >= threshold).astype(int)
                
                if metric == 'f1':
                    from sklearn.metrics import f1_score
                    score = f1_score(y_t, y_pred, zero_division=0)
                elif metric == 'accuracy':
                    from sklearn.metrics import accuracy_score
                    score = accuracy_score(y_t, y_pred)
                elif metric == 'balanced':
                    from sklearn.metrics import balanced_accuracy_score
                    score = balanced_accuracy_score(y_t, y_pred)
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            optimized_thresholds[str(group)] = float(best_threshold)
        
        # Calculate disparity reduction
        thresholds = list(optimized_thresholds.values())
        threshold_disparity = max(thresholds) - min(thresholds) if thresholds else 0
        
        # If disparity too high, constrain thresholds
        if threshold_disparity > fairness_constraint:
            mean_threshold = np.mean(thresholds)
            for group in optimized_thresholds:
                # Move toward mean
                current = optimized_thresholds[group]
                optimized_thresholds[group] = current * 0.7 + mean_threshold * 0.3
        
        self.group_thresholds = optimized_thresholds
        
        return ThresholdResult(
            default_threshold=0.5,
            optimized_thresholds=optimized_thresholds,
            disparity_reduction=0.5 - threshold_disparity
        )
    
    def calculate_disparity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
        metric: str = 'accuracy'
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate disparity in model performance across groups.
        
        Args:
            y_true: True labels
            y_pred: Binary predictions
            groups: Demographic group labels
            metric: 'accuracy', 'tpr', 'fpr', 'auc'
        
        Returns:
            Tuple of (max_disparity, group_metrics)
        """
        unique_groups = np.unique(groups)
        group_metrics = {}
        
        for group in unique_groups:
            mask = groups == group
            y_t = y_true[mask]
            y_p = y_pred[mask]
            
            if len(y_t) < 10:
                continue
            
            if metric == 'accuracy':
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_t, y_p)
            elif metric == 'tpr':
                pos_mask = y_t == 1
                score = y_p[pos_mask].mean() if pos_mask.any() else 0
            elif metric == 'fpr':
                neg_mask = y_t == 0
                score = y_p[neg_mask].mean() if neg_mask.any() else 0
            elif metric == 'auc':
                try:
                    score = roc_auc_score(y_t, y_p)
                except:
                    score = 0.5
            
            group_metrics[str(group)] = float(score)
        
        scores = list(group_metrics.values())
        max_disparity = max(scores) - min(scores) if scores else 0
        
        return float(max_disparity), group_metrics
    
    def stratified_c_index(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        groups: np.ndarray
    ) -> Dict[str, any]:
        """
        Calculate stratified C-index with fairness assessment.
        
        This is the primary competition metric.
        
        Args:
            y_true: True labels
            y_proba: Probability predictions
            groups: Demographic group labels
        
        Returns:
            Dictionary with C-index metrics and fairness assessment
        """
        unique_groups = np.unique(groups)
        group_c_indices = {}
        group_sizes = {}
        
        for group in unique_groups:
            mask = groups == group
            y_t = y_true[mask]
            y_p = y_proba[mask]
            
            group_sizes[str(group)] = int(mask.sum())
            
            if len(y_t) < 20:
                continue
            
            try:
                c_index = roc_auc_score(y_t, y_p)
                group_c_indices[str(group)] = float(c_index)
            except:
                pass
        
        # Calculate weighted average (by group size)
        total_samples = sum(group_sizes.values())
        weighted_sum = 0
        for group, c_idx in group_c_indices.items():
            weight = group_sizes[group] / total_samples
            weighted_sum += c_idx * weight
        
        # Calculate disparity
        c_values = list(group_c_indices.values())
        disparity = max(c_values) - min(c_values) if c_values else 0
        
        # Fairness check: disparity should be < 0.10
        fairness_passed = disparity < 0.10
        
        return {
            'stratified_c_index': float(weighted_sum),
            'group_c_indices': group_c_indices,
            'group_sizes': group_sizes,
            'c_index_disparity': float(disparity),
            'fairness_passed': fairness_passed,
            'recommendation': 'OK' if fairness_passed else 'Apply additional calibration or reweighting'
        }
    
    def apply_fairness_adjustment(
        self,
        y_proba: np.ndarray,
        groups: np.ndarray,
        y_true: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply all fairness adjustments to predictions.
        
        Args:
            y_proba: Raw probability predictions
            groups: Demographic group labels
            y_true: Optional true labels for calibration
        
        Returns:
            Fairness-adjusted probabilities
        """
        adjusted = y_proba.copy()
        
        # Apply calibration if fitted
        if self.calibrators:
            adjusted = self.transform_probas(adjusted, groups)
        
        return adjusted
    
    def generate_fairness_report(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        groups: np.ndarray
    ) -> Dict:
        """
        Generate comprehensive fairness report.
        
        Args:
            y_true: True labels
            y_proba: Probability predictions
            groups: Demographic group labels
        
        Returns:
            Dictionary with complete fairness analysis
        """
        # Stratified C-index
        c_index_results = self.stratified_c_index(y_true, y_proba, groups)
        
        # Disparity metrics
        y_pred = (y_proba >= 0.5).astype(int)
        acc_disparity, acc_by_group = self.calculate_disparity(y_true, y_pred, groups, 'accuracy')
        tpr_disparity, tpr_by_group = self.calculate_disparity(y_true, y_pred, groups, 'tpr')
        
        return {
            'c_index': c_index_results,
            'accuracy_disparity': acc_disparity,
            'accuracy_by_group': acc_by_group,
            'tpr_disparity': tpr_disparity,
            'tpr_by_group': tpr_by_group,
            'overall_fairness_score': 1 - (acc_disparity + tpr_disparity) / 2,
            'meets_competition_requirements': c_index_results['fairness_passed']
        }
