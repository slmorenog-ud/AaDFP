"""
Module M2: Equity Analysis
Performs stratified analysis, bias detection, and demographic reweighting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, accuracy_score


@dataclass
class BiasReport:
    """Report from bias detection analysis."""
    overall_event_rate: float
    group_event_rates: Dict[str, float]
    max_disparity: float
    groups_analyzed: List[str]
    bias_detected: bool
    recommendations: List[str]


@dataclass
class StratifiedAnalysis:
    """Results from stratified analysis."""
    group_counts: Dict[str, int]
    group_percentages: Dict[str, float]
    group_event_rates: Dict[str, float]
    missing_by_group: Dict[str, Dict[str, float]]


class EquityAnalyzer:
    """
    M2: Equity Analysis Module
    
    Performs comprehensive examination of data across demographic subgroups
    to identify disparities and potential biases.
    """
    
    def __init__(self, group_col: str = 'race_group'):
        self.group_col = group_col
        self.group_weights: Optional[Dict[str, float]] = None
        self.sample_weights: Optional[np.ndarray] = None
    
    def stratified_analysis(self, df: pd.DataFrame, target_col: str = 'efs') -> StratifiedAnalysis:
        """
        Perform stratified analysis across demographic groups.
        
        Args:
            df: Input DataFrame
            target_col: Target column for event rate calculation
        
        Returns:
            StratifiedAnalysis with group statistics
        """
        if self.group_col not in df.columns:
            raise ValueError(f"Group column '{self.group_col}' not found in data")
        
        # Group counts and percentages
        group_counts = df[self.group_col].value_counts().to_dict()
        total = len(df)
        group_percentages = {k: v / total * 100 for k, v in group_counts.items()}
        
        # Event rates by group
        group_event_rates = {}
        if target_col in df.columns:
            for group in df[self.group_col].unique():
                group_df = df[df[self.group_col] == group]
                if target_col == 'efs':
                    if df[target_col].dtype == 'object':
                        event_rate = (group_df[target_col] == 'Event').mean()
                    else:
                        event_rate = group_df[target_col].mean()
                else:
                    event_rate = group_df[target_col].mean()
                group_event_rates[str(group)] = float(event_rate)
        
        # Missing values by group
        missing_by_group = {}
        for group in df[self.group_col].unique():
            group_df = df[df[self.group_col] == group]
            missing_pct = (group_df.isnull().sum() / len(group_df) * 100).to_dict()
            missing_by_group[str(group)] = missing_pct
        
        return StratifiedAnalysis(
            group_counts={str(k): v for k, v in group_counts.items()},
            group_percentages={str(k): v for k, v in group_percentages.items()},
            group_event_rates=group_event_rates,
            missing_by_group=missing_by_group
        )
    
    def detect_bias(self, df: pd.DataFrame, target_col: str = 'efs') -> BiasReport:
        """
        Detect potential biases in the dataset.
        
        Args:
            df: Input DataFrame
            target_col: Target column for analysis
        
        Returns:
            BiasReport with bias detection results
        """
        groups_analyzed = df[self.group_col].unique().tolist()
        
        # Calculate event rates
        if target_col == 'efs':
            if df[target_col].dtype == 'object':
                overall_event_rate = (df[target_col] == 'Event').mean()
                group_event_rates = df.groupby(self.group_col).apply(
                    lambda x: (x[target_col] == 'Event').mean()
                ).to_dict()
            else:
                overall_event_rate = df[target_col].mean()
                group_event_rates = df.groupby(self.group_col)[target_col].mean().to_dict()
        else:
            overall_event_rate = df[target_col].mean()
            group_event_rates = df.groupby(self.group_col)[target_col].mean().to_dict()
        
        group_event_rates = {str(k): float(v) for k, v in group_event_rates.items()}
        
        # Calculate disparity
        rates = list(group_event_rates.values())
        max_disparity = max(rates) - min(rates) if rates else 0
        
        # Threshold from Workshop 3: disparity > 0.10 indicates bias
        bias_detected = max_disparity > 0.10
        
        # Generate recommendations
        recommendations = []
        if bias_detected:
            recommendations.append("Apply reweighting to balance demographic representation")
            recommendations.append("Consider fairness-aware training techniques")
        
        # Check for underrepresented groups (< 5% of total)
        total = len(df)
        for group, count in df[self.group_col].value_counts().items():
            if count / total < 0.05:
                recommendations.append(f"Group '{group}' is underrepresented ({count/total*100:.1f}%)")
        
        return BiasReport(
            overall_event_rate=float(overall_event_rate),
            group_event_rates=group_event_rates,
            max_disparity=float(max_disparity),
            groups_analyzed=[str(g) for g in groups_analyzed],
            bias_detected=bias_detected,
            recommendations=recommendations
        )
    
    def calculate_reweights(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate sample weights to balance demographic representation.
        
        Uses inverse frequency weighting to give more weight to
        underrepresented groups.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Array of sample weights
        """
        if self.group_col not in df.columns:
            return np.ones(len(df))
        
        # Calculate group frequencies
        group_counts = df[self.group_col].value_counts()
        total = len(df)
        n_groups = len(group_counts)
        
        # Calculate weights (inverse frequency, normalized)
        target_freq = 1.0 / n_groups
        self.group_weights = {}
        
        for group, count in group_counts.items():
            current_freq = count / total
            weight = target_freq / current_freq
            self.group_weights[group] = weight
        
        # Apply weights to samples
        self.sample_weights = df[self.group_col].map(self.group_weights).values
        
        # Normalize weights to sum to n_samples
        self.sample_weights = self.sample_weights * len(df) / self.sample_weights.sum()
        
        return self.sample_weights
    
    def demographic_parity_check(
        self, 
        y_pred: np.ndarray, 
        groups: np.ndarray,
        threshold: float = 0.10
    ) -> Tuple[float, bool, Dict[str, float]]:
        """
        Check demographic parity of predictions.
        
        Demographic parity requires that the positive prediction rate
        is similar across all demographic groups.
        
        Args:
            y_pred: Binary predictions
            groups: Group labels for each sample
            threshold: Maximum allowed disparity
        
        Returns:
            Tuple of (max_disparity, parity_satisfied, group_rates)
        """
        group_rates = {}
        unique_groups = np.unique(groups)
        
        for group in unique_groups:
            mask = groups == group
            rate = y_pred[mask].mean()
            group_rates[str(group)] = float(rate)
        
        rates = list(group_rates.values())
        max_disparity = max(rates) - min(rates) if rates else 0
        parity_satisfied = max_disparity <= threshold
        
        return max_disparity, parity_satisfied, group_rates
    
    def equalized_odds_check(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
        threshold: float = 0.10
    ) -> Dict[str, any]:
        """
        Check equalized odds across demographic groups.
        
        Equalized odds requires similar TPR and FPR across groups.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            groups: Group labels
            threshold: Maximum allowed disparity
        
        Returns:
            Dictionary with TPR and FPR by group and overall disparity
        """
        unique_groups = np.unique(groups)
        tpr_by_group = {}
        fpr_by_group = {}
        
        for group in unique_groups:
            mask = groups == group
            y_t = y_true[mask]
            y_p = y_pred[mask]
            
            # True Positive Rate
            pos_mask = y_t == 1
            if pos_mask.sum() > 0:
                tpr = y_p[pos_mask].mean()
            else:
                tpr = 0.0
            
            # False Positive Rate
            neg_mask = y_t == 0
            if neg_mask.sum() > 0:
                fpr = y_p[neg_mask].mean()
            else:
                fpr = 0.0
            
            tpr_by_group[str(group)] = float(tpr)
            fpr_by_group[str(group)] = float(fpr)
        
        tpr_values = list(tpr_by_group.values())
        fpr_values = list(fpr_by_group.values())
        
        tpr_disparity = max(tpr_values) - min(tpr_values) if tpr_values else 0
        fpr_disparity = max(fpr_values) - min(fpr_values) if fpr_values else 0
        
        return {
            'tpr_by_group': tpr_by_group,
            'fpr_by_group': fpr_by_group,
            'tpr_disparity': tpr_disparity,
            'fpr_disparity': fpr_disparity,
            'equalized_odds_satisfied': tpr_disparity <= threshold and fpr_disparity <= threshold
        }
    
    def generate_equity_report(
        self,
        df: pd.DataFrame,
        y_pred: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Generate comprehensive equity report.
        
        Args:
            df: Original DataFrame with group information
            y_pred: Optional predictions for fairness checks
            y_true: Optional true labels for fairness checks
        
        Returns:
            Dictionary with complete equity analysis
        """
        report = {
            'stratified_analysis': None,
            'bias_report': None,
            'demographic_parity': None,
            'equalized_odds': None,
            'recommendations': []
        }
        
        # Stratified analysis
        if 'efs' in df.columns:
            analysis = self.stratified_analysis(df, 'efs')
            report['stratified_analysis'] = {
                'group_counts': analysis.group_counts,
                'group_percentages': analysis.group_percentages,
                'group_event_rates': analysis.group_event_rates
            }
        
        # Bias detection
        if 'efs' in df.columns:
            bias = self.detect_bias(df, 'efs')
            report['bias_report'] = {
                'overall_event_rate': bias.overall_event_rate,
                'group_event_rates': bias.group_event_rates,
                'max_disparity': bias.max_disparity,
                'bias_detected': bias.bias_detected
            }
            report['recommendations'].extend(bias.recommendations)
        
        # Prediction fairness checks
        if y_pred is not None and self.group_col in df.columns:
            groups = df[self.group_col].values
            
            disparity, satisfied, rates = self.demographic_parity_check(y_pred, groups)
            report['demographic_parity'] = {
                'max_disparity': disparity,
                'parity_satisfied': satisfied,
                'group_rates': rates
            }
            
            if y_true is not None:
                report['equalized_odds'] = self.equalized_odds_check(y_true, y_pred, groups)
        
        return report
