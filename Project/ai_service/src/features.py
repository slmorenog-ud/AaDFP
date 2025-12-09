"""
Module M3: Feature Selection and Importance
Handles clinical domain integration, ML-based rankings, and equitable feature availability.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr
from dataclasses import dataclass


@dataclass
class FeatureImportance:
    """Feature importance results."""
    feature_name: str
    importance_score: float
    importance_type: str  # 'clinical', 'statistical', 'ml'
    availability_rate: float


class FeatureSelector:
    """
    M3: Feature Selection and Importance Module
    
    Determines which variables contribute most to survival predictions
    using clinical domain knowledge, statistical tests, and ML-based rankings.
    """
    
    # Clinical domain features based on HCT literature
    CLINICAL_PRIORITY_FEATURES = [
        # High priority - identified in Workshop 1 as sensitive parameters
        'age_at_hct',
        'dri_score',
        'conditioning_intensity',
        'comorbidity_score',
        'karnofsky_score',
        
        # HLA matching - critical for transplant outcomes
        'hla_high_res_8',
        'hla_high_res_10',
        'hla_match_quality',
        
        # Transplant-specific
        'graft_type',
        'donor_related',
        'donor_age',
        'sex_match',
        
        # Disease characteristics
        'prim_disease_hct',
        'cyto_score',
        'mrd_hct',
        
        # Year of transplant (captures medical advances)
        'year_hct',
    ]
    
    # Features that might introduce bias if not handled carefully
    SENSITIVE_FEATURES = [
        'race_group',
        'ethnicity',
    ]
    
    def __init__(self):
        self.selected_features: List[str] = []
        self.feature_importances: Dict[str, float] = {}
        self.availability_by_group: Dict[str, Dict[str, float]] = {}
    
    def get_clinical_domain_features(self, available_columns: List[str]) -> List[str]:
        """
        Get clinical domain priority features that are available in the dataset.
        
        Args:
            available_columns: List of columns in the dataset
        
        Returns:
            List of clinical priority features present in data
        """
        return [f for f in self.CLINICAL_PRIORITY_FEATURES if f in available_columns]
    
    def statistical_importance(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        method: str = 'mutual_info'
    ) -> Dict[str, float]:
        """
        Calculate feature importance using statistical methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: 'mutual_info' or 'correlation'
        
        Returns:
            Dictionary of feature importances
        """
        importances = {}
        
        if method == 'mutual_info':
            # Mutual information for classification
            mi_scores = mutual_info_classif(X, y, random_state=42)
            for i, col in enumerate(X.columns):
                importances[col] = float(mi_scores[i])
        
        elif method == 'correlation':
            # Spearman correlation for robustness
            for col in X.columns:
                corr, _ = spearmanr(X[col], y)
                importances[col] = abs(float(corr)) if not np.isnan(corr) else 0.0
        
        return importances
    
    def ml_importance(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_estimators: int = 100
    ) -> Dict[str, float]:
        """
        Calculate feature importance using Random Forest.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_estimators: Number of trees
        
        Returns:
            Dictionary of feature importances
        """
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        importances = {}
        for i, col in enumerate(X.columns):
            importances[col] = float(rf.feature_importances_[i])
        
        return importances
    
    def check_availability(
        self, 
        df: pd.DataFrame, 
        features: List[str], 
        group_col: str = 'race_group'
    ) -> Dict[str, Dict[str, float]]:
        """
        Check feature availability across demographic groups.
        
        Features with >20% availability disparity across groups should be flagged.
        
        Args:
            df: Original DataFrame with missing values
            features: List of features to check
            group_col: Demographic group column
        
        Returns:
            Dictionary with availability rates by group for each feature
        """
        availability = {}
        
        if group_col not in df.columns:
            # No group column, return overall availability
            for feat in features:
                if feat in df.columns:
                    availability[feat] = {'overall': 1 - df[feat].isnull().mean()}
            return availability
        
        groups = df[group_col].unique()
        
        for feat in features:
            if feat not in df.columns:
                continue
                
            availability[feat] = {}
            for group in groups:
                group_df = df[df[group_col] == group]
                avail_rate = 1 - group_df[feat].isnull().mean()
                availability[feat][str(group)] = float(avail_rate)
            
            # Check disparity
            rates = list(availability[feat].values())
            disparity = max(rates) - min(rates) if rates else 0
            availability[feat]['disparity'] = disparity
            availability[feat]['flag'] = disparity > 0.20
        
        self.availability_by_group = availability
        return availability
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        df_original: Optional[pd.DataFrame] = None,
        n_features: int = 20,
        method: str = 'combined',
        group_col: str = 'race_group'
    ) -> List[str]:
        """
        Select top features using multiple strategies.
        
        Args:
            X: Preprocessed feature matrix
            y: Target variable
            df_original: Original DataFrame for availability check
            n_features: Number of features to select
            method: 'clinical', 'statistical', 'ml', or 'combined'
            group_col: Demographic group column
        
        Returns:
            List of selected feature names
        """
        available_features = X.columns.tolist()
        
        # Clinical domain features
        clinical_features = self.get_clinical_domain_features(available_features)
        
        if method == 'clinical':
            self.selected_features = clinical_features[:n_features]
            return self.selected_features
        
        # Statistical importance
        stat_importance = self.statistical_importance(X, y, 'mutual_info')
        
        if method == 'statistical':
            sorted_features = sorted(stat_importance.items(), key=lambda x: x[1], reverse=True)
            self.selected_features = [f[0] for f in sorted_features[:n_features]]
            return self.selected_features
        
        # ML importance
        ml_importance = self.ml_importance(X, y)
        
        if method == 'ml':
            sorted_features = sorted(ml_importance.items(), key=lambda x: x[1], reverse=True)
            self.selected_features = [f[0] for f in sorted_features[:n_features]]
            return self.selected_features
        
        # Combined method (consensus approach)
        # Rank features by each method
        stat_ranks = {f: i for i, (f, _) in enumerate(
            sorted(stat_importance.items(), key=lambda x: x[1], reverse=True)
        )}
        ml_ranks = {f: i for i, (f, _) in enumerate(
            sorted(ml_importance.items(), key=lambda x: x[1], reverse=True)
        )}
        
        # Clinical features get bonus (lower rank = better)
        clinical_bonus = {f: -10 for f in clinical_features}
        
        # Combine ranks
        combined_ranks = {}
        for feat in available_features:
            stat_rank = stat_ranks.get(feat, len(available_features))
            ml_rank = ml_ranks.get(feat, len(available_features))
            bonus = clinical_bonus.get(feat, 0)
            combined_ranks[feat] = (stat_rank + ml_rank) / 2 + bonus
        
        # Sort by combined rank
        sorted_features = sorted(combined_ranks.items(), key=lambda x: x[1])
        
        # Check availability if original data provided
        if df_original is not None:
            self.check_availability(df_original, available_features, group_col)
            
            # Filter out features with high availability disparity
            filtered = []
            for feat, rank in sorted_features:
                if feat in self.availability_by_group:
                    if not self.availability_by_group[feat].get('flag', False):
                        filtered.append(feat)
                else:
                    filtered.append(feat)
                
                if len(filtered) >= n_features:
                    break
            
            self.selected_features = filtered
        else:
            self.selected_features = [f[0] for f in sorted_features[:n_features]]
        
        # Store importances
        self.feature_importances = {
            'statistical': stat_importance,
            'ml': ml_importance,
            'combined_ranks': combined_ranks
        }
        
        return self.selected_features
    
    def get_feature_report(self) -> Dict:
        """
        Generate feature selection report.
        
        Returns:
            Dictionary with feature selection details
        """
        report = {
            'selected_features': self.selected_features,
            'n_features': len(self.selected_features),
            'clinical_features_included': [
                f for f in self.selected_features 
                if f in self.CLINICAL_PRIORITY_FEATURES
            ],
            'sensitive_features_included': [
                f for f in self.selected_features 
                if f in self.SENSITIVE_FEATURES
            ],
            'availability_concerns': []
        }
        
        # Add availability concerns
        for feat, avail in self.availability_by_group.items():
            if avail.get('flag', False):
                report['availability_concerns'].append({
                    'feature': feat,
                    'disparity': avail.get('disparity', 0),
                    'rates_by_group': {k: v for k, v in avail.items() 
                                       if k not in ['disparity', 'flag']}
                })
        
        return report
    
    def recursive_feature_elimination(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 20
    ) -> List[str]:
        """
        Alternative feature selection using RFE.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
        
        Returns:
            List of selected features
        """
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        selector = RFE(estimator, n_features_to_select=n_features, step=5)
        selector.fit(X, y)
        
        selected = X.columns[selector.support_].tolist()
        return selected
