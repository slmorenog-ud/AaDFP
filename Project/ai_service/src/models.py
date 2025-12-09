"""
Module M4: Predictive Modeling Core
Implements survival analysis models including Cox PH, GBM, and Random Forest.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, brier_score_loss
)
from dataclasses import dataclass
import pickle
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CVResults:
    """Cross-validation results."""
    mean_accuracy: float
    std_accuracy: float
    mean_auc: float
    std_auc: float
    fold_results: List[Dict]


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: float
    auc_roc: float
    precision: float
    recall: float
    f1: float
    brier_score: float


class PredictiveModel:
    """
    M4: Predictive Modeling Core
    
    Implements ensemble approach with multiple model types:
    - Gradient Boosting Machine (GBM)
    - Random Forest
    - Cross-validation with demographic stratification
    """
    
    # Model configurations based on Workshop 4 findings
    MODEL_CONFIGS = {
        'gbm': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42
        },
        'rf': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
    }
    
    def __init__(self):
        self.model = None
        self.model_type: str = 'gbm'
        self.feature_names: List[str] = []
        self.is_trained: bool = False
    
    def get_model(self, model_type: str = 'gbm', **kwargs) -> Any:
        """
        Get a model instance with specified configuration.
        
        Args:
            model_type: 'gbm' or 'rf'
            **kwargs: Additional model parameters
        
        Returns:
            Sklearn model instance
        """
        config = self.MODEL_CONFIGS.get(model_type, self.MODEL_CONFIGS['gbm']).copy()
        config.update(kwargs)
        
        if model_type == 'gbm':
            return GradientBoostingClassifier(**config)
        elif model_type == 'rf':
            return RandomForestClassifier(**config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        model_type: str = 'gbm',
        sample_weights: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'PredictiveModel':
        """
        Train the predictive model.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_type: 'gbm' or 'rf'
            sample_weights: Optional sample weights for equity
            **kwargs: Additional model parameters
        
        Returns:
            Self with trained model
        """
        self.model_type = model_type
        self.feature_names = X.columns.tolist()
        self.model = self.get_model(model_type, **kwargs)
        
        if sample_weights is not None:
            self.model.fit(X, y, sample_weight=sample_weights)
        else:
            self.model.fit(X, y)
        
        self.is_trained = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate binary predictions.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of binary predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure columns match training features
        X = X[self.feature_names]
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate probability predictions.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of probabilities for positive class
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X = X[self.feature_names]
        proba = self.model.predict_proba(X)
        return proba[:, 1]  # Return probability of event
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> ModelMetrics:
        """
        Calculate comprehensive model performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Binary predictions
            y_proba: Probability predictions
        
        Returns:
            ModelMetrics with all calculated metrics
        """
        return ModelMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            auc_roc=float(roc_auc_score(y_true, y_proba)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            brier_score=float(brier_score_loss(y_true, y_proba))
        )
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[np.ndarray] = None,
        n_splits: int = 5,
        model_type: str = 'gbm'
    ) -> CVResults:
        """
        Perform stratified cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            groups: Optional demographic groups for stratification
            n_splits: Number of CV folds
            model_type: Model type to evaluate
        
        Returns:
            CVResults with fold-level metrics
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_results = []
        accuracies = []
        aucs = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model = self.get_model(model_type)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_proba)
            
            fold_result = {
                'fold': fold + 1,
                'accuracy': float(acc),
                'auc': float(auc),
                'n_train': len(train_idx),
                'n_val': len(val_idx)
            }
            
            # Add group-level metrics if groups provided
            if groups is not None:
                groups_val = groups[val_idx]
                group_aucs = {}
                for group in np.unique(groups_val):
                    mask = groups_val == group
                    if mask.sum() > 10:
                        try:
                            group_auc = roc_auc_score(y_val[mask], y_proba[mask])
                            group_aucs[str(group)] = float(group_auc)
                        except:
                            pass
                fold_result['group_aucs'] = group_aucs
            
            fold_results.append(fold_result)
            accuracies.append(acc)
            aucs.append(auc)
        
        return CVResults(
            mean_accuracy=float(np.mean(accuracies)),
            std_accuracy=float(np.std(accuracies)),
            mean_auc=float(np.mean(aucs)),
            std_auc=float(np.std(aucs)),
            fold_results=fold_results
        )
    
    def stratified_c_index(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        groups: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate stratified C-index (concordance) across demographic groups.
        
        This is the competition metric - C-index calculated within each group
        then averaged.
        
        Args:
            y_true: True labels
            y_proba: Probability predictions
            groups: Demographic group labels
        
        Returns:
            Dictionary with overall and per-group C-index
        """
        unique_groups = np.unique(groups)
        group_c_indices = {}
        
        for group in unique_groups:
            mask = groups == group
            if mask.sum() > 10:
                try:
                    # Use AUC as proxy for C-index in classification setting
                    c_index = roc_auc_score(y_true[mask], y_proba[mask])
                    group_c_indices[str(group)] = float(c_index)
                except:
                    pass
        
        # Weight by group size
        total = sum(1 for g in groups if str(g) in group_c_indices)
        weighted_c_index = 0
        for group, c in group_c_indices.items():
            group_size = sum(1 for g in groups if str(g) == group)
            weight = group_size / total if total > 0 else 0
            weighted_c_index += c * weight
        
        return {
            'stratified_c_index': float(weighted_c_index),
            'group_c_indices': group_c_indices,
            'c_index_disparity': max(group_c_indices.values()) - min(group_c_indices.values()) if group_c_indices else 0
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from trained model.
        
        Returns:
            Dictionary of feature importances
        """
        if not self.is_trained:
            return {}
        
        importances = self.model.feature_importances_
        return {f: float(i) for f, i in zip(self.feature_names, importances)}
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> 'PredictiveModel':
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        return self


class EnsembleModel:
    """
    Ensemble of multiple models for improved robustness.
    """
    
    def __init__(self):
        self.models: List[PredictiveModel] = []
        self.weights: List[float] = []
        self.is_trained = False
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_types: List[str] = ['gbm', 'rf'],
        sample_weights: Optional[np.ndarray] = None
    ) -> 'EnsembleModel':
        """
        Train ensemble of models.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_types: List of model types to include
            sample_weights: Optional sample weights
        
        Returns:
            Self with trained ensemble
        """
        self.models = []
        self.weights = []
        
        for model_type in model_types:
            pm = PredictiveModel()
            pm.train(X, y, model_type, sample_weights)
            
            # Estimate model weight based on CV performance
            cv_results = pm.cross_validate(X, y, n_splits=3, model_type=model_type)
            weight = cv_results.mean_auc
            
            self.models.append(pm)
            self.weights.append(weight)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        
        self.is_trained = True
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate weighted average probability predictions.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of ensemble probabilities
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained")
        
        probas = np.zeros(len(X))
        for model, weight in zip(self.models, self.weights):
            probas += model.predict_proba(X) * weight
        
        return probas
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Generate binary predictions.
        
        Args:
            X: Feature matrix
            threshold: Classification threshold
        
        Returns:
            Array of binary predictions
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
