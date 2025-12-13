"""
Module M1: Data Preprocessing
Handles data ingestion, validation, imputation, normalization, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass


@dataclass
class ValidationReport:
    """Report from data validation process."""
    is_valid: bool
    total_rows: int
    total_columns: int
    missing_summary: Dict[str, float]
    data_types: Dict[str, str]
    issues: List[str]


class DataPreprocessor:
    """
    M1: Data Preprocessing Module
    
    Handles initial data preparation through multiple stages:
    - Ingestion and validation
    - Equity-aware imputation
    - Standardization and normalization
    - Basic feature engineering
    """
    
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.imputers: Dict[str, SimpleImputer] = {}
        self.feature_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        df = pd.read_csv(filepath, encoding='utf-8', encoding_errors='replace')
        return df
    
    def validate_data(self, df: pd.DataFrame) -> ValidationReport:
        """
        Validate data integrity and format.
        
        Returns a ValidationReport with details about the dataset.
        """
        issues = []
        
        # Check for required columns
        required_columns = ['efs', 'efs_time', 'race_group']
        for col in required_columns:
            if col not in df.columns:
                issues.append(f"Missing required column: {col}")
        
        # Calculate missing value percentages
        missing_summary = (df.isnull().sum() / len(df) * 100).to_dict()
        
        # Check for high missing rates (>30% threshold from Workshop 3)
        for col, pct in missing_summary.items():
            if pct > 30:
                issues.append(f"High missing rate ({pct:.1f}%) in column: {col}")
        
        # Get data types
        data_types = df.dtypes.astype(str).to_dict()
        
        return ValidationReport(
            is_valid=len(issues) == 0,
            total_rows=len(df),
            total_columns=len(df.columns),
            missing_summary=missing_summary,
            data_types=data_types,
            issues=issues
        )
    
    def identify_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify categorical and numerical columns."""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove target columns from features
        target_cols = ['efs', 'efs_time', 'ID']
        categorical_cols = [c for c in categorical_cols if c not in target_cols]
        numerical_cols = [c for c in numerical_cols if c not in target_cols]
        
        self.categorical_columns = categorical_cols
        self.numerical_columns = numerical_cols
        
        return categorical_cols, numerical_cols
    
    def impute_missing(self, df: pd.DataFrame, strategy: str = 'equity_aware') -> pd.DataFrame:
        """
        Handle missing data with equity-aware imputation.
        
        Args:
            df: Input DataFrame
            strategy: 'simple' or 'equity_aware'
        
        For equity_aware strategy:
        - Numerical: median imputation within each demographic group
        - Categorical: mode imputation within each demographic group
        """
        df = df.copy()
        
        if strategy == 'equity_aware' and 'race_group' in df.columns:
            # Impute within demographic groups
            for col in self.numerical_columns:
                if df[col].isnull().any():
                    df[col] = df.groupby('race_group')[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                    # Fill remaining with overall median
                    df[col] = df[col].fillna(df[col].median())
            
            for col in self.categorical_columns:
                if df[col].isnull().any():
                    df[col] = df.groupby('race_group')[col].transform(
                        lambda x: x.fillna(x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown')
                    )
                    # Fill remaining with overall mode
                    mode_val = df[col].mode()
                    df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown')
        else:
            # Simple imputation
            for col in self.numerical_columns:
                if df[col].isnull().any():
                    imputer = SimpleImputer(strategy='median')
                    df[col] = imputer.fit_transform(df[[col]]).ravel()
                    self.imputers[col] = imputer
            
            for col in self.categorical_columns:
                if df[col].isnull().any():
                    mode_val = df[col].mode()
                    df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown')
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables using LabelEncoder.
        
        Args:
            df: Input DataFrame
            fit: If True, fit new encoders; if False, use existing
        """
        df = df.copy()
        
        for col in self.categorical_columns:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    # Handle unseen values by adding 'Unknown'
                    df[col] = df[col].astype(str)
                    le.fit(df[col].unique().tolist() + ['Unknown'])
                    self.label_encoders[col] = le
                
                if col in self.label_encoders:
                    df[col] = df[col].astype(str)
                    # Handle unseen values
                    df[col] = df[col].apply(
                        lambda x: x if x in self.label_encoders[col].classes_ else 'Unknown'
                    )
                    df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features using StandardScaler.
        
        Args:
            df: Input DataFrame
            fit: If True, fit new scaler; if False, use existing
        """
        df = df.copy()
        
        if len(self.numerical_columns) == 0:
            return df
        
        cols_to_scale = [c for c in self.numerical_columns if c in df.columns]
        
        if fit:
            self.scaler = StandardScaler()
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        elif self.scaler is not None:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features based on clinical domain knowledge.
        
        Features created:
        - age_donor_diff: Difference between patient and donor age
        - high_risk_comorbidity: Flag for severe comorbidities
        - hla_match_quality: Composite HLA matching score
        """
        df = df.copy()
        
        # Age difference between patient and donor
        if 'age_at_hct' in df.columns and 'donor_age' in df.columns:
            df['age_donor_diff'] = df['age_at_hct'] - df['donor_age']
        
        # High-risk comorbidity flag
        comorbidity_cols = ['cardiac', 'pulm_severe', 'hepatic_severe', 'renal_issue', 'diabetes']
        existing_comorbidity_cols = [c for c in comorbidity_cols if c in df.columns]
        if existing_comorbidity_cols:
            # After encoding, 'Yes' should be encoded to a specific value
            # For now, check if any comorbidity is present
            df['high_risk_comorbidity'] = 0
            for col in existing_comorbidity_cols:
                if df[col].dtype in ['int64', 'float64']:
                    df['high_risk_comorbidity'] = df['high_risk_comorbidity'] | (df[col] > 0)
        
        # HLA match quality (composite score)
        hla_cols = [c for c in df.columns if c.startswith('hla_') and 'match' in c]
        if hla_cols:
            df['hla_match_quality'] = df[hla_cols].mean(axis=1)
        
        return df
    
    def prepare_target(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Prepare target variables for survival analysis.
        
        Returns:
            event: Binary event indicator (1 = event, 0 = censored)
            time: Time to event
        """
        # Convert EFS to binary - handle both numeric and string formats
        if 'efs' in df.columns:
            if df['efs'].dtype == 'object':
                event = (df['efs'] == 'Event').astype(int)
            else:
                event = df['efs'].fillna(0).astype(int)
        else:
            event = pd.Series([0] * len(df))
        
        # Get time to event
        if 'efs_time' in df.columns:
            time = df['efs_time']
        else:
            time = pd.Series([0] * len(df))
        
        return event, time
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Complete preprocessing pipeline for training data.
        
        Returns:
            X: Preprocessed features
            y_event: Binary event indicator
            y_time: Time to event
        """
        # Validate
        report = self.validate_data(df)
        print(f"Validation: {report.total_rows} rows, {report.total_columns} columns")
        if report.issues:
            print(f"Issues found: {report.issues}")
        
        # Identify column types
        self.identify_column_types(df)
        
        # Store original feature columns
        self.feature_columns = self.categorical_columns + self.numerical_columns
        
        # Impute missing values
        df = self.impute_missing(df, strategy='equity_aware')
        
        # Create engineered features
        df = self.create_features(df)
        
        # Update numerical columns with new features
        new_features = ['age_donor_diff', 'high_risk_comorbidity', 'hla_match_quality']
        for feat in new_features:
            if feat in df.columns and feat not in self.numerical_columns:
                self.numerical_columns.append(feat)
        
        # Encode categorical
        df = self.encode_categorical(df, fit=True)
        
        # Prepare target
        y_event, y_time = self.prepare_target(df)
        
        # Normalize numerical features
        df = self.normalize_features(df, fit=True)
        
        # Select feature columns
        all_features = self.categorical_columns + self.numerical_columns
        X = df[[c for c in all_features if c in df.columns]]
        
        return X, y_event, y_time
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: New data to transform
        
        Returns:
            X: Preprocessed features
        """
        df = df.copy()
        
        # Ensure all expected columns exist (fill with defaults)
        for col in self.categorical_columns:
            if col not in df.columns:
                df[col] = 'Unknown'
        
        for col in self.numerical_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        # Convert numeric columns to proper type
        for col in self.numerical_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Convert categorical columns to string
        for col in self.categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('Unknown').replace('None', 'Unknown').replace('nan', 'Unknown')
        
        # Create features
        df = self.create_features(df)
        
        # Encode categorical (use existing encoders)
        for col in self.categorical_columns:
            if col in df.columns and col in self.label_encoders:
                df[col] = df[col].astype(str)
                # Handle unseen values
                df[col] = df[col].apply(
                    lambda x: x if x in self.label_encoders[col].classes_ else 'Unknown'
                )
                df[col] = self.label_encoders[col].transform(df[col])
        
        # Normalize numerical features (use existing scaler)
        if self.scaler is not None:
            cols_to_scale = [c for c in self.numerical_columns if c in df.columns]
            # Make sure we have the same columns as training
            for col in cols_to_scale:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            if cols_to_scale:
                df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        # Select features
        all_features = self.categorical_columns + self.numerical_columns
        available_features = [c for c in all_features if c in df.columns]
        X = df[available_features]
        
        return X
