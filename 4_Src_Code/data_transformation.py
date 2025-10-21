"""
Data Transformation Module for AI Academic Mentor
Handles feature engineering and data transformation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataTransformation:
    """Handles data transformation and feature engineering"""
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.feature_columns = []
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """Encode categorical features using label encoding"""
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_encoded[col] = self.encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    # Handle unseen categories
                    df_encoded[col] = df_encoded[col].astype(str)
                    unseen_mask = ~df_encoded[col].isin(self.encoders[col].classes_)
                    if unseen_mask.any():
                        df_encoded.loc[unseen_mask, col] = 'Unknown'
                    df_encoded[col] = self.encoders[col].transform(df_encoded[col])
        
        return df_encoded
    
    def create_academic_features(self, student_df: pd.DataFrame, assessment_df: pd.DataFrame) -> pd.DataFrame:
        """Create academic performance features"""
        features = student_df.copy()
        
        # Calculate assessment statistics per student
        assessment_stats = assessment_df.groupby('id_student').agg({
            'score': ['mean', 'std', 'count'],
            'date_submitted': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        assessment_stats.columns = ['id_student', 'avg_score', 'score_std', 'assessment_count', 
                                  'first_submission', 'last_submission']
        
        # Merge with student features
        features = features.merge(assessment_stats, on='id_student', how='left')
        
        # Fill missing values
        features['avg_score'] = features['avg_score'].fillna(0)
        features['score_std'] = features['score_std'].fillna(0)
        features['assessment_count'] = features['assessment_count'].fillna(0)
        
        return features
    
    def scale_numerical_features(self, df: pd.DataFrame, numerical_columns: List[str]) -> pd.DataFrame:
        """Scale numerical features using StandardScaler"""
        df_scaled = df.copy()
        
        for col in numerical_columns:
            if col in df_scaled.columns:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    df_scaled[col] = self.scalers[col].fit_transform(df_scaled[[col]])
                else:
                    df_scaled[col] = self.scalers[col].transform(df_scaled[[col]])
        
        return df_scaled
    
    def prepare_features_for_modeling(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare final feature matrix and target variable"""
        # Get student info as base
        student_df = datasets['studentInfo']
        assessment_df = datasets['studentAssessment']
        
        # Create features
        features = self.create_academic_features(student_df, assessment_df)
        
        # Define categorical and numerical columns
        categorical_cols = ['code_module', 'code_presentation', 'gender', 'region', 
                           'highest_education', 'imd_band', 'age_band', 'disability']
        numerical_cols = ['num_of_prev_attempts', 'studied_credits', 'avg_score', 
                         'score_std', 'assessment_count']
        
        # Encode categorical features
        features = self.encode_categorical_features(features, categorical_cols)
        
        # Scale numerical features
        features = self.scale_numerical_features(features, numerical_cols)
        
        # Prepare target variable
        target = features['final_result']
        features = features.drop(['final_result', 'id_student'], axis=1, errors='ignore')
        
        self.feature_columns = list(features.columns)
        
        return features, target
