"""
Data Cleaning Module for AI Academic Mentor
Handles data cleaning and preprocessing for OULAD dataset
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataCleaning:
    """Handles data cleaning and preprocessing"""
    
    def __init__(self):
        self.cleaning_log = []
    
    def clean_student_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean student information dataset"""
        df_clean = df.copy()
        
        # Handle missing values
        df_clean['imd_band'] = df_clean['imd_band'].fillna('Unknown')
        df_clean['num_of_prev_attempts'] = df_clean['num_of_prev_attempts'].fillna(0)
        
        # Convert date columns
        if 'date_unregistration' in df_clean.columns:
            df_clean['date_unregistration'] = pd.to_datetime(
                df_clean['date_unregistration'], errors='coerce'
            )
        
        self.cleaning_log.append("Cleaned student_info dataset")
        return df_clean
    
    def clean_student_assessment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean student assessment dataset"""
        df_clean = df.copy()
        
        # Handle missing scores
        df_clean['score'] = df_clean['score'].fillna(0)
        
        # Handle date columns
        if 'date_submitted' in df_clean.columns:
            df_clean['date_submitted'] = pd.to_datetime(
                df_clean['date_submitted'], errors='coerce'
            )
        
        self.cleaning_log.append("Cleaned studentAssessment dataset")
        return df_clean
    
    def clean_all_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Clean all datasets in the collection"""
        cleaned_datasets = {}
        
        for name, df in datasets.items():
            if name == 'studentInfo':
                cleaned_datasets[name] = self.clean_student_info(df)
            elif name == 'studentAssessment':
                cleaned_datasets[name] = self.clean_student_assessment(df)
            else:
                # Basic cleaning for other datasets
                df_clean = df.copy()
                df_clean = df_clean.dropna(subset=df_clean.columns[df_clean.isnull().any()])
                cleaned_datasets[name] = df_clean
                self.cleaning_log.append(f"Basic cleaning applied to {name}")
        
        return cleaned_datasets
    
    def get_cleaning_report(self) -> List[str]:
        """Get cleaning operations log"""
        return self.cleaning_log
