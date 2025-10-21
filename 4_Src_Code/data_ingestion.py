"""
Data Ingestion Module for AI Academic Mentor
Handles loading and initial processing of OULAD dataset
"""

import pandas as pd
import os
from typing import Dict, List, Optional
from pathlib import Path

class DataIngestion:
    """Handles data ingestion from OULAD CSV files"""
    
    def __init__(self, data_path: str = "Data_csv's"):
        self.data_path = Path(data_path)
        self.data_files = {
            'assessments': 'assessments.csv',
            'courses': 'courses.csv', 
            'studentAssessment': 'studentAssessment.csv',
            'studentInfo': 'studentInfo.csv',
            'studentRegistration': 'studentRegistration.csv',
            'studentVle': 'studentVle.csv',
            'vle': 'vle.csv'
        }
    
    def load_dataset(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files from the dataset"""
        datasets = {}
        
        for name, filename in self.data_files.items():
            file_path = self.data_path / filename
            if file_path.exists():
                datasets[name] = pd.read_csv(file_path)
                print(f"Loaded {name}: {datasets[name].shape}")
            else:
                print(f"Warning: {filename} not found at {file_path}")
        
        return datasets
    
    def get_data_info(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Get basic information about loaded datasets"""
        info = {}
        for name, df in datasets.items():
            info[name] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict()
            }
        return info
