"""
Configuration utility for AI Academic Mentor
"""

import os
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

class Config:
    """Configuration management for the application"""
    
    def __init__(self, env_file: str = ".env"):
        # Load environment variables
        load_dotenv(env_file)
        
        # Project paths
        self.PROJECT_ROOT = Path(__file__).parent.parent.parent
        self.DATA_DIR = self.PROJECT_ROOT / "2_Data"
        self.MODELS_DIR = self.PROJECT_ROOT / "6_Models"
        self.RESULTS_DIR = self.PROJECT_ROOT / "7_Results"
        
        # API Keys
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
        
        # Model settings
        self.MODEL_SETTINGS = {
            "groq_model": "llama-3.1-70b-versatile",
            "temperature": 0.7,
            "max_tokens": 2048,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        }
        
        # Vector store settings
        self.VECTORSTORE_SETTINGS = {
            "persist_directory": str(self.MODELS_DIR / "vectorstore"),
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
        
        # Data processing settings
        self.DATA_SETTINGS = {
            "train_size": 0.8,
            "val_size": 0.1,
            "test_size": 0.1,
            "random_state": 42
        }
        
        # Logging settings
        self.LOGGING_SETTINGS = {
            "log_level": "INFO",
            "log_dir": str(self.PROJECT_ROOT / "logs")
        }
    
    def get_groq_config(self) -> Dict[str, Any]:
        """Get Groq API configuration"""
        return {
            "groq_api_key": self.GROQ_API_KEY,
            "model_name": self.MODEL_SETTINGS["groq_model"],
            "temperature": self.MODEL_SETTINGS["temperature"],
            "max_tokens": self.MODEL_SETTINGS["max_tokens"]
        }
    
    def get_vectorstore_config(self) -> Dict[str, Any]:
        """Get vector store configuration"""
        return self.VECTORSTORE_SETTINGS
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data processing configuration"""
        return self.DATA_SETTINGS
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        if not self.GROQ_API_KEY:
            errors.append("GROQ_API_KEY not found in environment variables")
        
        if not self.DATA_DIR.exists():
            errors.append(f"Data directory not found: {self.DATA_DIR}")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.DATA_DIR / "raw",
            self.DATA_DIR / "cleaned", 
            self.DATA_DIR / "processed",
            self.MODELS_DIR / "vectorstore",
            self.RESULTS_DIR / "visualizations",
            Path(self.LOGGING_SETTINGS["log_dir"])
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# Global config instance
config = Config()

