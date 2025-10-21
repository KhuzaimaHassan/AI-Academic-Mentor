"""
Helper functions for AI Academic Mentor
Provides utility functions for student prediction and feature processing
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any, Optional
from pathlib import Path

class StudentPredictionHelper:
    """Helper class for student prediction operations"""
    
    def __init__(self, models_dir: str = "6_Models", data_path: str = "Data_csv's"):
        self.models_dir = models_dir
        self.data_path = data_path
        
        # Load encoders and model
        self._load_models()
    
    def _load_models(self):
        """Load all saved models and encoders"""
        try:
            # Load feature encoders
            feature_encoders_path = os.path.join(self.models_dir, "feature_encoders.pkl")
            if os.path.exists(feature_encoders_path):
                self.feature_encoders = joblib.load(feature_encoders_path)
            else:
                self.feature_encoders = {}
            
            # Load target encoder
            target_encoder_path = os.path.join(self.models_dir, "final_result_encoder.pkl")
            if os.path.exists(target_encoder_path):
                self.target_encoder = joblib.load(target_encoder_path)
            else:
                self.target_encoder = None
            
            # Load trained model
            model_path = os.path.join(self.models_dir, "trained_model.pkl")
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
            else:
                self.model = None
            
            # Load feature columns
            feature_columns_path = os.path.join(self.models_dir, "feature_columns.pkl")
            if os.path.exists(feature_columns_path):
                self.feature_columns = joblib.load(feature_columns_path)
            else:
                self.feature_columns = ['gender', 'disability', 'num_of_prev_attempts', 'studied_credits', 'avg_score', 'num_assessments']
            
            print("✅ All models and encoders loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.feature_encoders = {}
            self.target_encoder = None
            self.model = None
            self.feature_columns = []
    
    def get_student_features(self, student_id: int) -> Optional[pd.DataFrame]:
        """
        Get processed features for a specific student ID.
        
        Args:
            student_id: The student ID to process
            
        Returns:
            Processed feature row ready for prediction, or None if student not found
        """
        try:
            print(f"🔍 Processing features for student ID: {student_id}")
            
            # Load raw CSV files
            student_info = pd.read_csv(Path(self.data_path) / "studentInfo.csv")
            student_assessment = pd.read_csv(Path(self.data_path) / "studentAssessment.csv")
            
            # Find student in studentInfo
            student_row = student_info[student_info['id_student'] == student_id]
            
            if student_row.empty:
                print(f"❌ Student ID {student_id} not found in studentInfo")
                return None
            
            # Calculate aggregated features for this student from studentAssessment
            student_assessments = student_assessment[student_assessment['id_student'] == student_id]
            
            if not student_assessments.empty:
                # Calculate average score and number of assessments
                avg_score = student_assessments['score'].mean()
                num_assessments = len(student_assessments)
            else:
                # Student has no assessments
                avg_score = 0
                num_assessments = 0
            
            # Extract features from student info
            features = {
                'gender': student_row['gender'].iloc[0],
                'disability': student_row['disability'].iloc[0],
                'num_of_prev_attempts': student_row['num_of_prev_attempts'].iloc[0],
                'studied_credits': student_row['studied_credits'].iloc[0],
                'avg_score': avg_score,
                'num_assessments': num_assessments
            }
            
            # Handle missing values
            features['gender'] = features['gender'] if pd.notna(features['gender']) else 'Unknown'
            features['disability'] = features['disability'] if pd.notna(features['disability']) else 'Unknown'
            features['num_of_prev_attempts'] = features['num_of_prev_attempts'] if pd.notna(features['num_of_prev_attempts']) else 0
            features['studied_credits'] = features['studied_credits'] if pd.notna(features['studied_credits']) else 0
            
            # Create DataFrame with single row
            feature_df = pd.DataFrame([features])
            
            # Apply feature encoders
            for feature, encoder in self.feature_encoders.items():
                if feature in feature_df.columns:
                    # Handle unseen categories
                    feature_value = str(feature_df[feature].iloc[0])
                    if feature_value in encoder.classes_:
                        feature_df[feature] = encoder.transform([feature_value])[0]
                    else:
                        # Use the most common class as default
                        default_value = encoder.transform([encoder.classes_[0]])[0]
                        feature_df[feature] = default_value
            
            # Ensure all required features are present
            for feature in self.feature_columns:
                if feature not in feature_df.columns:
                    feature_df[feature] = 0  # Default value
            
            # Reorder columns to match training data
            feature_df = feature_df[self.feature_columns]
            
            print(f"✅ Features processed for student {student_id}")
            print(f"Features: {features}")
            
            return feature_df
            
        except Exception as e:
            print(f"❌ Error processing features for student {student_id}: {e}")
            return None
    
    def predict_student_result(self, student_id: int) -> Optional[str]:
        """
        Predict the final result for a specific student.
        
        Args:
            student_id: The student ID to predict
            
        Returns:
            Predicted result string (e.g., 'Pass', 'Fail', 'Withdrawn', 'Distinction'), 
            or None if prediction failed
        """
        try:
            print(f"🎯 Predicting result for student ID: {student_id}")
            
            # Check if model and encoder are loaded
            if self.model is None:
                print("❌ Model not loaded. Please train the model first.")
                return None
            
            if self.target_encoder is None:
                print("❌ Target encoder not loaded. Please train the model first.")
                return None
            
            # Get student features
            student_features = self.get_student_features(student_id)
            
            if student_features is None:
                return None
            
            # Make prediction
            prediction_encoded = self.model.predict(student_features)[0]
            prediction_proba = self.model.predict_proba(student_features)[0]
            
            # Convert encoded prediction back to original label
            prediction_original = self.target_encoder.inverse_transform([prediction_encoded])[0]
            
            # Get prediction confidence
            confidence = prediction_proba[prediction_encoded]
            
            print(f"✅ Prediction completed for student {student_id}")
            print(f"Predicted result: {prediction_original}")
            print(f"Confidence: {confidence:.4f}")
            print(f"All probabilities: {dict(zip(self.target_encoder.classes_, prediction_proba))}")
            
            return prediction_original
            
        except Exception as e:
            print(f"❌ Error predicting result for student {student_id}: {e}")
            return None
    
    def get_student_prediction_details(self, student_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed prediction information for a student.
        
        Args:
            student_id: The student ID to analyze
            
        Returns:
            Dictionary with prediction details, or None if prediction failed
        """
        try:
            # Check if model and encoder are loaded
            if self.model is None or self.target_encoder is None:
                return None
            
            # Get student features
            student_features = self.get_student_features(student_id)
            
            if student_features is None:
                return None
            
            # Make prediction
            prediction_encoded = self.model.predict(student_features)[0]
            prediction_proba = self.model.predict_proba(student_features)[0]
            
            # Convert encoded prediction back to original label
            prediction_original = self.target_encoder.inverse_transform([prediction_encoded])[0]
            
            # Get feature importance if available
            feature_importance = None
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            
            # Prepare detailed results
            details = {
                'student_id': student_id,
                'predicted_result': prediction_original,
                'confidence': prediction_proba[prediction_encoded],
                'all_probabilities': dict(zip(self.target_encoder.classes_, prediction_proba)),
                'features': student_features.iloc[0].to_dict(),
                'feature_importance': feature_importance
            }
            
            return details
            
        except Exception as e:
            print(f"❌ Error getting prediction details for student {student_id}: {e}")
            return None
    
    def batch_predict_students(self, student_ids: list) -> Dict[int, str]:
        """
        Predict results for multiple students.
        
        Args:
            student_ids: List of student IDs to predict
            
        Returns:
            Dictionary mapping student IDs to predicted results
        """
        results = {}
        
        for student_id in student_ids:
            prediction = self.predict_student_result(student_id)
            if prediction is not None:
                results[student_id] = prediction
        
        return results

# Convenience functions for direct use
def get_student_features(student_id: int, models_dir: str = "6_Models", data_path: str = "Data_csv's") -> Optional[pd.DataFrame]:
    """
    Get processed features for a specific student ID.
    
    Args:
        student_id: The student ID to process
        models_dir: Directory containing saved models
        data_path: Directory containing CSV files
        
    Returns:
        Processed feature row ready for prediction, or None if student not found
    """
    helper = StudentPredictionHelper(models_dir, data_path)
    return helper.get_student_features(student_id)

def predict_student_result(student_id: int, models_dir: str = "6_Models", data_path: str = "Data_csv's") -> Optional[str]:
    """
    Predict the final result for a specific student.
    
    Args:
        student_id: The student ID to predict
        models_dir: Directory containing saved models
        data_path: Directory containing CSV files
        
    Returns:
        Predicted result string, or None if prediction failed
    """
    helper = StudentPredictionHelper(models_dir, data_path)
    return helper.predict_student_result(student_id)
