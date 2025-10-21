"""
Model Training Module for AI Academic Mentor
Handles machine learning model training and validation
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from pathlib import Path

class ModelTraining:
    """Handles model training and validation"""
    
    def __init__(self, models_dir: str = "6_Models"):
        self.models_dir = models_dir
        self.models = {}
        self.model_scores = {}
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
    
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize different ML models"""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                learning_rate=0.1
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
        
        self.models = models
        return models
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Dict]:
        """Train all models and return performance metrics"""
        
        if not self.models:
            self.initialize_models()
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'classification_report': classification_report(y_val, y_pred, output_dict=True)
            }
            
            self.model_scores[name] = accuracy
            
            # Save model
            self.save_model(model, name)
            
            print(f"{name} - Accuracy: {accuracy:.4f}, CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def save_model(self, model: Any, model_name: str) -> None:
        """Save trained model to disk"""
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_name: str) -> Any:
        """Load trained model from disk"""
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model
        else:
            raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
    
    def get_best_model(self) -> Tuple[str, Any]:
        """Get the best performing model based on accuracy"""
        if not self.model_scores:
            raise ValueError("No models have been trained yet")
        
        best_model_name = max(self.model_scores, key=self.model_scores.get)
        best_model = self.models[best_model_name]
        
        return best_model_name, best_model
    
    def predict_student_outcome(self, student_features: pd.DataFrame, model_name: str = None) -> Dict:
        """Predict student outcome using trained model"""
        if model_name is None:
            model_name, model = self.get_best_model()
        else:
            model = self.models.get(model_name)
            if model is None:
                model = self.load_model(model_name)
        
        # Make prediction
        prediction = model.predict(student_features)[0]
        prediction_proba = model.predict_proba(student_features)[0]
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(student_features.columns, model.feature_importances_))
        
        return {
            'prediction': prediction,
            'probability': dict(zip(model.classes_, prediction_proba)),
            'model_used': model_name,
            'feature_importance': feature_importance
        }
    
    def train_model(self, data_path: str = "Data_csv's") -> Dict[str, Any]:
        """
        Complete training pipeline for OULAD dataset.
        Performs: load, merge, engineer, transform, train, and save all .pkl files.
        
        Args:
            data_path: Path to the directory containing CSV files
            
        Returns:
            Dictionary with training results and model information
        """
        print("🚀 Starting OULAD Model Training Pipeline...")
        
        # Step 1: Load data
        print("📊 Loading datasets...")
        student_info = pd.read_csv(Path(data_path) / "studentInfo.csv")
        student_assessment = pd.read_csv(Path(data_path) / "studentAssessment.csv")
        assessments = pd.read_csv(Path(data_path) / "assessments.csv")
        
        print(f"Loaded datasets:")
        print(f"  - studentInfo: {student_info.shape}")
        print(f"  - studentAssessment: {student_assessment.shape}")
        print(f"  - assessments: {assessments.shape}")
        
        # Step 2: Feature Engineering
        print("\n🔧 Feature Engineering...")
        
        # Calculate aggregated features from studentAssessment
        assessment_aggregated = student_assessment.groupby('id_student').agg({
            'score': ['mean', 'count'],
            'date_submitted': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        assessment_aggregated.columns = [
            'id_student', 'avg_score', 'num_assessments', 
            'first_submission', 'last_submission'
        ]
        
        # Handle missing scores
        assessment_aggregated['avg_score'] = assessment_aggregated['avg_score'].fillna(0)
        assessment_aggregated['num_assessments'] = assessment_aggregated['num_assessments'].fillna(0)
        
        print(f"Created aggregated features: {assessment_aggregated.shape}")
        
        # Step 3: Merge data
        print("\n🔗 Merging datasets...")
        merged_data = student_info.merge(assessment_aggregated, on='id_student', how='left')
        
        # Fill missing values for students with no assessments
        merged_data['avg_score'] = merged_data['avg_score'].fillna(0)
        merged_data['num_assessments'] = merged_data['num_assessments'].fillna(0)
        
        print(f"Merged data shape: {merged_data.shape}")
        
        # Step 4: Data Cleaning & Transformation
        print("\n🧹 Data Cleaning & Transformation...")
        
        # Handle missing values
        categorical_columns = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
        for col in categorical_columns:
            if col in merged_data.columns:
                merged_data[col] = merged_data[col].fillna('Unknown')
        
        numerical_columns = ['num_of_prev_attempts', 'studied_credits']
        for col in numerical_columns:
            if col in merged_data.columns:
                merged_data[col] = merged_data[col].fillna(0)
        
        # Feature Selection
        feature_columns = ['gender', 'disability', 'num_of_prev_attempts', 'studied_credits', 'avg_score', 'num_assessments']
        target_column = 'final_result'
        
        X = merged_data[feature_columns].copy()
        y = merged_data[target_column].copy()
        
        print(f"Selected features: {feature_columns}")
        print(f"Feature matrix shape: {X.shape}")
        
        # Label Encoding
        categorical_features = ['gender', 'disability']
        feature_encoders = {}
        
        for feature in categorical_features:
            if feature in X.columns:
                encoder = LabelEncoder()
                X[feature] = encoder.fit_transform(X[feature].astype(str))
                feature_encoders[feature] = encoder
                print(f"Encoded {feature}: {len(encoder.classes_)} unique values")
        
        # Encode target variable
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y.astype(str))
        
        print(f"Encoded target: {len(target_encoder.classes_)} classes")
        print(f"Target classes: {target_encoder.classes_}")
        
        # Step 5: Train-Test Split
        print("\n✂️ Train-Test Split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Step 6: Model Training
        print("\n🤖 Training GradientBoostingClassifier...")
        
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        gb_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = gb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        
        # Step 7: Save Models and Encoders
        print("\n💾 Saving models and encoders...")
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Save trained model
        model_path = os.path.join(self.models_dir, "trained_model.pkl")
        joblib.dump(gb_model, model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Save target encoder
        target_encoder_path = os.path.join(self.models_dir, "final_result_encoder.pkl")
        joblib.dump(target_encoder, target_encoder_path)
        print(f"✅ Target encoder saved: {target_encoder_path}")
        
        # Save feature encoders
        feature_encoders_path = os.path.join(self.models_dir, "feature_encoders.pkl")
        joblib.dump(feature_encoders, feature_encoders_path)
        print(f"✅ Feature encoders saved: {feature_encoders_path}")
        
        # Save feature columns
        feature_columns_path = os.path.join(self.models_dir, "feature_columns.pkl")
        joblib.dump(feature_columns, feature_columns_path)
        print(f"✅ Feature columns saved: {feature_columns_path}")
        
        # Prepare results
        results = {
            'model': gb_model,
            'accuracy': accuracy,
            'feature_columns': feature_columns,
            'target_encoder': target_encoder,
            'feature_encoders': feature_encoders,
            'test_predictions': y_pred,
            'test_actual': y_test,
            'classification_report': classification_report(y_test, y_pred, target_names=target_encoder.classes_, output_dict=True)
        }
        
        print("\n🎉 Training pipeline completed successfully!")
        return results
