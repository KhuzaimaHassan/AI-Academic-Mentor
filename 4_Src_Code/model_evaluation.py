"""
Model Evaluation Module for AI Academic Mentor
Handles model evaluation and performance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import joblib
import os

class ModelEvaluation:
    """Handles model evaluation and performance analysis"""
    
    def __init__(self, results_dir: str = "7_Results"):
        self.results_dir = results_dir
        self.evaluation_results = {}
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "visualizations"), exist_ok=True)
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: str) -> Dict[str, Any]:
        """Evaluate a single model comprehensively"""
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Calculate ROC AUC for multiclass
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            metrics['roc_auc'] = None
        
        # Classification report
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        evaluation_result = {
            'model_name': model_name,
            'metrics': metrics,
            'classification_report': classification_rep,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_test
        }
        
        self.evaluation_results[model_name] = evaluation_result
        
        return evaluation_result
    
    def compare_models(self, models_dict: Dict[str, Any], X_test: pd.DataFrame, 
                      y_test: pd.Series) -> pd.DataFrame:
        """Compare multiple models and return comparison dataframe"""
        
        comparison_results = []
        
        for model_name, model in models_dict.items():
            result = self.evaluate_model(model, X_test, y_test, model_name)
            
            comparison_results.append({
                'Model': model_name,
                'Accuracy': result['metrics']['accuracy'],
                'Precision': result['metrics']['precision'],
                'Recall': result['metrics']['recall'],
                'F1-Score': result['metrics']['f1_score'],
                'ROC-AUC': result['metrics']['roc_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        return comparison_df
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                             model_name: str, save_path: str = None) -> None:
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, model: Any, feature_names: List[str], 
                               model_name: str, top_n: int = 20, save_path: str = None) -> None:
        """Plot feature importance for tree-based models"""
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} does not support feature importance")
            return
        
        # Get feature importance
        importance = model.feature_importances_
        
        # Create dataframe and sort
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance_df, x='importance', y='feature')
        plt.title(f'Feature Importance - {model_name}')
        plt.xlabel('Importance')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, models_dict: Dict[str, Any], X_test: pd.DataFrame, 
                       y_test: pd.Series, save_path: str = None) -> None:
        """Plot ROC curves for multiple models"""
        plt.figure(figsize=(12, 8))
        
        for model_name, model in models_dict.items():
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                
                # For multiclass, use one-vs-rest
                from sklearn.preprocessing import label_binarize
                from sklearn.metrics import roc_curve, auc
                
                # Get unique classes
                classes = np.unique(y_test)
                y_test_bin = label_binarize(y_test, classes=classes)
                
                if len(classes) == 2:
                    # Binary classification
                    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
                else:
                    # Multiclass - average ROC
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    
                    for i in range(len(classes)):
                        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                    
                    # Compute micro-average ROC curve
                    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                    
                    plt.plot(fpr["micro"], tpr["micro"], 
                            label=f'{model_name} (AUC = {roc_auc["micro"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self, output_path: str = None) -> str:
        """Generate comprehensive evaluation report"""
        report_lines = []
        report_lines.append("# Model Evaluation Report\n")
        report_lines.append("## Performance Summary\n")
        
        # Model comparison table
        if self.evaluation_results:
            comparison_data = []
            for model_name, result in self.evaluation_results.items():
                metrics = result['metrics']
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1-Score': f"{metrics['f1_score']:.4f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            report_lines.append(comparison_df.to_string(index=False))
            report_lines.append("\n")
        
        # Detailed classification reports
        report_lines.append("## Detailed Classification Reports\n")
        for model_name, result in self.evaluation_results.items():
            report_lines.append(f"### {model_name}\n")
            report_lines.append("```")
            report_lines.append(classification_report(result['true_labels'], result['predictions']))
            report_lines.append("```\n")
        
        report_content = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_content)
            print(f"Evaluation report saved to {output_path}")
        
        return report_content
