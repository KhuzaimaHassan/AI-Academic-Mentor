"""
Prefect Orchestration Pipeline for AI Academic Mentor
Automated data pipeline using Prefect for academic monitoring and mentoring
"""

from prefect import flow, task, get_run_logger
import pandas as pd
import json
import os
from pathlib import Path

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent / "4_Src_Code"))

# Import required functions
from model_training import ModelTraining
from agentic_ai_pipeline import AgenticAIPipeline
from utils.helpers import predict_student_result

@task
def retrain_model_task():
    """Task to retrain the ML model"""
    logger = get_run_logger()
    logger.info("🔄 Starting model retraining...")
    
    try:
        # Initialize model training
        trainer = ModelTraining()
        
        # Call train_model function
        results = trainer.train_model()
        
        logger.info("✅ Model retraining completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"❌ Model retraining failed: {str(e)}")
        raise

@task
def load_student_list_task():
    """Task to load student list from CSV"""
    logger = get_run_logger()
    logger.info("📊 Loading student list...")
    
    try:
        # Try multiple data paths
        data_paths = [
            "2_Data/raw/studentInfo.csv",
            "Data_csv's/studentInfo.csv"
        ]
        
        df = None
        for path in data_paths:
            if Path(path).exists():
                df = pd.read_csv(path)
                logger.info(f"✅ Loaded student data from: {path}")
                break
        
        if df is None:
            raise FileNotFoundError("Student data file not found")
        
        # Return student IDs as list
        student_ids = df['id_student'].tolist()
        logger.info(f"✅ Loaded {len(student_ids)} student IDs")
        
        return student_ids
        
    except Exception as e:
        logger.error(f"❌ Failed to load student list: {str(e)}")
        raise

@task
def predict_results_task(student_ids: list):
    """Task to predict results for all students"""
    logger = get_run_logger()
    logger.info(f"🎯 Predicting results for {len(student_ids)} students...")
    
    try:
        predictions = {}
        
        for student_id in student_ids:
            try:
                # Call predict_student_result for each student
                result = predict_student_result(student_id)
                predictions[student_id] = result
                
                if len(predictions) % 1000 == 0:
                    logger.info(f"📊 Processed {len(predictions)} predictions...")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to predict for student {student_id}: {str(e)}")
                predictions[student_id] = "Error"
        
        logger.info(f"✅ Completed predictions for {len(predictions)} students")
        return predictions
        
    except Exception as e:
        logger.error(f"❌ Prediction task failed: {str(e)}")
        raise

@task
def identify_at_risk_task(predictions: dict):
    """Task to identify at-risk students"""
    logger = get_run_logger()
    logger.info("🚨 Identifying at-risk students...")
    
    try:
        at_risk_ids = []
        
        # Filter for students predicted as 'Fail' or 'Withdrawn'
        at_risk_outcomes = ['Fail', 'Withdrawn', 'Withdrawal']
        
        for student_id, prediction in predictions.items():
            if prediction in at_risk_outcomes:
                at_risk_ids.append(student_id)
        
        logger.info(f"✅ Identified {len(at_risk_ids)} at-risk students")
        logger.info(f"📊 At-risk outcomes: {at_risk_outcomes}")
        
        return at_risk_ids
        
    except Exception as e:
        logger.error(f"❌ At-risk identification failed: {str(e)}")
        raise

@task
def run_mentor_analysis_task(student_id: int):
    """Task to run mentor analysis for a single student"""
    logger = get_run_logger()
    logger.info(f"🤖 Running mentor analysis for student {student_id}...")
    
    try:
        # Initialize the agentic AI pipeline
        agent = AgenticAIPipeline()
        
        # Call run_student_analysis
        report = agent.run_student_analysis(student_id)
        
        logger.info(f"✅ Mentor analysis completed for student {student_id}")
        
        return {
            "student_id": student_id,
            "report": report,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Mentor analysis failed for student {student_id}: {str(e)}")
        return {
            "student_id": student_id,
            "report": f"Analysis failed: {str(e)}",
            "timestamp": pd.Timestamp.now().isoformat(),
            "error": True
        }

@task
def save_reports_task(reports: list):
    """Task to save mentor reports to JSON file"""
    logger = get_run_logger()
    logger.info(f"💾 Saving {len(reports)} mentor reports...")
    
    try:
        # Ensure results directory exists
        os.makedirs("7_Results", exist_ok=True)
        
        # Prepare reports data
        reports_data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_reports": len(reports),
            "reports": reports
        }
        
        # Save to JSON file
        output_path = "7_Results/mentor_reports.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(reports_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Reports saved to {output_path}")
        
        # Log summary statistics
        successful_reports = len([r for r in reports if not r.get('error', False)])
        failed_reports = len([r for r in reports if r.get('error', False)])
        
        logger.info(f"📊 Report Summary: {successful_reports} successful, {failed_reports} failed")
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Failed to save reports: {str(e)}")
        raise

@flow(name="AI Academic Mentor Pipeline")
def academic_monitoring_pipeline():
    """Main Prefect flow for academic monitoring and mentoring"""
    logger = get_run_logger()
    logger.info("🚀 Starting AI Academic Mentor Monitoring Pipeline...")
    
    try:
        # Step 1: Retrain the model
        model_task = retrain_model_task()
        
        # Step 2: Load student list (wait for model retraining to complete)
        student_ids = load_student_list_task(wait_for=[model_task])
        
        # Step 3: Predict results for all students
        predictions = predict_results_task(student_ids)
        
        # Step 4: Identify at-risk students
        at_risk_ids = identify_at_risk_task(predictions)
        
        # Step 5: Run mentor analysis for at-risk students (parallel processing)
        reports = run_mentor_analysis_task.map(at_risk_ids)
        
        # Step 6: Save all reports
        output_path = save_reports_task(reports)
        
        logger.info("🎉 AI Academic Mentor Pipeline completed successfully!")
        
        return {
            "status": "success",
            "total_students": len(student_ids),
            "at_risk_students": len(at_risk_ids),
            "reports_generated": len(reports),
            "output_path": output_path
        }
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the pipeline
    print("🚀 Starting AI Academic Mentor Monitoring Pipeline...")
    result = academic_monitoring_pipeline()
    print(f"📊 Pipeline Result: {result}")
