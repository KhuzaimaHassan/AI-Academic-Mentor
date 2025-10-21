"""
Agentic AI Pipeline Module for AI Academic Mentor
Implements LangGraph-based agentic AI system for academic mentoring using Groq
"""

import os
import pandas as pd
from typing import Dict, List, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import json
from pathlib import Path

# Import our helper functions
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "utils"))

try:
    from utils.helpers import predict_student_result
    from rag_integration import RAGIntegration
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback imports
    try:
        from helpers import predict_student_result
        from rag_integration import RAGIntegration
    except ImportError:
        print("Failed to import helper modules")
        predict_student_result = None
        RAGIntegration = None

# Define the Agent State
class AgentState(TypedDict):
    student_id: int
    student_info: dict
    predicted_result: str
    root_causes: str
    rag_query: str
    intervention_advice: str
    final_report: str

class AgenticAIPipeline:
    """Implements LangGraph-based agentic AI system for academic mentoring"""
    
    def __init__(self, groq_api_key: str = None):
        # Initialize the LLM
        if groq_api_key:
            api_key = groq_api_key
        else:
            # Try multiple sources for API key (in order of priority)
            api_key = None
            
            # 1. Try Streamlit secrets (for Streamlit Cloud deployment)
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and 'groq' in st.secrets:
                    api_key = st.secrets['groq']['GROQ_API_KEY']
                    print("✅ Using Groq API key from Streamlit secrets")
            except Exception:
                pass
            
            # 2. Try config file (for local development)
            if not api_key:
                try:
                    parent_dir = Path(__file__).parent.parent
                    sys.path.insert(0, str(parent_dir))
                    from config import GROQ_API_KEY
                    if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
                        api_key = GROQ_API_KEY
                        print("✅ Using Groq API key from config.py")
                except ImportError:
                    pass
            
            # 3. Try environment variable (fallback)
            if not api_key:
                api_key = os.getenv('GROQ_API_KEY')
                if api_key:
                    print("✅ Using Groq API key from environment variable")
        
        if api_key and api_key != "your_groq_api_key_here":
            try:
                self.llm = ChatGroq(
                    groq_api_key=api_key,
                    model="llama-3.1-8b-instant",
                    temperature=0.7
                )
                print("✅ Groq LLM initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize Groq LLM: {e}")
                self.llm = None
        else:
            print("⚠️ Groq API key not configured. Please add it via Streamlit secrets or config.py")
            self.llm = None
        
        # Initialize RAG system
        self.rag_system = RAGIntegration()
        
        # Build the LangGraph workflow
        self.app = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow with nodes and edges"""
        
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add all nodes
        workflow.add_node('get_info', self.get_student_info)
        workflow.add_node('predict', self.predict_performance)
        workflow.add_node('analyze', self.analyze_root_cause)
        workflow.add_node('generate_query', self.generate_rag_query)
        workflow.add_node('get_advice', self.get_rag_advice)
        workflow.add_node('compile_report', self.compile_final_report)
        
        # Set entry point
        workflow.set_entry_point('get_info')
        
        # Add edges
        workflow.add_edge('get_info', 'predict')
        workflow.add_edge('predict', 'analyze')
        workflow.add_edge('analyze', 'generate_query')
        workflow.add_edge('generate_query', 'get_advice')
        workflow.add_edge('get_advice', 'compile_report')
        workflow.add_edge('compile_report', END)
        
        # Compile the graph
        return workflow.compile()
    
    def get_student_info(self, state: AgentState) -> AgentState:
        """Node: Get student information from CSV"""
        try:
            student_id = state["student_id"]
            print(f"🔍 Getting student info for ID: {student_id}")
            
            # Load studentInfo.csv
            student_info_path = Path("Data_csv's/studentInfo.csv")
            student_df = pd.read_csv(student_info_path)
            
            # Find the student's row
            student_row = student_df[student_df['id_student'] == student_id]
            
            if student_row.empty:
                raise ValueError(f"Student ID {student_id} not found in dataset")
            
            # Convert to dictionary
            student_info = student_row.iloc[0].to_dict()
            
            print(f"✅ Retrieved student info for {student_id}")
            
            return {
                **state,
                "student_info": student_info
            }
            
        except Exception as e:
            print(f"❌ Error getting student info: {e}")
            return {
                **state,
                "student_info": {"error": str(e)}
            }
    
    def predict_performance(self, state: AgentState) -> AgentState:
        """Node: Predict student performance using ML model"""
        try:
            student_id = state["student_id"]
            print(f"🎯 Predicting performance for student {student_id}")
            
            # Use our helper function to predict student result
            predicted_result = predict_student_result(student_id)
            
            if predicted_result is None:
                predicted_result = "Unable to predict - model not available"
            
            print(f"✅ Predicted result: {predicted_result}")
            
            return {
                **state,
                "predicted_result": predicted_result
            }
            
        except Exception as e:
            print(f"❌ Error predicting performance: {e}")
            return {
                **state,
                "predicted_result": f"Prediction failed: {str(e)}"
            }
    
    def analyze_root_cause(self, state: AgentState) -> AgentState:
        """Node: Analyze root causes using LLM"""
        try:
            student_info = state["student_info"]
            predicted_result = state["predicted_result"]
            
            print("🧠 Analyzing root causes...")
            
            # Create prompt for root cause analysis
            prompt = f"""
            You are an expert academic advisor analyzing a student's situation.
            
            Student Information:
            - Final Result: {predicted_result}
            - Previous Attempts: {student_info.get('num_of_prev_attempts', 'Unknown')}
            - Disability Status: {student_info.get('disability', 'Unknown')}
            - Gender: {student_info.get('gender', 'Unknown')}
            - Region: {student_info.get('region', 'Unknown')}
            - Education Level: {student_info.get('highest_education', 'Unknown')}
            - Age Band: {student_info.get('age_band', 'Unknown')}
            - IMD Band: {student_info.get('imd_band', 'Unknown')}
            - Studied Credits: {student_info.get('studied_credits', 'Unknown')}
            
            Based on this information, identify the likely root causes for the student's academic outcome.
            Focus on factors like:
            - Academic preparation and background
            - Personal circumstances
            - Study patterns and engagement
            - Support systems and resources
            
            Provide a concise analysis of the key factors influencing this student's performance.
            """
            
            # Get LLM response
            if self.llm is None:
                root_causes = "LLM not available - using fallback analysis. Student shows typical patterns based on their demographic and academic history."
            else:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                root_causes = response.content
            
            print(f"✅ Root cause analysis completed")
            
            return {
                **state,
                "root_causes": root_causes
            }
            
        except Exception as e:
            print(f"❌ Error analyzing root causes: {e}")
            return {
                **state,
                "root_causes": f"Analysis failed: {str(e)}"
            }
    
    def generate_rag_query(self, state: AgentState) -> AgentState:
        """Node: Generate RAG query based on root causes"""
        try:
            root_causes = state["root_causes"]
            predicted_result = state["predicted_result"]
            
            print("🔍 Generating RAG query...")
            
            # Create prompt for query generation
            prompt = f"""
            You are creating a search query for an academic intervention knowledge base.
            
            Student's Predicted Result: {predicted_result}
            Root Causes Identified: {root_causes}
            
            Based on this analysis, generate a specific search query that would help find relevant 
            academic intervention advice. Focus on:
            - Time management strategies
            - Study techniques
            - Academic support methods
            - Overcoming specific challenges
            
            Generate a concise, targeted search query (2-5 words) that would retrieve relevant intervention advice.
            Examples: "time management students", "overcoming procrastination", "study strategies"
            """
            
            # Get LLM response
            if self.llm is None:
                rag_query = "academic success strategies"
            else:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                rag_query = response.content.strip()
            
            print(f"✅ Generated RAG query: '{rag_query}'")
            
            return {
                **state,
                "rag_query": rag_query
            }
            
        except Exception as e:
            print(f"❌ Error generating RAG query: {e}")
            return {
                **state,
                "rag_query": f"Query generation failed: {str(e)}"
            }
    
    def get_rag_advice(self, state: AgentState) -> AgentState:
        """Node: Get intervention advice from RAG system"""
        try:
            rag_query = state["rag_query"]
            
            print(f"📚 Getting RAG advice for query: '{rag_query}'")
            
            # Use RAG system to get intervention advice
            intervention_advice = self.rag_system.get_intervention_advice(rag_query)
            
            print(f"✅ Retrieved intervention advice")
            
            return {
                **state,
                "intervention_advice": intervention_advice
            }
            
        except Exception as e:
            print(f"❌ Error getting RAG advice: {e}")
            return {
                **state,
                "intervention_advice": f"RAG advice retrieval failed: {str(e)}"
            }
    
    def compile_final_report(self, state: AgentState) -> AgentState:
        """Node: Compile final empathetic report"""
        try:
            student_info = state["student_info"]
            predicted_result = state["predicted_result"]
            root_causes = state["root_causes"]
            intervention_advice = state["intervention_advice"]
            
            print("📝 Compiling final report...")
            
            # Create prompt for final report
            prompt = f"""
            You are an empathetic academic mentor writing a personalized report for a student.
            
            Student Information:
            - Student ID: {state['student_id']}
            - Predicted Result: {predicted_result}
            - Root Causes: {root_causes}
            - Intervention Advice: {intervention_advice}
            
            Write a comprehensive, empathetic, and actionable final report that:
            1. Acknowledges the student's situation with empathy
            2. Explains the analysis in accessible terms
            3. Provides specific, actionable recommendations
            4. Offers encouragement and support
            5. Includes concrete next steps
            
            Make the tone supportive, professional, and motivating. Focus on helping the student succeed.
            """
            
            # Get LLM response
            if self.llm is None:
                final_report = f"""
# 📋 AI Academic Mentor Report

## Student Analysis Summary
**Student ID:** {state['student_id']}  
**Predicted Result:** {predicted_result}

## Key Findings
{root_causes}

## Intervention Recommendations
{intervention_advice}

## Next Steps
1. Review the intervention recommendations above
2. Consider implementing time management strategies
3. Seek additional academic support if needed
4. Maintain regular study schedules

*Note: This report was generated using fallback analysis due to LLM unavailability.*
"""
            else:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                final_report = response.content
            
            print(f"✅ Final report compiled")
            
            return {
                **state,
                "final_report": final_report
            }
            
        except Exception as e:
            print(f"❌ Error compiling final report: {e}")
            return {
                **state,
                "final_report": f"Report compilation failed: {str(e)}"
            }
    
    def run_student_analysis(self, student_id: int) -> str:
        """
        Main function to run complete student analysis
        
        Args:
            student_id: The student ID to analyze
            
        Returns:
            Final report string
        """
        try:
            print(f"🚀 Starting student analysis for ID: {student_id}")
            
            # Initialize state
            initial_state = {
                "student_id": student_id,
                "student_info": {},
                "predicted_result": "",
                "root_causes": "",
                "rag_query": "",
                "intervention_advice": "",
                "final_report": ""
            }
            
            # Run the agentic workflow
            final_state = self.app.invoke(initial_state)
            
            print(f"✅ Student analysis completed for ID: {student_id}")
            
            return final_state.get("final_report", "Analysis failed to complete.")
            
        except Exception as e:
            error_msg = f"❌ Student analysis failed: {str(e)}"
            print(error_msg)
            return error_msg
