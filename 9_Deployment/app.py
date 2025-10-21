"""
Streamlit Demo Application for AI Academic Mentor
Main deployment interface for the academic mentoring system
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src code to path
sys.path.append(str(Path(__file__).parent.parent / "4_Src_Code"))

# Import the LangGraph agent function
from agentic_ai_pipeline import AgenticAIPipeline

# Page configuration
st.set_page_config(
    page_title="AI Academic Mentor",
    page_icon="🎓",
    layout="wide"
)

def load_student_data():
    """Load the original, un-encoded student data"""
    try:
        # Try to load from 2_Data/raw first, then fallback to Data_csv's
        data_paths = [
            "2_Data/raw/studentInfo.csv",
            "Data_csv's/studentInfo.csv"
        ]
        
        for path in data_paths:
            if Path(path).exists():
                df = pd.read_csv(path)
                return df
        
        st.error("❌ Student data file not found. Please ensure studentInfo.csv exists.")
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading student data: {e}")
        return None

def display_student_info(student_id, df):
    """Display demographic information for selected student"""
    try:
        student_row = df[df['id_student'] == student_id].iloc[0]
        
        st.subheader("👤 Student Information")
        
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Student ID:** {student_id}")
            st.write(f"**Gender:** {student_row.get('gender', 'Unknown')}")
            st.write(f"**Region:** {student_row.get('region', 'Unknown')}")
            st.write(f"**Age Band:** {student_row.get('age_band', 'Unknown')}")
        
        with col2:
            st.write(f"**Education Level:** {student_row.get('highest_education', 'Unknown')}")
            st.write(f"**IMD Band:** {student_row.get('imd_band', 'Unknown')}")
            st.write(f"**Disability:** {student_row.get('disability', 'Unknown')}")
            st.write(f"**Previous Attempts:** {student_row.get('num_of_prev_attempts', 'Unknown')}")
        
        with col3:
            st.write(f"**Studied Credits:** {student_row.get('studied_credits', 'Unknown')}")
            st.write(f"**Final Result:** {student_row.get('final_result', 'Unknown')}")
            
    except Exception as e:
        st.error(f"❌ Error displaying student info: {e}")

def main():
    """Main demo application function"""
    
    # Set page title
    st.title('🤖 AI Academic Mentor')
    st.markdown("---")
    
    # Load student data
    st.write("📊 Loading student data...")
    df = load_student_data()
    
    if df is None:
        st.stop()
    
    st.success(f"✅ Loaded data for {len(df)} students")
    
    # Create student selection dropdown
    st.subheader("🎯 Select Student for Analysis")
    selected_id = st.selectbox(
        'Select Student ID:', 
        df['id_student'].unique(),
        help="Choose a student ID to generate a personalized mentor report"
    )
    
    # Display selected student's demographic info
    if selected_id:
        display_student_info(selected_id, df)
        
        st.markdown("---")
        
        # Add Generate Mentor Report button
        if st.button('🚀 Generate Mentor Report', type="primary", use_container_width=True):
            
            # Show spinner with Groq branding
            with st.spinner('🤖 AI Mentor is analyzing... (Powered by Groq)'):
                try:
                    # Initialize the LangGraph agent
                    agent = AgenticAIPipeline()
                    
                    # Call the LangGraph function
                    final_report = agent.run_student_analysis(selected_id)
                    
                    # Display the result using markdown
                    st.markdown("---")
                    st.subheader("📋 AI Mentor Report")
                    st.markdown(final_report)
                    
                except Exception as e:
                    st.error(f"❌ Error generating mentor report: {e}")
                    st.write("💡 **Troubleshooting Tips:**")
                    st.write("1. Ensure Groq API key is configured")
                    st.write("2. Check that all dependencies are installed")
                    st.write("3. Verify that ML models are trained")
                    st.write("4. Ensure RAG knowledge base is built")
    
    # Add footer information
    st.markdown("---")
    st.markdown("""
    ### 🔧 System Information
    - **AI Model:** Groq Llama 3.1 8B Instant
    - **Framework:** LangGraph Agentic AI
    - **Knowledge Base:** ChromaDB RAG System
    - **ML Pipeline:** GradientBoostingClassifier
    - **Dataset:** OULAD (Open University Learning Analytics Dataset)
    """)

if __name__ == "__main__":
    main()
