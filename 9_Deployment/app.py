"""
Streamlit Demo Application for AI Academic Mentor
Main deployment interface for the academic mentoring system
"""

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import os

# Suppress Streamlit file watcher warnings
os.environ['STREAMLIT_WATCHER_LOG_LEVEL'] = 'ERROR'

# Add src code to path
sys.path.append(str(Path(__file__).parent.parent / "4_Src_Code"))

# Import the LangGraph agent function
from agentic_ai_pipeline import AgenticAIPipeline

# Page configuration
st.set_page_config(
    page_title="AI Academic Mentor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Card styling */
    .student-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    
    .info-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    /* Metric card */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Title styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 50px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Report container */
    .report-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 10px;
    }
    
    .stSelectbox label {
        color: #333 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    .stSelectbox input {
        color: #333 !important;
        font-size: 1.1rem !important;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background-color: white !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        color: #333 !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    
    /* Dropdown menu items */
    [role="option"] {
        color: #333 !important;
        font-size: 1rem !important;
    }
    
    [role="option"]:hover {
        background-color: #667eea !important;
        color: white !important;
    }
    
    /* Selected value in dropdown */
    [data-baseweb="select"] span {
        color: #333 !important;
    }
    
    /* Dropdown icon/arrow */
    .stSelectbox svg {
        fill: #333 !important;
    }
    
    /* Input placeholder */
    .stSelectbox input::placeholder {
        color: #999 !important;
    }
    
    /* Listbox (dropdown menu) */
    [data-baseweb="popover"] {
        background-color: white !important;
    }
    
    [data-baseweb="menu"] {
        background-color: white !important;
    }
    
    /* Menu items in dropdown list */
    [data-baseweb="menu"] li {
        color: #333 !important;
        background-color: white !important;
    }
    
    [data-baseweb="menu"] li:hover {
        background-color: #667eea !important;
        color: white !important;
    }
    
    /* Success/Error message styling */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

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
        
        # Header with student ID
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; 
                    border-radius: 15px; 
                    text-align: center; 
                    margin-bottom: 2rem;
                    box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);">
            <h2 style="color: white; 
                       margin: 0; 
                       font-size: 2rem;
                       font-weight: 700;
                       text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                👤 Student Profile
            </h2>
            <p style="color: white; 
                      font-size: 1.4rem; 
                      font-weight: 600;
                      margin: 0.8rem 0 0 0;
                      text-shadow: 1px 1px 3px rgba(0,0,0,0.3);">
                ID: {student_id}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            edu_value = student_row.get('highest_education', 'Unknown')
            # Shorten education labels for better display
            edu_short = {
                'Lower Than A Level': 'Below A-Level',
                'A Level or Equivalent': 'A-Level',
                'HE Qualification': 'HE Qual',
                'Post Graduate Qualification': 'Post-Grad'
            }.get(edu_value, edu_value)
            st.metric(
                label="🎓 Education",
                value=edu_short,
                delta=None
            )
        
        with col2:
            st.metric(
                label="📚 Credits",
                value=int(student_row.get('studied_credits', 0)),
                delta=None
            )
        
        with col3:
            st.metric(
                label="🔄 Attempts",
                value=int(student_row.get('num_of_prev_attempts', 0)),
                delta="First try" if int(student_row.get('num_of_prev_attempts', 0)) == 0 else None
            )
        
        with col4:
            result = student_row.get('final_result', 'Unknown')
            result_color = {
                'Pass': '🟢',
                'Fail': '🔴', 
                'Distinction': '🌟',
                'Withdrawn': '⚠️'
            }.get(result, '⚪')
            st.metric(
                label="📊 Result",
                value=f"{result_color} {result}"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Detailed Information in Cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="background: white; 
                        padding: 1.5rem; 
                        border-radius: 12px; 
                        box-shadow: 0 4px 15px rgba(0,0,0,0.12); 
                        height: 100%;
                        border-top: 4px solid #667eea;">
                <h4 style="color: #667eea; 
                           margin-top: 0; 
                           font-size: 1.3rem;
                           font-weight: 700;
                           margin-bottom: 1rem;">
                    🧑 Personal Information
                </h4>
                <p style="color: #333; font-size: 1rem; margin: 0.7rem 0; line-height: 1.6;">
                    <strong style="color: #555;">Gender:</strong> {student_row.get('gender', 'Unknown')}
                </p>
                <p style="color: #333; font-size: 1rem; margin: 0.7rem 0; line-height: 1.6;">
                    <strong style="color: #555;">Age Band:</strong> {student_row.get('age_band', 'Unknown')}
                </p>
                <p style="color: #333; font-size: 1rem; margin: 0.7rem 0; line-height: 1.6;">
                    <strong style="color: #555;">Region:</strong> {student_row.get('region', 'Unknown')}
                </p>
                <p style="color: #333; font-size: 1rem; margin: 0.7rem 0; line-height: 1.6;">
                    <strong style="color: #555;">Disability:</strong> {student_row.get('disability', 'N/A')}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: white; 
                        padding: 1.5rem; 
                        border-radius: 12px; 
                        box-shadow: 0 4px 15px rgba(0,0,0,0.12); 
                        height: 100%;
                        border-top: 4px solid #764ba2;">
                <h4 style="color: #764ba2; 
                           margin-top: 0; 
                           font-size: 1.3rem;
                           font-weight: 700;
                           margin-bottom: 1rem;">
                    📖 Academic Background
                </h4>
                <p style="color: #333; font-size: 1rem; margin: 0.7rem 0; line-height: 1.6;">
                    <strong style="color: #555;">Education:</strong> {student_row.get('highest_education', 'Unknown')}
                </p>
                <p style="color: #333; font-size: 1rem; margin: 0.7rem 0; line-height: 1.6;">
                    <strong style="color: #555;">Attempts:</strong> {student_row.get('num_of_prev_attempts', 'Unknown')}
                </p>
                <p style="color: #333; font-size: 1rem; margin: 0.7rem 0; line-height: 1.6;">
                    <strong style="color: #555;">Credits:</strong> {student_row.get('studied_credits', 'Unknown')}
                </p>
                <p style="color: #333; font-size: 1rem; margin: 0.7rem 0; line-height: 1.6;">
                    <strong style="color: #555;">Module:</strong> {student_row.get('code_module', 'N/A')}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            result = student_row.get('final_result', 'Unknown')
            result_color = {
                'Pass': '#28a745',
                'Fail': '#dc3545',
                'Distinction': '#ffc107',
                'Withdrawn': '#6c757d'
            }.get(result, '#333')
            st.markdown(f"""
            <div style="background: white; 
                        padding: 1.5rem; 
                        border-radius: 12px; 
                        box-shadow: 0 4px 15px rgba(0,0,0,0.12); 
                        height: 100%;
                        border-top: 4px solid #667eea;">
                <h4 style="color: #667eea; 
                           margin-top: 0; 
                           font-size: 1.3rem;
                           font-weight: 700;
                           margin-bottom: 1rem;">
                    📍 Socioeconomic Context
                </h4>
                <p style="color: #333; font-size: 1rem; margin: 0.7rem 0; line-height: 1.6;">
                    <strong style="color: #555;">IMD Band:</strong> {student_row.get('imd_band', 'Unknown')}
                </p>
                <p style="color: #333; font-size: 1rem; margin: 0.7rem 0; line-height: 1.6;">
                    <strong style="color: #555;">Region:</strong> {student_row.get('region', 'Unknown')}
                </p>
                <p style="color: #333; font-size: 1rem; margin: 0.7rem 0; line-height: 1.6;">
                    <strong style="color: #555;">Presentation:</strong> {student_row.get('code_presentation', 'N/A')}
                </p>
                <p style="color: #333; font-size: 1rem; margin: 0.7rem 0; line-height: 1.6;">
                    <strong style="color: #555;">Final Result:</strong> 
                    <span style="color: {result_color}; font-weight: 700;">{result}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Create a simple visualization
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Quick stats visualization
        if 'studied_credits' in student_row:
            fig = go.Figure()
            
            categories = ['Credits', 'Attempts', 'IMD Score']
            values = [
                int(student_row.get('studied_credits', 0)) / 10,  # Scale down for comparison
                int(student_row.get('num_of_prev_attempts', 0)) * 10,  # Scale up for visibility
                int(student_row.get('imd_band', '50-75%').split('-')[0]) if isinstance(student_row.get('imd_band'), str) else 50
            ]
            
            fig.add_trace(go.Bar(
                x=categories,
                y=values,
                marker=dict(
                    color=['#667eea', '#764ba2', '#a8a3ea'],
                    line=dict(color='white', width=2)
                ),
                text=[f'{v:.0f}' for v in values],
                textposition='outside',
                textfont=dict(size=16, color='#000000', family='Arial Black, sans-serif'),
            ))
            
            fig.update_layout(
                title={
                    'text': "Student Quick Stats Overview",
                    'font': {'size': 22, 'color': '#000000', 'family': 'Arial Black, sans-serif'},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                template="plotly_white",
                height=350,
                showlegend=False,
                plot_bgcolor='rgba(255,255,255,1)',
                paper_bgcolor='rgba(255,255,255,1)',
                margin=dict(t=70, b=50, l=50, r=50),
                font=dict(size=14, color='#000000', family='Arial, sans-serif'),
                xaxis=dict(
                    title='',
                    tickfont=dict(size=14, color='#000000', family='Arial Black, sans-serif'),
                    showgrid=False
                ),
                yaxis=dict(
                    title=dict(
                        text='Value',
                        font=dict(size=14, color='#000000', family='Arial Black, sans-serif')
                    ),
                    tickfont=dict(size=12, color='#000000', family='Arial, sans-serif'),
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Error displaying student info: {e}")

def main():
    """Main demo application function"""
    
    # Sidebar with information and settings
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h1 style="font-size: 3rem; margin: 0;">🎓</h1>
            <h2 style="margin: 0.5rem 0;">AI Academic Mentor</h2>
            <p style="font-size: 0.9rem; opacity: 0.8;">Powered by Advanced AI</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        ### 📊 About This System
        
        This AI-powered academic mentoring system uses:
        
        - 🤖 **Groq LLM** for intelligent analysis
        - 🧠 **LangGraph** for agentic workflows
        - 📚 **RAG System** for knowledge retrieval
        - 📈 **ML Models** for performance prediction
        
        ### 🎯 How It Works
        
        1. **Select** a student from the dropdown
        2. **Review** their profile and stats
        3. **Generate** AI-powered mentor report
        4. **Get** personalized recommendations
        
        ### 🔧 System Status
        """)
        
        st.success("✅ AI Model: Active")
        st.success("✅ RAG System: Ready")
        st.success("✅ ML Pipeline: Loaded")
        
    st.markdown("---")
    
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; opacity: 0.7;">
            <p>Dataset: OULAD</p>
            <p>Version 1.0.0</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    # Animated header with better contrast
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; 
                background: rgba(255,255,255,0.1); 
                border-radius: 20px; 
                margin-bottom: 2rem;">
        <h1 style="font-size: 3.5rem; font-weight: 800; 
                   color: white;
                   margin: 0; 
                   text-shadow: 3px 3px 6px rgba(0,0,0,0.4);">
            🤖 AI Academic Mentor
        </h1>
        <p style="font-size: 1.5rem; 
                  color: white; 
                  font-weight: 600;
                  margin-top: 1rem; 
                  text-shadow: 2px 2px 5px rgba(0,0,0,0.5);
                  letter-spacing: 0.5px;">
            Empowering Students with Intelligent Insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load student data with progress indicator
    with st.spinner('📊 Loading student database...'):
    df = load_student_data()
    
    if df is None:
        st.stop()
    
    # Display data stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; 
                    box-shadow: 0 4px 20px rgba(0,0,0,0.15); text-align: center;
                    border-left: 5px solid #667eea;">
            <h3 style="color: #667eea; margin: 0; font-size: 2.5rem; font-weight: 700;">{}</h3>
            <p style="color: #333; margin: 0.5rem 0 0 0; font-size: 1rem; font-weight: 600;">Total Students</p>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        pass_rate = (df['final_result'] == 'Pass').sum() / len(df) * 100 if 'final_result' in df.columns else 0
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; 
                    box-shadow: 0 4px 20px rgba(0,0,0,0.15); text-align: center;
                    border-left: 5px solid #28a745;">
            <h3 style="color: #28a745; margin: 0; font-size: 2.5rem; font-weight: 700;">{:.1f}%</h3>
            <p style="color: #333; margin: 0.5rem 0 0 0; font-size: 1rem; font-weight: 600;">Pass Rate</p>
        </div>
        """.format(pass_rate), unsafe_allow_html=True)
    
    with col3:
        avg_credits = df['studied_credits'].mean() if 'studied_credits' in df.columns else 0
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; 
                    box-shadow: 0 4px 20px rgba(0,0,0,0.15); text-align: center;
                    border-left: 5px solid #764ba2;">
            <h3 style="color: #764ba2; margin: 0; font-size: 2.5rem; font-weight: 700;">{:.0f}</h3>
            <p style="color: #333; margin: 0.5rem 0 0 0; font-size: 1rem; font-weight: 600;">Avg Credits</p>
        </div>
        """.format(avg_credits), unsafe_allow_html=True)
    
    with col4:
        unique_courses = df['code_module'].nunique() if 'code_module' in df.columns else 0
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; 
                    box-shadow: 0 4px 20px rgba(0,0,0,0.15); text-align: center;
                    border-left: 5px solid #ffa502;">
            <h3 style="color: #ffa502; margin: 0; font-size: 2.5rem; font-weight: 700;">{}</h3>
            <p style="color: #333; margin: 0.5rem 0 0 0; font-size: 1rem; font-weight: 600;">Courses</p>
        </div>
        """.format(unique_courses), unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Student selection in a beautiful container
    st.markdown("""
    <div style="background: white; 
                padding: 2.5rem; 
                border-radius: 15px; 
                box-shadow: 0 8px 32px rgba(0,0,0,0.15); 
                margin-bottom: 2rem;
                border-top: 5px solid #667eea;">
        <h2 style="color: #667eea; 
                   text-align: center; 
                   margin-top: 0;
                   font-size: 2rem;
                   font-weight: 700;">
            🎯 Select Student for Analysis
        </h2>
        <p style="text-align: center; 
                  color: #666; 
                  font-size: 1.1rem;
                  margin-bottom: 1.5rem;">
            Choose a student to generate personalized AI-powered insights
        </p>
    """, unsafe_allow_html=True)
    
    # Create searchable dropdown
    selected_id = st.selectbox(
        'Student ID:', 
        options=sorted(df['id_student'].unique()),
        help="🔍 Choose a student ID to generate a personalized AI mentor report",
        label_visibility="collapsed"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display selected student's demographic info
    if selected_id:
        display_student_info(selected_id, df)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Add Generate Mentor Report button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_button = st.button(
                '🚀 Generate AI Mentor Report', 
                type="primary", 
                use_container_width=True,
                help="Click to generate a comprehensive AI-powered analysis and recommendations"
            )
        
        if generate_button:
            # Animated progress
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            progress_text.markdown("""
            <h3 style="color: #667eea; font-size: 1.5rem; font-weight: 700; text-align: center;">
                🤖 AI Mentor is analyzing...
            </h3>
            """, unsafe_allow_html=True)
            progress_bar.progress(20)
            
                try:
                    # Initialize the LangGraph agent
                progress_text.markdown("""
                <h3 style="color: #764ba2; font-size: 1.5rem; font-weight: 700; text-align: center;">
                    🧠 Initializing AI agents...
                </h3>
                """, unsafe_allow_html=True)
                    agent = AgenticAIPipeline()
                progress_bar.progress(40)
                    
                    # Call the LangGraph function
                progress_text.markdown("""
                <h3 style="color: #28a745; font-size: 1.5rem; font-weight: 700; text-align: center;">
                    📊 Analyzing student data...
                </h3>
                """, unsafe_allow_html=True)
                progress_bar.progress(60)
                
                    final_report = agent.run_student_analysis(selected_id)
                progress_bar.progress(100)
                
                # Clear progress indicators
                progress_text.empty()
                progress_bar.empty()
                
                # Display the result in a beautiful container
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                <div style="background: white; 
                            padding: 2.5rem; 
                            border-radius: 15px; 
                            box-shadow: 0 8px 32px rgba(0,0,0,0.15);
                            border-top: 5px solid #667eea;">
                    <h2 style="color: #667eea; 
                               text-align: center; 
                               margin-top: 0;
                               font-size: 2.2rem;
                               font-weight: 700;
                               margin-bottom: 1.5rem;">
                        📋 AI Mentor Report
                    </h2>
                    <div style="color: #333; font-size: 1.05rem; line-height: 1.8;">
                """, unsafe_allow_html=True)
                
                    st.markdown(final_report)
                st.markdown("</div></div>", unsafe_allow_html=True)
                
                # Download button for report
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.download_button(
                        label="📥 Download Report",
                        data=final_report,
                        file_name=f"mentor_report_{selected_id}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                    
                except Exception as e:
                progress_text.empty()
                progress_bar.empty()
                
                    st.error(f"❌ Error generating mentor report: {e}")
    
                with st.expander("💡 Troubleshooting Tips"):
    st.markdown("""
                    1. ✅ Ensure Groq API key is configured in `config.py`
                    2. ✅ Check that all dependencies are installed
                    3. ✅ Verify that ML models are trained
                    4. ✅ Ensure RAG knowledge base is built
                    """)
    
    # Footer with system information
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 15px; 
                box-shadow: 0 8px 32px rgba(0,0,0,0.1); margin-top: 3rem;">
        <h3 style="color: #667eea; text-align: center; margin-top: 0;">🔧 System Architecture</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1.5rem;">
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem;">🤖</div>
                <strong>AI Model</strong>
                <p style="font-size: 0.9rem; color: #666;">Groq Llama 3.1 8B</p>
            </div>
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem;">🧠</div>
                <strong>Framework</strong>
                <p style="font-size: 0.9rem; color: #666;">LangGraph Agentic AI</p>
            </div>
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem;">📚</div>
                <strong>Knowledge Base</strong>
                <p style="font-size: 0.9rem; color: #666;">ChromaDB RAG</p>
            </div>
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem;">📈</div>
                <strong>ML Pipeline</strong>
                <p style="font-size: 0.9rem; color: #666;">GradientBoosting</p>
            </div>
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem;">📊</div>
                <strong>Dataset</strong>
                <p style="font-size: 0.9rem; color: #666;">OULAD</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
