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

# Modern, Professional CSS - Fully Compatible with Light & Dark Themes
st.markdown("""
<style>
    /* ============================================
       CORE BACKGROUND & LAYOUT
       ============================================ */
    
    /* Force gradient background across all themes */
    .main, .block-container, .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        background-attachment: fixed !important;
    }
    
    /* Ensure proper spacing */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    /* ============================================
       TYPOGRAPHY - Enhanced Readability
       ============================================ */
    
    /* Global font improvements */
    * {
        font-family: 'Inter', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Headings on gradient background - white with strong shadow */
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.6), 0px 0px 20px rgba(0,0,0,0.4) !important;
        letter-spacing: -0.5px;
    }
    
    /* ============================================
       SIDEBAR - Modern Purple Gradient
       ============================================ */
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%) !important;
        box-shadow: 4px 0 15px rgba(0,0,0,0.1);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.3) !important;
        margin: 1.5rem 0 !important;
    }
    
    /* ============================================
       BUTTONS - Purple Gradient with Hover
       ============================================ */
    
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2.5rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border-radius: 50px !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        text-transform: none !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6) !important;
        background: linear-gradient(135deg, #7b8ff0 0%, #8a5fb8 100%) !important;
    }
    
    .stButton>button:active, .stDownloadButton>button:active {
        transform: translateY(-1px) !important;
    }
    
    /* ============================================
       SELECT BOX / DROPDOWN - White with Dark Text
       ============================================ */
    
    .stSelectbox label {
        color: white !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.5) !important;
        margin-bottom: 0.75rem !important;
    }
    
    .stSelectbox > div > div {
        background-color: white !important;
        border-radius: 12px !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important;
        transition: all 0.3s ease !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: rgba(255,255,255,0.6) !important;
        box-shadow: 0 6px 25px rgba(0,0,0,0.2) !important;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background-color: white !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        color: #2d3748 !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        padding: 0.75rem 1rem !important;
    }
    
    .stSelectbox input {
        color: #2d3748 !important;
        font-size: 1.1rem !important;
    }
    
    /* Dropdown arrow */
    .stSelectbox svg {
        fill: #667eea !important;
        width: 24px !important;
        height: 24px !important;
    }
    
    /* Dropdown menu */
    [data-baseweb="popover"], [data-baseweb="menu"] {
        background-color: white !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.2) !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Target the option list item */
    [role="option"] {
        color: #2d3748 !important;
        font-size: 1.05rem !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.2s ease !important;
    }

    /* NEW: Force the text *inside* the option to be dark */
    [role="option"] div {
        color: #2d3748 !important;
    }
    
    /* Target the hover state */
    [role="option"]:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 8px !important;
        margin: 0 0.5rem !important;
    }

    /* NEW: Force the text *inside* the hovered option to be white */
    [role="option"]:hover div {
        color: white !important;
    }
    
    /* ============================================
       METRICS / INFO CARDS - Clean White Cards
       ============================================ */
    
    .stMetric {
        background: white !important;
        padding: 1.5rem !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.12) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stMetric:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.18) !important;
    }
    
    .stMetric label {
        color: #4a5568 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #667eea !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
    }
    
    /* ============================================
       PROGRESS & SPINNER - Brand Colors
       ============================================ */
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 10px !important;
    }
    
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* ============================================
       ALERTS & NOTIFICATIONS
       ============================================ */
    
    .stSuccess {
        background-color: rgba(72, 187, 120, 0.1) !important;
        border-left: 4px solid #48bb78 !important;
        border-radius: 10px !important;
        padding: 1rem 1.5rem !important;
        color: #22543d !important;
    }
    
    .stError {
        background-color: rgba(245, 101, 101, 0.1) !important;
        border-left: 4px solid #f56565 !important;
        border-radius: 10px !important;
        padding: 1rem 1.5rem !important;
        color: #742a2a !important;
    }
    
    .stWarning {
        background-color: rgba(237, 137, 54, 0.1) !important;
        border-left: 4px solid #ed8936 !important;
        border-radius: 10px !important;
        padding: 1rem 1.5rem !important;
        color: #7c2d12 !important;
    }
    
    .stInfo {
        background-color: rgba(66, 153, 225, 0.1) !important;
        border-left: 4px solid #4299e1 !important;
        border-radius: 10px !important;
        padding: 1rem 1.5rem !important;
        color: #2c5282 !important;
    }
    
    /* ============================================
       EXPANDER - Collapsible Sections
       ============================================ */
    
    .streamlit-expanderHeader {
        background: white !important;
        border-radius: 10px !important;
        padding: 1rem 1.5rem !important;
        font-weight: 600 !important;
        color: #2d3748 !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08) !important;
        transition: all 0.2s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: #f7fafc !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.12) !important;
    }
    
    .streamlit-expanderContent {
        background: white !important;
        border-radius: 0 0 10px 10px !important;
        padding: 1.5rem !important;
        border: 1px solid rgba(102, 126, 234, 0.1) !important;
        border-top: none !important;
    }
    
    /* ============================================
       CUSTOM ELEMENTS - For Inline HTML
       ============================================ */
    
    /* Main header box */
    .header-box {
        background: linear-gradient(135deg, rgba(102,126,234,0.25) 0%, rgba(118,75,162,0.25) 100%) !important;
        backdrop-filter: blur(20px) !important;
        border: 2px solid rgba(255,255,255,0.4) !important;
        border-radius: 20px !important;
        padding: 3rem 2rem !important;
        margin-bottom: 2rem !important;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2) !important;
    }
    
    /* White content cards */
    .content-card {
        background: white !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        box-shadow: 0 6px 25px rgba(0,0,0,0.12) !important;
        border: 1px solid rgba(102, 126, 234, 0.1) !important;
        margin: 1rem 0 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .content-card:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 35px rgba(0,0,0,0.16) !important;
    }
    
    /* Stat cards */
    .stat-card {
        background: white !important;
        border-radius: 15px !important;
        padding: 2rem !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
        border: 2px solid transparent !important;
    }
    
    .stat-card:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.15) !important;
        border-color: rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Profile info cards */
    .info-card {
        background: white !important;
        border-radius: 12px !important;
        padding: 1.75rem !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
        border-top: 4px solid #667eea !important;
        height: 100% !important;
        transition: all 0.3s ease !important;
    }
    
    .info-card:hover {
        box-shadow: 0 8px 25px rgba(0,0,0,0.15) !important;
        transform: translateY(-3px) !important;
    }
    
    .info-card h4 {
        color: #667eea !important;
        margin: 0 0 1rem 0 !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        text-shadow: none !important;
    }
    
    .info-card p {
        color: #2d3748 !important;
        margin: 0.6rem 0 !important;
        line-height: 1.7 !important;
        font-size: 1rem !important;
        text-shadow: none !important;
    }
    
    .info-card strong {
        color: #4a5568 !important;
        font-weight: 600 !important;
    }
    
    /* System architecture cards */
    .arch-card {
        text-align: center !important;
        padding: 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .arch-card:hover {
        transform: translateY(-5px) !important;
    }
    
    .arch-card strong {
        color: #2d3748 !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        display: block !important;
        margin: 0.75rem 0 0.5rem 0 !important;
        text-shadow: none !important;
    }
    
    .arch-card p {
        color: #4a5568 !important;
        font-size: 0.95rem !important;
        margin: 0 !important;
        text-shadow: none !important;
    }
    
    /* ============================================
       PLOTLY CHARTS - Dark Text
       ============================================ */
    
    .js-plotly-plot {
        background: white !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
    }
    
    /* ============================================
       MARKDOWN CONTENT - Proper Contrast
       ============================================ */
    
    .stMarkdown {
        color: #2d3748 !important;
    }
    
    /* In white cards, ensure text is dark and no shadows */
    [style*="background: white"] h1,
    [style*="background: white"] h2,
    [style*="background: white"] h3,
    [style*="background: white"] h4,
    [style*="background: white"] h5,
    [style*="background: white"] h6,
    [style*="background: white"] p,
    [style*="background: white"] span,
    [style*="background: white"] div,
    [style*="background: white"] strong {
        color: #2d3748 !important;
        text-shadow: none !important;
    }
    
    /* Override for stat cards and content cards */
    .stat-card h3, .stat-card p,
    .content-card h2, .content-card p,
    .content-card h3, .content-card h4 {
        color: #2d3748 !important;
        text-shadow: none !important;
    }
    
    /* ============================================
       ACCESSIBILITY & POLISH
       ============================================ */
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Focus states for accessibility */
    *:focus {
        outline: 2px solid #667eea !important;
        outline-offset: 2px !important;
    }
    
    /* Remove Streamlit branding adjustments */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ============================================
       RESPONSIVE DESIGN
       ============================================ */
    
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
        h1 {
            font-size: 2rem !important;
        }
        
        .stButton>button {
            width: 100% !important;
        }
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
            <div class="info-card">
                <h4>🧑 Personal Information</h4>
                <p><strong>Gender:</strong> {student_row.get('gender', 'Unknown')}</p>
                <p><strong>Age Band:</strong> {student_row.get('age_band', 'Unknown')}</p>
                <p><strong>Region:</strong> {student_row.get('region', 'Unknown')}</p>
                <p><strong>Disability:</strong> {student_row.get('disability', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-card" style="border-top-color: #764ba2;">
                <h4 style="color: #764ba2 !important;">📖 Academic Background</h4>
                <p><strong>Education:</strong> {student_row.get('highest_education', 'Unknown')}</p>
                <p><strong>Attempts:</strong> {student_row.get('num_of_prev_attempts', 'Unknown')}</p>
                <p><strong>Credits:</strong> {student_row.get('studied_credits', 'Unknown')}</p>
                <p><strong>Module:</strong> {student_row.get('code_module', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            result = student_row.get('final_result', 'Unknown')
            result_color = {
                'Pass': '#28a745',
                'Fail': '#dc3545',
                'Distinction': '#ffc107',
                'Withdrawn': '#6c757d'
            }.get(result, '#2d3748')
            st.markdown(f"""
            <div class="info-card" style="border-top-color: #8b5cf6;">
                <h4 style="color: #8b5cf6 !important;">📍 Socioeconomic Context</h4>
                <p><strong>IMD Band:</strong> {student_row.get('imd_band', 'Unknown')}</p>
                <p><strong>Region:</strong> {student_row.get('region', 'Unknown')}</p>
                <p><strong>Presentation:</strong> {student_row.get('code_presentation', 'N/A')}</p>
                <p><strong>Final Result:</strong> <span style="color: {result_color}; font-weight: 700;">{result}</span></p>
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
    # Animated header with gradient background - works in both themes
    st.markdown("""
    <div class="header-box" style="text-align: center;">
        <h1 style="font-size: 3.5rem; font-weight: 800; 
                   color: white !important;
                   margin: 0;">
            🤖 AI Academic Mentor
        </h1>
        <p style="font-size: 1.5rem; 
                  color: white !important; 
                  font-weight: 600;
                  margin-top: 1rem; 
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
        <div class="stat-card" style="border-left: 5px solid #667eea;">
            <h3 style="color: #667eea !important; margin: 0; font-size: 2.5rem; font-weight: 700; text-shadow: none !important;">{}</h3>
            <p style="color: #4a5568 !important; margin: 0.5rem 0 0 0; font-size: 1rem; font-weight: 600; text-shadow: none !important;">Total Students</p>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        pass_rate = (df['final_result'] == 'Pass').sum() / len(df) * 100 if 'final_result' in df.columns else 0
        st.markdown("""
        <div class="stat-card" style="border-left: 5px solid #28a745;">
            <h3 style="color: #28a745 !important; margin: 0; font-size: 2.5rem; font-weight: 700; text-shadow: none !important;">{:.1f}%</h3>
            <p style="color: #4a5568 !important; margin: 0.5rem 0 0 0; font-size: 1rem; font-weight: 600; text-shadow: none !important;">Pass Rate</p>
        </div>
        """.format(pass_rate), unsafe_allow_html=True)
    
    with col3:
        avg_credits = df['studied_credits'].mean() if 'studied_credits' in df.columns else 0
        st.markdown("""
        <div class="stat-card" style="border-left: 5px solid #764ba2;">
            <h3 style="color: #764ba2 !important; margin: 0; font-size: 2.5rem; font-weight: 700; text-shadow: none !important;">{:.0f}</h3>
            <p style="color: #4a5568 !important; margin: 0.5rem 0 0 0; font-size: 1rem; font-weight: 600; text-shadow: none !important;">Avg Credits</p>
        </div>
        """.format(avg_credits), unsafe_allow_html=True)
    
    with col4:
        unique_courses = df['code_module'].nunique() if 'code_module' in df.columns else 0
        st.markdown("""
        <div class="stat-card" style="border-left: 5px solid #ffa502;">
            <h3 style="color: #ffa502 !important; margin: 0; font-size: 2.5rem; font-weight: 700; text-shadow: none !important;">{}</h3>
            <p style="color: #4a5568 !important; margin: 0.5rem 0 0 0; font-size: 1rem; font-weight: 600; text-shadow: none !important;">Courses</p>
        </div>
        """.format(unique_courses), unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Student selection in a beautiful container
    st.markdown("""
    <div class="content-card" style="border-top: 5px solid #667eea; margin-bottom: 2rem;">
        <h2 style="color: #667eea !important; 
                   text-align: center; 
                   margin-top: 0;
                   font-size: 2rem;
                   font-weight: 700;
                   text-shadow: none !important;">
            🎯 Select Student for Analysis
        </h2>
        <p style="text-align: center; 
                  color: #4a5568 !important; 
                  font-size: 1.1rem;
                  margin-bottom: 1.5rem;
                  text-shadow: none !important;">
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

                # --- START: NEW REPORT RENDERING ---
                
                # Build the entire HTML string in Python first
                # We use your 'content-card' class for consistent styling
                report_html = f"""
                <div class="content-card" style="border-top: 5px solid #667eea; margin-top: 2rem;">
                    <h2 style="color: #667eea !important; 
                               text-align: center; 
                               margin-top: 0;
                               font-size: 2.2rem;
                               font-weight: 700;
                               text-shadow: none !important;
                               margin-bottom: 1.5rem;">
                        📋 AI Mentor Report
                    </h2>
                    
                    <div style="color: #2d3748; font-size: 1.05rem; line-height: 1.8;">
                        {final_report}
                    </div>
                </div>
                """
                
                # Render the entire HTML block in ONE single call
                st.markdown(report_html, unsafe_allow_html=True)
                
                # --- END: NEW REPORT RENDERING ---
                
               
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
    <div class="content-card" style="margin-top: 3rem;">
        <h3 style="color: #667eea; text-align: center; margin-top: 0; font-size: 2rem; font-weight: 700; text-shadow: none !important;">
            🔧 System Architecture
        </h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1.5rem;">
            <div class="arch-card">
                <div style="font-size: 2.5rem;">🤖</div>
                <strong>AI Model</strong>
                <p>Groq Llama 3.1 8B</p>
            </div>
            <div class="arch-card">
                <div style="font-size: 2.5rem;">🧠</div>
                <strong>Framework</strong>
                <p>LangGraph Agentic AI</p>
            </div>
            <div class="arch-card">
                <div style="font-size: 2.5rem;">📚</div>
                <strong>Knowledge Base</strong>
                <p>ChromaDB RAG</p>
            </div>
            <div class="arch-card">
                <div style="font-size: 2.5rem;">📈</div>
                <strong>ML Pipeline</strong>
                <p>GradientBoosting</p>
            </div>
            <div class="arch-card">
                <div style="font-size: 2.5rem;">📊</div>
                <strong>Dataset</strong>
                <p>OULAD</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
