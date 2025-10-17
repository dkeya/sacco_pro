# sacco_core/ui.py
import streamlit as st

def hide_default_streamlit_elements():
    """Hide default Streamlit elements across all pages"""
    hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {visibility: hidden;}
        
        /* Hide Streamlit's default sidebar navigation */
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        
        /* Consistent padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Main container adjustments */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        
        /* Collapsible sidebar styling */
        .sidebar-category {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 0.5rem;
            margin-bottom: 0.5rem;
            border: 1px solid #e9ecef;
        }
        
        /* Indented page buttons */
        .indented-button {
            margin-left: 1rem;
        }
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def apply_custom_styling():
    """Apply custom styling across all pages"""
    custom_css = """
    <style>
        /* Custom header styling */
        .custom-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        
        /* Button styling */
        .stButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
            color: white;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Category toggle buttons */
        .category-toggle {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        }
        
        /* Metric card styling */
        [data-testid="metric-container"] {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 10px;
            padding: 1rem;
        }
        
        /* Indented page buttons */
        .indented-page {
            margin-left: 1.5rem;
            background: rgba(102, 126, 234, 0.1) !important;
            border: 1px solid rgba(102, 126, 234, 0.2) !important;
        }
        
        .indented-page:hover {
            background: rgba(102, 126, 234, 0.2) !important;
            border: 1px solid rgba(102, 126, 234, 0.4) !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)