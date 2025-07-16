"""
Streamlit Cloud deployment configuration
This file helps optimize the app for Streamlit Cloud deployment
"""

import streamlit as st
import os
import sys

def setup_streamlit_cloud():
    """Configure app for Streamlit Cloud deployment"""
    
    # Set page config for better mobile experience
    st.set_page_config(
        page_title="EEG Healthcare VLM",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/tirth8205/EEGSpeech-Healthcare',
            'Report a bug': "https://github.com/tirth8205/EEGSpeech-Healthcare/issues",
            'About': "# EEG Healthcare VLM\nAdvanced brain-computer interface with VLM integration"
        }
    )
    
    # Add custom CSS for better UI
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton > button {
        width: 100%;
    }
    .stSelectbox > div > div {
        min-height: 2.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check if running on Streamlit Cloud
    if 'STREAMLIT_CLOUD' in os.environ:
        st.sidebar.success("🌐 Running on Streamlit Cloud")
    
    # Handle file upload limits
    st.sidebar.info("📁 Max file size: 200MB")
    
    return True

if __name__ == "__main__":
    setup_streamlit_cloud()