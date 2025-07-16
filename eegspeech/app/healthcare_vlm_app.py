"""
Healthcare VLM-Enhanced EEG Speech Classifier App
Advanced Streamlit application with Vision Language Model integration for clinical insights
"""

import streamlit as st
import numpy as np
import torch
import time
import plotly.graph_objects as go
from PIL import Image
import pandas as pd
from datetime import datetime
import io
import base64
import json

# Local imports
from eegspeech.models.model import EEGSpeechClassifier
from eegspeech.models.dataset import create_synthetic_eeg_data, preprocess_real_eeg
from eegspeech.models.vlm_integration import HealthcareVLMPipeline
from eegspeech.models.utils import plot_training_history

# Configure page
st.set_page_config(
    page_title="Healthcare VLM EEG Analyzer",
    page_icon="🧠💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for healthcare theme
st.markdown("""
<style>
    .healthcare-header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .risk-low { color: #28a745; font-weight: bold; }
    .risk-moderate { color: #ffc107; font-weight: bold; }
    .risk-high { color: #dc3545; font-weight: bold; }
    .clinical-insight {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 10px 0;
    }
    .patient-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the EEG model and VLM pipeline"""
    try:
        # Load EEG model
        model = EEGSpeechClassifier(14, 8, 1000)
        model.load_state_dict(torch.load('eeg_speech_classifier.pth', map_location=torch.device('cpu')))
        model.eval()
        
        # Load VLM pipeline
        vlm_pipeline = HealthcareVLMPipeline()
        
        return model, vlm_pipeline
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

def initialize_session_state():
    """Initialize session state variables"""
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = {}
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'current_eeg' not in st.session_state:
        X, y, _ = create_synthetic_eeg_data(n_samples=1)
        st.session_state.current_eeg = X[0]
        st.session_state.current_label = y[0]

def render_patient_info_form():
    """Render patient information form"""
    st.markdown("### 👤 Patient Information")
    
    with st.form("patient_info_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            patient_id = st.text_input("Patient ID", value="P001")
            age = st.number_input("Age", min_value=0, max_value=120, value=45)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
        with col2:
            medical_history = st.text_area("Medical History", 
                                         placeholder="Previous conditions, surgeries, etc.")
            medications = st.text_area("Current Medications", 
                                     placeholder="List current medications")
            
        submitted = st.form_submit_button("Update Patient Info")
        
        if submitted:
            st.session_state.patient_data = {
                "id": patient_id,
                "age": age,
                "gender": gender,
                "medical_history": medical_history,
                "medications": medications,
                "last_updated": datetime.now().isoformat()
            }
            st.success("Patient information updated!")

def render_risk_assessment(risk_data):
    """Render risk assessment dashboard"""
    st.markdown("### 🎯 Clinical Risk Assessment")
    
    # Create risk level indicators
    cols = st.columns(len(risk_data))
    
    for i, (risk_type, risk_value) in enumerate(risk_data.items()):
        with cols[i]:
            # Determine risk level and color
            if risk_value < 0.3:
                risk_level = "LOW"
                color = "#28a745"
            elif risk_value < 0.7:
                risk_level = "MODERATE"
                color = "#ffc107"
            else:
                risk_level = "HIGH"
                color = "#dc3545"
            
            # Display risk metric
            st.metric(
                label=risk_type.replace('_', ' ').title(),
                value=f"{risk_value:.1%}",
                delta=f"{risk_level}",
                delta_color="normal" if risk_level == "LOW" else "inverse"
            )
            
            # Progress bar
            st.progress(risk_value)

def render_clinical_insights(insights):
    """Render clinical insights section"""
    st.markdown("### 🔍 Clinical Insights")
    
    for question, insight in insights.items():
        with st.expander(f"🔸 {question}"):
            st.markdown(f'<div class="clinical-insight">{insight}</div>', 
                       unsafe_allow_html=True)

def render_vlm_analysis_results(analysis_results):
    """Render VLM analysis results"""
    st.markdown("### 🧠 VLM Analysis Results")
    
    # Create tabs for different analysis types
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Spectrogram", "🧠 Brain Map", "⏱️ Temporal", "📋 Report"])
    
    with tab1:
        st.subheader("Clinical Spectrogram Analysis")
        st.image(analysis_results["images"]["spectrogram"], 
                caption="EEG Spectrogram with Clinical Annotations")
        
        st.subheader("Spectrogram Insights")
        render_clinical_insights(analysis_results["spectrogram_analysis"]["clinical_insights"])
        
        st.subheader("Risk Assessment")
        render_risk_assessment(analysis_results["spectrogram_analysis"]["risk_assessment"])
    
    with tab2:
        st.subheader("Topographic Brain Map")
        st.image(analysis_results["images"]["brain_map"], 
                caption="Brain Activity Topography")
        
        st.subheader("Brain Map Insights")
        render_clinical_insights(analysis_results["brain_map_analysis"]["clinical_insights"])
    
    with tab3:
        st.subheader("Temporal Dynamics")
        st.image(analysis_results["images"]["temporal_plot"], 
                caption="Temporal Evolution of EEG Patterns")
        
        st.subheader("Temporal Insights")
        render_clinical_insights(analysis_results["temporal_analysis"]["clinical_insights"])
    
    with tab4:
        st.subheader("Comprehensive Clinical Report")
        st.markdown(f"""
        <div style="background: white; padding: 20px; border-radius: 10px; 
                   box-shadow: 0 2px 4px rgba(0,0,0,0.1); font-family: monospace;">
        <pre>{analysis_results["clinical_report"]}</pre>
        </div>
        """, unsafe_allow_html=True)
        
        # Download report button
        report_bytes = analysis_results["clinical_report"].encode()
        st.download_button(
            label="📥 Download Clinical Report",
            data=report_bytes,
            file_name=f"clinical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def render_analysis_history():
    """Render analysis history"""
    st.markdown("### 📈 Analysis History")
    
    if not st.session_state.analysis_results:
        st.info("No analysis history available yet.")
        return
    
    # Create summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Analyses", len(st.session_state.analysis_results))
    
    with col2:
        avg_risk = np.mean([
            result["analysis"]["spectrogram_analysis"]["risk_assessment"]["overall_risk"]
            for result in st.session_state.analysis_results
        ])
        st.metric("Average Risk", f"{avg_risk:.1%}")
    
    with col3:
        latest_analysis = st.session_state.analysis_results[-1]
        st.metric("Latest Analysis", latest_analysis["timestamp"][:10])
    
    # Display history table
    history_data = []
    for result in st.session_state.analysis_results:
        history_data.append({
            "Timestamp": result["timestamp"][:19],
            "Phoneme": result["phoneme"] or "Unknown",
            "Overall Risk": f"{result['analysis']['spectrogram_analysis']['risk_assessment']['overall_risk']:.1%}",
            "Speech Risk": f"{result['analysis']['spectrogram_analysis']['risk_assessment']['speech_disorder_risk']:.1%}",
            "Cognitive Risk": f"{result['analysis']['spectrogram_analysis']['risk_assessment']['cognitive_decline_risk']:.1%}"
        })
    
    df = pd.DataFrame(history_data)
    st.dataframe(df, use_container_width=True)

def main():
    """Main application function"""
    # Header
    st.markdown("""
    <div class="healthcare-header">
        <h1>🧠💊 Healthcare VLM EEG Analyzer</h1>
        <p>Advanced neural speech decoding with clinical insights powered by Vision Language Models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Load models
    model, vlm_pipeline = load_model()
    if model is None or vlm_pipeline is None:
        st.error("Failed to load models. Please check model files.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Patient information form
        render_patient_info_form()
        
        st.markdown("---")
        
        # Data source selection
        data_source = st.radio("📊 Data Source", ["Synthetic EEG", "Upload Real EEG"])
        
        if data_source == "Synthetic EEG":
            phoneme_options = ['a', 'e', 'i', 'o', 'u', 'p', 't', 'k']
            selected_phoneme = st.selectbox("Select Phoneme", phoneme_options)
            
            if st.button("🔄 Generate New EEG Sample"):
                # Generate specific phoneme
                phoneme_idx = phoneme_options.index(selected_phoneme)
                X, y, _ = create_synthetic_eeg_data(n_samples=8)
                st.session_state.current_eeg = X[phoneme_idx]
                st.session_state.current_label = phoneme_idx
                st.success(f"Generated new EEG sample for phoneme: {selected_phoneme}")
        
        else:
            uploaded_file = st.file_uploader("📁 Upload EDF File", type=["edf"])
            if uploaded_file:
                if st.button("🔍 Process EEG File"):
                    with st.spinner("Processing EEG file..."):
                        with open("temp.edf", "wb") as f:
                            f.write(uploaded_file.getvalue())
                        X, y, _ = preprocess_real_eeg("temp.edf")
                        if X is not None:
                            st.session_state.current_eeg = X[0]
                            st.session_state.current_label = y[0] if y is not None else -1
                            st.success("EEG file processed successfully!")
                        else:
                            st.error("Failed to process EEG file")
        
        st.markdown("---")
        
        # Analysis controls
        st.subheader("🔬 Analysis Controls")
        
        if st.button("🚀 Run VLM Analysis", type="primary"):
            if st.session_state.patient_data:
                with st.spinner("Running VLM analysis..."):
                    # Get current phoneme label
                    phoneme_labels = ['a', 'e', 'i', 'o', 'u', 'p', 't', 'k']
                    current_phoneme = phoneme_labels[st.session_state.current_label] if st.session_state.current_label >= 0 else None
                    
                    # Run VLM analysis
                    analysis_results = vlm_pipeline.process_eeg_for_healthcare(
                        st.session_state.current_eeg,
                        current_phoneme,
                        st.session_state.patient_data
                    )
                    
                    # Store results
                    st.session_state.analysis_results.append({
                        "timestamp": datetime.now().isoformat(),
                        "phoneme": current_phoneme,
                        "patient_id": st.session_state.patient_data.get("id", "Unknown"),
                        "analysis": analysis_results
                    })
                    
                    st.success("✅ VLM analysis completed!")
            else:
                st.error("Please enter patient information first!")
    
    # Main content area
    if st.session_state.analysis_results:
        # Get latest analysis
        latest_analysis = st.session_state.analysis_results[-1]
        
        # Display current EEG
        st.subheader("📊 Current EEG Signal")
        
        # EEG visualization
        fig = go.Figure()
        channel_names = ['Fz', 'F3', 'F4', 'Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'Oz']
        
        for i in range(min(4, len(channel_names))):  # Show first 4 channels
            fig.add_trace(go.Scatter(
                y=st.session_state.current_eeg[i],
                name=channel_names[i],
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="EEG Signal - Key Channels",
            xaxis_title="Time (ms)",
            yaxis_title="Amplitude (µV)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display model prediction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(st.session_state.current_eeg).unsqueeze(0)
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            _, pred = torch.max(output, 1)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "🗣️ Predicted Phoneme",
                ['a','e','i','o','u','p','t','k'][pred.item()],
                f"{probs[pred].item():.1%} confidence"
            )
        
        with col2:
            true_phoneme = ['a','e','i','o','u','p','t','k'][st.session_state.current_label] if st.session_state.current_label >= 0 else "Unknown"
            st.metric("📋 True Phoneme", true_phoneme)
        
        # Display VLM analysis results
        render_vlm_analysis_results(latest_analysis["analysis"])
        
        # Display analysis history
        render_analysis_history()
        
    else:
        # Welcome message
        st.markdown("""
        ### 🚀 Welcome to Healthcare VLM EEG Analyzer
        
        This advanced system combines traditional EEG signal processing with cutting-edge Vision Language Models 
        to provide comprehensive clinical insights for speech-related neurological conditions.
        
        **Key Features:**
        - 🧠 **Advanced EEG Analysis**: Hybrid CNN-LSTM model for phoneme classification
        - 👁️ **Vision Language Models**: Clinical interpretation of EEG visualizations
        - 📊 **Risk Assessment**: Automated clinical risk scoring
        - 📋 **Clinical Reports**: Comprehensive medical documentation
        - 📈 **Progress Tracking**: Patient monitoring over time
        
        **To get started:**
        1. Enter patient information in the sidebar
        2. Select or upload EEG data
        3. Click "Run VLM Analysis" to generate clinical insights
        """)
        
        # Show sample analysis
        st.subheader("📝 Sample Clinical Report")
        st.markdown("""
        ```
        CLINICAL EEG SPEECH ANALYSIS REPORT
        Generated: 2024-01-15T14:30:00
        
        PATIENT INFORMATION:
        Patient ID: P001
        Age: 45
        Gender: Male
        
        CLINICAL FINDINGS:
        ✅ Normal speech motor cortex function
        ✅ No significant abnormalities detected
        ✅ Appropriate hemispheric activation patterns
        
        RISK ASSESSMENT:
        Speech Disorder Risk: 12% (LOW)
        Cognitive Decline Risk: 8% (LOW)
        Overall Risk: 7% (LOW)
        
        RECOMMENDATIONS:
        1. Continue regular monitoring
        2. Follow-up EEG in 6 months
        3. Patient education on prevention
        ```
        """)

if __name__ == "__main__":
    main()