import streamlit as st
import numpy as np
import torch
import time
import plotly.graph_objects as go
from PIL import Image

# Local imports
from eegspeech.models.model import EEGSpeechClassifier
from eegspeech.models.dataset import create_synthetic_eeg_data
from eegspeech.models.utils import plot_training_history, visualize_eeg_and_predictions

# Configure page
st.set_page_config(
    page_title="🧠 EEG Speech Decoder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Loads trained model with error handling"""
    try:
        model = EEGSpeechClassifier(14, 5, 1000)
        model.load_state_dict(torch.load('eeg_speech_classifier.pth'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def main():
    st.title("🧠 Real-time Neural Speech Decoding")
    st.markdown("""
    This interactive demo shows how brain activity (EEG) can be decoded into speech phonemes.
    """)
    
    # Initialize session state
    if 'eeg_data' not in st.session_state:
        X, y, _ = create_synthetic_eeg_data(n_samples=1)
        st.session_state.eeg_data = X[0]
        st.session_state.true_class = y[0]
        st.session_state.last_update = time.time()
    
    # Load model
    model = load_model()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        if st.button("Generate New Sample"):
            X, y, _ = create_synthetic_eeg_data(n_samples=1)
            st.session_state.eeg_data = X[0]
            st.session_state.true_class = y[0]
            st.session_state.last_update = time.time()
        
        st.write(f"Sample age: {time.time() - st.session_state.last_update:.1f}s")
        st.write(f"True phoneme: **{['a','e','i','o','u'][st.session_state.true_class]}**")

    # Main display columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Neural Signals")
        # Plot EEG channels
        fig = go.Figure()
        for ch in range(4):  # Show first 4 channels
            fig.add_trace(go.Scatter(
                y=st.session_state.eeg_data[ch],
                name=f'Channel {ch}'
            ))
        fig.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show predictions if model loaded
        if model:
            with torch.no_grad():
                input_tensor = torch.FloatTensor(st.session_state.eeg_data).unsqueeze(0)
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                _, pred = torch.max(output, 1)
            
            st.subheader("Model Prediction")
            confidence = probs[pred].item()
            st.metric(
                label="Predicted Phoneme",
                value=['a','e','i','o','u'][pred.item()],
                delta=f"{confidence:.1%} confidence",
                delta_color="normal"
            )
            
            # Confidence chart
            st.subheader("Class Probabilities")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['a','e','i','o','u'],
                y=probs.numpy(),
                marker_color=['#1f77b4']*5
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Brain Activation Map")
        # Create heatmap
        channel_activity = np.mean(np.abs(st.session_state.eeg_data), axis=1)
        fig = go.Figure(go.Heatmap(
            z=[channel_activity],
            colorscale='Viridis',
            showscale=False
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
