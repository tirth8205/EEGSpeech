import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import datetime
from scipy.ndimage import gaussian_filter

# Local imports
from models.model import EEGSpeechClassifier
from models.dataset import create_synthetic_eeg_data
from models.processing import preprocess_eeg, apply_filters

# Configure page
st.set_page_config(
    page_title="🧠 EEGSpeech - Neural Decoder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'filter_settings' not in st.session_state:
    st.session_state.filter_settings = {
        'highpass': 1.0,
        'lowpass': 50.0,
        'notch': True
    }
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'recordings' not in st.session_state:
    st.session_state.recordings = []

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = EEGSpeechClassifier(14, 5, 1000)
        model.load_state_dict(torch.load('models/eeg_speech_classifier.pth'))
        model.eval()
        return model, True, ['a', 'e', 'i', 'o', 'u']
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, False, []

def generate_sample():
    """Generate a new EEG sample with a random class"""
    X, y, _ = create_synthetic_eeg_data(n_samples=1, n_channels=14, time_points=1000)
    return X[0], y[0]

def plot_eeg_channels(eeg_data):
    """Plot EEG channels using plotly"""
    fig = go.Figure()
    channels_to_plot = [0, 1, 7, 8]
    channel_names = {
        0: "Frontal 1 (F3)",
        1: "Frontal 2 (F4)", 
        7: "Temporal 1 (T7)",
        8: "Temporal 2 (T8)"
    }
    
    time = np.arange(eeg_data.shape[1]) / 1000
    for i in channels_to_plot:
        fig.add_trace(go.Scatter(
            x=time, y=eeg_data[i], mode='lines', name=channel_names[i]
        ))
    
    fig.update_layout(
        title="EEG Channel Activity",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (μV)",
        height=400
    )
    return fig

def plot_brain_activity(eeg_data):
    """Create a heatmap of brain activity"""
    channel_activity = np.mean(np.abs(eeg_data), axis=1)
    positions = {
        0: (4, 8), 1: (6, 8), 2: (3, 7), 3: (7, 7),
        4: (2, 6), 5: (8, 6), 6: (3, 5), 7: (7, 5),
        8: (4, 4), 9: (6, 4), 10: (2, 3), 11: (8, 3),
        12: (4, 2), 13: (6, 2)
    }
    
    grid_size = 10
    activity_grid = np.zeros((grid_size, grid_size))
    for ch in range(14):
        x, y = positions[ch]
        activity_grid[y, x] = channel_activity[ch]
    
    smoothed_grid = gaussian_filter(activity_grid, sigma=1)
    fig = px.imshow(smoothed_grid, color_continuous_scale='inferno')
    
    for ch in range(14):
        x, y = positions[ch]
        fig.add_annotation(x=x, y=y, text=str(ch), showarrow=False)
    
    fig.update_layout(title='Neural Activity Heatmap', height=400)
    return fig

def run_model_prediction(model, eeg_data, class_names):
    """Run model prediction on EEG data"""
    with torch.no_grad():
        input_tensor = torch.FloatTensor(eeg_data).unsqueeze(0)
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        _, predicted = torch.max(output, 1)
    return predicted.item(), probabilities.numpy()

def main():
    """Main application function"""
    st.title("🧠 Real-time Neural Speech Decoding")
    
    # Load model
    model, model_loaded, class_names = load_model()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        if st.button("Generate New Sample"):
            eeg_data, true_class = generate_sample()
            st.session_state.current_eeg = eeg_data
            st.session_state.true_class = true_class
            st.session_state.time_generated = time.time()
        
        if 'current_eeg' not in st.session_state:
            eeg_data, true_class = generate_sample()
            st.session_state.current_eeg = eeg_data
            st.session_state.true_class = true_class
            st.session_state.time_generated = time.time()
        
        st.write(f"Sample age: {time.time() - st.session_state.time_generated:.1f}s")
        st.write(f"True phoneme: **{class_names[st.session_state.true_class]}**")

    # Main display
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Neural Signals")
        eeg_fig = plot_eeg_channels(st.session_state.current_eeg)
        st.plotly_chart(eeg_fig, use_container_width=True)
        
        if model_loaded:
            predicted, probs = run_model_prediction(model, st.session_state.current_eeg, class_names)
            st.subheader("Prediction Results")
            st.metric("Predicted Phoneme", class_names[predicted], 
                     delta="✓ Correct" if predicted == st.session_state.true_class else "✗ Incorrect")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=class_names,
                y=probs,
                marker_color=px.colors.qualitative.Plotly
            ))
            fig.update_layout(title="Confidence Scores", height=300)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.header("Brain Activation")
        brain_fig = plot_brain_activity(st.session_state.current_eeg)
        st.plotly_chart(brain_fig, use_container_width=True)

if __name__ == "__main__":
    main()
