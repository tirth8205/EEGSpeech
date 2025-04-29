import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from scipy.ndimage import gaussian_filter
import os

# Local imports
from eegspeech.models.model import EEGSpeechClassifier
from eegspeech.models.dataset import create_synthetic_eeg_data
from eegspeech.models.processing import preprocess_eeg, apply_filters

# Set page configuration
st.set_page_config(
    page_title="🧠 EEG Speech Decoder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = EEGSpeechClassifier(n_channels=14, n_classes=5, time_points=1000)
        model_path = os.path.join(os.path.dirname(__file__), '../../../outputs/eeg_speech_classifier.pth')
        model.load_state_dict(torch.load(model_path))
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
    """Plot EEG channels using plotly for interactive visualization"""
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
            x=time,
            y=eeg_data[i],
            mode='lines',
            name=channel_names[i]
        ))
    
    fig.update_layout(
        title="EEG Channel Activity",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (μV)",
        legend_title="Channel",
        hovermode="x unified",
        height=400
    )
    return fig

def plot_brain_activity(eeg_data):
    """Create a heatmap of brain activity based on EEG channels"""
    channel_activity = np.mean(np.abs(eeg_data), axis=1)
    positions = {
        0: (4, 8), 1: (6, 8), 2: (3, 7), 3: (7, 7),
        4: (2, 6), 5: (8, 6), 6: (3, 5), 7: (7, 5),
        8: (4, 4), 9: (6, 4), 10: (2, 3), 11: (8, 3),
        12: (4, 2), 13: (6, 2)
    }
    
    grid_size = 10
    activity_grid = np.zeros((grid_size, grid_size))
    for ch in range(len(channel_activity)):
        x, y = positions[ch]
        activity_grid[y, x] = channel_activity[ch]
    
    smoothed_grid = gaussian_filter(activity_grid, sigma=1)
    fig = px.imshow(smoothed_grid, color_continuous_scale='inferno')
    
    for ch in range(len(channel_activity)):
        x, y = positions[ch]
        fig.add_annotation(
            x=x, y=y,
            text=str(ch),
            showarrow=False,
            font=dict(color="white", size=10)
        )
    
    fig.update_layout(
        title='Brain Activity Heatmap',
        height=400,
        coloraxis_colorbar=dict(title="Activity")
    )
    return fig

def run_model_prediction(model, eeg_data, class_names):
    """Run the model prediction on EEG data"""
    with torch.no_grad():
        input_tensor = torch.FloatTensor(eeg_data).unsqueeze(0)
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        _, predicted = torch.max(output, 1)
        return predicted.item(), probabilities.numpy()

def display_confidence_chart(probabilities, class_names):
    """Display a bar chart of prediction confidence for each class"""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=class_names,
        y=probabilities,
        marker_color=['#2C3E50', '#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    ))
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Phoneme",
        yaxis_title="Confidence",
        yaxis=dict(range=[0, 1]),
        height=300
    )
    return fig

def visualize_filter_activation(model, eeg_data):
    """Visualize first layer filter activations"""
    input_tensor = torch.FloatTensor(eeg_data).unsqueeze(0)
    first_conv = model.feature_extractor[0]
    with torch.no_grad():
        activations = first_conv(input_tensor)
    activations = activations.squeeze(0).numpy()
    
    fig = plt.figure(figsize=(10, 5))
    for i in range(min(4, activations.shape[0])):
        plt.subplot(2, 2, i+1)
        plt.plot(activations[i])
        plt.title(f'Filter {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Activation')
    plt.tight_layout()
    return fig

def main():
    """Main application function"""
    st.title("🧠 Neural Speech Decoder")
    st.markdown("""
    This application demonstrates how neural signals (EEG) can be decoded to identify speech phonemes.
    The system uses a lightweight convolutional neural network to analyze EEG patterns associated with vowel sounds.
    """)
    
    # Load model
    model, model_loaded, class_names = load_model()
    
    # Sidebar
    st.sidebar.title("Controls")
    if not model_loaded:
        st.sidebar.warning("⚠️ Trained model not found. Run `python -m EEGSpeech.app.main` first to train the model.")
    
    if st.sidebar.button("Generate New EEG Sample"):
        eeg_data, true_class = generate_sample()
        st.session_state['current_eeg'] = eeg_data
        st.session_state['true_class'] = true_class
        st.session_state['time_generated'] = time.time()
    
    if 'current_eeg' not in st.session_state:
        eeg_data, true_class = generate_sample()
        st.session_state['current_eeg'] = eeg_data
        st.session_state['true_class'] = true_class
        st.session_state['time_generated'] = time.time()
    
    eeg_data = st.session_state['current_eeg']
    true_class = st.session_state['true_class']
    
    elapsed = time.time() - st.session_state['time_generated']
    st.sidebar.write(f"Sample generated: {elapsed:.1f} seconds ago")
    st.sidebar.write(f"True phoneme: **{class_names[true_class]}**")
    
    # Removed unused filter sensitivity slider
    # with st.sidebar.expander("Advanced Options"):
    #     st.write("Filter sensitivity")
    #     sensitivity = st.slider("Adjust filter sensitivity", 0.1, 2.0, 1.0, 0.1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("EEG Signal")
        eeg_fig = plot_eeg_channels(eeg_data)
        st.plotly_chart(eeg_fig, use_container_width=True)
        
        if model_loaded:
            predicted_class, probabilities = run_model_prediction(model, eeg_data, class_names)
            st.subheader("Model Prediction")
            st.metric("Predicted Phoneme", class_names[predicted_class],
                     delta="✓ Correct" if predicted_class == true_class else "✗ Incorrect")
            conf_fig = display_confidence_chart(probabilities, class_names)
            st.plotly_chart(conf_fig, use_container_width=True)
        else:
            st.warning("⚠️ Please train the model first by running `python -m EEGSpeech.app.main`")
    
    with col2:
        st.subheader("Brain Activity")
        brain_fig = plot_brain_activity(eeg_data)
        st.plotly_chart(brain_fig, use_container_width=True)
        
        st.subheader("Neural Network Filter Activations")
        if model_loaded:
            act_fig = visualize_filter_activation(model, eeg_data)
            st.pyplot(act_fig)
        else:
            st.info("Model not loaded. Filter activations unavailable.")
    
    st.markdown("---")
    st.subheader("How it Works")
    
    with st.expander("Neural Speech Decoding Explained"):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            ### Speech Processing in the Brain
            Speech processing involves multiple brain regions:
            - **Temporal Lobe**: Processes initial sound in the primary auditory cortex.
            - **Wernicke's Area**: Crucial for speech comprehension.
            - **Broca's Area**: Involved in speech production.
            - **Motor Cortex**: Controls articulation muscles.
            This EEG decoder captures activity across these regions to identify speech phonemes.
            
            ### Deep Learning Architecture
            The convolutional neural network:
            1. Extracts temporal patterns from EEG signals.
            2. Identifies neural signatures of phonemes.
            3. Classifies the intended speech sound.
            The lightweight design ensures efficiency on standard hardware.
            """)
        with col2:
            st.image("https://upload.wikimedia.org/wikipedia/commons/0/0e/Cerebral_Cortex.png", caption="Speech areas in the brain")
    
    with st.expander("Technical Performance"):
        st.markdown("""
        | Metric | Value |
        | --- | --- |
        | Model Size | 33,165 parameters |
        | Test Accuracy | 73.33% |
        | Inference Time | <10ms per sample |
        | Memory Usage | ~0.5MB |
        This implementation is optimized for efficiency and real-time decoding.
        """)

if __name__ == "__main__":
    main()