import streamlit as st
import numpy as np
import torch
import time
import plotly.graph_objects as go
from PIL import Image
import mne
import io
import matplotlib.pyplot as plt

# Local imports
from eegspeech.models.model import EEGSpeechClassifier
from eegspeech.models.dataset import create_synthetic_eeg_data, preprocess_real_eeg
from eegspeech.models.utils import plot_training_history

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
        model = EEGSpeechClassifier(14, 8, 1000)
        model.load_state_dict(torch.load('eeg_speech_classifier.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def plot_topographic_map(data, sfreq=1000):
    """Creates topographic EEG map using MNE"""
    # Create MNE info structure for 14 channels (approximating 10-20 system)
    ch_names = ['Fz', 'F3', 'F4', 'Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6', 'Oz']
    info = mne.create_info(ch_names=ch_names[:data.shape[0]], sfreq=sfreq, ch_types='eeg')
    info.set_montage('standard_1020')
    
    # Create MNE Raw object
    raw = mne.io.RawArray(data, info, verbose=False)
    
    # Plot topographic map
    fig, ax = plt.subplots(figsize=(6, 4))
    mne.viz.plot_topomap(np.mean(np.abs(data), axis=1), info, axes=ax, show=False)
    plt.tight_layout()
    
    # Convert to image for Streamlit
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return Image.open(buf)

def main():
    st.title("🧠 Real-time Neural Speech Decoder")
    st.markdown("""
    Decode speech phonemes (/a/, /e/, /i/, /o/, /u/, /p/, /t/, /k/) from EEG signals using a hybrid CNN-LSTM model.
    Upload real EEG data (EDF) or generate synthetic data for predictions.
    """)
    
    # Initialize session state
    if 'eeg_data' not in st.session_state:
        X, y, _ = create_synthetic_eeg_data(n_samples=1)
        st.session_state.eeg_data = X[0]
        st.session_state.true_class = y[0]
        st.session_state.data_type = 'synthetic'
        st.session_state.last_update = time.time()
    
    # Load model
    model = load_model()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        data_source = st.radio("Data Source", ["Synthetic", "Real EEG"])
        
        if data_source == "Synthetic":
            if st.button("Generate New Sample"):
                X, y, _ = create_synthetic_eeg_data(n_samples=1)
                st.session_state.eeg_data = X[0]
                st.session_state.true_class = y[0]
                st.session_state.data_type = 'synthetic'
                st.session_state.last_update = time.time()
        else:
            uploaded_file = st.file_uploader("Upload EDF File", type=["edf"])
            if uploaded_file and st.button("Process EEG"):
                with open("temp.edf", "wb") as f:
                    f.write(uploaded_file.getvalue())
                X, y, _ = preprocess_real_eeg("temp.edf")
                if X is not None:
                    st.session_state.eeg_data = X[0]
                    st.session_state.true_class = y[0] if y is not None else -1
                    st.session_state.data_type = 'real'
                    st.session_state.last_update = time.time()
                else:
                    st.error("Failed to process EEG file.")
        
        st.write(f"Sample age: {time.time() - st.session_state.last_update:.1f}s")
        if st.session_state.data_type == 'synthetic':
            st.write(f"True phoneme: **{['a','e','i','o','u','p','t','k'][st.session_state.true_class]}**")
        
        # Channel and time selection
        channels = st.multiselect("Select Channels", 
                                [f'Ch {i}' for i in range(14)], 
                                default=[f'Ch {i}' for i in range(4)])
        time_range = st.slider("Time Range (ms)", 0, 1000, (0, 1000), step=10)

    # Main display columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Neural Signals")
        # Plot selected channels
        fig = go.Figure()
        channel_indices = [int(ch.split()[1]) for ch in channels]
        start_idx, end_idx = int(time_range[0] * 1), int(time_range[1] * 1)
        for ch in channel_indices:
            fig.add_trace(go.Scatter(
                y=st.session_state.eeg_data[ch, start_idx:end_idx],
                name=f'Channel {ch}',
                line=dict(width=2)
            ))
        fig.update_layout(height=300, showlegend=True, xaxis_title="Time (ms)")
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
                value=['a','e','i','o','u','p','t','k'][pred.item()],
                delta=f"{confidence:.1%} confidence",
                delta_color="normal"
            )
            
            # Confidence chart
            st.subheader("Class Probabilities")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['a','e','i','o','u','p','t','k'],
                y=probs.numpy(),
                marker_color=['#1f77b4']*8
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        # Training history
        st.subheader("Training History")
        try:
            img = Image.open('training_history.png')
            st.image(img, caption="Loss and Accuracy Curves", use_column_width=True)
        except FileNotFoundError:
            st.info("Training history not available. Run training first.")

    with col2:
        st.subheader("Brain Activation Map")
        # Topographic map
        topo_img = plot_topographic_map(st.session_state.eeg_data)
        st.image(topo_img, caption="Topographic EEG Activity", use_column_width=True)
        
        # Confusion matrix
        st.subheader("Model Performance")
        try:
            img = Image.open('confusion_matrix.png')
            st.image(img, caption="Confusion Matrix", use_column_width=True)
        except FileNotFoundError:
            st.info("Confusion matrix not available. Run training first.")

if __name__ == "__main__":
    main()