# EEGSpeech: Neural Speech Decoding from EEG Signals

![Python](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Docker](https://img.shields.io/badge/docker-supported-blue)
![Status](https://img.shields.io/badge/status-active-brightgreen)

A robust brain-computer interface (BCI) for decoding speech phonemes from EEG signals using a hybrid CNN-LSTM model. Supports synthetic and real EEG data, with interactive visualizations and containerized deployment for research and production.

## 📌 Project Overview

**Objective**: Decode 8 speech phonemes (/a/, /e/, /i/, /o/, /u/, /p/, /t/, /k/) from EEG signals using deep learning.

**Key Features**:
- Hybrid CNN-LSTM model for accurate phoneme classification.
- Synthetic EEG data with vowel and consonant patterns, plus real EEG (EDF) support.
- Streamlit app with topographic scalp maps, Grad-CAM, and performance metrics.
- Robust training with k-fold cross-validation and metrics (accuracy, F1-score).
- Dockerized for consistent deployment.
- CLI for training, evaluation, prediction, and analysis.
- Grad-CAM for model interpretability.

## 🗂 Project Structure

```
eegspeech/
├── eegspeech/
│   ├── app/
│   │   ├── app.py        # Streamlit web application
│   │   └── cli.py        # Command-line interface
│   ├── models/
│   │   ├── dataset.py    # Data generation and preprocessing
│   │   ├── model.py      # CNN-LSTM model
│   │   ├── train.py      # Training with k-fold
│   │   └── utils.py      # Visualization utilities
├── Dockerfile            # Containerized deployment
├── requirements.txt      # Dependencies
├── setup.py             # Package configuration
├── README.md            # Documentation
└── LICENSE              # MIT License
```

## 🚀 Installation

### Prerequisites
- Python 3.8–3.10
- Git
- Docker (optional, for containerized deployment)

### Local Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/eegspeech.git
   cd eegspeech
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

### Docker Installation
1. Build the Docker image:
   ```bash
   docker build -t eegspeech .
   ```
2. Run the Streamlit app:
   ```bash
   docker run -p 8501:8501 eegspeech
   ```
   Access at `http://localhost:8501`.
3. Run CLI commands:
   ```bash
   docker run eegspeech eegspeech train --epochs 50
   ```

## 🧠 Usage

### Training the Model
Train with synthetic data and k-fold cross-validation:
```bash
eegspeech train --epochs 50 --batch-size 32 --lr 0.001 --kfold
```
Train with real EEG data:
```bash
eegspeech train --data-type real --file-path path/to/eeg.edf --output models/model.pth
```

**Example Output**:
```
Fold 1/5
Epoch 1/50 | Train Loss: 2.079 | Val Loss: 1.845 | Val Acc: 0.325 | Val F1: 0.310 | Time: 12.3s
...
Test Accuracy: 0.892 | Precision: 0.895 | Recall: 0.892 | F1: 0.893
Model saved to models/model.pth
```

### Running the Streamlit App
```bash
streamlit run eegspeech/app/app.py
```
**Features**:
- Upload real EEG data (EDF) or generate synthetic samples.
- Visualize EEG signals, topographic maps, and Grad-CAM.
- View training history and confusion matrix.
- Select channels and time ranges interactively.

### Making Predictions
Predict phonemes from an EDF file:
```bash
eegspeech predict --input-file path/to/eeg.edf --output-file predictions.txt
```

### Analyzing the Model
Check model complexity and generate Grad-CAM:
```bash
eegspeech analyze --grad-cam
```

## 🧬 Technical Details

### Model Architecture
- **Hybrid CNN-LSTM**:
  - 3-layer CNN (32→64→128 filters) with batch normalization.
  - 2-layer bidirectional LSTM (256 hidden units).
  - Classifier (512→128→8).
  - ~1.5M parameters, ~500M FLOPs.
- **Regularization**: Dropout (0.4), weight decay (0.001).

### Data Processing
- **Synthetic**: 14 channels, 1000 Hz, with vowel (/a/, /e/, /i/, /o/, /u/) and consonant (/p/, /t/, /k/) patterns, augmented with noise and time warping.
- **Real EEG**: EDF files, 1–50 Hz band-pass filtering, 1-second epochs.

### Training Pipeline
- **Optimizer**: Adam (tuned learning rate).
- **Loss**: CrossEntropy.
- **Validation**: 5-fold cross-validation.
- **Metrics**: Accuracy, precision, recall, F1-score, confusion matrix.

## 🔮 Potential Extensions
- Integrate with BCI headsets (e.g., OpenBCI).
- Expand to syllable or word decoding.
- Deploy as a FastAPI service.
- Support real-time EEG streaming.

## 📚 References
- Jayalath, D., Landau, G., Shillingford, B., Woolrich, M., & Parker Jones, O. (2024). "The Brain's Bitter Lesson: Scaling Speech Decoding With Self-Supervised Learning." *arXiv preprint arXiv:2406.04328*. https://arxiv.org/abs/2406.04328[](https://www.researchgate.net/publication/381227083_The_Brain%27s_Bitter_Lesson_Scaling_Speech_Decoding_With_Self-Supervised_Learning)
- MNE-Python for EEG processing: https://mne.tools
- PyTorch for deep learning: https://pytorch.org
- Streamlit for UI: https://streamlit.io

## 🤝 Contributing
See `CONTRIBUTING.md` for guidelines. Issues and PRs welcome on GitHub.

## 📜 License
MIT License. See `LICENSE` for details.

## 📸 Screenshots
*(Add screenshots of the Streamlit app, e.g., topographic map or Grad-CAM, via GitHub-hosted images.)*
```