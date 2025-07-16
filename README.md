# EEG Speech Classifier with VLM Healthcare Integration

A brain-computer interface that classifies imagined speech from EEG signals using deep learning, enhanced with Vision Language Models for healthcare applications.

## Features

- **EEG Speech Classification**: CNN-LSTM model for imagined speech recognition
- **VLM Integration**: Clinical analysis using Vision Language Models
- **Healthcare Dashboard**: Interactive web interface for medical analysis
- **Natural Language Queries**: Ask questions about EEG data in plain English
- **Clinical Reports**: Automated healthcare risk assessment

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/tirth8205/EEGSpeech.git
cd EEGSpeech

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Run the Application

```bash
# Healthcare VLM Dashboard
streamlit run eegspeech/app/healthcare_vlm_app.py

# Original EEG App
streamlit run eegspeech/app/app.py
```

## Usage

### Healthcare Analysis

```bash
# CLI healthcare analysis
eegspeech-healthcare healthcare-analyze --patient-id Demo --age 45 --gender Male

# With real EEG file
eegspeech-healthcare healthcare-analyze --data-type real --file-path your_eeg.edf
```

### Training

```bash
# Train the model
eegspeech train --epochs 50 --batch-size 32

# Predict on new data
eegspeech predict --file-path test_data.edf
```

### Natural Language Interface

```python
from eegspeech.models.natural_language_interface import ConversationalEEGInterface

interface = ConversationalEEGInterface()
response = interface.chat("What is the stroke risk for this patient?", eeg_data)
```

## Project Structure

```text
├── eegspeech/
│   ├── app/
│   │   ├── healthcare_vlm_app.py    # VLM Healthcare Dashboard
│   │   ├── healthcare_cli.py        # Healthcare CLI
│   │   └── app.py                   # Original EEG App
│   ├── models/
│   │   ├── vlm_integration.py       # VLM Integration
│   │   ├── natural_language_interface.py  # Conversational AI
│   │   ├── predictive_healthcare.py # Healthcare Predictions
│   │   ├── model.py                 # CNN-LSTM Model
│   │   ├── dataset.py               # Data Processing
│   │   └── utils.py                 # Utilities
│   └── __init__.py
├── requirements.txt
├── setup.py
└── README.md
```

## Docker Deployment

```bash
# Build and run with Docker
docker build -t eeg-classifier .
docker run -p 8501:8501 eeg-classifier

# Or use docker-compose
docker-compose up
```

## Healthcare Applications

- **Risk Assessment**: Automated clinical risk scoring
- **Speech Pathology**: Early detection of speech disorders
- **Cognitive Screening**: Neurological condition assessment
- **Treatment Planning**: Personalized therapy recommendations

## Requirements

- Python 3.8+
- PyTorch
- Streamlit
- MNE-Python
- Transformers (for VLM)
- scikit-learn
- pandas, numpy, matplotlib

## Important Notes

### Healthcare Disclaimer

**FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY**

This software is intended for research and educational use only. It is not intended for clinical diagnosis, treatment, or patient care. All clinical decisions should be made by qualified healthcare professionals.

## License

MIT License - see LICENSE for details.
