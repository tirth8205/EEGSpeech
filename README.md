# EEGSpeech Classifier

A neural speech decoding application using EEG data.

## Setup
1. Clone the repository: `git clone https://github.com/tirth8205/EEGSpeech.git`
2. Create a virtual environment: `python -m venv .venv`
3. Activate the virtual environment: `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Training the Model
1. Run the training script: `python -m src.eegspeech.scripts.train`
2. This generates `outputs/eeg_speech_classifier.pth`.

## Running the Application
1. Ensure `outputs/eeg_speech_classifier.pth` exists (run training script if needed).
2. Run the Streamlit app: `streamlit run src/eegspeech/app/app.py`

## Notes
- Model checkpoint (`eeg_speech_classifier.pth`) and plots (`*.png`) are stored in `outputs/` and ignored by `.gitignore`.
- Place EEG data files in `src/eegspeech/data/`.
