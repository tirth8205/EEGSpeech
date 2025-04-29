# Contributing to EEGSpeech

Thank you for your interest in contributing to EEGSpeech! This project aims to advance brain-computer interfaces (BCIs) for speech decoding, and we welcome contributions from the community to make it better. Whether you're fixing bugs, adding features, improving documentation, or suggesting ideas, your help is appreciated.

## How to Contribute

### 1. Reporting Issues
If you find a bug, have a feature request, or encounter any issues:
- Check the [GitHub Issues](https://github.com/tirth8205/EEGSpeech/issues) page to see if the issue has already been reported.
- If not, open a new issue with a descriptive title and include:
  - A clear description of the problem or feature request.
  - Steps to reproduce the issue (if applicable).
  - Your environment (e.g., Python version, OS, Docker or local setup).
  - Any error messages or logs.

### 2. Submitting Pull Requests (PRs)
To contribute code, documentation, or other improvements:
1. **Fork the Repository**:
   - Fork the project on GitHub: https://github.com/tirth8205/EEGSpeech
   - Clone your fork:
     ```
     git clone https://github.com/tirth8205/EEGSpeech.git
     cd EEGSpeech
     ```
2. **Set Up the Environment**:
   - Follow the installation instructions in `README.md` (local or Docker setup).
   - Create a virtual environment and install dependencies:
     ```
     python3.10 -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     pip install -e .
     ```
3. **Create a Branch**:
   - Create a new branch for your changes:
     ```
     git checkout -b feature/your-feature-name
     ```
4. **Make Changes**:
   - Follow the coding standards below.
   - Test your changes locally (e.g., run the Streamlit app, train the model, or add unit tests).
5. **Commit and Push**:
   - Commit your changes with a clear message:
     ```
     git commit -m "Add feature: description of your change"
     ```
   - Push to your fork:
     ```
     git push origin feature/your-feature-name
     ```
6. **Open a Pull Request**:
   - Go to the original repository on GitHub and open a PR from your branch.
   - Provide a detailed description of your changes, referencing any related issues (e.g., "Fixes #123").
   - Wait for review and address any feedback.

### 3. Coding Standards
- **Code Style**: Follow PEP 8 guidelines. Use tools like `black` and `flake8` for formatting and linting (install via `pip install -e ".[dev]"`).
- **Documentation**: Update `README.md` or add comments for new features. Use docstrings for functions and classes.
- **Testing**: Add unit tests if possible (place in a `tests/` directory). Ensure existing functionality isn’t broken.
- **Dependencies**: If adding new dependencies, update `requirements.txt` and `setup.py`, and test compatibility.

### 4. Development Setup
- **Training**: Use the CLI to train the model (`eegspeech train --help` for options).
- **Running the App**: Test the Streamlit app locally (`streamlit run eegspeech/app/app.py`).
- **Docker**: Test changes in a Docker container if applicable (`docker build -t eegspeech .`).

### 5. Areas for Contribution
- Add support for real EEG datasets (e.g., OpenNeuro integration).
- Expand the phoneme set or decode syllables/words.
- Improve model performance (e.g., new architectures, hyperparameter tuning).
- Enhance visualizations (e.g., more interactive Streamlit features).
- Write unit tests for core functionality (`dataset.py`, `model.py`, etc.).

## Code of Conduct
- Be respectful and inclusive in all interactions.
- Avoid offensive language or behavior.
- Collaborate constructively and provide helpful feedback.

## Contact
For questions or discussions, open an issue on GitHub or reach out to the maintainers:
- Tirth (tirthkanani18@gmail.com)

## License
Contributions are licensed under the MIT License. See `LICENSE` for details.