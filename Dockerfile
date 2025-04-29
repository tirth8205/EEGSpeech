# Use a slim Python base image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for mne, matplotlib, and other libraries
RUN apt-get update && apt-get install -y \
    libatlas-base-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install the eegspeech package
RUN pip install -e .

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Default command to run Streamlit app
CMD ["streamlit", "run", "eegspeech/app/app.py"]
