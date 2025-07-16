#!/bin/bash

# EEGSpeech Healthcare - One-Click Setup Script
# This script sets up the complete VLM-enhanced EEG healthcare system

set -e  # Exit on any error

echo "🧠 EEGSpeech Healthcare Setup"
echo "==============================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.8+ and try again.${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${BLUE}🐍 Python version: $PYTHON_VERSION${NC}"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git is not installed. Please install Git and try again.${NC}"
    exit 1
fi

# Create project directory
PROJECT_DIR="EEGSpeech-Healthcare"
if [ -d "$PROJECT_DIR" ]; then
    echo -e "${YELLOW}📁 Project directory already exists. Updating...${NC}"
    cd "$PROJECT_DIR"
    git pull origin main
else
    echo -e "${BLUE}📥 Cloning repository...${NC}"
    git clone https://github.com/tirth8205/EEGSpeech-Healthcare.git
    cd "$PROJECT_DIR"
fi

# Create virtual environment
echo -e "${BLUE}🔧 Creating virtual environment...${NC}"
python3 -m venv .venv

# Activate virtual environment
echo -e "${BLUE}⚡ Activating virtual environment...${NC}"
source .venv/bin/activate

# Upgrade pip
echo -e "${BLUE}📦 Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "${BLUE}📚 Installing dependencies...${NC}"
pip install -r requirements.txt

# Install package in development mode
echo -e "${BLUE}🔧 Installing EEGSpeech Healthcare package...${NC}"
pip install -e .

# Install additional dependencies for VLM
echo -e "${BLUE}🧠 Installing VLM dependencies...${NC}"
pip install kaleido  # For plotly image export

# Create necessary directories
echo -e "${BLUE}📁 Creating necessary directories...${NC}"
mkdir -p data
mkdir -p results
mkdir -p models

# Test installation
echo -e "${BLUE}🧪 Testing installation...${NC}"
python -c "
import eegspeech
from eegspeech.models.vlm_integration import HealthcareVLMPipeline
from eegspeech.models.natural_language_interface import ConversationalEEGInterface
print('✅ All imports successful!')
"

# Check if CLI commands work
echo -e "${BLUE}🔍 Testing CLI commands...${NC}"
eegspeech-healthcare --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ CLI commands working correctly${NC}"
else
    echo -e "${YELLOW}⚠️  CLI commands may need shell restart${NC}"
fi

# Success message
echo ""
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "==============================="
echo ""
echo -e "${GREEN}🎉 EEGSpeech Healthcare is now installed and ready to use!${NC}"
echo ""
echo -e "${BLUE}🚀 Quick Start Commands:${NC}"
echo ""
echo "1. Healthcare Dashboard:"
echo "   streamlit run eegspeech/app/healthcare_vlm_app.py"
echo ""
echo "2. CLI Analysis:"
echo "   eegspeech-healthcare healthcare-analyze --patient-id Demo --age 45 --gender Male"
echo ""
echo "3. Original EEG App:"
echo "   streamlit run eegspeech/app/app.py"
echo ""
echo "4. Train Model:"
echo "   eegspeech train --epochs 20 --batch-size 32"
echo ""
echo -e "${YELLOW}📚 Documentation: README.md${NC}"
echo -e "${YELLOW}🐛 Issues: https://github.com/tirth8205/EEGSpeech-Healthcare/issues${NC}"
echo ""
echo -e "${GREEN}Happy analyzing! 🧠💊${NC}"

# Create a launcher script
echo -e "${BLUE}📝 Creating launcher script...${NC}"
cat > run_healthcare_app.sh << 'EOF'
#!/bin/bash
# EEGSpeech Healthcare Launcher

echo "🧠 Starting EEGSpeech Healthcare Dashboard..."
echo "Access the app at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Run the healthcare app
streamlit run eegspeech/app/healthcare_vlm_app.py
EOF

chmod +x run_healthcare_app.sh

echo -e "${GREEN}✅ Launcher script created: run_healthcare_app.sh${NC}"
echo ""
echo -e "${BLUE}🎯 To start the app anytime, just run:${NC}"
echo "   ./run_healthcare_app.sh"
echo ""
echo -e "${GREEN}Setup completed successfully! 🎉${NC}"