#!/bin/bash

# PPO-CodeT5 Installation Script
echo "🚀 Installing PPO-CodeT5..."

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected"
else
    echo "❌ Python 3.8+ is required. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "📥 Downloading NLTK data..."
python3 -c "
import nltk
import os
nltk.data.path.append('./cache/nltk')
os.makedirs('./cache/nltk', exist_ok=True)
nltk.download('wordnet', download_dir='./cache/nltk', quiet=True)
nltk.download('averaged_perceptron_tagger', download_dir='./cache/nltk', quiet=True)
nltk.download('stopwords', download_dir='./cache/nltk', quiet=True)
print('✅ NLTK data downloaded successfully')
"

# Create cache directories
echo "📁 Creating cache directories..."
mkdir -p cache/hf cache/gensim cache/nltk
mkdir -p 01_Test outputs results

echo "✅ Installation completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "source venv/bin/activate"
echo ""
echo "To run the project, execute the Jupyter notebook:"
echo "jupyter notebook 01_ppo_codet5.ipynb"
