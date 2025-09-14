@echo off
REM PPO-CodeT5 Installation Script for Windows

echo 🚀 Installing PPO-CodeT5...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH. Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python detected

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Download NLTK data
echo 📥 Downloading NLTK data...
python -c "import nltk; import os; nltk.data.path.append('./cache/nltk'); os.makedirs('./cache/nltk', exist_ok=True); nltk.download('wordnet', download_dir='./cache/nltk', quiet=True); nltk.download('averaged_perceptron_tagger', download_dir='./cache/nltk', quiet=True); nltk.download('stopwords', download_dir='./cache/nltk', quiet=True); print('✅ NLTK data downloaded successfully')"

REM Create cache directories
echo 📁 Creating cache directories...
if not exist "cache" mkdir cache
if not exist "cache\hf" mkdir cache\hf
if not exist "cache\gensim" mkdir cache\gensim
if not exist "cache\nltk" mkdir cache\nltk
if not exist "01_Test" mkdir 01_Test
if not exist "outputs" mkdir outputs
if not exist "results" mkdir results

echo ✅ Installation completed successfully!
echo.
echo To activate the environment, run:
echo venv\Scripts\activate.bat
echo.
echo To run the project, execute the Jupyter notebook:
echo jupyter notebook 01_ppo_codet5.ipynb
pause
