@echo off
echo Setting up Construction RAG Assistant with 3 documents...
echo.

REM Create necessary folders
if not exist "documents" mkdir documents
if not exist "vector_store" mkdir vector_store

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo Python is not installed. Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install streamlit==1.28.1 openai==1.3.0 python-dotenv==1.0.0 numpy==1.24.3 pandas==2.1.4

echo.
echo ✅ SETUP COMPLETE!
echo.
echo IMPORTANT NEXT STEPS:
echo 1. Get free API key from https://openrouter.ai
echo 2. Edit .env file with your key: OPENROUTER_API_KEY=your-key
echo 3. Download 3 documents to documents/ folder
echo 4. Run: streamlit run app.py
echo.
pause