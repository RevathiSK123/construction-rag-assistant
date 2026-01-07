@echo off
echo ========================================
echo 🏗️  CONSTRUCTION RAG - ULTRA SIMPLE INSTALL
echo ========================================
echo.

echo Deleting old virtual environment...
if exist "venv" rmdir /s /q venv

echo Creating new virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo Upgrading pip and setuptools...
python -m pip install --upgrade pip setuptools wheel

echo Installing FAISS (pre-compiled wheel)...
pip install faiss-cpu --no-deps --no-cache-dir
pip install numpy==1.24.3

echo Installing minimal requirements...
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install pandas==2.1.4
pip install streamlit==1.28.1
pip install openai==1.3.0
pip install python-dotenv==1.0.0
pip install httpx==0.25.2

echo.
echo ✅ INSTALLATION COMPLETE!
echo.
pause