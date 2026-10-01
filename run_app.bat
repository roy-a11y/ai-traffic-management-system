@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
if errorlevel 1 (
  echo Failed to activate the Python virtual environment.
  pause
  exit /b 1
)

python -m streamlit run app.py --server.port 8501
if errorlevel 1 pause