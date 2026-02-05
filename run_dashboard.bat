@echo off
echo Starting Customer Intelligence Dashboard...
echo.

REM Go to project folder
cd /d "%~dp0"

REM Run Streamlit with venv Python
".\.venv\Scripts\python.exe" -m streamlit run app/dashboard.py

pause
