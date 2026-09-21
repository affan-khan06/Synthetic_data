@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe goto missing
.venv\Scripts\python.exe -c "import streamlit, torch, pandas, plotly, scipy, sklearn" >nul 2>nul
if errorlevel 1 goto missing
echo Open http://127.0.0.1:8501 after the server starts. Keep this window open.
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
if errorlevel 1 pause
exit /b
:missing
echo Run setup_windows.bat first and wait for Setup complete.
pause
exit /b 1
