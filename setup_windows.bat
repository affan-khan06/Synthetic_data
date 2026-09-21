@echo off
setlocal
cd /d "%~dp0"
echo Borrower Data - real UCI dataset prototype setup
set "PROJECT_PY="
py -3.12 -c "import sys; assert sys.maxsize > 2**32" >nul 2>nul
if not errorlevel 1 set "PROJECT_PY=py -3.12"
if defined PROJECT_PY goto found
py -3.11 -c "import sys; assert sys.maxsize > 2**32" >nul 2>nul
if not errorlevel 1 set "PROJECT_PY=py -3.11"
if defined PROJECT_PY goto found
python -c "import sys; assert sys.version_info[:2] in [(3,11),(3,12)] and sys.maxsize > 2**32" >nul 2>nul
if not errorlevel 1 set "PROJECT_PY=python"
if not defined PROJECT_PY goto missing
:found
if exist .venv\Scripts\python.exe goto checkvenv
%PROJECT_PY% -m venv .venv
if errorlevel 1 goto failed
:checkvenv
.venv\Scripts\python.exe -c "import sys; assert sys.version_info[:2] in [(3,11),(3,12)]"
if errorlevel 1 goto wrongvenv
echo Installing dependencies. This can take several minutes; progress is recorded in setup.log.
echo Keep this window open until Setup complete.
.venv\Scripts\python.exe -m pip install --upgrade pip > setup.log 2>&1
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cpu >> setup.log 2>&1
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -r requirements.txt >> setup.log 2>&1
if errorlevel 1 goto failed
.venv\Scripts\python.exe -c "import torch, streamlit, plotly, pandas, scipy, sklearn; print('Dependencies ready.')"
if errorlevel 1 goto failed
echo Setup complete. Run start.bat. No activation needed.
pause
exit /b 0
:missing
echo Install 64-bit Python 3.12 from python.org, then rerun setup.
echo If you have the Python install manager, you can use: py install 3.12
pause
exit /b 1
:wrongvenv
echo This folder contains a virtual environment from another Python version.
echo Rename .venv to .venv_old, then rerun setup to create a Python 3.11/3.12 environment.
pause
exit /b 1
:failed
echo Setup failed. Full installation details are in setup.log.
if exist setup.log type setup.log
pause
exit /b 1
