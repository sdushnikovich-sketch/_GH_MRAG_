@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo   PMOOS-RAG v0.14.0 Modular - install (Windows)
echo ============================================================

where python >nul 2>nul
if errorlevel 1 goto NOPY

echo.
echo [1/4] Creating virtual environment .venv ...
if exist .venv\Scripts\activate.bat goto VENVOK
python -m venv .venv
:VENVOK
call .venv\Scripts\activate.bat

echo.
echo [2/4] Upgrading pip ...
python -m pip install --upgrade pip

echo.
echo [3/4] Installing PyTorch with CUDA 12.4 ...
echo     For NVIDIA GPU such as RTX 3070 Ti. No GPU - see README, section GPU/CPU.
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
if not errorlevel 1 goto TORCHOK
echo WARNING: GPU torch install failed. Installing CPU build instead ...
pip install torch==2.6.0
:TORCHOK

echo.
echo [4/4] Installing dependencies ...
pip install -r requirements.txt

echo.
echo Checking GPU availability ...
python -c "import torch;print('CUDA available:',torch.cuda.is_available())"

echo.
echo ============================================================
echo   Install finished.
echo   Start the UI:      run.bat
echo   Run one module:    run_module.bat modules\module1_inventory.py --project "Name"
echo ------------------------------------------------------------
echo   Set API keys in the UI sidebar, or copy .env.example to:
echo   %USERPROFILE%\.pmoos-rag\.env
echo ============================================================
pause
goto END

:NOPY
echo [ERROR] Python not found.
echo Install Python 3.10-3.12 from https://python.org
echo and enable "Add Python to PATH" during setup, then re-run install.bat
pause

:END
