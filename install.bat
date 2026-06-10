@echo off
chcp 65001 >nul
cd /d "%~dp0"
REM ---- Robust against slow / unreliable internet (pythonhosted read timeouts) ----
set PIP_DEFAULT_TIMEOUT=120
echo ============================================================
echo   PMOOS-RAG v0.14.6 Modular - install (Windows)
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
python -m pip install --upgrade pip --timeout 30 --retries 2

echo.
echo [3/4] Installing PyTorch with CUDA 12.4 ...
echo     For NVIDIA GPU such as RTX 3070 Ti. No GPU - see README, section GPU/CPU.
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --timeout 180 --retries 10
if not errorlevel 1 goto TORCHOK
echo WARNING: GPU torch failed. Installing CPU build instead ...
pip install torch==2.6.0 --timeout 180 --retries 10
:TORCHOK

echo.
echo [4/4] Installing dependencies (NumPy + the rest) ...
echo     Attempt 1: default index, fail-fast ...
pip install --prefer-binary --timeout 60 --retries 2 -r requirements.txt
if not errorlevel 1 goto DEPSOK

echo.
echo     Default index (pythonhosted) is unreliable on this network.
echo     Switching to a reliable mirror for ALL remaining downloads ...
set PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
set PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

echo     Attempt 2: same packages from the mirror ...
pip install --prefer-binary --timeout 120 --retries 10 -r requirements.txt
if not errorlevel 1 goto DEPSOK

echo.
echo     Attempt 3: installing packages ONE BY ONE from the mirror (kept on success) ...
for /f "usebackq eol=# tokens=1 delims= " %%P in ("requirements.txt") do (
  echo        package: %%P
  pip install --prefer-binary --timeout 120 --retries 10 "%%P"
)
:DEPSOK

echo.
echo Verifying NumPy + GPU ...
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo.
echo ============================================================
echo   If a package was still missed, just run install.bat again
echo   (it resumes from cache / mirror).
echo   Start the UI:      run.bat
echo   Run one module:    run_module.bat modules\module1_inventory.py --project "Name"
echo ------------------------------------------------------------
echo   Set API keys in each module tab, or copy .env.example to:
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
