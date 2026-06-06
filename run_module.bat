@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat goto NOVENV
call .venv\Scripts\activate.bat
REM Example:
REM   run_module.bat modules\module1_inventory.py --project "OPOCHKA 83-26S" --uploads files
python %*
pause
goto END
:NOVENV
echo [ERROR] Environment not found. Run install.bat first.
pause
:END
