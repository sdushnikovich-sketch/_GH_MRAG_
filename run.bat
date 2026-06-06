@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat goto NOVENV
call .venv\Scripts\activate.bat
echo Starting PMOOS-RAG ... a browser tab will open.
echo To stop: press Ctrl+C in this window.
streamlit run app\hub.py
pause
goto END
:NOVENV
echo [ERROR] Environment not found. Run install.bat first.
pause
:END
