@echo off
echo ========================================================
echo   Starting PathSnap Evaluator Dashboard on Laptop 2
echo ========================================================
cd %~dp0\..
call .venv\Scripts\activate
streamlit run ui/app.py
pause
