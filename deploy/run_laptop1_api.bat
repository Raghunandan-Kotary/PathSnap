@echo off
echo ========================================================
echo   Starting PathSnap Model Server on Laptop 1 (Port 8000)
echo ========================================================
cd %~dp0\..
call .venv\Scripts\activate
echo Bound to 0.0.0.0 (accessible across LAN)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
pause
