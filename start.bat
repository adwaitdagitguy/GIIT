@echo off
echo ===================================================
echo   SciML Intervention Workbench - Launcher
echo ===================================================

echo [1/2] Launching FastAPI Backend on port 8000...
start cmd /k "python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

echo [2/2] Launching React Vite Frontend on port 5173...
cd frontend
start cmd /k "npm run dev"

echo Done! Open your browser at http://localhost:5173
pause
