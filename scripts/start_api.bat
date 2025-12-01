@echo off
REM Start the Production FastAPI API Server
REM Adaptive Traffic Control System

echo.
echo ========================================
echo    ADAPTIVE TRAFFIC CONTROL API SERVER
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv .venv
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if API dependencies are installed
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo Installing API dependencies...
    pip install -r requirements-api.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies!
        echo.
        pause
        exit /b 1
    )
)

echo.
echo Starting API server...
echo.
echo API will be available at:
echo   - Documentation: http://localhost:8000/api/docs
echo   - Health Check: http://localhost:8000/health
echo   - API Root: http://localhost:8000/
echo.

python scripts\start_api.py --host 0.0.0.0 --port 8000

pause

