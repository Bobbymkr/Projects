@echo off
REM Test Runner Script for Adaptive Traffic Control System
REM Phase 1.3: Test Coverage Expansion

echo.
echo ========================================
echo    API TEST RUNNER
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

REM Check if test dependencies are installed
python -c "import pytest" 2>nul
if errorlevel 1 (
    echo Installing test dependencies...
    pip install -r requirements-test.txt
    if errorlevel 1 (
        echo ERROR: Failed to install test dependencies!
        echo.
        pause
        exit /b 1
    )
)

echo.
echo Running tests...
echo.

REM Run tests with coverage by default
python scripts\run_tests.py --coverage

if errorlevel 1 (
    echo.
    echo Tests failed! Check output above.
    pause
    exit /b 1
)

echo.
echo Tests completed successfully!
echo.
echo Coverage report available at: htmlcov\index.html
echo.
pause

