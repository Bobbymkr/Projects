@echo off
REM ============================================================================
REM Adaptive Traffic Control System - Proof of Concept Launcher (Windows)
REM ============================================================================

echo.
echo ╔══════════════════════════════════════════════════════════════════════╗
echo ║     ADAPTIVE TRAFFIC CONTROL SYSTEM - POC LAUNCHER                   ║
echo ║     Comprehensive Demonstration - All Branches Integrated            ║
echo ╚══════════════════════════════════════════════════════════════════════╝
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✓ Python detected
echo.

REM Check for NumPy (required dependency)
python -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  NumPy not found. Installing...
    python -m pip install numpy
    if errorlevel 1 (
        echo ❌ ERROR: Failed to install NumPy
        pause
        exit /b 1
    )
    echo ✓ NumPy installed
)

echo ✓ Dependencies checked
echo.
echo Starting POC demonstration...
echo Runtime: ~10 minutes
echo Output: Console + HTML Dashboard + JSON Metrics
echo.
echo ════════════════════════════════════════════════════════════════════════
echo.

REM Run the POC
python "%~dp0proof_of_concept_comprehensive.py"

if errorlevel 1 (
    echo.
    echo ❌ POC execution failed
    pause
    exit /b 1
)

echo.
echo ════════════════════════════════════════════════════════════════════════
echo ✓ POC Complete!
echo.
echo Results saved to: poc_results\
echo   - comprehensive_metrics.json
echo   - poc_dashboard.html
echo   - regional_configs\
echo.
pause
