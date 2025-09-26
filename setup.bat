@echo off
echo Setting up Adaptive Traffic Project...
echo.

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment and install dependencies
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing core dependencies...
pip install --upgrade pip
pip install numpy matplotlib opencv-python tqdm
pip install gymnasium stable-baselines3
pip install pytest ultralytics
pip install optuna

echo Installing project in development mode...
pip install -e .

echo.
echo Setup complete! You can now run:
echo   python demo.py
echo.
pause