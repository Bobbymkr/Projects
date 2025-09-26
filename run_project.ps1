# PowerShell script to run the Adaptive Traffic project
Write-Host "🚦 Adaptive Traffic Signal Control System" -ForegroundColor Green
Write-Host "=" * 60

# Test Python availability
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python first." -ForegroundColor Red
    exit 1
}

# Check if virtual environment exists
if (Test-Path ".venv") {
    Write-Host "✅ Virtual environment found" -ForegroundColor Green
    
    # Try to activate and run
    try {
        & .venv\Scripts\python.exe simple_test.py
    } catch {
        Write-Host "⚠️ Running with system Python..." -ForegroundColor Yellow
        python simple_test.py
    }
} else {
    Write-Host "⚠️ No virtual environment found. Running with system Python..." -ForegroundColor Yellow
    python simple_test.py
}

Write-Host "`nProject execution completed!" -ForegroundColor Green