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
    $py = ".venv\Scripts\python.exe"
} else {
    Write-Host "⚠️ No virtual environment found. Using system Python..." -ForegroundColor Yellow
    $py = "python"
}

# 1) Run quick smoke test
try {
    & $py simple_test.py
} catch {
    Write-Host "⚠️ Smoke test encountered an issue: $($_.Exception.Message)" -ForegroundColor Yellow
}

# 2) Run a focused pytest suite and export JUnit XML
try {
    if (-not (Test-Path "reports")) { New-Item -ItemType Directory -Path "reports" | Out-Null }
    & $py -m pytest -q --maxfail=1 --junitxml "reports\junit.xml" tests
} catch {
    Write-Host "⚠️ Tests encountered an issue: $($_.Exception.Message)" -ForegroundColor Yellow
}

# 3) Update Excel workbook with achievements (optional) and test results
try {
    $workbook = "AdaptiveTraffic_Workbook.xlsx"
    $junit = "reports\junit.xml"
    $achievements = "reports\achievements.json"
    $args = @("--workbook-path", $workbook, "--junit-xml", $junit)
    if (Test-Path $achievements) { $args += @("--achievements-json", $achievements) }
    & $py scripts\update_excel_workbook.py @args
} catch {
    Write-Host "⚠️ Workbook update encountered an issue: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host "`nProject execution completed!" -ForegroundColor Green