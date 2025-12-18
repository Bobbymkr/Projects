#!/bin/bash
# ============================================================================
# Adaptive Traffic Control System - Proof of Concept Launcher (Linux/Mac)
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║     ADAPTIVE TRAFFIC CONTROL SYSTEM - POC LAUNCHER                   ║"
echo "║     Comprehensive Demonstration - All Branches Integrated            ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python 3 is not installed"
    echo ""
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "✓ Python detected: $(python3 --version)"
echo ""

# Check for NumPy (required dependency)
if ! python3 -c "import numpy" &> /dev/null; then
    echo "⚠️  NumPy not found. Installing..."
    python3 -m pip install numpy
    if [ $? -ne 0 ]; then
        echo "❌ ERROR: Failed to install NumPy"
        exit 1
    fi
    echo "✓ NumPy installed"
fi

echo "✓ Dependencies checked"
echo ""
echo "Starting POC demonstration..."
echo "Runtime: ~10 minutes"
echo "Output: Console + HTML Dashboard + JSON Metrics"
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# Run the POC
python3 "$(dirname "$0")/proof_of_concept_comprehensive.py"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ POC execution failed"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "✓ POC Complete!"
echo ""
echo "Results saved to: poc_results/"
echo "  - comprehensive_metrics.json"
echo "  - poc_dashboard.html"
echo "  - regional_configs/"
echo ""
