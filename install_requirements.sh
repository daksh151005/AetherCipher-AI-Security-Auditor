#!/bin/bash
# ========================================================
# Script 1: Download & Install All Requirements
# Project: Classical Image Encryption (AetherCipher)
# ========================================================

set -e

# Change directory to the script's directory
cd "$(dirname "$0")"

echo "========================================================"
echo "  AetherCipher - Installing Dependencies"
echo "========================================================"

# 1. Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed or not in PATH."
    exit 1
fi

echo "✔ Found Python: $(python3 --version)"

# 2. Check Pip
if ! python3 -m pip --version &> /dev/null; then
    echo "❌ Error: pip is not available for python3."
    exit 1
fi

echo "📦 Installing required Python libraries from requirements.txt..."
python3 -m pip install --upgrade pip --quiet 2>/dev/null || true

# 3. Install packages
if python3 -m pip install -r requirements.txt; then
    echo ""
    echo "========================================================"
    echo "✔ All dependencies installed successfully!"
    echo "  You can now run: ./run_app.sh"
    echo "========================================================"
else
    echo "Retrying with --break-system-packages (for managed environments)..."
    python3 -m pip install -r requirements.txt --break-system-packages
    echo ""
    echo "========================================================"
    echo "✔ All dependencies installed successfully!"
    echo "  You can now run: ./run_app.sh"
    echo "========================================================"
fi
