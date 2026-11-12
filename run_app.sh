#!/bin/bash
# ========================================================
# Script 2: Run Application (Web App & Desktop GUI)
# Project: Classical Image Encryption (AetherCipher)
# ========================================================

cd "$(dirname "$0")"

echo "========================================================"
echo "  AetherCipher - Starting Application"
echo "========================================================"

# If user passes --gui flag, launch Tkinter Desktop GUI
if [ "$1" == "--gui" ] || [ "$1" == "-g" ]; then
    echo "🖥 Launching Desktop GUI (gui_image_encryptor.py)..."
    exec python3 gui_image_encryptor.py
fi

# Default: Start Web Application
echo "🌐 Starting Web Application on http://127.0.0.1:5000 ..."
echo "💡 Tip: You can also launch the Desktop GUI using: ./run_app.sh --gui"
echo ""

# Automatically open the browser tab on macOS or Linux
(
    sleep 1.5
    if command -v open &> /dev/null; then
        open "http://127.0.0.1:5000"
    elif command -v xdg-open &> /dev/null; then
        xdg-open "http://127.0.0.1:5000"
    fi
) &

# Run Flask Web App
exec python3 app.py
