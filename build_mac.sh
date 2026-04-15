#!/bin/bash
echo "======================================================"
echo "BUILDING STANDALONE MACOS APP (Saturn V5)"
echo "======================================================"
echo "Note: This must be run on a MacBook."
# Check for python3
if ! command -v python3 &> /dev/null
then
    echo "ERROR: python3 could not be found."
    echo "Please install it from https://www.python.org/ or run 'xcode-select --install' in Terminal."
    exit
fi

echo "Installing requirements if needed..."
python3 -m pip install pyinstaller numpy pandas matplotlib tifffile scikit-image opencv-python scipy Pillow xlsxwriter python-pptx requests seaborn

echo "Starting build..."
python3 -m PyInstaller sperm_tool.spec --noconfirm

echo ""
echo "======================================================"
echo "BUILD COMPLETE!"
echo "Your app is in: dist/SpermAnalysisTool.app"
echo "======================================================"
