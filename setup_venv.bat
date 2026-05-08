@echo off
setlocal EnableDelayedExpansion

echo ================================================================
echo   Navigation Assistance System - Virtual Environment Setup
echo ================================================================
echo.

REM Check Python is available
python --version >NUL 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH.
    echo Please install Python 3.9+ from https://python.org and try again.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python --version') do echo Found: %%v

REM Create virtual environment
if exist venv (
    echo venv already exists - skipping creation.
) else (
    echo [1/5] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [4/5] Installing PyTorch (CPU build)...
echo       This may take several minutes...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch.
    echo Try manually: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pause
    exit /b 1
)

echo [5/5] Installing runtime dependencies...
pip install onnxruntime opencv-python numpy matplotlib pyttsx3 scikit-image pytest
if errorlevel 1 (
    echo ERROR: Failed to install runtime dependencies.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo   Setup complete!
echo ================================================================
echo.
echo NEXT STEPS (run in order):
echo.
echo   Step 1 - Activate venv (do this in every new terminal):
echo     venv\Scripts\activate.bat
echo.
echo   Step 2 - Convert TopFormer to ONNX (one-time, ~1 min):
echo     python convert_topformer_onnx.py
echo.
echo   Step 3 - Launch the navigation system:
echo     python navigation_app.py
echo.
echo   OR simply double-click run_navigation.bat
echo.
pause
