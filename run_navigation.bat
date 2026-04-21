@echo off
setlocal

REM ── Activate the project venv and run the navigation app ─────────────────

if not exist venv\Scripts\activate.bat (
    echo ERROR: Virtual environment not found.
    echo Please run setup_venv.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

if not exist topformer.onnx (
    echo TopFormer ONNX model not found.
    echo Running conversion now ^(one-time, ~1 min^)...
    echo.
    python convert_topformer_onnx.py
    if errorlevel 1 (
        echo ERROR: ONNX conversion failed. See messages above.
        pause
        exit /b 1
    )
    echo.
)

echo Starting Obs-tackle Navigation Assistance System...
echo Press Q or Esc to quit, S for screenshot, M to mute/unmute.
echo.
python main.py

if errorlevel 1 (
    echo.
    echo Application exited with an error. See messages above.
    pause
)
