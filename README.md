# EYE-CUE FYP

EYE-CUE is a real-time navigation assistance prototype for visually impaired users.
It combines monocular depth estimation and semantic segmentation, then fuses both streams
to detect nearby obstacles and generate spoken movement guidance.

## What the system does

- Captures live camera frames.
- Runs **Depth Anything V2** for depth perception.
- Runs **TopFormer (ONNX)** for ADE20K semantic segmentation.
- Fuses depth + class information into obstacle reasoning.
- Uses a path-planning module to produce guidance instructions.
- Provides audio feedback and optional assistive utilities (time/weather, SOS, short clip save).

## Main project structure

- `main.py` - app entrypoint.
- `nav_assist/` - core pipeline modules (workers, planner, visualization, audio, AI helpers).
- `tests/` - test suite for planning/obstacle logic.
- `convert_topformer_onnx.py` - exports TopFormer to ONNX.
- `run_navigation.bat` - Windows helper script to launch the app.
- `setup_venv.bat` - Windows helper script to create and prepare a virtual environment.

## Requirements

- Python 3.10+ (recommended).
- Webcam.
- Optional GPU for better real-time performance.
- Model assets:
  - Depth checkpoint (expected by config in project root).
  - `topformer.onnx` (auto-generated if missing via conversion script).

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Windows (recommended in this repo):

```bat
setup_venv.bat
run_navigation.bat
```

Or directly:

```bash
python main.py
```

## Keyboard controls (current app)

- `Q` / `Esc` - quit
- `S` - save screenshot
- `M` - mute/unmute guidance
- `D` - AI surrounding description
- `W` - AI wardrobe suggestion
- `T` - announce time and weather
- `C` - announce current time/date (offline)
- `A` - toggle sonar beeps for proximity
- `V` - save last 30 seconds as clip
- `E` - send SOS alert

## Notes

- Large model/artifact files are intentionally ignored by `.gitignore`.
- Snapshot archive folders (`fyp-*`) are local version archives and are not part of the tracked app source.
