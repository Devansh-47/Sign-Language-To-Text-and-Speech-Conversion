# HandSpeak.ai Backend and App Pipeline

## What This Project Does
HandSpeak.ai provides real-time sign-language to text using a CNN model and MediaPipe hand landmarks. A FastAPI backend receives frames from the Android client, detects the hand, draws a skeleton on a 400x400 canvas, and classifies the gesture. The desktop app (final_pred.py) uses the same model and drawing pipeline for local webcam prediction.

## High-Level Pipeline
1) Capture frame
   - Desktop: webcam via OpenCV in final_pred.py
   - Mobile: Android app sends base64 frames over WebSocket to /ws/recognize
2) Detect hand bounding box with cvzone.HandDetector (hd)
3) Crop region around hand with offset padding
4) Detect 21 hand landmarks inside the crop with another HandDetector (hd2)
5) Draw skeleton on 400x400 white canvas using bbox-based offsets
6) CNN (cnn8grps_rad1_model.h5) predicts top classes; rule-based post-processing maps to final character/command (A-Z, space, backspace, next)
7) Suggestions/word handling done via enchant dictionary and core suggestion logic

## Key Components
- final_pred.py: Desktop GUI app (Tkinter) + webcam inference; reference pipeline
- server.py: FastAPI + WebSocket backend for mobile clients; mirrors final_pred.py logic
- app/core/*: Autocorrect, suggestions, TTS, session management, phrase DTW
- app/static/index.html: Frontend asset (if used)
- Models & data: models/cnn8grps_rad1_model.h5, app/data/hand_landmarker.task, lexicon_small.txt, phrase_signatures.json

## Files of Interest
- server.py: API + websocket; predict_sign() draws skeleton and classifies
- final_pred.py: Baseline working reference
- app/core: Suggestion, TTS, phrase logic
- requirements.txt: Backend + desktop dependencies
- runtime-requirements-gui.txt: GUI app deps

## How the Drawing/Prediction Works
- BBox: (x, y, w, h) from first detector on full frame
- Crop: frame[y-offset:y+h+offset, x-offset:x+w+offset]
- Landmarks: hd2 on cropped image -> pts (21 x [x,y,z])
- Offsets: os = ((400 - w) // 2) - 15; os1 likewise
- Skeleton: draw pts+offset on 400x400 white canvas
- Model input: canvas reshaped to (1, 400, 400, 3) -> CNN
- Post rules: Rule-set maps coarse class to final letter/command

## Running the Backend (Windows, PowerShell)
Prereqs: Python 3.x, git, build tools; optionally conda/venv

1) Create/activate venv (example)
   python -m venv .venv-server
   .venv-server\Scripts\Activate.ps1

2) Install deps
   pip install -r requirements.txt

3) Start server
   python server.py

4) API docs
   http://localhost:8000/docs

## WebSocket Usage
- Endpoint: ws://<HOST>:8000/ws/recognize
- Client sends: base64-encoded JPEG frame (string)
- Server responds JSON:
  {
    "success": true/false,
    "character": "A"|"B"|...|" "|"Backspace"|"next"|null,
    "handDetected": true/false,
    "message": "..."
  }

## REST Endpoint (upload image)
POST /api/recognize
- multipart/form-data file: image (jpeg/png)
- Response: { character, confidence?, suggestions? }

## Desktop App (reference)
- Activate GUI env (if separate)
- pip install -r runtime-requirements-gui.txt
- python final_pred.py

## Troubleshooting
- Unicode prints on Windows CP1252: use ASCII-only logs (already adjusted)
- If landmarks look out-of-bounds: ensure offsets use bbox w,h and crop matches final_pred.py
- Performance: disable TF OneDNN via TF_ENABLE_ONEDNN_OPTS=0 if needed
- Model load warnings about optimizer state are expected (inference only)

## Notes on Resolution Differences
- Pipeline is bbox-relative: offsets rely on bbox w,h, so it is resolution-agnostic as long as the crop and offsets mirror final_pred.py. Ensure the Android capture plus server crop matches the desktop reference.

## Suggested Next Steps
- Compare debug_skeleton_*.jpg (saved by server) with desktop visualization to confirm identical skeletons
- Add unit tests for predict_sign with recorded landmark sets
- Profile WebSocket throughput and JPEG quality/size
