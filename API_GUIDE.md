# HandSpeak.ai Backend Guide

## Running the server

1) Install deps: `pip install -r requirements.txt`
2) Start API (auto-loads cnn8grps_rad1_model.h5): `uvicorn app.main:app --host 0.0.0.0 --port 8000`

## Endpoints (prefixed with /v1)
- POST /frame: body {session_id, frame_b64, timestamp_ms?}; returns live letter hypotheses, smoothed confidence, suggestions, and session state.
- POST /commit: body {session_id, letter}; force-commit a letter (space triggers word boundary).
- POST /feedback: body {session_id, suggestion, accepted}; increments personalization counts when accepted.
- GET /session/{id}: fetch current partial word, committed letters, words, and sentence.
- GET /health: returns status + model path/version.
- POST /tts: body {text, return_audio?}; returns text and optional base64 audio.

## Session + smoothing logic
- Per-session buffer of 10 frames with exponential smoothing (lambda=0.15).
- Commit rule: stable >=300 ms, velocity <0.1, confidence >0.6. Holds otherwise.
- Multi-hypothesis: top1/top2/top5 returned for each frame.

## Suggestions
- Prefix-based lookup over app/data/lexicon_small.txt.
- Scoring: 0.4*P_gesture + 0.4*P_language + 0.2*P_personal.
- Autocorrect adds confusion-matrix variants, max 200.

## Phrase fast-path
- Optional DTW match against app/data/phrase_signatures.json (threshold <0.3, conf >0.75). If hit, phrase is appended to the sentence.

## Evaluation harness
- scripts/benchmark.py writes out/report.json and out/report.md with stub metrics; plug in dataset later.

## Android integration hints
- Send frames as base64-encoded JPEG/PNG via POST /v1/frame at ~15–30 FPS with a stable session_id.
- Use suggestions immediately for mid-word autocomplete; accept via /v1/feedback.
- Perform TTS on-device or call /v1/tts when audio is needed from backend.
