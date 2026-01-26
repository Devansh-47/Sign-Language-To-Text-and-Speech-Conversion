   <div align="center">

# 🤝 HandSpeak.ai

### Real-Time Sign Language Recognition System

*Bridging communication through AI-powered gesture recognition*

---

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-orange.svg)](https://google.github.io/mediapipe/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Pipeline Workflow](#-pipeline-workflow)
- [Key Components](#-key-components)
- [Getting Started](#-getting-started)
- [API Documentation](#-api-documentation)
- [Desktop Application](#-desktop-application)
- [Technical Details](#-technical-details)
- [Troubleshooting](#-troubleshooting)
- [Next Steps](#-next-steps)

---

## 🎯 Overview

**HandSpeak.ai** is an intelligent sign language recognition system that converts hand gestures into text in real-time. The system uses:

- 🤖 **Deep Learning**: CNN model for gesture classification
- ✋ **MediaPipe**: 21-point hand landmark detection
- 🚀 **FastAPI Backend**: WebSocket & REST endpoints for mobile clients
- 🖥️ **Desktop App**: Standalone Tkinter application for local webcam inference
- 📱 **Mobile Support**: Android app integration via WebSocket

### Supported Gestures
- ✅ Alphabet (A-Z)
- ✅ Space & Backspace commands
- ✅ "Next" navigation gesture
- ✅ Word suggestions & autocorrect

---

## 🏗️ Architecture

```
┌─────────────────┐         ┌──────────────────┐
│  Android App    │────────▶│  FastAPI Server  │
│  (Mobile)       │ WebSocket│  (server.py)     │
└─────────────────┘         └──────────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │  Hand Detection  │
                            │  (cvzone + MP)   │
                            └──────────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │ Skeleton Drawing │
                            │  (400x400 canvas)│
                            └──────────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │  CNN Classifier  │
                            │ (cnn8grps model) │
                            └──────────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │  Post-Processing │
                            │  (Rules + Logic) │
                            └──────────────────┘

┌─────────────────┐
│  Desktop App    │────────▶ Same Pipeline ────▶ Tkinter GUI
│  (Webcam)       │         (local inference)
└─────────────────┘
```

---

## 🔄 Pipeline Workflow

### Step-by-Step Process

1. **📸 Frame Capture**
   - **Desktop**: OpenCV webcam capture (`final_pred.py`)
   - **Mobile**: Base64-encoded JPEG frames via WebSocket

2. **👋 Hand Detection**
   - Primary detector (`hd`) finds bounding box `(x, y, w, h)` on full frame
   - 29-pixel offset padding applied

3. **✂️ Region Cropping**
   ```python
   image = frame[y - offset : y + h + offset,
                 x - offset : x + w + offset]
   ```

4. **🎯 Landmark Detection**
   - Secondary detector (`hd2`) finds 21 hand landmarks in cropped region
   - Output: `pts[21][3]` → `[x, y, z]` coordinates

5. **🎨 Skeleton Drawing**
   - Load 400×400 white canvas
   - Calculate offsets:
     ```python
     os  = ((400 - w) // 2) - 15  # horizontal offset
     os1 = ((400 - h) // 2) - 15  # vertical offset
     ```
   - Draw hand skeleton by connecting landmarks with offsets

6. **🧠 CNN Classification**
   - Input: Skeleton image `(1, 400, 400, 3)`
   - Model: `cnn8grps_rad1_model.h5`
   - Output: 8 coarse gesture groups

7. **⚙️ Post-Processing**
   - Rule-based mapping to final character/command
   - Autocorrect & suggestions via enchant dictionary
   - Text-to-speech (TTS) output

---

## 🔧 Key Components

### Core Files

| File | Purpose |
|------|---------|
| `server.py` | FastAPI backend with WebSocket + REST endpoints |
| `final_pred.py` | Desktop GUI reference implementation (Tkinter) |
| `app/core/hand_engine.py` | Core hand detection & prediction logic |
| `app/core/suggestions.py` | Word suggestions & autocorrect |
| `app/core/tts.py` | Text-to-speech engine |
| `app/core/phrase_dtw.py` | Phrase recognition using DTW |

### Models & Data

| Asset | Description |
|-------|-------------|
| `cnn8grps_rad1_model.h5` | CNN model for gesture classification |
| `app/data/hand_landmarker.task` | MediaPipe hand landmark model |
| `app/data/lexicon_small.txt` | Dictionary for autocorrect |
| `app/data/phrase_signatures.json` | Pre-computed phrase signatures |

### Dependencies

- **Backend**: `requirements.txt`
- **Desktop GUI**: `runtime-requirements-gui.txt`

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- Windows build tools (for some packages)
- Optional: conda/virtualenv

### Installation (Windows PowerShell)

#### 1️⃣ Clone Repository
```powershell
git clone https://github.com/krish-rRay23/HandSpeak.ai.git
cd HandSpeak.ai
```

#### 2️⃣ Create Virtual Environment
```powershell
python -m venv .venv-server
.\.venv-server\Scripts\Activate.ps1
```

#### 3️⃣ Install Dependencies
```powershell
pip install -r requirements.txt
```

#### 4️⃣ Start Server
```powershell
python server.py
```

#### 5️⃣ Access API Documentation
Open browser: **http://localhost:8000/docs**

---

## 📡 API Documentation

### WebSocket Endpoint

**Connect**: `ws://<HOST>:8000/ws/recognize`

#### Client Message Format
```
Base64-encoded JPEG image (string)
```

#### Server Response Format
```json
{
  "success": true,
  "character": "A",
  "handDetected": true,
  "message": "Prediction successful"
}
```

**Possible Characters**: `A-Z`, `" "` (space), `"Backspace"`, `"next"`, `null`

---

### REST Endpoint

**Upload Image**: `POST /api/recognize`

#### Request
```bash
curl -X POST "http://localhost:8000/api/recognize" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@hand_gesture.jpg"
```

#### Response
```json
{
  "character": "B",
  "confidence": 0.95,
  "suggestions": ["hello", "help", "hand"]
}
```

---

## 🖥️ Desktop Application

Run the standalone desktop app with webcam:

```powershell
# Activate GUI environment (if separate)
.\.venv-gui\Scripts\Activate.ps1

# Install GUI dependencies
pip install -r runtime-requirements-gui.txt

# Launch application
python final_pred.py
```

---

## 🔬 Technical Details

### Coordinate System

- **Bounding Box**: `(x, y, w, h)` from primary detector
- **Crop Dimensions**: Original bbox dimensions used for offset calculations
- **Landmark Space**: 21 points in cropped image coordinates
- **Offset Formula**: Centers hand skeleton on 400×400 canvas

### Critical Implementation Notes

⚠️ **Exact Pipeline Match Required**
```python
# CORRECT (matches final_pred.py)
image = frame[y - offset : y + h + offset,
              x - offset : x + w + offset]

# WRONG (adds bounds checking - breaks predictions)
y1 = max(0, y - offset)
y2 = min(frame.shape[0], y + h + offset)
image = frame[y1:y2, x1:x2]  # ❌ Different crop dimensions
```

### Resolution Agnostic Design

The pipeline is **resolution-independent**:
- Offsets computed from bbox dimensions, not absolute frame size
- Works across different cameras/screens (desktop 1920×1080, mobile varies)
- As long as crop extraction matches reference implementation

---

## 🛠️ Troubleshooting

### Common Issues

#### ❌ Wrong Predictions / Random Characters

**Symptom**: Server predicts X, C, Y, E regardless of actual sign

**Solution**: Ensure crop extraction exactly matches `final_pred.py` (no bounds checking)

**Debug**: Check `debug_skeleton_*.jpg` files for visual comparison

---

#### ❌ Landmarks Out of Bounds

**Symptom**: Landmark coordinates exceed bbox dimensions

**Solution**: Use original bbox `w, h` for offset calculations (not actual crop dimensions)

---

#### ❌ Unicode Encoding Errors

**Symptom**: `UnicodeEncodeError: 'charmap' codec can't encode character`

**Solution**: All logging uses ASCII characters (already fixed in code)

---

#### ❌ TensorFlow Performance Warnings

**Symptom**: OneDNN optimization messages

**Solution** (optional):
```powershell
$env:TF_ENABLE_ONEDNN_OPTS = "0"
python server.py
```

---

#### ⚠️ Model Loading Warnings

**Expected**: Optimizer state warnings during model load (inference-only mode)

---

## 📊 Debug Features

### Skeleton Visualization

The server saves debug skeleton images (1 in ~30 frames):

```python
# Saved in project root
debug_skeleton_1234.jpg
debug_skeleton_5678.jpg
```

**Use**: Compare with desktop app visualization to verify identical rendering

---

## 🎯 Next Steps

### Recommended Actions

- [ ] **Test with Mobile App**: Verify predictions match desktop behavior
- [ ] **Compare Debug Skeletons**: Confirm identical rendering across platforms  
- [ ] **Unit Testing**: Add tests for `predict_sign()` with recorded landmarks
- [ ] **Performance Profiling**: Measure WebSocket throughput & latency
- [ ] **Image Optimization**: Tune JPEG quality vs size for mobile bandwidth
- [ ] **Expand Gesture Set**: Train model on additional signs/phrases

---

## 📄 License

This project is part of HandSpeak.ai gesture recognition research.

---

## 👥 Contributors

Developed with ❤️ for bridging communication barriers

---

<div align="center">

**Made with** 🤖 **AI** • **Powered by** 🧠 **Deep Learning** • **Built for** 🌍 **Accessibility**

</div>
