# 🖐️ HandSpeak.ai

<div align="center">

![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Android%20%7C%20Web-blue)
![License](https://img.shields.io/badge/License-MIT-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)

**A Real-Time AI Sign Language Recognition System**

[**⬇️ Download Latest APK**](https://github.com/krish-rRay23/HandSpeak.ai/blob/main/app-release.apk)

</div>

---

## 📄 Abstract

**HandSpeak.ai** is a production-ready, real-time American Sign Language (ASL) recognition system that converts hand gestures into text and speech. This project balances privacy, latency, and accuracy by combining an on-device landmark-first strategy with a cloud-based inference service. It utilizes a hybrid Machine Learning + geometric-rule pipeline and features a roadmap toward end-to-end temporal models for improved generalization.

**Keywords:** ASL recognition, MediaPipe, landmarks, Transformer, CNN, FastAPI, temporal smoothing, privacy-preserving inference, real-time.

---

## 📚 Table of Contents

- [Introduction](#-introduction)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Data Pipeline & Preprocessing](#-data-pipeline--preprocessing)
- [Model Architectures](#-model-architectures)
- [Client Application](#-client-application--ux)
- [Performance & Evaluation](#-performance--evaluation)
- [Installation & Setup](#-installation--setup)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 🚀 Introduction

The goal of HandSpeak.ai is to provide a **low-latency, high-accuracy letter-level ASL recognizer** that works across mobile and desktop clients while preserving user privacy and offering a strong User Experience (real-time suggestions, autocorrect, and TTS). The system currently supports 26 static letters and several command gestures, deployed via a dual-backend architecture.

### Objectives
- **Real-time responsiveness:** 15–30 FPS, sub-100ms frame-to-prediction latency.
- **High Accuracy:** Target 95–98% for static letters in production.
- **Privacy:** Landmark-only mobile uploads (no raw images sent to server).
- **Robustness:** Generalization across different users, lighting conditions, and hand sizes.

---

## 🏗️ System Architecture

The system implements a **dual-backend design** to satisfy divergent client constraints:

### 1. Primary Server (Full Image Pipeline)
- Accepts images (Base64 JPEG).
- Runs MediaPipe locally.
- Generates skeleton images for CNN processing.
- **Stack:** FastAPI, OpenCV, MediaPipe, TensorFlow/Keras.

### 2. Landmark Server (Mobile-Optimized)
- Accepts **21×3 landmark arrays** from clients.
- Applies canonicalization, rule engine, and temporal smoothing.
- **Advantages:** Sub-50ms latency, minimal bandwidth, privacy-preserving.

---

## 🔄 Data Pipeline & Preprocessing

Robust preprocessing is central to generalization. We follow a strict **canonicalization pipeline** on the 21 MediaPipe joints across a 30-frame window:

1.  **Translation:** Re-center coordinates using the wrist as origin (0,0,0).
2.  **Scale Normalization:** Scale by Euclidean distance (wrist to middle-finger MCP).
3.  **Rotation Normalization:** Align wrist→index MCP with the Y-axis.
4.  **Left-hand Mirroring:** Mirror x-axis for left-handed inputs.

**Temporal Handling:**
- Rolling buffer of **30 frames**.
- Stride = 10 (inference every 10 frames).

---

## 🧠 Model Architectures

The system uses a hybrid approach for optimal performance:

### Current Hybrid: CNN (8-group) + Rule Engine
- **Core:** Image-based pipeline converting landmarks to 400x400 skeleton images fed to a coarse 8-group CNN.
- **Post-processing:** ~600 lines of deterministic geometric rules for intra-group disambiguation.
- **Temporal Smoothing:** Exponential Moving Average (λ=0.7) with stability requirements.

### Advanced Roadmap: Temporal Transformer
- **Input:** Linear projection of flattened landmark sequences (30×63).
- **Architecture:** 4-layer Transformer encoder with positional encoding.
- **Benefit:** Captures temporal context, reducing jitter and manual rule maintenance.

---

## 📱 Client Application & UX

The **Android Client** is built for speed and usability:

- **CameraX:** High-performance camera stream.
- **On-Device MediaPipe:** Extracts 21 3D landmarks locally.
- **Bandwidth Efficient:** Only landmarks are sent to the cloud.
- **Smart Word Engine:**
    - Prefix-based suggestions.
    - Personal-frequency weighting.
    - Smart scoring (Gesture Confidence × Lexicon Weight).
- **UI:** Jetpack Compose with a 'Cyberpunk Modern' aesthetic, animated skeleton overlay, and accessibility features like Text-to-Speech (TTS).

---

## 📊 Performance & Evaluation

- **Latency:** 50–100ms (Frame → Prediction).
- **Throughput:** 15–30 FPS real-time processing.
- **Model Footprint:** ≈50MB.
- **Accuracy:** ~85–90% (Internal Validation), targeting 95%+ with expanded datasets.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- Android Studio (for mobile app)

### Backend Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/krish-rRay23/HandSpeak.ai.git
    cd HandSpeak.ai
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements_prod.txt
    ```

3.  **Run the server:**
    ```bash
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
    ```

### Mobile App
1.  Open the `app/` directory in **Android Studio**.
2.  Build and run on a physical device (Emulators may lack camera support).
3.  Ensure the backend URL is configured in the app settings.

---

## 🗺️ Roadmap

- [x] Real-time static letter recognition (A-Z).
- [x] Dual-backend architecture (Image & Landmark).
- [ ] **Short-term:** Train 26-class CNN baseline & Landmark-Sequence Transformer.
- [ ] **Mid-term:** Replace rule engine with learned sub-classifiers.
- [ ] **Long-term:** Dynamic sign recognition (words/phrases) and end-to-end temporal models.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  Made with ❤️ by the HandSpeak.ai Team
</div>
