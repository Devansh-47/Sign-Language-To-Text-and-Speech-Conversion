# 🚀 HandSpeak.ai - Complete Setup Guide

## Overview
This guide will help you set up the complete HandSpeak.ai system with:
1. ✅ Desktop application (Python + Tkinter)
2. ✅ Backend API server (FastAPI)
3. ✅ Android mobile app (Kotlin)

---

## Part 1: Backend Server Setup

### Step 1: Install Backend Dependencies
```bash
cd HandSpeak.ai
pip install -r requirements-server.txt
```

### Step 2: Start the Backend Server
```bash
python server.py
```

The server will start on: `http://0.0.0.0:8000`

### Step 3: Find Your IP Address
**Windows:**
```bash
ipconfig
```
Look for `IPv4 Address` under your active network adapter (e.g., `192.168.1.100`)

**Note this IP - you'll need it for the Android app!**

### Step 4: Test the API
Open browser and go to: `http://localhost:8000/docs`
You should see the interactive API documentation (Swagger UI)

---

## Part 2: Android App Setup

### Step 1: Open Android Studio
1. Launch Android Studio
2. Click "New Project"
3. Select "Empty Activity"
4. Name: `HandSpeakAI`
5. Package: `com.handspeak.ai`
6. Language: **Kotlin**
7. Minimum SDK: **API 24 (Android 7.0)**

### Step 2: Use the AI Assistant
1. Open the Gemini/AI assistant in Android Studio
2. Copy the **ENTIRE CONTENT** from `ANDROID_STUDIO_PROMPT.md`
3. Paste it into the AI assistant
4. Let it generate the complete app

### Step 3: Configure the Server URL
In your Android app, update the API base URL:

**File: `app/src/main/java/com/handspeak/ai/network/RetrofitClient.kt`**
```kotlin
private const val BASE_URL = "http://192.168.1.100:8000/"  // ← Use YOUR IP here
```

### Step 4: Enable Internet Permission
The AI should add this, but verify in `AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />
```

### Step 5: Build and Run
1. Connect your Android phone via USB (enable USB debugging)
2. Click "Run" button in Android Studio
3. App will install on your phone

---

## Part 3: Testing the Complete System

### Test Flow:
1. ✅ Start backend server: `python server.py`
2. ✅ Open Android app on phone
3. ✅ Grant camera permissions
4. ✅ Make a hand sign (e.g., "A", "B", "C")
5. ✅ Tap "Capture" button
6. ✅ App sends image to server
7. ✅ Server recognizes sign and returns character
8. ✅ Character appears in app
9. ✅ Word suggestions appear as you build words
10. ✅ Tap "Speak" to hear the sentence

---

## Part 4: Troubleshooting

### Problem: Android app can't connect to server
**Solution:**
- Ensure both phone and computer are on the **same WiFi network**
- Check Windows Firewall allows port 8000
- Use your computer's **local IP**, not `localhost`
- Test by opening `http://YOUR_IP:8000` in phone's browser

### Problem: "No hand detected" error
**Solution:**
- Ensure good lighting
- Hold hand clearly in frame
- Try different hand positions
- Make sure camera is focused

### Problem: Server crashes or slow
**Solution:**
- Check GPU/CPU usage
- Restart server
- Check model files are present (`cnn8grps_rad1_model.h5`, `white.jpg`)

### Problem: Camera permission denied
**Solution:**
- Go to Android Settings → Apps → HandSpeakAI → Permissions
- Enable Camera permission

---

## Part 5: Desktop App (Already Working)

You already have the desktop version working:
```bash
python final_pred.py
```

---

## Architecture Overview

```
┌─────────────────┐
│  Desktop App    │  ← Python + Tkinter (Local processing)
│  (final_pred.py)│
└─────────────────┘

┌─────────────────┐      HTTP/REST API      ┌─────────────────┐
│  Android App    │ ◄───────────────────► │  Backend Server │
│  (Kotlin)       │                         │  (FastAPI)      │
└─────────────────┘                         └─────────────────┘
     📱 Mobile                                    💻 Server
                                                  (Python)
```

### Data Flow:
1. Android app captures image 📸
2. Sends to backend via HTTP POST 📤
3. Backend runs ML model 🧠
4. Returns predicted character 🔤
5. Android app displays result 📱
6. User builds sentences 📝
7. Text-to-Speech reads aloud 🔊

---

## Part 6: Advanced Configuration

### For Production Deployment:

#### Deploy Backend to Cloud:
1. **Heroku:**
   ```bash
   heroku create handspeak-api
   git push heroku main
   ```

2. **AWS EC2:**
   - Launch Ubuntu instance
   - Install Python and dependencies
   - Run server with `uvicorn`

3. **Google Cloud Run:**
   - Containerize with Docker
   - Deploy to Cloud Run
   - Get public URL

#### Update Android App:
```kotlin
private const val BASE_URL = "https://your-app.herokuapp.com/"
```

---

## Part 7: File Structure

```
HandSpeak.ai/
├── final_pred.py              # Desktop app (Tkinter)
├── server.py                  # Backend API server
├── mediapipe_compat.py        # MediaPipe compatibility layer
├── cnn8grps_rad1_model.h5     # Trained ML model
├── white.jpg                  # White background for skeleton
├── requirements.txt           # Desktop app dependencies
├── requirements-server.txt    # Server dependencies
├── SERVER_README.md           # Server documentation
├── ANDROID_STUDIO_PROMPT.md   # Android app specification
└── QUICK_START_GUIDE.md       # This file
```

---

## Summary

You now have:
✅ **Desktop App** - Full-featured Python application
✅ **Backend Server** - REST API for mobile apps
✅ **Android App Template** - Complete specification for Android Studio

### Next Steps:
1. ⚡ Start backend server
2. 📱 Generate Android app using AI assistant
3. 🔗 Connect them together
4. 🎉 Test and enjoy!

---

## Support

For issues:
1. Check server logs for backend errors
2. Check Android Logcat for app errors
3. Test API endpoints in browser
4. Verify network connectivity

---

**Happy Coding! 🚀**
