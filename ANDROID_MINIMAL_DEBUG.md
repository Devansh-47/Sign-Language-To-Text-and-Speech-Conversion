# 🐛 Android App Debug Guide - No Hand Detection Issue

## Problem
WebSocket connects successfully but:
- ❌ No hand detected
- ❌ No predictions received
- ❌ No response from backend

## Root Cause Analysis

### Most Likely Issues:
1. **App is NOT sending landmarks** (just connected, but idle)
2. **MediaPipe isn't running** (no hand detection on device)
3. **Wrong JSON format** (backend rejects silently)
4. **Missing result listener** (landmarks detected but not sent)

---

## 🔍 Quick Diagnosis

### Step 1: Add Debug Logging

Add this IMMEDIATELY after WebSocket connection:

```kotlin
webSocket = client.newWebSocket(request, object : WebSocketListener() {
    override fun onOpen(webSocket: WebSocket, response: Response) {
        Log.d("DEBUG", "✓ WebSocket CONNECTED")
        
        // TEST: Send a dummy message immediately
        val testPayload = JSONObject().apply {
            put("handDetected", false)
            put("handedness", "Right")
            put("landmarks", JSONObject.NULL)
        }
        webSocket.send(testPayload.toString())
        Log.d("DEBUG", "✓ Sent test payload: ${testPayload}")
    }
    
    override fun onMessage(webSocket: WebSocket, text: String) {
        Log.d("DEBUG", "✓ RECEIVED from backend: $text")
        // Your existing code...
    }
    
    override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
        Log.e("DEBUG", "✗ WebSocket FAILED: ${t.message}")
    }
})
```

**Expected Backend Response:**
```json
{"success": true, "handDetected": false, "character": ""}
```

**If you DON'T see "RECEIVED from backend" → Backend isn't responding. Check server logs.**

---

### Step 2: Verify MediaPipe is Running

Add this to your MediaPipe setup:

```kotlin
private fun setupMediaPipe() {
    Log.d("DEBUG", "Setting up MediaPipe...")
    
    try {
        handLandmarker = HandLandmarker.createFromOptions(
            this,
            HandLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder()
                    .setModelAssetPath("hand_landmarker.task")
                    .build())
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumHands(1)
                .setMinHandDetectionConfidence(0.5f)  // Lower threshold for testing
                .setResultListener { result, image ->
                    Log.d("DEBUG", "✓ MediaPipe callback triggered!")
                    Log.d("DEBUG", "Hands detected: ${result.landmarks().size}")
                    handleLandmarkResult(result)
                }
                .setErrorListener { error ->
                    Log.e("DEBUG", "✗ MediaPipe ERROR: ${error.message}")
                }
                .build()
        )
        Log.d("DEBUG", "✓ MediaPipe initialized successfully")
    } catch (e: Exception) {
        Log.e("DEBUG", "✗ MediaPipe setup FAILED: ${e.message}")
    }
}
```

**Expected Log Output (when hand is visible):**
```
DEBUG: ✓ MediaPipe callback triggered!
DEBUG: Hands detected: 1
```

**If you DON'T see "MediaPipe callback triggered" → Camera frames aren't reaching MediaPipe.**

---

### Step 3: Verify Camera Pipeline

Add logging to your camera frame processing:

```kotlin
private fun processCameraFrame(imageProxy: ImageProxy) {
    Log.d("DEBUG", "Processing camera frame ${imageProxy.imageInfo.timestamp}")
    
    try {
        // Convert to MediaPipe format
        val bitmap = imageProxy.toBitmap()
        val mpImage = BitmapImageBuilder(bitmap).build()
        
        // Send to MediaPipe
        handLandmarker.detectAsync(mpImage, imageProxy.imageInfo.timestamp)
        Log.d("DEBUG", "✓ Frame sent to MediaPipe")
    } catch (e: Exception) {
        Log.e("DEBUG", "✗ Frame processing failed: ${e.message}")
    } finally {
        imageProxy.close()
    }
}
```

**Expected:** Log spam every frame (~30 FPS)

**If you DON'T see "Processing camera frame" → Camera isn't started.**

---

### Step 4: Verify Landmark Extraction & Sending

```kotlin
private fun handleLandmarkResult(result: HandLandmarkerResult) {
    Log.d("DEBUG", "handleLandmarkResult called")
    
    if (result.landmarks().isEmpty()) {
        Log.d("DEBUG", "No hands in frame")
        val payload = LandmarkPayload(false, "Right", null)
        sendLandmarks(payload)
        return
    }
    
    Log.d("DEBUG", "✓ Hand detected! Extracting landmarks...")
    val payload = extractLandmarks(result)
    Log.d("DEBUG", "Landmarks: ${payload.landmarks?.size} points")
    Log.d("DEBUG", "Handedness: ${payload.handedness}")
    
    sendLandmarks(payload)
}

private fun sendLandmarks(payload: LandmarkPayload) {
    Log.d("DEBUG", "sendLandmarks called")
    
    if (webSocket == null) {
        Log.e("DEBUG", "✗ WebSocket is NULL!")
        return
    }
    
    val json = JSONObject().apply {
        put("handDetected", payload.handDetected)
        put("handedness", payload.handedness)
        
        if (payload.landmarks != null) {
            val landmarksArray = JSONArray()
            payload.landmarks.forEach { point ->
                val pointArray = JSONArray()
                point.forEach { coord -> pointArray.put(coord) }
                landmarksArray.put(pointArray)
            }
            put("landmarks", landmarksArray)
        } else {
            put("landmarks", JSONObject.NULL)
        }
    }
    
    Log.d("DEBUG", "Sending JSON (${json.toString().length} bytes)")
    val sent = webSocket?.send(json.toString())
    Log.d("DEBUG", if (sent == true) "✓ Message sent" else "✗ Send FAILED")
}
```

---

## 🎯 Minimal Working Code (Copy-Paste Ready)

If nothing above works, use this MINIMAL test activity:

```kotlin
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.google.mediapipe.tasks.vision.handlandmarker.*
import okhttp3.*
import org.json.JSONArray
import org.json.JSONObject

class TestActivity : AppCompatActivity() {
    private var webSocket: WebSocket? = null
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Step 1: Connect WebSocket
        connectWebSocket()
        
        // Step 2: Send fake landmarks every 2 seconds
        sendFakeLandmarks()
    }
    
    private fun connectWebSocket() {
        val request = Request.Builder()
            .url("ws://YOUR_SERVER_IP:8000/ws/landmarks")
            .build()
        
        webSocket = OkHttpClient().newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d("TEST", "✓ Connected!")
            }
            
            override fun onMessage(webSocket: WebSocket, text: String) {
                Log.d("TEST", "✓ Backend says: $text")
                
                // Parse and show in UI
                runOnUiThread {
                    val json = JSONObject(text)
                    val char = json.optString("character", "")
                    Log.d("TEST", "Character: '$char'")
                }
            }
            
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e("TEST", "✗ Failed: ${t.message}")
            }
        })
    }
    
    private fun sendFakeLandmarks() {
        // Send fake "A" gesture every 2 seconds
        android.os.Handler(mainLooper).postDelayed({
            val json = JSONObject().apply {
                put("handDetected", true)
                put("handedness", "Right")
                
                // Fake 21 landmarks
                val landmarks = JSONArray()
                for (i in 0..20) {
                    val point = JSONArray()
                    point.put(0.5 + i * 0.01) // x
                    point.put(0.5 + i * 0.01) // y
                    point.put(0.0)             // z
                    landmarks.put(point)
                }
                put("landmarks", landmarks)
            }
            
            Log.d("TEST", "Sending fake landmarks...")
            webSocket?.send(json.toString())
            
            // Repeat
            sendFakeLandmarks()
        }, 2000)
    }
    
    override fun onDestroy() {
        super.onDestroy()
        webSocket?.close(1000, "Activity destroyed")
    }
}
```

**Replace:** `YOUR_SERVER_IP` with your actual server IP (e.g., `192.168.1.100`)

**Expected Output:**
```
TEST: ✓ Connected!
TEST: Sending fake landmarks...
TEST: ✓ Backend says: {"success":true,"handDetected":true,"character":"A"}
TEST: Character: 'A'
```

**If this works → Your WebSocket + Backend are fine. Issue is MediaPipe integration.**

**If this doesn't work → Check server IP and firewall.**

---

## 🔧 Common Fixes

### Fix 1: Camera Not Starting
```kotlin
// In your Activity
override fun onResume() {
    super.onResume()
    startCamera()
}

private fun startCamera() {
    val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
    cameraProviderFuture.addListener({
        val cameraProvider = cameraProviderFuture.get()
        
        val preview = Preview.Builder().build()
        val imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(ContextCompat.getMainExecutor(this)) { image ->
                    processCameraFrame(image)
                }
            }
        
        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(
            this,
            CameraSelector.DEFAULT_FRONT_CAMERA,
            preview,
            imageAnalysis
        )
        
        Log.d("CAMERA", "✓ Camera started")
    }, ContextCompat.getMainExecutor(this))
}
```

### Fix 2: MediaPipe Model Missing
```
Error: java.io.FileNotFoundException: hand_landmarker.task
```

**Solution:** Download model and add to `assets` folder:
1. Download: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
2. Place in: `app/src/main/assets/hand_landmarker.task`

### Fix 3: WebSocket Null
```kotlin
// Keep WebSocket reference
private var webSocket: WebSocket? = null

// Don't create new instance every time
fun ensureWebSocketConnected() {
    if (webSocket == null) {
        connectWebSocket()
    }
}
```

---

## ✅ Success Checklist

Run through these in order:

1. [ ] **Logcat shows:** `✓ WebSocket CONNECTED`
2. [ ] **Send test payload** → Backend responds with JSON
3. [ ] **Logcat shows:** `✓ MediaPipe initialized successfully`
4. [ ] **Logcat shows:** `Processing camera frame` spam
5. [ ] **Put hand in front of camera**
6. [ ] **Logcat shows:** `✓ MediaPipe callback triggered!`
7. [ ] **Logcat shows:** `Hands detected: 1`
8. [ ] **Logcat shows:** `Landmarks: 21 points`
9. [ ] **Logcat shows:** `✓ Message sent`
10. [ ] **Logcat shows:** `✓ Backend says: {...}`
11. [ ] **UI updates with character**

**Stop at the first checkbox that fails. That's your issue.**

---

## 🆘 Emergency Contact

If still stuck, provide these logs:

```bash
# Backend logs
python landmark_server.py

# Android Logcat (filtered)
adb logcat | grep -E "DEBUG|TEST|MediaPipe|WebSocket"
```

Send both outputs to diagnose the exact failure point.

---

**Next Step:** Run the minimal test activity above. If that works, the issue is in your MediaPipe integration, not the backend.
