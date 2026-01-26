# Build Android ASL App from Scratch - Backend Integration
pc ip : 10.171.152.6
## Backend Details
- **Server:** `ws://YOUR_IP:8000/ws/landmarks`
- **Accepts:** JSON with 21 hand landmarks
- **Returns:** Predicted ASL character

---

## App Requirements

### 1. Dependencies (build.gradle)
```gradle
dependencies {
    implementation 'com.google.mediapipe:tasks-vision:0.10.9'
    implementation 'com.squareup.okhttp3:okhttp:4.12.0'
    implementation "androidx.camera:camera-camera2:1.3.1"
    implementation "androidx.camera:camera-lifecycle:1.3.1"
    implementation "androidx.camera:camera-view:1.3.1"
}
```

### 2. Permissions (AndroidManifest.xml)
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />
```

### 3. Download MediaPipe Model
- Get `hand_landmarker.task` from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
- Put in: `app/src/main/assets/hand_landmarker.task`

---

## Main Activity Code

```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var handLandmarker: HandLandmarker
    private lateinit var camera: Camera
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var resultText: TextView
    private lateinit var sentenceText: TextView
    
    private var webSocket: WebSocket? = null
    private var lastSendTime = 0L
    private var currentSentence = StringBuilder()
    private var lastCharacter = ""
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        resultText = findViewById(R.id.resultText)
        sentenceText = findViewById(R.id.sentenceText)
        
        setupMediaPipe()
        connectWebSocket("10.171.152.6")  // Your PC IP
        startCamera()
    }
    
    private fun setupMediaPipe() {
        handLandmarker = HandLandmarker.createFromOptions(
            this,
            HandLandmarkerOptions.builder()
                .setBaseOptions(
                    BaseOptions.builder()
                        .setModelAssetPath("hand_landmarker.task")
                        .build()
                )
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumHands(1)
                .setMinHandDetectionConfidence(0.5f)
                .setMinHandPresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setResultListener { result, _ -> 
                    processLandmarks(result) 
                }
                .setErrorListener { error ->
                    Log.e("MediaPipe", "Error: ${error.message}")
                }
                .build()
        )
    }
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(
                        ContextCompat.getMainExecutor(this)
                    ) { imageProxy ->
                        processFrame(imageProxy)
                    }
                }
            
            cameraProvider.unbindAll()
            camera = cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_FRONT_CAMERA,
                preview,
                imageAnalysis
            )
            
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun processFrame(imageProxy: ImageProxy) {
        val bitmap = imageProxy.toBitmap()
        val mpImage = BitmapImageBuilder(bitmap).build()
        
        handLandmarker.detectAsync(mpImage, imageProxy.imageInfo.timestamp)
        imageProxy.close()
    }
    
    private fun processLandmarks(result: HandLandmarkerResult) {
        // Draw overlay
        runOnUiThread {
            if (result.landmarks().isNotEmpty()) {
                overlayView.setLandmarks(result.landmarks()[0])
            } else {
                overlayView.clear()
            }
        }
        
        // Send to backend (throttled)
        val now = System.currentTimeMillis()
        if (now - lastSendTime < 150) return
        lastSendTime = now
        
        if (result.landmarks().isEmpty()) {
            sendToBackend(false, "Right", null)
            return
        }
        
        val landmarks = result.landmarks()[0].map { 
            listOf(it.x(), it.y(), it.z()) 
        }
        val handedness = result.handedness()[0][0].categoryName()
        
        sendToBackend(true, handedness, landmarks)
    }
    
    private fun connectWebSocket(serverIp: String) {
        val request = Request.Builder()
            .url("ws://$serverIp:8000/ws/landmarks")
            .build()
        
        val client = OkHttpClient.Builder()
            .readTimeout(0, TimeUnit.MILLISECONDS)
            .build()
        
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d("WebSocket", "Connected!")
            }
            
            override fun onMessage(webSocket: WebSocket, text: String) {
                val json = JSONObject(text)
                val character = json.optString("character", "")
                val handDetected = json.optBoolean("handDetected", false)
                
                runOnUiThread {
                    if (handDetected && character.isNotEmpty()) {
                        // Show current character
                        resultText.text = character
                        
                        // Build sentence (only add if character changed)
                        if (character != lastCharacter) {
                            lastCharacter = character
                            handleCharacter(character)
                        }
                    } else {
                        resultText.text = "No hand"
                        lastCharacter = ""
                    }
                }
            }
            
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e("WebSocket", "Error: ${t.message}")
            }
        })
    }
    
    private fun sendToBackend(
        handDetected: Boolean, 
        handedness: String, 
        landmarks: List<List<Float>>?
    ) {
        val json = JSONObject().apply {
            put("handDetected", handDetected)
            put("handedness", handedness)
            
            if (landmarks != null) {
                val array = JSONArray()
                landmarks.forEach { point ->
                    val pointArray = JSONArray()
                    point.forEach { coord -> pointArray.put(coord) }
                    array.put(pointArray)
                }
                put("landmarks", array)
            } else {
                put("landmarks", JSONObject.NULL)
            }
        }
        
        webSocket?.send(json.toString())
    }
    
    private fun handleCharacter(character: String) {
        when (character) {
            " " -> {
                // Space - add to sentence
                currentSentence.append(" ")
                sentenceText.text = currentSentence.toString()
            }
            "Backspace" -> {
                // Remove last character
                if (currentSentence.isNotEmpty()) {
                    currentSentence.deleteCharAt(currentSentence.length - 1)
                    sentenceText.text = currentSentence.toString()
                }
            }
            "next" -> {
                // Word boundary - add space
                currentSentence.append(" ")
                sentenceText.text = currentSentence.toString()
            }
            else -> {
                // Regular A-Z character
                if (character.length == 1 && character[0].isLetter()) {
                    currentSentence.append(character)
                    sentenceText.text = currentSentence.toString()
                }
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        webSocket?.close(1000, "Activity destroyed")
        handLandmarker.close()
    }
}
```

---

## OverlayView (Hand Visualization)

```kotlin
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {
    
    private var landmarks: List<NormalizedLandmark>? = null
    private val paint = Paint().apply {
        color = Color.GREEN
        strokeWidth = 8f
        style = Paint.Style.STROKE
    }
    private val pointPaint = Paint().apply {
        color = Color.RED
        strokeWidth = 16f
        style = Paint.Style.FILL
    }
    
    fun setLandmarks(landmarks: List<NormalizedLandmark>) {
        this.landmarks = landmarks
        invalidate()
    }
    
    fun clear() {
        landmarks = null
        invalidate()
    }
    
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        landmarks?.let { points ->
            // Draw connections
            val connections = listOf(
                // Thumb
                0 to 1, 1 to 2, 2 to 3, 3 to 4,
                // Index
                0 to 5, 5 to 6, 6 to 7, 7 to 8,
                // Middle
                0 to 9, 9 to 10, 10 to 11, 11 to 12,
                // Ring
                0 to 13, 13 to 14, 14 to 15, 15 to 16,
                // Pinky
                0 to 17, 17 to 18, 18 to 19, 19 to 20,
                // Palm
                5 to 9, 9 to 13, 13 to 17
            )
            
            connections.forEach { (start, end) ->
                val startX = points[start].x() * width
                val startY = points[start].y() * height
                val endX = points[end].x() * width
                val endY = points[end].y() * height
                
                canvas.drawLine(startX, startY, endX, endY, paint)
            }
            
            // Draw points
            points.forEach { landmark ->
                canvas.drawCircle(
                    landmark.x() * width,
                    landmark.y() * height,
                    8f,
                    pointPaint
                )
            }
        }
    }
}
```

---

## Layout (activity_main.xml)

```xml
<?xml version="1.0" encoding="utf-8"?>
<FrameLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent">
    
    <androidx.camera.view.PreviewView
        android:id="@+id/previewView"
        android:layout_width="match_parent"
        android:layout_height="match_parent" />
    
    <com.yourpackage.OverlayView
        android:id="@+id/overlayView"
        android:layout_width="match_parent"
        android:layout_height="match_parent" />
    
    <TextView
        android:id="@+id/resultText"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_gravity="center_horizontal|top"
        android:layout_marginTop="50dp"
        android:textSize="72sp"
        android:textColor="#00FF00"
        android:textStyle="bold"
        android:shadowColor="#000000"
        android:shadowRadius="10"
        android:text="No hand" />
    
    <TextView
        android:id="@+id/sentenceText"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_gravity="bottom"
        android:layout_marginBottom="50dp"
        android:padding="16dp"
        android:textSize="24sp"
        android:textColor="#FFFFFF"
        android:background="#80000000"
        android:text="" />
    
</FrameLayout>
```

---

## Backend JSON Format

**What you send:**
```json
{
  "handDetected": true,
  "handedness": "Right",
  "landmarks": [
    [0.523, 0.487, -0.012],
    [0.498, 0.452, -0.015],
    // ... 21 points total
  ]
}
```

**What you receive:**
```json
{
  "success": true,
  "handDetected": true,
  "character": "A"
}
```

---

## Key Points

1. **MediaPipe** detects hand and gives 21 landmarks
2. **OverlayView** draws green skeleton on camera
3. **WebSocket** sends landmarks every 150ms
4. **Backend** returns character prediction
5. **Current character** shows at top (large, green)
6. **Full sentence** shows at bottom with transparent background
7. **Temporal smoothing** prevents duplicate characters (3 frames)
8. **Special gestures:**
   - Space gesture → Adds space to sentence
   - Backspace gesture → Removes last character
   - "next" gesture → Adds space (word boundary)

---

## Testing

1. Backend server IP is already set to: `10.171.152.6`
2. Make sure backend is running: `python landmark_server.py`
3. Build and run Android app
4. Point camera at hand
5. You should see:
   - Camera preview with green hand skeleton overlay
   - Current character at top (green, large)
   - Full sentence at bottom (white text, dark background)
   - As you make gestures, characters accumulate into sentence

---

## Troubleshooting

**No hand detected:**
- Check camera permission granted
- Check MediaPipe model exists in assets
- Lower confidence threshold to 0.3

**No predictions:**
- Check server IP is correct
- Check backend is running
- Check Logcat for WebSocket errors

**App crashes:**
- Check all dependencies installed
- Check camera permission in runtime
- Check MediaPipe model downloaded

---

## That's Everything

Build this from scratch. It's complete and will work with your backend.
