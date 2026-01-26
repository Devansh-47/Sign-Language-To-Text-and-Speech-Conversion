# Android Studio Project Setup - HandSpeak.ai

## PROMPT FOR ANDROID STUDIO (Copy and paste this into Android Studio's Gemini/AI assistant):

---

**Create a complete Android application for HandSpeak.ai - a Sign Language to Text conversion app. Here are the requirements:**

## App Overview
Build a native Android app that captures hand signs via camera and converts them to text using a backend API.

## Core Features Required:

### 1. Camera Integration (REAL-TIME, NOT CAPTURE)
- **Continuous camera preview using CameraX**
- **Real-time frame processing (like desktop app)**
- Front-facing camera for sign language detection
- **NO capture button - continuous recognition**
- Process frames every 100-200ms and send to server via WebSocket
- Display live video feed with hand detection overlay

### 2. Main Screen UI Components:
- **Top Section**: Live camera preview (full width, continuous)
- **Middle Section**: 
  - Detected character display (large text, updates in real-time)
  - Current sentence/text being built
  - Hand detection status indicator (green = hand detected, red = no hand)
- **Bottom Section**:
  - 4 suggestion buttons for word completion (auto-updates)
  - "Speak" button (uses Android TTS)
  - "Clear" button to reset sentence
  - "Space" button to add space between words

### 3. Backend API Integration:
- Base URL: `http://YOUR_SERVER_IP:8000`
- **Primary: WebSocket for real-time recognition**
- HTTP endpoints for word suggestions

#### a) WebSocket Connection for Real-Time Recognition
```kotlin
WebSocket URL: ws://YOUR_SERVER_IP:8000/ws/recognize

Client sends (continuously, every 100-200ms):
- Base64 encoded JPEG frame from camera

Server responds (for each frame):
{
  "success": true,
  "character": "A",
  "handDetected": true,
  "message": "Character recognized"
}
```

#### b) Get Word Suggestions (HTTP)
```kotlin
POST /api/suggest-words
Content-Type: application/json
Body: {"word": "current_word"}

Response:
{
  "word": "hel",
  "suggestions": ["hello", "help", "held", "hell"]
}
```

### 4. App Flow (CONTINUOUS REAL-TIME):
1. App starts with live camera feed
2. **Every 100-200ms, app captures current frame**
3. **Frame is encoded to base64 JPEG and sent via WebSocket**
4. Backend processes frame and returns character (if hand detected)
5. **If same character detected for 3+ consecutive frames, add to sentence**
6. App requests word suggestions for current incomplete word
7. User can tap suggestion buttons to complete word
8. User can tap "Speak" to hear the sentence using Android TTS
9. User can tap "Clear" to start over

**CRITICAL: This mimics the desktop app's `video_loop()` function - continuous processing, not button-triggered capture!**

### 5. Technical Requirements:

#### Dependencies (build.gradle):
```gradle
dependencies {
    // CameraX
    implementation "androidx.camera:camera-camera2:1.3.1"
    implementation "androidx.camera:camera-lifecycle:1.3.1"
    implementation "androidx.camera:camera-view:1.3.1"
    
    // WebSocket for real-time communication
    implementation 'com.squareup.okhttp3:okhttp:4.12.0'
    
    // Networking (for word suggestions)
    implementation 'com.squareup.retrofit2:retrofit:2.9.0'
    implementation 'com.squareup.retrofit2:converter-gson:2.9.0'
    implementation 'com.squareup.okhttp3:logging-interceptor:4.11.0'
    
    // Image processing
    implementation 'com.github.bumptech.glide:glide:4.16.0'
    
    // Kotlin coroutines
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
    
    // ViewModel and LiveData
    implementation 'androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0'
    implementation 'androidx.lifecycle:lifecycle-livedata-ktx:2.7.0'
}
```

#### Permissions (AndroidManifest.xml):
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-feature android:name="android.hardware.camera.any" />
```

### 6. Architecture:
- **MVVM Pattern**
- **Repository pattern for API calls**
- **LiveData for UI updates**
- **Coroutines for async operations**

### 7. Key Classes to Create:

#### a) MainActivity.kt
- Manages live camera preview with CameraX
- **Implements ImageAnalysis.Analyzer for frame-by-frame processing**
- Handles UI interactions
- Displays real-time results

#### b) SignLanguageViewModel.kt
- Manages app state (current sentence, detected characters)
- **Character debouncing logic (same char 3+ times = confirmed)**
- Handles business logic
- Communicates with WebSocket manager

#### c) WebSocketManager.kt
```kotlin
class WebSocketManager(private val serverUrl: String) {
    private var webSocket: WebSocket? = null
    
    fun connect(onMessageReceived: (RecognitionResponse) -> Unit)
    fun sendFrame(base64Frame: String)
    fun disconnect()
}
```

#### d) ApiService.kt (Retrofit interface for word suggestions)
```kotlin
interface ApiService {
    @POST("/api/suggest-words")
    suspend fun getSuggestions(@Body request: WordRequest): SuggestionResponse
}
```

#### e) CameraFrameAnalyzer.kt
```kotlin
class CameraFrameAnalyzer(
    private val onFrameProcessed: (ByteArray) -> Unit
) : ImageAnalysis.Analyzer {
    private var lastAnalyzedTimestamp = 0L
    private val frameIntervalMs = 150 // Process every 150ms
    
    override fun analyze(image: ImageProxy) {
        val currentTimestamp = System.currentTimeMillis()
        if (currentTimestamp - lastAnalyzedTimestamp >= frameIntervalMs) {
            // Convert ImageProxy to JPEG bytes
            // Encode to base64
            // Send to WebSocket
            lastAnalyzedTimestamp = currentTimestamp
        }
        image.close()
    }
}
```

#### f) Repository.kt
- Handles API calls
- Error handling
- Data transformation

#### e) Models (data classes)
- RecognitionResponse
- SuggestionResponse
- WordRequest

### 8. UI Layout (activity_main.xml):
```xml
<androidx.constraintlayout.widget.ConstraintLayout>
    
    <!-- Camera Preview -->
    <androidx.camera.view.PreviewView
        android:id="@+id/cameraPreview"
        android:layout_width="match_parent"
        android:layout_height="400dp" />
    
    <!-- Character Display -->
    <TextView
        android:id="@+id/characterText"
        android:text="Character: "
        android:textSize="24sp" />
    
    <!-- Sentence Display -->
    <TextView
        android:id="@+id/sentenceText"
        android:text="Sentence: "
        android:textSize="20sp" />
    
    <!-- Suggestion Buttons -->
    <LinearLayout
        android:orientation="horizontal">
        <Button android:id="@+id/suggestion1" />
        <Button android:id="@+id/suggestion2" />
        <Button android:id="@+id/suggestion3" />
        <Button android:id="@+id/suggestion4" />
    </LinearLayout>
    
    <!-- Action Buttons -->
    <LinearLayout
        android:orientation="horizontal">
        <Button 
            android:id="@+id/captureButton"
            android:text="Capture" />
        <Button 
            android:id="@+id/speakButton"
            android:text="Speak" />
        <Button 
            android:id="@+id/clearButton"
            android:text="Clear" />
    </LinearLayout>
    
</androidx.constraintlayout.widget.ConstraintLayout>
```

### 9. Settings Screen:
- API server URL configuration
- Camera settings
- Speech rate adjustment
- Theme selection (Light/Dark)

### 10. Additional Features:
- Loading indicators during API calls
- Error messages with retry option
- Offline mode detection
- Image preview before sending
- History of recognized characters
- Save sentences to clipboard

### 11. Material Design:
- Use Material 3 components
- Follow Android design guidelines
- Smooth animations and transitions
- Haptic feedback on button clicks

### 12. Error Handling:
- Network errors (show toast/snackbar)
- Camera permission denied
- Server connection issues
- Invalid image detection

## Project Structure:
```
app/src/main/
├── java/com/handspeak/ai/
│   ├── ui/
│   │   ├── MainActivity.kt
│   │   ├── SettingsActivity.kt
│   │   └── adapters/
│   ├── viewmodel/
│   │   └── SignLanguageViewModel.kt
│   ├── repository/
│   │   └── SignLanguageRepository.kt
│   ├── network/
│   │   ├── ApiService.kt
│   │   ├── RetrofitClient.kt
│   │   └── models/
│   ├── utils/
│   │   ├── CameraHelper.kt
│   │   ├── TTSHelper.kt
│   │   └── ImageUtils.kt
│   └── Application.kt
└── res/
    ├── layout/
    ├── values/
    └── drawable/
```

## Testing Requirements:
- Test with different lighting conditions
- Test with multiple hand positions
- Test offline behavior
- Test error scenarios

## Deliverables:
1. Complete working Android app
2. Proper error handling
3. Clean, documented code
4. User-friendly interface
5. Smooth performance (60 FPS camera preview)

---

**Please create this complete Android application with all the necessary files, configurations, and code.**
