# Android App - Quick Integration Guide

## What Changed
- **OLD:** Send Base64 JPEG frames to `/ws/recognize`
- **NEW:** Send JSON landmarks to `/ws/landmarks`

---

## 1. WebSocket Endpoint
Change connection URL:
```
ws://YOUR_IP:8000/ws/landmarks
```

---

## 2. JSON Format

**Send this:**
```json
{
  "handDetected": true,
  "handedness": "Right",
  "landmarks": [
    [x, y, z],  // 21 points
    [x, y, z],
    ...
  ]
}
```

**Receive this:**
```json
{
  "success": true,
  "handDetected": true,
  "character": "A"
}
```

---

## 3. Code Changes

### Extract landmarks from MediaPipe:
```kotlin
val landmarks = result.landmarks()[0].map { 
    listOf(it.x(), it.y(), it.z()) 
}
val handedness = result.handedness()[0][0].categoryName()
```

### Build JSON and send:
```kotlin
val json = JSONObject().apply {
    put("handDetected", true)
    put("handedness", handedness)
    put("landmarks", JSONArray(landmarks))
}
webSocket.send(json.toString())
```

### Draw hand overlay (for visualization):
```kotlin
// In your camera preview
canvas.drawCircle(landmark.x() * width, landmark.y() * height, 8f, paint)

// Draw connections between landmarks
val connections = listOf(
    0 to 1, 1 to 2, 2 to 3, 3 to 4,  // Thumb
    0 to 5, 5 to 6, 6 to 7, 7 to 8,  // Index
    // ... etc for all fingers
)
connections.forEach { (start, end) ->
    canvas.drawLine(
        landmarks[start].x() * width, 
        landmarks[start].y() * height,
        landmarks[end].x() * width, 
        landmarks[end].y() * height,
        paint
    )
}
```

---

## 4. Key Points

- MediaPipe gives you 21 landmarks (0-20)
- Landmark 0 = Wrist, 1-4 = Thumb, 5-8 = Index, etc.
- Send as 3D array: `[[x,y,z], [x,y,z], ...]`
- Backend responds with character instantly
- Draw circles + lines on canvas to show hand skeleton

---

## Debugging

**Test without camera:**
```kotlin
// Send fake landmarks
val fake = (0..20).map { listOf(0.5f, 0.5f, 0.0f) }
send(JSONObject().apply {
    put("handDetected", true)
    put("handedness", "Right")
    put("landmarks", JSONArray(fake))
})
```

**Check logs:**
```
Log.d("WS", "Sent: ${json.length()} bytes")
Log.d("WS", "Received: $response")
```

---

## That's It

1. Change URL to `/ws/landmarks`
2. Extract 21 landmarks from MediaPipe
3. Send as JSON array
4. Draw overlay for visual feedback
5. Done.
