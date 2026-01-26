# HandSpeak.ai Backend Server

## Installation

1. Install required packages:
```bash
pip install fastapi uvicorn python-multipart
```

2. Run the server:
```bash
python server.py
```

The server will start on `http://0.0.0.0:8000`

## API Endpoints

### 1. Health Check
- **GET** `/`
- Returns server status

### 2. Recognize Sign Language
- **POST** `/api/recognize`
- **Body**: `multipart/form-data` with image file
- **Returns**: 
```json
{
  "success": true,
  "character": "A",
  "confidence": 0.95,
  "message": "Character recognized successfully"
}
```

### 3. Get Word Suggestions
- **POST** `/api/suggest-words`
- **Body**: `{"word": "hel"}`
- **Returns**:
```json
{
  "word": "hel",
  "suggestions": ["hello", "help", "held", "hell"]
}
```

### 4. Text to Speech
- **POST** `/api/text-to-speech`
- **Body**: `{"text": "Hello world"}`
- **Returns**: Audio data or instruction to use device TTS

## API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI)

## For Android Development

1. Find your computer's IP address:
   - Windows: `ipconfig` → Look for IPv4 Address
   - Example: `192.168.1.100`

2. Update Android app to use: `http://192.168.1.100:8000`

3. Make sure both devices are on the same WiFi network

4. For production, deploy server to cloud (AWS, Heroku, etc.)
