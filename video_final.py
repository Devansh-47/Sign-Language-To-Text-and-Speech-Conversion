import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from gtts import gTTS
import os
import tempfile
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load the trained model
model = load_model('sign_language_model.h5')

def play_audio(text):
    # Convert text to speech
    tts = gTTS(text=text, lang='en', slow=False)
    
    # Save the audio to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tts.save(temp_file.name)
    
    # Load and play the audio file
    pygame.mixer.music.load(temp_file.name)
    pygame.mixer.music.play()
    
    # Wait until the audio finishes playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
    # Clean up
    os.remove(temp_file.name)

def predict_from_video(video_path, model, frame_size=(64, 64)):
    cap = cv2.VideoCapture(video_path)
    
    last_predicted_class = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        resized_frame = cv2.resize(frame, frame_size)
        resized_frame = resized_frame / 255.0  # Normalize
        frame_array = img_to_array(resized_frame)
        frame_array = np.expand_dims(frame_array, axis=0)
        
        # Predict
        prediction = model.predict(frame_array)
        predicted_class = np.argmax(prediction)
        
        # Convert predicted class to a readable string (e.g., "Class 1")
        predicted_text = f"Class {predicted_class}"
        
        # Check if the predicted class has changed
        if predicted_class != last_predicted_class:
            play_audio(predicted_text)
            last_predicted_class = predicted_class
        
        # Display the result on the frame
        cv2.putText(frame, f"Prediction: {predicted_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the video frame
        cv2.imshow('Video Prediction', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Predict using a new video
predict_from_video('/Users/shriya/Desktop/isl_video.mp4', model)
