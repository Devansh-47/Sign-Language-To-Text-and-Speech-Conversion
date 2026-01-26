"""
Compatibility layer for mediapipe 0.10.x with cvzone
Mediapipe 0.10.x uses a task-based API instead of solutions.
This layer creates a backwards-compatible solutions API for cvzone.
"""

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
import cv2
import numpy as np


# Create the Hands detector with proper configuration
try:
    options = vision.HandLandmarkerOptions(
        base_options=base_options.BaseOptions(
            model_asset_path='app/data/hand_landmarker.task'),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    hand_detector = vision.HandLandmarker.create_from_options(options)
    detector_available = True
except Exception as e:
    print(f"Warning: Could not load hand detector model: {e}")
    print("Falling back to test mode - hand detection disabled")
    detector_available = False
    hand_detector = None


class NormalizedLandmark:
    """Mimic mediapipe's NormalizedLandmark"""
    def __init__(self, x, y, z=0.0, presence=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.presence = presence


class HandLandmarks:
    """Mimic mediapipe's hand landmarks list"""
    def __init__(self, landmarks_data):
        self.landmark = [NormalizedLandmark(lm.x, lm.y, lm.z) for lm in landmarks_data]


class Handedness:
    """Mimic mediapipe's Handedness with classification"""
    def __init__(self, label="Right", score=1.0):
        self.label = label
        self.score = score


class Classification:
    """Mimic mediapipe's Classification"""
    def __init__(self, label="Right", score=1.0):
        self.label = label
        self.score = score


class HandednessWrapper:
    """Wrapper for handedness to match old API format"""
    def __init__(self, label="Right", score=1.0):
        self.classification = [Classification(label, score)]


class Hands:
    """Wrapper for HandLandmarker to match old solutions.hands API"""
    
    def __init__(self, static_image_mode=False, max_num_hands=2,
                 model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
    
    def process(self, image_rgb):
        """Process image and return results in old API format"""
        # Create Results object matching old API
        class Results:
            def __init__(self):
                self.multi_hand_landmarks = []
                self.multi_handedness = []
        
        results = Results()
        
        if not detector_available or hand_detector is None:
            return results
        
        try:
            # Convert numpy array to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Detect hands
            detection_result = hand_detector.detect(mp_image)
            
            if detection_result.hand_landmarks:
                # Convert new format to old format
                for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                    # Create a landmarks object with .landmark attribute for cvzone compatibility
                    landmarks_obj = HandLandmarks(hand_landmarks)
                    results.multi_hand_landmarks.append(landmarks_obj)
                
                # Wrap handedness in the expected format
                if detection_result.handedness:
                    for handedness in detection_result.handedness:
                        # handedness is a list of Classification objects
                        # Extract the label from the first classification
                        if handedness:
                            label = handedness[0].category_name  # e.g., "Right" or "Left"
                            score = handedness[0].score
                            results.multi_handedness.append(HandednessWrapper(label, score))
        
        except Exception as e:
            # Silently handle errors to allow app to continue
            pass
        
        return results


class Solutions:
    """Container for solution modules"""
    class hands:
        """Hands solution module"""
        Hands = Hands
    
    class drawing_utils:
        """Drawing utilities stub"""
        @staticmethod
        def draw_landmarks(image, landmarks, connections=None, **kwargs):
            return image


# Monkey-patch into mediapipe if solutions doesn't exist
if not hasattr(mp, 'solutions'):
    mp.solutions = Solutions()
    print("[OK] MediaPipe compatibility layer loaded")



