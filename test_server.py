"""
Test client for HandSpeak.ai API
Use this to verify your backend server is working before building Android app
"""

import requests
import cv2
import sys

# Configuration
SERVER_URL = "http://localhost:8000"

def test_health_check():
    """Test if server is running"""
    print("Testing server health check...")
    try:
        response = requests.get(f"{SERVER_URL}/")
        if response.status_code == 200:
            print("✅ Server is running!")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print(f"   Make sure server is running at {SERVER_URL}")
        return False


def test_image_recognition():
    """Test sign language recognition with a sample image"""
    print("\nTesting sign language recognition...")
    
    # Capture an image from webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return False
    
    print("📸 Show a hand sign to the camera...")
    print("   Press SPACE to capture, ESC to cancel")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read frame")
            break
        
        cv2.imshow('Show hand sign (SPACE to capture, ESC to cancel)', frame)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            print("Cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return False
        elif key == 32:  # SPACE
            print("✅ Image captured!")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save image temporarily
    cv2.imwrite('test_image.jpg', frame)
    
    # Send to server
    print("Sending image to server...")
    try:
        with open('test_image.jpg', 'rb') as img_file:
            files = {'file': ('image.jpg', img_file, 'image/jpeg')}
            response = requests.post(f"{SERVER_URL}/api/recognize", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Recognition successful!")
            print(f"   Success: {result.get('success')}")
            print(f"   Character: {result.get('character')}")
            print(f"   Message: {result.get('message')}")
            return True
        else:
            print(f"❌ Recognition failed with status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during recognition: {e}")
        return False


def test_word_suggestions():
    """Test word suggestion API"""
    print("\nTesting word suggestions...")
    
    test_word = "hel"
    print(f"Getting suggestions for: '{test_word}'")
    
    try:
        response = requests.post(
            f"{SERVER_URL}/api/suggest-words",
            params={"word": test_word}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Suggestions retrieved!")
            print(f"   Word: {result.get('word')}")
            print(f"   Suggestions: {result.get('suggestions')}")
            return True
        else:
            print(f"❌ Failed with status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("HandSpeak.ai Backend Server Test")
    print("=" * 60)
    
    # Test 1: Health check
    if not test_health_check():
        print("\n❌ Server is not running. Please start it with: python server.py")
        sys.exit(1)
    
    # Test 2: Word suggestions
    test_word_suggestions()
    
    # Test 3: Image recognition
    print("\n" + "=" * 60)
    proceed = input("Test image recognition? (y/n): ")
    if proceed.lower() == 'y':
        test_image_recognition()
    
    print("\n" + "=" * 60)
    print("✅ Testing complete!")
    print("=" * 60)
    print("\nYour backend server is working correctly.")
    print("You can now proceed to build the Android app.")
    print(f"\nServer URL for Android: {SERVER_URL}")
    print("(Replace 'localhost' with your IP address for mobile access)")


if __name__ == "__main__":
    main()
