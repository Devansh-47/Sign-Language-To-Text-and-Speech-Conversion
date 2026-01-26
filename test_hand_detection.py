import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe_compat  # Import compatibility layer first

cap = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)

print("Testing hand detection...")
ret, frame = cap.read()
if ret:
    frame = cv2.flip(frame, 1)
    hands = hd.findHands(frame, draw=False, flipType=True)
    print(f"hands type: {type(hands)}")
    print(f"hands length: {len(hands)}")
    print(f"hands content: {hands}")
    if isinstance(hands, list) and len(hands) > 0:
        print(f"hands[0] type: {type(hands[0])}")
        if hands[0]:
            print("Hand detected!")
        else:
            print("No hand detected")

cap.release()
