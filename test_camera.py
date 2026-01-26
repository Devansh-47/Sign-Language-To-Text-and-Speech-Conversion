    import cv2

print("Testing camera access...")
cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("✓ Camera is opened successfully")
    ret, frame = cap.read()
    if ret:
        print(f"✓ Can read frames: {ret}")
        print(f"✓ Frame shape: {frame.shape}")
    else:
        print("✗ Cannot read frame")
else:
    print("✗ Camera failed to open")

cap.release()
