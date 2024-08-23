import cv2
import os

# Initialize webcam capture
capture = cv2.VideoCapture(0)

# Check if the capture opened successfully
if not capture.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Directory to save captured images
save_dir = "data_ka_nya/jha"
os.makedirs(save_dir, exist_ok=True)

# Image counter
img_count = 0
max_captures = 100  # Maximum number of captures

print(f"Press 's' to save an image. Limiting to {max_captures} captures.")

while img_count < max_captures:
    # Capture frame-by-frame
    ret, frame = capture.read()
    
    # Check if the frame was captured successfully
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Flip the frame horizontally to correct the mirror image
    frame = cv2.flip(frame, 1)

    # Display the resulting frame
    cv2.imshow("Webcam", frame)
    
    # Wait for key press
    key = cv2.waitKey(1) & 0xFF
    
    # Save image on 's' key press
    if key == ord('s'):
        img_name = os.path.join(save_dir, f"image_{img_count}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Image saved as {img_name}")
        img_count += 1
    
    # Exit on 'q' key press
    elif key == ord('q'):
        print("Exiting...")
        break

# Release the capture and close windows
capture.release()
cv2.destroyAllWindows()

print(f"Captured {img_count} images.")
