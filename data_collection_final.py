import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os as oss
import traceback

# Initialize video capture and hand detectors
capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

# Initialize variables
count = len(oss.listdir("data_ka_nya/ka"))
c_dir = '1'
offset = 15
step = 1
flag = False
suv = 0

# Create a white background image (400x400 pixels)
white = np.ones((400, 400), np.uint8) * 255

while True:
    try:
        # Capture a frame from the webcam
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally

        # Detect hands in the frame
        hands = hd.findHands(frame, draw=False, flipType=True)

        # Reset the white background for each iteration
        white = np.ones((400, 400), np.uint8) * 255

        if hands:
            hand = hands[0]
            # Ensure bbox exists and has exactly 4 elements
            if 'bbox' in hand and len(hand['bbox']) == 4:
                x, y, w, h = hand['bbox']

                # Extract the region of interest (ROI) from the frame
                image = np.array(frame[y - offset:y + h + offset, x - offset:x + w + offset])

                # Detect hands again in the ROI
                handz, imz = hd2.findHands(image, draw=True, flipType=True)
                if handz:
                    hand = handz[0]
                    pts = hand['lmList']

                    # Offset calculations for drawing
                    os = ((400 - w) // 2) - 15
                    os1 = ((400 - h) // 2) - 15

                    # Draw lines between specific landmarks to form the hand skeleton
                    for t in range(0, 4):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1),
                                 (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(5, 8):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1),
                                 (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(9, 12):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1),
                                 (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(13, 16):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1),
                                 (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(17, 20):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1),
                                 (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)

                    # Additional lines to complete the skeleton
                    cv2.line(white, (pts[5][0] + os, pts[5][1] + os1),
                             (pts[9][0] + os, pts[9][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[9][0] + os, pts[9][1] + os1),
                             (pts[13][0] + os, pts[13][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[13][0] + os, pts[13][1] + os1),
                             (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0] + os, pts[0][1] + os1),
                             (pts[5][0] + os, pts[5][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0] + os, pts[0][1] + os1),
                             (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)

                    # Draw circles on the landmarks
                    for i in range(21):
                        cv2.circle(white, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)

                    # Display the skeleton image
                    cv2.imshow("1", white)

        # Add text overlay on the frame
        frame = cv2.putText(frame, "dir=" + str(c_dir) + "  count=" + str(count), (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("frame", frame)

        # Handle keypress events
        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:  # ESC key to exit
            break
        if interrupt & 0xFF == ord('n'):  # Move to the next directory
            c_dir = chr(ord(c_dir) + 1)
            if ord(c_dir) == ord('Z') + 1:
                c_dir = 'A'
            flag = False
            count = len(oss.listdir("data_ka_nya/ka/" + c_dir + "/"))

        if interrupt & 0xFF == ord('a'):  # Toggle flag
            flag = not flag
            if flag:
                suv = 0

        # Save images when the flag is set
        if flag:
            if suv == 180:
                flag = False
            if step % 3 == 0:
                cv2.imwrite(f"data_ka_nya/ka/{c_dir}/{count}.jpg", white)
                count += 1
                suv += 1
            step += 1

    except Exception:
        print("==", traceback.format_exc())

# Release resources
capture.release()
cv2.destroyAllWindows()
