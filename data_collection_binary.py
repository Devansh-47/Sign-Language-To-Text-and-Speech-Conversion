import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import os
from keras.models import load_model
import traceback

# Initialize the webcam
capture = cv2.VideoCapture(0)

# Initialize hand detectors
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

# Directory paths
base_dir = "data_ka_nya"
p_dir = "ka"
c_dir = "a"
count = len(os.listdir(os.path.join(base_dir, p_dir)))

offset = 30
step = 1
flag = False
suv = 0

# Create a white image for visual reference
white = np.ones((400, 400), np.uint8) * 255
cv2.imwrite("white.jpg", white)

while True:
    try:
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        hands = hd.findHands(frame, draw=False, flipType=True)

        img_final = img_final1 = img_final2 = None

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Ensure bounding box dimensions are valid
            if w > 0 and h > 0:
                x1, y1 = max(x - offset, 0), max(y - offset, 0)
                x2, y2 = min(x + w + offset, frame.shape[1]), min(y + h + offset, frame.shape[0])
                roi = frame[y1:y2, x1:x2]

                # Image processing for gray and binary images
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (1, 1), 2)

                gray2 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur2 = cv2.GaussianBlur(gray2, (5, 5), 2)
                th3 = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                ret, test_image = cv2.threshold(th3, 27, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Resize processed images to fit 400x400 dimensions
                img_final1 = np.ones((400, 400), np.uint8) * 148
                h, w = blur.shape
                img_final1[((400 - h) // 2):((400 - h) // 2) + h, ((400 - w) // 2):((400 - w) // 2) + w] = blur

                img_final = np.ones((400, 400), np.uint8) * 255
                h, w = test_image.shape
                img_final[((400 - h) // 2):((400 - h) // 2) + h, ((400 - w) // 2):((400 - w) // 2) + w] = test_image

                # Draw hand landmarks on a white background
                white = cv2.imread("white.jpg")
                handz = hd2.findHands(roi, draw=False, flipType=True)
                if handz:
                    hand = handz[0]
                    pts = hand['lmList']
                    os = ((400 - w) // 2) - 15
                    os1 = ((400 - h) // 2) - 15
                    for t in range(0, 4):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(5, 8):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(9, 12):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(13, 16):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(17, 20):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[5][0] + os, pts[5][1] + os1), (pts[9][0] + os, pts[9][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[9][0] + os, pts[9][1] + os1), (pts[13][0] + os, pts[13][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[13][0] + os, pts[13][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[5][0] + os, pts[5][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)

                    for i in range(21):
                        cv2.circle(white, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)

                    cv2.imshow("skeleton", white)

            # Additional hand detection in the white background image
            hands = hd.findHands(white, draw=False, flipType=True)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                cv2.rectangle(white, (x - offset, y - offset), (x + w, y + h), (3, 255, 25), 3)

            # Prepare images with drawing for processing
            image1 = frame[y1:y2, x1:x2]
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            blur1 = cv2.GaussianBlur(gray1, (1, 1), 2)

            test_image2 = blur1
            img_final2 = np.ones((400, 400), np.uint8) * 148
            h, w = test_image2.shape
            img_final2[((400 - h) // 2):((400 - h) // 2) + h, ((400 - w) // 2):((400 - w) // 2) + w] = test_image2

            # Display processed images
            cv2.imshow("binary", img_final)
            # cv2.imshow("gray w/o draw", img_final1)

            # Image saving functionality
            if flag:
                if suv == 50:
                    flag = False
                if step % 2 == 0:
                    # Save images
                    cv2.imwrite(os.path.join(base_dir, p_dir, c_dir + str(count) + ".jpg"), img_final1)
                    cv2.imwrite(os.path.join(base_dir, "Gray_imgs_with_drawing", p_dir, c_dir + str(count) + ".jpg"), img_final2)

                    count += 1
                    suv += 1
                step += 1

        # Display the current frame
        cv2.imshow("frame", frame)

        # Key
        key = cv2.waitKey(1)
        if key == ord('s'):
            flag = True
        if key == ord('q'):
            break

    except Exception as e:
        traceback.print_exc()

# Release resources
capture.release()
cv2.destroyAllWindows()