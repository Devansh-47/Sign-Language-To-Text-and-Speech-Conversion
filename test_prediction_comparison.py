"""
Test to compare predictions between desktop app logic and server logic
"""
import cv2
import numpy as np
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import mediapipe_compat
import math

# Load model
model = load_model('cnn8grps_rad1_model.h5')
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
offset = 29

def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

# Read a test image
print("Please show a sign to your webcam and press 's' to capture...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    cv2.imshow("Press 's' to capture", frame)
    key = cv2.waitKey(1)
    
    if key == ord('s'):
        # Process this frame
        cv2image = cv2.flip(frame, 1)
        hands = hd.findHands(cv2image, draw=False, flipType=True)
        
        if hands[0]:
            hand = hands[0][0]
            x, y, w, h = hand['bbox']
            image = cv2image[y - offset:y + h + offset, x - offset:x + w + offset]
            
            if image is not None and image.size > 0:
                handz = hd2.findHands(image, draw=False, flipType=True)
                
                if handz[0]:
                    handmap = handz[0][0]
                    pts = handmap['lmList']
                    
                    print(f"\n=== Hand Detection Info ===")
                    print(f"BBox: x={x}, y={y}, w={w}, h={h}")
                    print(f"Number of landmarks: {len(pts)}")
                    print(f"First landmark (index 0): {pts[0]}")
                    print(f"Fourth landmark (index 4): {pts[4]}")
                    print(f"Landmark type: {type(pts[0])}")
                    
                    # Draw skeleton on white
                    white = cv2.imread("white.jpg")
                    os = ((400 - w) // 2) - 15
                    os1 = ((400 - h) // 2) - 15
                    
                    for t in range(0, 4, 1):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), 
                                (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(5, 8, 1):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), 
                                (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(9, 12, 1):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), 
                                (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(13, 16, 1):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), 
                                (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(17, 20, 1):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), 
                                (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    
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
                    
                    for i in range(21):
                        cv2.circle(white, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)
                    
                    # Show skeleton
                    cv2.imshow("Skeleton", white)
                    
                    # Predict
                    white = white.reshape(1, 400, 400, 3)
                    prob = np.array(model.predict(white, verbose=0)[0], dtype='float32')
                    ch1 = int(np.argmax(prob, axis=0))
                    prob[ch1] = 0
                    ch2 = int(np.argmax(prob, axis=0))
                    prob[ch2] = 0
                    ch3 = int(np.argmax(prob, axis=0))
                    
                    print(f"\n=== Model Output ===")
                    print(f"Top 3 predictions: ch1={ch1}, ch2={ch2}, ch3={ch3}")
                    print(f"pl = [{ch1}, {ch2}]")
                    
                    # Test first few conditions
                    pl = [ch1, ch2]
                    l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
                         [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
                         [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
                    
                    if pl in l:
                        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                            print("CONDITION MET: Changed ch1 from", ch1, "to 0")
                            ch1 = 0
                    
                    print(f"\nFinal ch1 after first condition: {ch1}")
                    
                    cv2.waitKey(0)
                    break
        
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
