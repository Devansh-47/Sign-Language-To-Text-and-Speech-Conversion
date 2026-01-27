"""
FastAPI Backend Server for HandSpeak.ai
Provides REST API and WebSocket for real-time sign language recognition from mobile apps
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import cv2
import numpy as np
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import mediapipe_compat
import base64
import io
from PIL import Image
import math
from string import ascii_uppercase
import enchant

from pydantic import BaseModel

app = FastAPI(title="HandSpeak.ai API", version="1.0.0")

# Request model for word suggestions
class WordRequest(BaseModel):
    word: str

# Enable CORS for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Android app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models and detectors
print("Loading models...")
model = load_model('cnn8grps_rad1_model.h5')
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
ddd = enchant.Dict("en-US")
print("[OK] Models loaded successfully")

# Constants
offset = 29

def distance(x, y):
    """Calculate Euclidean distance between two points"""
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))


def predict_sign(image, pts, w, h):
    """
    Predict sign language character from hand landmarks
    NORMALIZED to match final_pred.py pixel-for-pixel
    """
    # CRITICAL: Load fresh copy for each request (FastAPI is async - no shared state)
    white = cv2.imread("white.jpg")
    if white is None:
        return "Error"
    # CRITICAL: Create independent copy to avoid memory corruption in async context
    white = white.copy()
    
    os = ((400 - w) // 2) - 15
    os1 = ((400 - h) // 2) - 15
    
    # Draw skeleton - EXACT code from final_pred.py
    for t in range(0, 4, 1):
        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
    for t in range(5, 8, 1):
        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
    for t in range(9, 12, 1):
        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
    for t in range(13, 16, 1):
        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
    for t in range(17, 20, 1):
        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
    cv2.line(white, (pts[5][0] + os, pts[5][1] + os1), (pts[9][0] + os, pts[9][1] + os1), (0, 255, 0), 3)
    cv2.line(white, (pts[9][0] + os, pts[9][1] + os1), (pts[13][0] + os, pts[13][1] + os1), (0, 255, 0), 3)
    cv2.line(white, (pts[13][0] + os, pts[13][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)
    cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[5][0] + os, pts[5][1] + os1), (0, 255, 0), 3)
    cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)
    
    for i in range(21):
        cv2.circle(white, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)
    
    # DEBUG: Save skeleton every ~30 frames for pixel-comparison with final_pred.py
    # This verifies that preprocessing is pixel-identical between local GUI and backend
    import random
    if random.randint(1, 30) == 1:
        debug_filename = f"debug_backend_skeleton_{random.randint(1000,9999)}.jpg"
        cv2.imwrite(debug_filename, white)
        print(f"[DEBUG] Saved skeleton: {debug_filename}")
    
    # Predict using model (UNCHANGED)
    white = white.reshape(1, 400, 400, 3)
    prob = np.array(model.predict(white, verbose=0)[0], dtype='float32')
    ch1 = int(np.argmax(prob, axis=0))
    prob[ch1] = 0
    ch2 = int(np.argmax(prob, axis=0))
    prob[ch2] = 0
    ch3 = int(np.argmax(prob, axis=0))
    prob[ch3] = 0

    pl = [ch1, ch2]

    # EXACT conditions from final_pred.py
    # condition for [Aemnst]
    l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
         [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
         [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][
            1]):
            ch1 = 0

    # condition for [o][s]
    l = [[2, 2], [2, 1]]
    if pl in l:
        if (pts[5][0] < pts[4][0]):
            ch1 = 0

    # condition for [c0][aemnst]
    l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][
            0] and pts[0][0] > pts[20][0]) and pts[5][0] > pts[4][0]:
            ch1 = 2

    # condition for [c0][aemnst]
    l = [[6, 0], [6, 6], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[8], pts[16]) < 52:
            ch1 = 2


    # condition for [gh][bdfikruvw]
    l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]

    if pl in l:
        if pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[0][0] < pts[8][
            0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
            ch1 = 3



    # con for [gh][l]
    l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 3

    # con for [gh][pqz]
    l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[2][1] + 15 < pts[16][1]:
            ch1 = 3

    # con for [l][x]
    l = [[6, 4], [6, 1], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[4], pts[11]) > 55:
            ch1 = 4

    # con for [l][d]
    l = [[1, 4], [1, 6], [1, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (distance(pts[4], pts[11]) > 50) and (
                pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <
                pts[20][1]):
            ch1 = 4

    # con for [l][gh]
    l = [[3, 6], [3, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[4][0] < pts[0][0]):
            ch1 = 4

    # con for [l][c0]
    l = [[2, 2], [2, 5], [2, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[1][0] < pts[12][0]):
            ch1 = 4

    # con for [l][c0]
    l = [[2, 2], [2, 5], [2, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[1][0] < pts[12][0]):
            ch1 = 4

    # con for [gh][z]
    l = [[3, 6], [3, 5], [3, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][
            1]) and pts[4][1] > pts[10][1]:
            ch1 = 5

    # con for [gh][pq]
    l = [[3, 2], [3, 1], [3, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][1] + 17 > pts[8][1] and pts[4][1] + 17 > pts[12][1] and pts[4][1] + 17 > pts[16][1] and pts[4][
            1] + 17 > pts[20][1]:
            ch1 = 5

    # con for [l][pqz]
    l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 5

    # con for [pqz][aemnst]
    l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
            ch1 = 5

    # con for [pqz][yj]
    l = [[5, 7], [5, 2], [5, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[3][0] < pts[0][0]:
            ch1 = 7

    # con for [l][yj]
    l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] < pts[8][1]:
            ch1 = 7

    # con for [x][yj]
    l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[18][1] > pts[20][1]:
            ch1 = 7

    # condition for [x][aemnst]
    l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] > pts[16][0]:
            ch1 = 6


    # condition for [yj][x]
    l = [[7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[18][1] < pts[20][1] and pts[8][1] < pts[10][1]:
            ch1 = 6

    # condition for [c0][x]
    l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[8], pts[16]) > 50:
            ch1 = 6

    # con for [l][x]

    l = [[4, 6], [4, 2], [4, 1], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[4], pts[11]) < 60:
            ch1 = 6

    # con for [x][d]
    l = [[1, 4], [1, 6], [1, 0], [1, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 > 0:
            ch1 = 6

    # con for [b][pqz]
    l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
         [6, 3], [6, 4], [7, 5], [7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][
            1]):
            ch1 = 1

    # con for [f][pqz]
    l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
         [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and
                pts[18][1] > pts[20][1]):
            ch1 = 1

    l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and
                pts[18][1] > pts[20][1]):
            ch1 = 1

    # con for [d][pqz]
    l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
             pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[4][1] > pts[14][1]):
            ch1 = 1

    l = [[4, 1], [4, 2], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (distance(pts[4], pts[11]) < 50) and (
                pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <
                pts[20][1]):
            ch1 = 1

    l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
             pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[14][1] < pts[4][1]):
            ch1 = 1

    l = [[6, 6], [6, 4], [6, 1], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 < 0:
            ch1 = 1

    # con for [i][pqz]
    l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
             pts[18][1] > pts[20][1])):
            ch1 = 1

    # con for [yj][bfdi]
    l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[4][0] < pts[5][0] + 15) and (
        (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
         pts[18][1] > pts[20][1])):
            ch1 = 7

    # con for [uvr]
    l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and
             pts[18][1] < pts[20][1])) and pts[4][1] > pts[14][1]:
            ch1 = 1

    # con for [w]
    fg = 13
    l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if not (pts[0][0] + fg < pts[8][0] and pts[0][0] + fg < pts[12][0] and pts[0][0] + fg < pts[16][0] and
                pts[0][0] + fg < pts[20][0]) and not (
                pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][
            0]) and distance(pts[4], pts[11]) < 50:
            ch1 = 1

    # con for [w]

    l = [[5, 0], [5, 5], [0, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1]:
            ch1 = 1

    # -------------------------condn for 8 groups  ends

    # -------------------------condn for subgroups  starts
    #
    if ch1 == 0:
        ch1 = 'S'
        if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
            ch1 = 'A'
        if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][
            0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]:
            ch1 = 'T'
        if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
            ch1 = 'E'
        if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]:
            ch1 = 'M'
        if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
            ch1 = 'N'

    if ch1 == 2:
        if distance(pts[12], pts[4]) > 42:
            ch1 = 'C'
        else:
            ch1 = 'O'

    if ch1 == 3:
        if (distance(pts[8], pts[12])) > 72:
            ch1 = 'G'
        else:
            ch1 = 'H'

    if ch1 == 7:
        if distance(pts[8], pts[4]) > 42:
            ch1 = 'Y'
        else:
            ch1 = 'J'

    if ch1 == 4:
        ch1 = 'L'

    if ch1 == 6:
        ch1 = 'X'

    if ch1 == 5:
        if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
            if pts[8][1] < pts[5][1]:
                ch1 = 'Z'
            else:
                ch1 = 'Q'
        else:
            ch1 = 'P'

    if ch1 == 1:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][
            1]):
            ch1 = 'B'
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][
            1]):
            ch1 = 'D'
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][
            1]):
            ch1 = 'F'
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][
            1]):
            ch1 = 'I'
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][
            1]):
            ch1 = 'W'
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][
            1]) and pts[4][1] < pts[9][1]:
            ch1 = 'K'
        if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8) and (
                pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <
                pts[20][1]):
            ch1 = 'U'
        if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8) and (
                pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <
                pts[20][1]) and (pts[4][1] > pts[9][1]):
            ch1 = 'V'

        if (pts[8][0] > pts[12][0]) and (
                pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <
                pts[20][1]):
            ch1 = 'R'

    if ch1 == 1 or ch1 =='E' or ch1 =='S' or ch1 =='X' or ch1 =='Y' or ch1 =='B':
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1=" "

    if ch1 == 'E' or ch1=='Y' or ch1=='B':
        if (pts[4][0] < pts[5][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1="next"

    if ch1 == 'Next' or 'B' or 'C' or 'H' or 'F' or 'X':
        if (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and (pts[4][1] < pts[8][1] and pts[4][1] < pts[12][1] and pts[4][1] < pts[16][1] and pts[4][1] < pts[20][1]) and (pts[4][1] < pts[6][1] and pts[4][1] < pts[10][1] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]):
            ch1 = 'Backspace'
    
    return ch1


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "HandSpeak.ai API",
        "version": "1.0.0"
    }


@app.post("/api/recognize")
async def recognize_sign(file: UploadFile = File(...)):
    """
    Recognize sign language from uploaded image
    
    Args:
        file: Image file (JPEG, PNG) containing hand gesture
    
    Returns:
        JSON with recognized character, confidence, and suggestions
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # CRITICAL STEP 1: Resize to 640x480 to match final_pred.py webcam resolution
        # This ensures bounding boxes, landmarks, and rule thresholds are in SAME coordinate space
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        
        # CRITICAL STEP 2: Horizontal flip to match webcam orientation from final_pred.py
        cv2image = cv2.flip(frame, 1)
        
        # Detect hands
        hands = hd.findHands(cv2image, draw=False, flipType=True)
        
        if not hands[0]:
            return JSONResponse({
                "success": False,
                "message": "No hand detected",
                "character": None,
                "suggestions": []
            })
        
        hand = hands[0][0]
        x, y, w, h = hand['bbox']
        
        # Extract hand region
        image = cv2image[y - offset:y + h + offset, x - offset:x + w + offset]
        
        if image is None or image.size == 0:
            return JSONResponse({
                "success": False,
                "message": "Could not extract hand region",
                "character": None,
                "suggestions": []
            })
        
        # Detect hand landmarks in cropped region
        handz = hd2.findHands(image, draw=False, flipType=True)
        
        if not handz[0]:
            return JSONResponse({
                "success": False,
                "message": "Could not detect hand landmarks",
                "character": None,
                "suggestions": []
            })
        
        handmap = handz[0][0]
        pts = handmap['lmList']
        
        # Predict character
        character = predict_sign(image, pts, w, h)
        
        return JSONResponse({
            "success": True,
            "character": character,
            "confidence": 0.95,  # Add actual confidence from model
            "message": "Character recognized successfully"
        })
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict_sign_legacy(file: UploadFile = File(...)):
    """
    Legacy endpoint for backward compatibility
    Redirects to /api/recognize
    """
    return await recognize_sign(file)


@app.post("/api/suggest-words")
async def suggest_words(request: WordRequest):
    """
    Get word suggestions based on partial word
    
    Args:
        request: WordRequest with word field
    
    Returns:
        List of suggested complete words
    """
    try:
        word = request.word
        if not word or not word.strip():
            return {"suggestions": []}
        
        word = word.strip().lower()
        ddd.check(word)
        suggestions = ddd.suggest(word)
        
        return {
            "word": word,
            "suggestions": suggestions[:4] if suggestions else []
        }
        
    except Exception as e:
        return {
            "word": request.word if hasattr(request, 'word') else "",
            "suggestions": [],
            "error": str(e)
        }


@app.post("/api/text-to-speech")
async def text_to_speech(text: str):
    """
    Convert text to speech (returns audio URL or file)
    
    Args:
        text: Text to convert to speech
    
    Returns:
        Audio data or URL
    """
    try:
        # Implement TTS logic here
        # For mobile, you might want to return audio file or use device's TTS
        return {
            "success": True,
            "message": "Use device native TTS",
            "text": text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/recognize")
async def websocket_recognize(websocket: WebSocket):
    """
    WebSocket endpoint for real-time continuous sign language recognition
    Client sends base64 encoded frames continuously, server responds with predictions
    """
    await websocket.accept()
    print("[CONNECT] Mobile client connected via WebSocket")
    
    try:
        while True:
            # Receive base64 encoded frame from client
            data = await websocket.receive_text()
            
            try:
                # Decode base64 image
                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None or frame.size == 0:
                    await websocket.send_json({
                        "success": False,
                        "message": "Invalid frame",
                        "character": None
                    })
                    continue
                
                # CRITICAL STEP 1: Resize to 640x480 to match final_pred.py webcam resolution
                # This ensures bounding boxes, landmarks, and rule thresholds are in SAME coordinate space
                # Without this, Android frames at different resolutions break all distance-based rules
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
                
                # CRITICAL STEP 2: Horizontal flip to match webcam orientation from final_pred.py
                # Webcam shows mirror image; this makes Android frames behave identically
                cv2image = cv2.flip(frame, 1)
                
                # Detect hands
                hands = hd.findHands(cv2image, draw=False, flipType=True)
                
                if not hands[0]:
                    await websocket.send_json({
                        "success": True,
                        "message": "No hand detected",
                        "character": None,
                        "handDetected": False
                    })
                    continue
                
                hand = hands[0][0]
                x, y, w, h = hand['bbox']
                
                # Extract hand region - EXACT same as final_pred.py (NO bounds checking)
                image = cv2image[y - offset:y + h + offset, x - offset:x + w + offset]
                
                if image is None or image.size == 0:
                    await websocket.send_json({
                        "success": True,
                        "message": "Hand region extraction failed",
                        "character": None,
                        "handDetected": True
                    })
                    continue
                
                # Detect hand landmarks in cropped region
                handz = hd2.findHands(image, draw=False, flipType=True)
                
                if not handz[0]:
                    await websocket.send_json({
                        "success": True,
                        "message": "Could not detect landmarks",
                        "character": None,
                        "handDetected": True
                    })
                    continue
                
                handmap = handz[0][0]
                pts = handmap['lmList']
                
                # Predict character
                character = predict_sign(image, pts, w, h)
                
                # Send result back to client
                await websocket.send_json({
                    "success": True,
                    "character": character,
                    "handDetected": True,
                    "message": "Character recognized"
                })
                
            except Exception as e:
                print(f"Frame processing error: {e}")
                await websocket.send_json({
                    "success": False,
                    "message": f"Error: {str(e)}",
                    "character": None
                })
                
    except WebSocketDisconnect:
        print("[DISCONNECT] Mobile client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("[SERVER] Starting HandSpeak.ai Backend Server")
    print("=" * 60)
    print("[MOBILE] Mobile apps can connect to: http://YOUR_IP:8000")
    print("[API] API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,
        log_level="info"
    )
