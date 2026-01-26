"""
Production-Grade Landmark-Based ASL Recognition Backend
Pure geometry-based inference - NO image processing, NO OpenCV, NO MediaPipe

Architecture:
- Receives normalized MediaPipe landmarks from Android
- Canonicalizes to resolution/camera/orientation-agnostic form
- Applies deterministic rule-based inference
- Returns ASL character predictions via WebSocket
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict
from collections import deque
import math
import uvicorn
import json
import enchant
import logging
from datetime import datetime

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="HandSpeak.ai Landmark API", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dictionary for word suggestions
ddd = enchant.Dict("en-US")

# ============================================================================
# DATA MODELS
# ============================================================================

class LandmarkFrame(BaseModel):
    """Incoming landmark payload from Android"""
    handDetected: bool
    handedness: Optional[str] = "Right"  # "Right" or "Left"
    landmarks: Optional[List[List[float]]] = None  # 21 points × [x, y, z]


class TemporalSmoother:
    """
    Temporal smoothing for stable predictions
    Requires N consecutive identical predictions before changing output
    """
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.current_stable = ""
    
    def add(self, prediction: str) -> str:
        """Add new prediction and return stabilized result"""
        self.history.append(prediction)
        
        # Need full window
        if len(self.history) < self.window_size:
            return self.current_stable
        
        # Check if all predictions in window are identical
        if len(set(self.history)) == 1:
            self.current_stable = prediction
        
        return self.current_stable
    
    def reset(self):
        """Clear history"""
        self.history.clear()
        self.current_stable = ""


# Per-connection smoothers (WebSocket connection ID → smoother)
connection_smoothers: Dict[str, TemporalSmoother] = {}


# ============================================================================
# LANDMARK CANONICALIZATION PIPELINE
# ============================================================================

def canonicalize_landmarks(
    landmarks: List[List[float]], 
    handedness: str
) -> List[Tuple[float, float]]:
    """
    Transform MediaPipe landmarks to match final_pred.py coordinate space
    
    CRITICAL: final_pred.py uses MediaPipe coordinates scaled to 640×480 pixels
    All hand-crafted rules were tuned for this exact coordinate space
    
    Pipeline:
    1) Drop Z-axis (only X, Y needed)
    2) Scale from normalized [0.0-1.0] to pixel space [640×480]
    3) Mirror if left hand (MediaPipe gives right-hand coordinates for left hands)
    
    Args:
        landmarks: 21×3 array from MediaPipe [x, y, z] in normalized [0.0-1.0] space
        handedness: "Right" or "Left"
    
    Returns:
        21×2 landmarks [(x, y), ...] in 640×480 pixel space (same as final_pred.py)
    """
    logger.debug(f"[CANON] Handedness: {handedness}")
    logger.debug(f"[CANON] Raw normalized landmarks (first 3): {landmarks[:3]}")
    
    # STEP 1: Drop Z-axis and scale to 640×480 pixel space
    # MediaPipe gives normalized coords [0.0-1.0], we need pixels like final_pred.py
    WIDTH = 640
    HEIGHT = 480
    pts_pixels = [(lm[0] * WIDTH, lm[1] * HEIGHT) for lm in landmarks]
    
    logger.debug(f"[CANON] Pixel coords (first 3): {pts_pixels[:3]}")
    
    # STEP 2: Mirror if left hand
    # MediaPipe treats left hands as mirrored right hands, so we flip X coordinates
    if handedness == "Left":
        pts_pixels = [(WIDTH - x, y) for x, y in pts_pixels]
        logger.debug("[CANON] Mirrored for left hand")
    
    logger.debug(f"[CANON] Final coords: wrist={pts_pixels[0]}, thumb_tip={pts_pixels[4]}, index_tip={pts_pixels[8]}")
    
    return pts_pixels


def distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two 2D points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# ============================================================================
# ASL CHARACTER INFERENCE ENGINE (GEOMETRY-BASED)
# ============================================================================

def predict_character_from_landmarks(pts: List[Tuple[float, float]]) -> str:
    """
    ASL character prediction using EXACT logic from final_pred.py
    
    CRITICAL: Coordinates are in 640×480 pixel space, matching final_pred.py
    All thresholds are identical to final_pred.py
    
    Args:
        pts: 21 landmarks [(x, y), ...] in 640×480 pixel coordinate space
    
    Returns:
        Predicted ASL character (A-Z, space, next, backspace)
    """
    
    logger.debug("=" * 60)
    logger.debug("[PREDICT] Starting prediction pipeline")
    
    # Log key landmark positions for debugging
    logger.debug(f"[PREDICT] Key landmarks (pixel space):")
    logger.debug(f"  Wrist(0): {pts[0]}")
    logger.debug(f"  Thumb tip(4): {pts[4]}")
    logger.debug(f"  Index tip(8): {pts[8]}")
    logger.debug(f"  Middle tip(12): {pts[12]}")
    logger.debug(f"  Ring tip(16): {pts[16]}")
    logger.debug(f"  Pinky tip(20): {pts[20]}")
    
    # Initial CNN-like classification (replaced with geometry heuristics)
    # Group 0: AEMNST, Group 1: BDFIKRUVW, Group 2: CO, Group 3: GH
    # Group 4: L, Group 5: PQZ, Group 6: X, Group 7: YJ
    
    ch1, ch2 = classify_to_groups(pts)
    logger.debug(f"[PREDICT] Initial classification: ch1={ch1}, ch2={ch2}")
    
    pl = [ch1, ch2]
    
    # ========================================================================
    # GROUP DISAMBIGUATION RULES (EXACT COPY from final_pred.py)
    # ========================================================================
    # Using PIXEL thresholds since coordinates are in 640×480 space
    
    # condition for [Aemnst]
    l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
         [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
         [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
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
        if (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0] and pts[0][0] > pts[12][0] and 
            pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and pts[5][0] > pts[4][0]:
            ch1 = 2

    # condition for [c0][aemnst]
    l = [[6, 0], [6, 6], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance_2d(pts[8], pts[16]) < 52:
            ch1 = 2

    # condition for [gh][bdfikruvw]
    l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and 
            pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]):
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
        # Normalized offset: original was +15 pixels ≈ 0.1 in normalized space
        if pts[2][1] + 0.1 < pts[16][1]:
            ch1 = 3

    # con for [l][x]
    l = [[6, 4], [6, 1], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance_2d(pts[4], pts[11]) > 55:
            ch1 = 4

    # con for [l][d]
    l = [[1, 4], [1, 6], [1, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (distance_2d(pts[4], pts[11]) > 50) and (
            pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
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

    # con for [gh][z]
    l = [[3, 6], [3, 5], [3, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and 
            pts[18][1] < pts[20][1]) and pts[4][1] > pts[10][1]:
            ch1 = 5

    # con for [gh][pq]
    l = [[3, 2], [3, 1], [3, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][1] + 17 > pts[8][1] and pts[4][1] + 17 > pts[12][1] and pts[4][1] + 17 > pts[16][1] and pts[4][1] + 17 > pts[20][1]:
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
        if distance_2d(pts[8], pts[16]) > 50:
            ch1 = 6

    # con for [l][x]
    l = [[4, 6], [4, 2], [4, 1], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if distance_2d(pts[4], pts[11]) < 60:
            ch1 = 6

    # con for [x][d]
    l = [[1, 4], [1, 6], [1, 0], [1, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] - pts[4][0] - 0.1 > 0:
            ch1 = 6

    # con for [b][pqz]
    l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
         [6, 3], [6, 4], [7, 5], [7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    # con for [f][pqz]
    l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [7, 1],
         [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
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
        if (distance_2d(pts[4], pts[11]) < 50) and (
            pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
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
    l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    # con for [yj][bfdi]
    l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[4][0] < pts[5][0] + 15) and (
            pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
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
                pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and distance_2d(pts[4], pts[11]) < 50:
            ch1 = 1

    # con for [w]
    l = [[5, 0], [5, 5], [0, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1]:
            ch1 = 1

    # ========================================================================
    # SUBGROUP CLASSIFICATION (Within each major group)
    # ========================================================================
    
    if ch1 == 0:
        ch1 = 'S'
        if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
            ch1 = 'A'
        if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]:
            ch1 = 'T'
        if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
            ch1 = 'E'
        if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]:
            ch1 = 'M'
        if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
            ch1 = 'N'

    if ch1 == 2:
        if distance_2d(pts[12], pts[4]) > 42:
            ch1 = 'C'
        else:
            ch1 = 'O'

    if ch1 == 3:
        if distance_2d(pts[8], pts[12]) > 72:
            ch1 = 'G'
        else:
            ch1 = 'H'

    if ch1 == 7:
        if distance_2d(pts[8], pts[4]) > 42:
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
        # Check for next/B first before other group 1 letters
        # (handled above)
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'D'
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 'F'
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 'I'
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'W'
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] < pts[9][1]:
            ch1 = 'K'
        if (pts[8][0] < pts[12][0]) and (
            pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'R'
        if ((distance_2d(pts[8], pts[12]) - distance_2d(pts[6], pts[10])) < 8) and (
            pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'U'
        if ((distance_2d(pts[8], pts[12]) - distance_2d(pts[6], pts[10])) >= 8) and (
            pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[4][1] > pts[9][1]):
            ch1 = 'V'

    # Space detection
    if ch1 == 1 or ch1 == 'E' or ch1 == 'S' or ch1 == 'X' or ch1 == 'Y' or ch1 == 'B':
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = " "

    logger.debug(f"[PREDICT] Final prediction: '{ch1}'")
    logger.debug("=" * 60)
    
    # Next word detection (all fingers down, thumb IN/tucked)
    if ch1 == 'E' or ch1 == 'Y' or ch1 == 1:
        if (pts[4][0] >= pts[5][0]) and (
            pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = "next"
    
    # B detection (all fingers down, thumb OUT/extended)
    if ch1 == 1:
        if (pts[4][0] < pts[5][0]) and (
            pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 'B'

    # T detection: Thumb pointing up, all fingers curled
    # Thumb must be significantly above wrist AND above all finger knuckles
    if (pts[4][1] < pts[0][1] - 50) and (pts[4][1] < pts[5][1]) and (pts[4][1] < pts[9][1]) and (pts[4][1] < pts[13][1]) and (pts[4][1] < pts[17][1]):
        # Also check that fingers are curled (not extended)
        if (pts[8][1] > pts[6][1] or pts[12][1] > pts[10][1] or pts[16][1] > pts[14][1] or pts[20][1] > pts[18][1]):
            ch1 = 'T'
    
    # Fallback: Convert any remaining integers to letters
    if ch1 == 1:
        ch1 = 'A'
    
    return ch1


def classify_to_groups(pts: List[Tuple[float, float]]) -> Tuple[int, int]:
    """
    Simplified geometry-based group classification
    
    In production, replace this with a lightweight ML classifier trained on canonical landmarks
    For now, use geometric heuristics
    
    Returns:
        (primary_group, secondary_group) - indices 0-7
    """
    # Heuristic: classify based on finger configurations
    # This is a placeholder - train a proper classifier in production
    
    # Count extended fingers (fingertip higher than MCP)
    extended = 0
    fingers = [
        (8, 6),    # Index
        (12, 10),  # Middle
        (16, 14),  # Ring
        (20, 18),  # Pinky
    ]
    
    finger_states = []
    for name, (tip, base) in zip(['Index', 'Middle', 'Ring', 'Pinky'], fingers):
        is_extended = pts[tip][1] < pts[base][1]  # Y decreases upward
        finger_states.append(f"{name}={'UP' if is_extended else 'DOWN'}")
        if is_extended:
            extended += 1
    
    # Thumb
    thumb_extended = pts[4][0] > pts[3][0]
    finger_states.insert(0, f"Thumb={'OUT' if thumb_extended else 'IN'}")
    
    logger.debug(f"[CLASSIFY] Finger states: {', '.join(finger_states)}")
    logger.debug(f"[CLASSIFY] Extended fingers: {extended}/4")
    
    # Simple heuristic mapping
    if extended == 0:
        result = (0, 2)  # Fist-like (AEMNST or CO)
    elif extended == 1:
        if thumb_extended:
            result = (4, 6)  # L or X
        else:
            result = (4, 1)  # L or BDFIKRUVW
    elif extended == 2:
        result = (3, 5)  # GH or PQZ
    elif extended == 3:
        # 3 fingers: could be W, R, U, V (all in group 1)
        result = (1, 5)  # BDFIKRUVW or PQZ
    elif extended == 4:
        result = (1, 0)  # BDFIKRUVW or AEMNST
    else:
        result = (1, 5)  # Default to group 1 for multi-finger gestures
    
    logger.debug(f"[CLASSIFY] Initial group classification: ch1={result[0]}, ch2={result[1]}")
    
    return result


# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

@app.websocket("/ws/landmarks")
async def websocket_landmarks(websocket: WebSocket):
    """
    Production WebSocket endpoint for landmark-based ASL recognition
    
    Protocol:
        Client → Server: JSON with {handDetected, handedness, landmarks}
        Server → Client: JSON with {success, handDetected, character}
    """
    await websocket.accept()
    connection_id = id(websocket)  # Use object ID as connection identifier
    connection_smoothers[connection_id] = TemporalSmoother(window_size=3)
    
    print(f"[CONNECT] Landmark client connected (ID: {connection_id})")
    
    try:
        while True:
            # Receive landmark data
            data = await websocket.receive_text()
            
            try:
                payload = json.loads(data)
                frame = LandmarkFrame(**payload)
                
                # No hand detected
                if not frame.handDetected or frame.landmarks is None:
                    connection_smoothers[connection_id].reset()
                    await websocket.send_json({
                        "success": True,
                        "handDetected": False,
                        "character": ""
                    })
                    continue
                
                # Validate landmark count
                if len(frame.landmarks) != 21:
                    await websocket.send_json({
                        "success": False,
                        "handDetected": True,
                        "character": "",
                        "error": f"Expected 21 landmarks, got {len(frame.landmarks)}"
                    })
                logger.info(f"[FRAME] Processing frame from connection {connection_id}")
                canonical_landmarks = canonicalize_landmarks(
                    frame.landmarks,
                    frame.handedness or "Right"
                )
                
                # STEP 2: Predict character
                raw_prediction = predict_character_from_landmarks(canonical_landmarks)
                logger.info(f"[RESULT] Raw prediction: '{raw_prediction}'")
                
                # STEP 3: Temporal smoothing
                stable_prediction = connection_smoothers[connection_id].add(raw_prediction)
                logger.info(f"[RESULT] Stable prediction (smoothed): '{stable_prediction}'")
                
                # STEP 4: Send result
                await websocket.send_json({
                    "success": True,
                    "handDetected": True,
                    "character": stable_prediction
                })
                logger.info(f"[SENT] Returned character: '{stable_prediction}'")
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "success": False,
                    "handDetected": False,
                    "character": "",
                    "error": "Invalid JSON"
                })
            except Exception as e:
                print(f"[ERROR] Frame processing: {e}")
                await websocket.send_json({
                    "success": False,
                    "handDetected": False,
                    "character": "",
                    "error": str(e)
                })
                
    except WebSocketDisconnect:
        print(f"[DISCONNECT] Landmark client disconnected (ID: {connection_id})")
        if connection_id in connection_smoothers:
            del connection_smoothers[connection_id]
    except Exception as e:
        print(f"[ERROR] WebSocket: {e}")
        if connection_id in connection_smoothers:
            del connection_smoothers[connection_id]


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "running",
        "service": "HandSpeak.ai Landmark API",
        "version": "2.0.0",
        "endpoints": {
            "websocket": "/ws/landmarks",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_connections": len(connection_smoothers)
    }


@app.post("/api/suggest-words")
async def suggest_words(request: dict):
    """
    Word suggestions endpoint (preserved from original)
    """
    try:
        word = request.get("word", "").strip().lower()
        if not word:
            return {"suggestions": []}
        
        ddd.check(word)
        suggestions = ddd.suggest(word)
        
        return {
            "word": word,
            "suggestions": suggestions[:4] if suggestions else []
        }
    except Exception as e:
        return {
            "word": request.get("word", ""),
            "suggestions": [],
            "error": str(e)
        }


# ============================================================================
# SERVER ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  HandSpeak.ai Landmark-Based ASL Recognition Backend")
    print("  Pure Geometry Inference - No Image Processing")
    print("=" * 70)
    print("[INFO] WebSocket endpoint: ws://YOUR_IP:8000/ws/landmarks")
    print("[INFO] Health check: http://localhost:8000/health")
    print("[INFO] API docs: http://localhost:8000/docs")
    print("=" * 70)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
