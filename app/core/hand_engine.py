import base64
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from app.core.mp_hand_detector import MPHandDetector
from keras.models import load_model

from app.config import MODEL_PATH, FRAME_BUFFER_SIZE, SMOOTHING_LAMBDA
from app.core.session_manager import SessionData


def _distance(x: List[int], y: List[int]) -> float:
    return float(np.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2)))


def _velocity(prev: Optional[List[List[int]]], curr: List[List[int]]) -> float:
    if prev is None or len(prev) != len(curr):
        return 0.0
    deltas = [_distance(p, c) for p, c in zip(prev, curr)]
    mean_delta = float(np.mean(deltas))
    bbox_diag = float(np.linalg.norm(np.ptp(np.array(curr), axis=0))) + 1e-3
    return mean_delta / bbox_diag


class LegacyPostProcessor:
    """Reuses the legacy rule set to map 8-group outputs to 26 letters."""

    def __init__(self) -> None:
        pass

    def map_groups_to_letter(self, prob: List[float], pts: List[List[int]]) -> Tuple[str, float, List[Tuple[str, float]]]:
        prob_arr = np.array(prob, dtype="float32")
        ch1 = int(np.argmax(prob_arr))
        prob_arr[ch1] = 0
        ch2 = int(np.argmax(prob_arr))
        prob_arr[ch2] = 0
        ch3 = int(np.argmax(prob_arr))

        pl = [ch1, ch2]

        def cond_list(vals: List[List[int]]) -> bool:
            return pl in vals

        # Large block mirrors the GUI script behaviour.
        if cond_list([[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
                      [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
                      [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]):
            if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                ch1 = 0

        if cond_list([[2, 2], [2, 1]]):
            if pts[5][0] < pts[4][0]:
                ch1 = 0

        pl = [ch1, ch2]
        if cond_list([[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]):
            if (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and pts[5][0] > pts[4][0]:
                ch1 = 2

        pl = [ch1, ch2]
        if cond_list([[6, 0], [6, 6], [6, 2]]):
            if _distance(pts[8], pts[16]) < 52:
                ch1 = 2

        pl = [ch1, ch2]
        if cond_list([[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]):
            if pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
                ch1 = 3

        pl = [ch1, ch2]
        if cond_list([[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]):
            if pts[4][0] > pts[0][0]:
                ch1 = 3

        pl = [ch1, ch2]
        if cond_list([[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]):
            if pts[2][1] + 15 < pts[16][1]:
                ch1 = 3

        pl = [ch1, ch2]
        if cond_list([[6, 4], [6, 1], [6, 2]]):
            if _distance(pts[4], pts[11]) > 55:
                ch1 = 4

        pl = [ch1, ch2]
        if cond_list([[1, 4], [1, 6], [1, 1]]):
            if (_distance(pts[4], pts[11]) > 50) and (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                ch1 = 4

        pl = [ch1, ch2]
        if cond_list([[3, 6], [3, 4]]):
            if pts[4][0] < pts[0][0]:
                ch1 = 4

        pl = [ch1, ch2]
        if cond_list([[2, 2], [2, 5], [2, 4]]):
            if pts[1][0] < pts[12][0]:
                ch1 = 4

        pl = [ch1, ch2]
        if cond_list([[2, 2], [2, 5], [2, 4]]):
            if pts[1][0] < pts[12][0]:
                ch1 = 4

        pl = [ch1, ch2]
        if cond_list([[3, 6], [3, 5], [3, 4]]):
            if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] > pts[10][1]:
                ch1 = 5

        pl = [ch1, ch2]
        if cond_list([[3, 2], [3, 1], [3, 6]]):
            if pts[4][1] + 17 > pts[8][1] and pts[4][1] + 17 > pts[12][1] and pts[4][1] + 17 > pts[16][1] and pts[4][1] + 17 > pts[20][1]:
                ch1 = 5

        pl = [ch1, ch2]
        if cond_list([[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]):
            if pts[4][0] > pts[0][0]:
                ch1 = 5

        pl = [ch1, ch2]
        if cond_list([[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]):
            if pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
                ch1 = 5

        pl = [ch1, ch2]
        if cond_list([[5, 7], [5, 2], [5, 6]]):
            if pts[3][0] < pts[0][0]:
                ch1 = 7

        pl = [ch1, ch2]
        if cond_list([[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]):
            if pts[6][1] < pts[8][1]:
                ch1 = 7

        pl = [ch1, ch2]
        if cond_list([[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]):
            if pts[18][1] > pts[20][1]:
                ch1 = 7

        pl = [ch1, ch2]
        if cond_list([[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]):
            if pts[5][0] > pts[16][0]:
                ch1 = 6

        pl = [ch1, ch2]
        if cond_list([[7, 2]]):
            if pts[18][1] < pts[20][1] and pts[8][1] < pts[10][1]:
                ch1 = 6

        pl = [ch1, ch2]
        if cond_list([[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]):
            if _distance(pts[8], pts[16]) > 50:
                ch1 = 6

        pl = [ch1, ch2]
        if cond_list([[4, 6], [4, 2], [4, 1], [4, 4]]):
            if _distance(pts[4], pts[11]) < 60:
                ch1 = 6

        pl = [ch1, ch2]
        if cond_list([[1, 4], [1, 6], [1, 0], [1, 2]]):
            if pts[5][0] - pts[4][0] - 15 > 0:
                ch1 = 6

        pl = [ch1, ch2]
        if cond_list([[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0], [6, 3], [6, 4], [7, 5], [7, 2]]):
            if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
                ch1 = 1

        pl = [ch1, ch2]
        if cond_list([[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]):
            if pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
                ch1 = 1

        pl = [ch1, ch2]
        if cond_list([[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]):
            if pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
                ch1 = 1

        pl = [ch1, ch2]
        if cond_list([[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]):
            if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[4][1] > pts[14][1]):
                ch1 = 1

        pl = [ch1, ch2]
        if cond_list([[4, 1], [4, 2], [4, 4]]):
            if (_distance(pts[4], pts[11]) < 50) and (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                ch1 = 1

        pl = [ch1, ch2]
        if cond_list([[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]):
            if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[14][1] < pts[4][1]):
                ch1 = 1

        pl = [ch1, ch2]
        if cond_list([[6, 6], [6, 4], [6, 1], [6, 2]]):
            if pts[5][0] - pts[4][0] - 15 < 0:
                ch1 = 1

        pl = [ch1, ch2]
        if cond_list([[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]):
            if pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]:
                ch1 = 1

        pl = [ch1, ch2]
        if cond_list([[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]):
            if (pts[4][0] < pts[5][0] + 15) and (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
                ch1 = 7

        pl = [ch1, ch2]
        if cond_list([[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]):
            if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] > pts[14][1]:
                ch1 = 1

        pl = [ch1, ch2]
        fg = 13
        if cond_list([[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]):
            if not (pts[0][0] + fg < pts[8][0] and pts[0][0] + fg < pts[12][0] and pts[0][0] + fg < pts[16][0] and pts[0][0] + fg < pts[20][0]) and not (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and _distance(pts[4], pts[11]) < 50:
                ch1 = 1

        pl = [ch1, ch2]
        if cond_list([[5, 0], [5, 5], [0, 1]]):
            if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1]:
                ch1 = 1

        # subgroup mapping
        if ch1 == 0:
            ch = "S"
            if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
                ch = "A"
            if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]:
                ch = "T"
            if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
                ch = "E"
            if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]:
                ch = "M"
            if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
                ch = "N"
            ch1 = ch

        if ch1 == 2:
            if _distance(pts[12], pts[4]) > 42:
                ch1 = "C"
            else:
                ch1 = "O"

        if ch1 == 3:
            if _distance(pts[8], pts[12]) > 72:
                ch1 = "G"
            else:
                ch1 = "H"

        if ch1 == 7:
            if _distance(pts[8], pts[4]) > 42:
                ch1 = "Y"
            else:
                ch1 = "J"

        if ch1 == 4:
            ch1 = "L"

        if ch1 == 6:
            ch1 = "X"

        if ch1 == 5:
            if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
                if pts[8][1] < pts[5][1]:
                    ch1 = "Z"
                else:
                    ch1 = "Q"
            else:
                ch1 = "P"

        if ch1 == 1:
            if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
                ch1 = "B"
            if pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]:
                ch1 = "D"
            if pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]:
                ch1 = "F"
            if pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]:
                ch1 = "I"
            if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]:
                ch1 = "W"
            if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] < pts[9][1]:
                ch1 = "K"
            if (_distance(pts[8], pts[12]) - _distance(pts[6], pts[10]) < 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                ch1 = "U"
            if (_distance(pts[8], pts[12]) - _distance(pts[6], pts[10]) >= 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[4][1] > pts[9][1]):
                ch1 = "V"
            if pts[8][0] > pts[12][0] and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                ch1 = "R"

        # Debounce blanks
        if isinstance(ch1, str) and ch1 in ["E", "S", "X", "Y", "B"]:
            if pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]:
                ch1 = " "

        conf = float(max(prob)) if len(prob) else 0.0
        ranked = sorted([(idx, p) for idx, p in enumerate(prob)], key=lambda x: x[1], reverse=True)
        letters = [self._group_index_to_label(i, pts) for i, _ in ranked[:5]]
        topk = list(zip(letters, [p for _, p in ranked[:5]]))
        return str(ch1), conf, topk

    @staticmethod
    def _group_index_to_label(idx: int, pts: List[List[int]]) -> str:
        # Lightweight mapping for hypotheses; uses coarse mapping.
        mapping = {
            0: "A",
            1: "B",
            2: "C",
            3: "G",
            4: "L",
            5: "P",
            6: "X",
            7: "Y",
        }
        # fallback to subgroup logic for top hypothesis
        return mapping.get(idx, "")


class HandInferenceEngine:
    def __init__(self) -> None:
        self.detector = MPHandDetector(maxHands=1)
        self.model = load_model(str(MODEL_PATH))
        self.post = LegacyPostProcessor()

    def decode_frame(self, frame_b64: str) -> np.ndarray:
        raw = base64.b64decode(frame_b64)
        arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Unable to decode frame")
        return frame

    def _extract_hand(self, frame: np.ndarray) -> Tuple[np.ndarray, List[List[int]], float]:
        offset = 29
        hands = self.detector.findHands(frame, draw=False, flipType=True)
        if not hands or not hands[0]:
            raise ValueError("No hand detected")
        hand = hands[0]
        meta = hand[0]
        handmap = hand[1]
        x, y, w, h = meta["bbox"]
        image = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        white = np.ones((400, 400, 3), dtype=np.uint8) * 255
        # Convert absolute landmark coords to ROI-relative
        pts = [[int(px - x + offset), int(py - y + offset)] for px, py in handmap["lmList"]]

        os_x = ((400 - w) // 2) - 15
        os_y = ((400 - h) // 2) - 15
        fingers = [(0, 4), (5, 8), (9, 12), (13, 16), (17, 20)]
        for start, end in fingers:
            for t in range(start, end, 1):
                cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
        cv2.line(white, (pts[5][0] + os_x, pts[5][1] + os_y), (pts[9][0] + os_x, pts[9][1] + os_y), (0, 255, 0), 3)
        cv2.line(white, (pts[9][0] + os_x, pts[9][1] + os_y), (pts[13][0] + os_x, pts[13][1] + os_y), (0, 255, 0), 3)
        cv2.line(white, (pts[13][0] + os_x, pts[13][1] + os_y), (pts[17][0] + os_x, pts[17][1] + os_y), (0, 255, 0), 3)
        cv2.line(white, (pts[0][0] + os_x, pts[0][1] + os_y), (pts[5][0] + os_x, pts[5][1] + os_y), (0, 255, 0), 3)
        cv2.line(white, (pts[0][0] + os_x, pts[0][1] + os_y), (pts[17][0] + os_x, pts[17][1] + os_y), (0, 255, 0), 3)

        for i in range(21):
            cv2.circle(white, (pts[i][0] + os_x, pts[i][1] + os_y), 2, (0, 0, 255), 1)

        return white, pts, 0.0

    def predict(self, frame_b64: str, session: SessionData) -> Dict[str, object]:
        frame = self.decode_frame(frame_b64)
        skeleton, pts, _ = self._extract_hand(frame)
        velocity = _velocity(session.last_pts, pts)
        session.last_velocity = velocity
        session.last_pts = pts

        sample = skeleton.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(sample)[0], dtype="float32").tolist()
        letter, raw_conf, topk = self.post.map_groups_to_letter(prob, pts)

        smoothed = session.push_probs(prob)
        ranked = sorted([(i, p) for i, p in enumerate(smoothed)], key=lambda x: x[1], reverse=True)
        smooth_top = ranked[0][1] if ranked else 0.0

        commit_ready = session.is_stable(letter, smooth_top)
        if commit_ready and letter.strip():
            session.commit_letter(letter)
        else:
            session.commit_state = "HOLDING"
        session.update_timestamp(None)

        top2_letters = []
        for idx, p in ranked[:2]:
            top2_letters.append({"letter": self.post._group_index_to_label(idx, pts), "confidence": float(p)})
        top5_letters = []
        for idx, p in ranked[:5]:
            top5_letters.append({"letter": self.post._group_index_to_label(idx, pts), "confidence": float(p)})

        return {
            "letter": letter,
            "confidence": smooth_top,
            "top2": top2_letters,
            "top5": top5_letters,
            "raw_confidence": raw_conf,
        }


def create_engine() -> HandInferenceEngine:
    return HandInferenceEngine()
