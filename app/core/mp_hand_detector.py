import os
import urllib.request
from typing import List, Tuple

import cv2
import numpy as np

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe import Image, ImageFormat
except Exception:
    # Fallback import names in some builds
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe import Image, ImageFormat


MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../data/hand_landmarker.task")


class MPHandDetector:
    def __init__(self, maxHands: int = 1) -> None:
        self.maxHands = maxHands
        self.model_path = os.path.abspath(DEFAULT_MODEL_PATH)
        self._ensure_model()
        base_opts = BaseOptions(model_asset_path=self.model_path)
        opts = vision.HandLandmarkerOptions(base_options=base_opts, num_hands=self.maxHands)
        self.detector = vision.HandLandmarker.create_from_options(opts)

    def _ensure_model(self) -> None:
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        if not os.path.exists(self.model_path):
            try:
                urllib.request.urlretrieve(MODEL_URL, self.model_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download Mediapipe model: {e}")

    def findHands(self, frame: np.ndarray, draw: bool = False, flipType: bool = True) -> List[Tuple[dict, dict]]:
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        res = self.detector.detect(mp_img)
        hands_out: List[Tuple[dict, dict]] = []
        h, w, _ = frame.shape
        for i, landmarks in enumerate(res.hand_landmarks or []):
            # Landmarks are normalized floats [0,1]; convert to pixel coords
            pts = []
            xs, ys = [], []
            for lm in landmarks:
                px = int(lm.x * w)
                py = int(lm.y * h)
                pts.append([px, py])
                xs.append(px)
                ys.append(py)
            if not pts:
                continue
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            meta = {"bbox": bbox, "type": "Right" if flipType else "Unknown"}
            handmap = {"lmList": pts}
            hands_out.append((meta, handmap))
        return hands_out
