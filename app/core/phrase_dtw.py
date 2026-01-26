from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import PHRASE_SIGNATURE_PATH


def _dtw(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(a[i - 1] - b[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            )
    return float(dtw_matrix[n, m]) / (n + m)


class PhraseMatcher:
    def __init__(self, path: Path = PHRASE_SIGNATURE_PATH, threshold: float = 0.3, min_conf: float = 0.75) -> None:
        self.path = path
        self.threshold = threshold
        self.min_conf = min_conf
        self.signatures: Dict[str, List[List[float]]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        for item in data:
            self.signatures[item["phrase"]] = item["signature"]

    def match(self, seq: List[List[float]], confidence: float) -> Optional[str]:
        if not self.signatures or confidence < self.min_conf:
            return None
        current = np.array(seq)
        for phrase, signature in self.signatures.items():
            target = np.array(signature)
            score = _dtw(current, target)
            if score < self.threshold:
                return phrase
        return None


matcher = PhraseMatcher()
