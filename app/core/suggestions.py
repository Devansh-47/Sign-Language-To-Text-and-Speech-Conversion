from __future__ import annotations

import math
from pathlib import Path
from typing import Counter as CounterType, Dict, List

from app.config import LEXICON_PATH, SUGGESTION_LIMIT


class Lexicon:
    def __init__(self, path: Path) -> None:
        self.words: Dict[str, float] = {}
        if path.exists():
            for idx, line in enumerate(path.read_text().splitlines()):
                word = line.strip().lower()
                if not word:
                    continue
                # simple descending weight
                self.words[word] = 1.0 / (1.0 + idx)
        else:
            seed = [
                "hello",
                "help",
                "hand",
                "speak",
                "thanks",
                "you",
                "please",
                "yes",
                "no",
                "good",
                "morning",
                "afternoon",
                "night",
            ]
            for idx, word in enumerate(seed):
                self.words[word] = 1.0 / (1.0 + idx)

    def prefix(self, prefix: str) -> Dict[str, float]:
        prefix_lower = prefix.lower()
        return {w: s for w, s in self.words.items() if w.startswith(prefix_lower)}


lexicon = Lexicon(LEXICON_PATH)


def _personal_weight(word: str, personal_counts: CounterType[str]) -> float:
    if not personal_counts:
        return 0.0
    max_seen = max(personal_counts.values()) or 1
    return personal_counts.get(word, 0) / max_seen


def suggest(prefix: str, gesture_confidence: float, personal_counts: CounterType[str]) -> List[Dict[str, object]]:
    if not prefix:
        return []
    candidates = lexicon.prefix(prefix)
    scored = []
    for word, lm_score in candidates.items():
        p_gesture = gesture_confidence
        p_language = lm_score
        p_personal = _personal_weight(word, personal_counts)
        score = 0.4 * p_gesture + 0.4 * p_language + 0.2 * p_personal
        scored.append({"word": word, "score": score, "source": "blend"})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:SUGGESTION_LIMIT]
