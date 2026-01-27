from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque, Counter

from app.config import FRAME_BUFFER_SIZE, SMOOTHING_LAMBDA, STABLE_MS, COMMIT_CONFIDENCE, VELOCITY_THRESHOLD


@dataclass
class SessionData:
    session_id: str
    partial_word: str = ""
    committed_letters: List[str] = field(default_factory=list)
    words: List[str] = field(default_factory=list)
    sentence: str = ""
    commit_state: str = "IN_TRANSITION"
    last_updated_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    prob_buffer: Deque[List[float]] = field(default_factory=lambda: deque(maxlen=FRAME_BUFFER_SIZE))
    smoothed: Optional[List[float]] = None
    stable_since: Optional[int] = None
    last_letter: Optional[str] = None
    last_velocity: float = 0.0
    personal_counts: Counter = field(default_factory=Counter)
    last_pts: Optional[List[List[int]]] = None

    def update_timestamp(self, ts: Optional[int]) -> None:
        self.last_updated_ms = ts or int(time.time() * 1000)

    def push_probs(self, probs: List[float]) -> List[float]:
        self.prob_buffer.append(probs)
        if self.smoothed is None:
            self.smoothed = list(probs)
        else:
            self.smoothed = [
                SMOOTHING_LAMBDA * p + (1 - SMOOTHING_LAMBDA) * s
                for p, s in zip(probs, self.smoothed)
            ]
        return self.smoothed

    def is_stable(self, letter: str, confidence: float) -> bool:
        now = int(time.time() * 1000)
        if letter != self.last_letter:
            self.stable_since = now
            self.last_letter = letter
            return False
        if self.stable_since is None:
            self.stable_since = now
            return False
        stable_ms = now - self.stable_since
        return stable_ms >= STABLE_MS and confidence >= COMMIT_CONFIDENCE and self.last_velocity <= VELOCITY_THRESHOLD

    def commit_letter(self, letter: str) -> None:
        if letter == " ":
            if self.partial_word:
                self.words.append(self.partial_word)
                self.sentence = " ".join(self.words)
                self.partial_word = ""
        else:
            self.partial_word += letter
            self.committed_letters.append(letter)
        self.commit_state = "COMMITTED"

    def commit_word(self) -> None:
        if self.partial_word:
            self.words.append(self.partial_word)
            self.partial_word = ""
            self.sentence = " ".join(self.words)
            self.commit_state = "COMMITTED"


class SessionManager:
    def __init__(self) -> None:
        self.sessions: Dict[str, SessionData] = {}

    def get(self, session_id: str) -> SessionData:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionData(session_id=session_id)
        return self.sessions[session_id]

    def as_dict(self, session: SessionData) -> Dict[str, object]:
        return {
            "session_id": session.session_id,
            "partial_word": session.partial_word,
            "committed_letters": session.committed_letters,
            "words": session.words,
            "sentence": session.sentence,
            "commit_state": session.commit_state,
            "last_updated_ms": session.last_updated_ms,
        }


session_manager = SessionManager()
