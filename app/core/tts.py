from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Optional

try:
    import pyttsx3
except ImportError:  # pragma: no cover
    pyttsx3 = None

from app.config import DEFAULT_TTS_FORMAT


class TTSService:
    def __init__(self) -> None:
        self.engine = pyttsx3.init() if pyttsx3 else None
        if self.engine:
            self.engine.setProperty("rate", 150)

    def synthesize(self, text: str, fmt: str = DEFAULT_TTS_FORMAT) -> Optional[str]:
        if not self.engine:
            return None
        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            self.engine.save_to_file(text, str(path))
            self.engine.runAndWait()
            data = path.read_bytes()
            return base64.b64encode(data).decode()
        finally:
            if path.exists():
                path.unlink()


tts_service = TTSService()
