"""
Central configuration for the FastAPI backend.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "cnn8grps_rad1_model.h5"
PHRASE_SIGNATURE_PATH = BASE_DIR / "app" / "data" / "phrase_signatures.json"
LEXICON_PATH = BASE_DIR / "app" / "data" / "lexicon_small.txt"

# Temporal smoothing
SMOOTHING_LAMBDA = 0.15
FRAME_BUFFER_SIZE = 10
STABLE_MS = 300
COMMIT_CONFIDENCE = 0.6
VELOCITY_THRESHOLD = 0.1

# Suggestions
SUGGESTION_LIMIT = 5
AUTOCORRECT_VARIANT_CAP = 200

# TTS
DEFAULT_TTS_FORMAT = "wav"

# API versioning
API_PREFIX = "/v1"

# Health metadata
SERVICE_NAME = "HandSpeak.ai"
SERVICE_VERSION = "1.0.0"
