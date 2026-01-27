from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionHypothesis(BaseModel):
    letter: str
    confidence: float


class PredictionResult(BaseModel):
    top1: Optional[PredictionHypothesis] = None
    top2: List[PredictionHypothesis] = Field(default_factory=list)
    top5: List[PredictionHypothesis] = Field(default_factory=list)
    phrase: Optional[str] = None
    raw_confidence: float = 0.0


class Suggestion(BaseModel):
    word: str
    score: float
    source: str = "ensemble"


class SessionState(BaseModel):
    session_id: str
    partial_word: str = ""
    committed_letters: List[str] = Field(default_factory=list)
    words: List[str] = Field(default_factory=list)
    sentence: str = ""
    commit_state: str = "IN_TRANSITION"
    last_updated_ms: Optional[int] = None


class FramePayload(BaseModel):
    session_id: str
    frame_b64: str
    timestamp_ms: Optional[int] = None


class FrameResponse(BaseModel):
    session: SessionState
    prediction: PredictionResult
    suggestions: List[Suggestion]


class CommitPayload(BaseModel):
    session_id: str
    letter: str


class FeedbackPayload(BaseModel):
    session_id: str
    suggestion: str
    accepted: bool = True


class SessionResponse(BaseModel):
    session: SessionState
    suggestions: List[Suggestion] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    model_path: str
    service_version: str


class TTSPayload(BaseModel):
    session_id: Optional[str] = None
    text: str
    return_audio: bool = False


class TTSResponse(BaseModel):
    text: str
    audio_b64: Optional[str] = None
    format: str = "wav"
