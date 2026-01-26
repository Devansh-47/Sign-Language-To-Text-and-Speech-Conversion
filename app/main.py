from __future__ import annotations

import logging
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app import config
from app.core.autocorrect import autocorrect
from app.core.hand_engine import create_engine
from app.core.phrase_dtw import matcher
from app.core.session_manager import session_manager
from app.core.suggestions import lexicon, suggest
from app.core.tts import tts_service
from app.models.schemas import (
    CommitPayload,
    FeedbackPayload,
    FramePayload,
    FrameResponse,
    HealthResponse,
    PredictionHypothesis,
    PredictionResult,
    SessionResponse,
    SessionState,
    Suggestion,
    TTSPayload,
    TTSResponse,
)

logger = logging.getLogger("handspeak")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title=config.SERVICE_NAME, version=config.SERVICE_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"]
    ,
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

engine = create_engine()


@app.get(f"{config.API_PREFIX}/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", model_path=str(config.MODEL_PATH), service_version=config.SERVICE_VERSION)


@app.post(f"{config.API_PREFIX}/frame", response_model=FrameResponse)
async def post_frame(payload: FramePayload) -> FrameResponse:
    session = session_manager.get(payload.session_id)
    try:
        pred = engine.predict(payload.frame_b64, session)
    except ValueError as exc:  # invalid input
        raise HTTPException(status_code=400, detail=str(exc))
    session.update_timestamp(payload.timestamp_ms)

    phrase = None
    if session.last_pts:
        phrase = matcher.match(session.last_pts, pred["confidence"])
        if phrase:
            session.words.append(phrase)
            session.partial_word = ""
            session.sentence = " ".join(session.words)

    suggestions = suggest(session.partial_word, pred["confidence"], session.personal_counts)
    autocorrected = autocorrect(session.partial_word, lexicon.words)
    for word in autocorrected:
        suggestions.append({"word": word, "score": pred["confidence"], "source": "autocorrect"})
    suggestions = sorted(suggestions, key=lambda x: x["score"], reverse=True)[: config.SUGGESTION_LIMIT]

    prediction = PredictionResult(
        top1=PredictionHypothesis(letter=pred["letter"], confidence=pred["confidence"]),
        top2=[PredictionHypothesis(**item) for item in pred["top2"]],
        top5=[PredictionHypothesis(**item) for item in pred["top5"]],
        phrase=phrase,
        raw_confidence=pred["raw_confidence"],
    )
    session_model = SessionState(**session_manager.as_dict(session))
    suggestion_models: List[Suggestion] = [Suggestion(**s) for s in suggestions]
    return FrameResponse(session=session_model, prediction=prediction, suggestions=suggestion_models)


@app.post(f"{config.API_PREFIX}/commit", response_model=SessionResponse)
async def commit(payload: CommitPayload) -> SessionResponse:
    session = session_manager.get(payload.session_id)
    session.commit_letter(payload.letter)
    session_model = SessionState(**session_manager.as_dict(session))
    suggestions = suggest(session.partial_word, 1.0, session.personal_counts)
    return SessionResponse(session=session_model, suggestions=[Suggestion(**s) for s in suggestions])


@app.post(f"{config.API_PREFIX}/feedback", response_model=SessionResponse)
async def feedback(payload: FeedbackPayload) -> SessionResponse:
    session = session_manager.get(payload.session_id)
    if payload.accepted:
        session.personal_counts[payload.suggestion] += 1
    session_model = SessionState(**session_manager.as_dict(session))
    return SessionResponse(session=session_model)


@app.get(f"{config.API_PREFIX}/session/{{session_id}}", response_model=SessionResponse)
async def get_session(session_id: str) -> SessionResponse:
    session = session_manager.get(session_id)
    session_model = SessionState(**session_manager.as_dict(session))
    suggestions = suggest(session.partial_word, 1.0, session.personal_counts)
    return SessionResponse(session=session_model, suggestions=[Suggestion(**s) for s in suggestions])


@app.post(f"{config.API_PREFIX}/tts", response_model=TTSResponse)
async def tts(payload: TTSPayload) -> TTSResponse:
    audio_b64 = tts_service.synthesize(payload.text) if payload.return_audio else None
    return TTSResponse(text=payload.text, audio_b64=audio_b64)


@app.get("/")
async def root():
    static_dir = Path(__file__).parent / "static"
    index_file = static_dir / "index.html"
    if index_file.exists():
        from fastapi.responses import FileResponse
        return FileResponse(index_file, media_type="text/html")
    return {"message": "HandSpeak.ai API. Docs: /docs, UI: /static/index.html"}
    return TTSResponse(text=payload.text, audio_b64=audio_b64)


# Optional websocket stub can be added if streaming is required.
