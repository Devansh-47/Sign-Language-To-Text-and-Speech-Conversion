from typing import List, Optional
from pydantic import BaseModel, Field, validator

class LandmarkFrame(BaseModel):
    """
    Represents a single frame containing landmarks.
    Expects 21 landmarks for a hand, each having x, y, z coordinates.
    """
    landmarks: List[List[float]] = Field(..., description="List of 21 landmarks [x, y, z]")

    @validator('landmarks')
    def check_landmarks_shape(cls, v):
        if len(v) != 21:
            raise ValueError(f"Expected 21 landmarks, got {len(v)}")
        for i, point in enumerate(v):
            if len(point) < 2: 
                raise ValueError(f"Landmark {i} must have at least x, y coordinates")
        return v

class PredictionRequest(BaseModel):
    """
    Request body for the prediction endpoint.
    """
    sequence: List[LandmarkFrame] = Field(..., description="Sequence of frames (temporal window)")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata from Android client")

    @validator('sequence')
    def check_sequence_length(cls, v):
        if len(v) == 0:
            raise ValueError("Sequence cannot be empty")
        return v
