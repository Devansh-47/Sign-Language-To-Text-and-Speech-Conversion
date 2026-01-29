import numpy as np
from typing import List

class Preprocessor:
    def __init__(self, target_frames: int = 30):
        self.target_frames = target_frames

    def process(self, raw_sequence: List[List[List[float]]]) -> np.ndarray:
        """
        Main processing pipeline.
        Args:
            raw_sequence: List of frames, where each frame is a list of 21 landmarks [x, y, z]
        Returns:
            Normalized numpy array of shape (target_frames, 21, 3)
        """
        # 1. Convert to numpy
        data = np.array(raw_sequence, dtype=np.float32) # (T, 21, 3)

        if data.ndim == 2: # Single frame case (1, 21, 3)
            data = data[np.newaxis, ...]

        # 2. Temporal Resampling (Pad or Truncate)
        data = self._resample(data)

        # 3. Spatial Normalization (Frame by Frame)
        processed_frames = []
        for frame in data:
            processed_frames.append(self._normalize_frame(frame))
        
        return np.stack(processed_frames)

    def _resample(self, data: np.ndarray) -> np.ndarray:
        """
        Resamples the sequence to `target_frames`.
        If T < target: repeat/pad
        If T > target: uniform sample
        """
        T = data.shape[0]
        if T == self.target_frames:
            return data
        
        if T < self.target_frames:
            # Pad with last frame
            padding = np.tile(data[-1], (self.target_frames - T, 1, 1))
            return np.concatenate([data, padding], axis=0)
        else:
            # Downsample
            indices = np.linspace(0, T - 1, self.target_frames, dtype=int)
            return data[indices]

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies translation, scale, and rotation normalization to a single frame.
        Expects frame shape (21, 3).
        """
        # MediaPipe Hands Keypoints: 0 is Wrist, 9 is Middle Finger MCP, 5 is Index MCP
        WRIST = 0
        INDEX_MCP = 5
        MIDDLE_MCP = 9

        # A. Translation: Center around wrist
        wrist = frame[WRIST]
        frame = frame - wrist

        # B. Scale: Normalize by size (dist wrist -> middle_mcp)
        # Using a stable reference bone for scale
        palm_size = np.linalg.norm(frame[MIDDLE_MCP])
        if palm_size < 1e-6:
            return frame # Avoid division by zero
        frame = frame / palm_size

        # C. Rotation: Align Wrist->Index_MCP vector to Y-axis
        # Current vector
        v = frame[INDEX_MCP]
        # Project to XY plane for simple alignment (assuming Z is depth)
        angle = np.arctan2(v[0], v[1]) # Angle with Y-axis
        
        # Rotation Matrix (around Z-axis)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        
        # Apply rotation (transposed because vectors are rows or we simply dot)
        # frame is (21, 3). we want to rotate x,y.
        # simpler:
        # x_new = x cos - y sin
        # y_new = x sin + y cos
        # To align v to Y-axis (x=0), we need to rotate by -angle? 
        # Actually let's do a proper 3D alignment if possible, but 2D is often enough for sign.
        # Let's strictly follow plan: "Rotation alignment".
        # We rotate such that Index MCP x-coordinate becomes 0.
        
        # Re-calc angle to rotate TO the Y axis (positive Y)
        # alpha = atan2(x, y). We want to rotate by -alpha.
        theta = -angle
        c, s = np.cos(theta), np.sin(theta)
        R_z = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        
        frame = np.dot(frame, R_z.T)

        return frame
