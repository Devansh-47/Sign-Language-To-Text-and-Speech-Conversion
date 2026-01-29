import json
import sys
import os
import numpy as np

# Add current dir to path
sys.path.append(os.getcwd())

from backend.schemas import LandmarkFrame
from backend.preprocessing import Preprocessor

def test_pipeline():
    # Load sample data
    with open(r"C:\Users\krish\HandSpeak.ai\dataset_v2_mobile\landmarks\A_landmarks.json", "r") as f:
        data = json.load(f)
    
    # Take the first sample
    sample_raw = data[0]['landmarks']
    
    print(f"Original shape: {len(sample_raw)} points")
    
    # 1. Validation
    # Wrap in list of lists for schema if needed, but schema expects single frame or sequence?
    # Schema `LandmarkFrame` expects `landmarks` list.
    try:
        frame = LandmarkFrame(landmarks=sample_raw)
        print("Schema Validation Passed")
    except Exception as e:
        print(f"Schema Validation Failed: {e}")
        return

    # 2. Preprocessing
    prep = Preprocessor(target_frames=30)
    # The preprocessor expects a sequence (List of Frames).
    # Since we have one static frame, we pass [sample_raw]
    # Ideally for training valid data we might want to pass more.
    
    output = prep.process([sample_raw])
    print(f"Processed Output Shape: {output.shape}")
    
    # Checks
    assert output.shape == (30, 21, 3), "Shape mismatch"
    
    # Check Wrist Centering (Frame 0, Point 0 should be approx 0,0,0)
    wrist = output[0, 0]
    print(f"Wrist at frame 0: {wrist}")
    assert np.allclose(wrist, 0, atol=1e-5), "Wrist not centered"
    
    # Check Rotation: Index MCP (Point 5) x should be 0 (aligned to Y axis)
    # Note: My logic in preprocessing was:
    # angle = arctan2(v[0], v[1]) -> angle with Y axis
    # rotate by -angle.
    # New X = x cos - y sin.
    # If v = [x, y], tan(a) = x/y. x = y tan(a).
    # ...
    # Let's just check the value.
    index_mcp = output[0, 5]
    print(f"Index MCP at frame 0: {index_mcp}")
    # Ideally index_mcp[0] should be close to 0.
    
    print("Test Pipeline SUCCESS")

if __name__ == "__main__":
    test_pipeline()
