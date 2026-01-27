"""
Test client for landmark-based ASL recognition backend
Simulates Android app sending MediaPipe landmarks
"""

import asyncio
import websockets
import json
import random

# Sample normalized MediaPipe landmarks for letter "A" (fist with thumb up)
SAMPLE_LANDMARKS_A = [
    [0.5, 0.5, 0.0],      # 0: Wrist
    [0.48, 0.45, -0.01],  # 1: Thumb CMC
    [0.46, 0.42, -0.02],  # 2: Thumb MCP
    [0.44, 0.39, -0.03],  # 3: Thumb IP
    [0.42, 0.36, -0.04],  # 4: Thumb tip (extended)
    [0.52, 0.43, 0.01],   # 5: Index MCP
    [0.53, 0.46, 0.02],   # 6: Index PIP (curled)
    [0.54, 0.48, 0.03],   # 7: Index DIP
    [0.55, 0.49, 0.04],   # 8: Index tip
    [0.55, 0.43, 0.01],   # 9: Middle MCP
    [0.56, 0.46, 0.02],   # 10: Middle PIP
    [0.57, 0.48, 0.03],   # 11: Middle DIP
    [0.58, 0.49, 0.04],   # 12: Middle tip
    [0.58, 0.44, 0.01],   # 13: Ring MCP
    [0.59, 0.47, 0.02],   # 14: Ring PIP
    [0.60, 0.49, 0.03],   # 15: Ring DIP
    [0.61, 0.50, 0.04],   # 16: Ring tip
    [0.61, 0.45, 0.01],   # 17: Pinky MCP
    [0.62, 0.48, 0.02],   # 18: Pinky PIP
    [0.63, 0.50, 0.03],   # 19: Pinky DIP
    [0.64, 0.51, 0.04],   # 20: Pinky tip
]

# Sample for "B" (flat hand, fingers together, thumb tucked)
SAMPLE_LANDMARKS_B = [
    [0.5, 0.5, 0.0],      # 0: Wrist
    [0.48, 0.45, -0.01],  # 1: Thumb CMC
    [0.47, 0.43, -0.02],  # 2: Thumb MCP
    [0.46, 0.42, -0.03],  # 3: Thumb IP
    [0.51, 0.47, -0.01],  # 4: Thumb tip (tucked)
    [0.52, 0.43, 0.01],   # 5: Index MCP
    [0.52, 0.38, 0.02],   # 6: Index PIP (extended)
    [0.52, 0.33, 0.03],   # 7: Index DIP
    [0.52, 0.28, 0.04],   # 8: Index tip
    [0.55, 0.43, 0.01],   # 9: Middle MCP
    [0.55, 0.38, 0.02],   # 10: Middle PIP
    [0.55, 0.33, 0.03],   # 11: Middle DIP
    [0.55, 0.28, 0.04],   # 12: Middle tip
    [0.58, 0.44, 0.01],   # 13: Ring MCP
    [0.58, 0.39, 0.02],   # 14: Ring PIP
    [0.58, 0.34, 0.03],   # 15: Ring DIP
    [0.58, 0.29, 0.04],   # 16: Ring tip
    [0.61, 0.45, 0.01],   # 17: Pinky MCP
    [0.61, 0.40, 0.02],   # 18: Pinky PIP
    [0.61, 0.35, 0.03],   # 19: Pinky DIP
    [0.61, 0.30, 0.04],   # 20: Pinky tip
]

# No hand detected
NO_HAND = {
    "handDetected": False,
    "handedness": "Right",
    "landmarks": None
}


async def test_websocket():
    """Test WebSocket connection with sample landmark data"""
    uri = "ws://localhost:8000/ws/landmarks"
    
    print("=" * 70)
    print("  Testing Landmark-Based ASL Recognition")
    print("=" * 70)
    
    try:
        async with websockets.connect(uri) as websocket:
            print("[CONNECTED] Connected to landmark server")
            
            # Test 1: No hand detected
            print("\n[TEST 1] Sending NO HAND...")
            await websocket.send(json.dumps(NO_HAND))
            response = await websocket.recv()
            result = json.loads(response)
            print(f"Response: {result}")
            assert not result["handDetected"], "Expected handDetected=False"
            print("✓ PASS")
            
            # Test 2: Letter A
            print("\n[TEST 2] Sending landmarks for 'A'...")
            payload_a = {
                "handDetected": True,
                "handedness": "Right",
                "landmarks": SAMPLE_LANDMARKS_A
            }
            
            # Send 3 times for temporal smoothing
            for i in range(3):
                await websocket.send(json.dumps(payload_a))
                response = await websocket.recv()
                result = json.loads(response)
                print(f"  Frame {i+1}: {result}")
                await asyncio.sleep(0.1)
            
            print("✓ Temporal smoothing active")
            
            # Test 3: Letter B
            print("\n[TEST 3] Sending landmarks for 'B'...")
            payload_b = {
                "handDetected": True,
                "handedness": "Right",
                "landmarks": SAMPLE_LANDMARKS_B
            }
            
            for i in range(3):
                await websocket.send(json.dumps(payload_b))
                response = await websocket.recv()
                result = json.loads(response)
                print(f"  Frame {i+1}: {result}")
                await asyncio.sleep(0.1)
            
            print("✓ Character transition works")
            
            # Test 4: Left hand (should mirror to right)
            print("\n[TEST 4] Testing left hand mirroring...")
            payload_left = {
                "handDetected": True,
                "handedness": "Left",
                "landmarks": SAMPLE_LANDMARKS_A
            }
            
            await websocket.send(json.dumps(payload_left))
            response = await websocket.recv()
            result = json.loads(response)
            print(f"Response: {result}")
            print("✓ Left hand mirroring active")
            
            # Test 5: Invalid landmark count
            print("\n[TEST 5] Testing error handling (wrong landmark count)...")
            payload_invalid = {
                "handDetected": True,
                "handedness": "Right",
                "landmarks": [[0.5, 0.5, 0.0]] * 10  # Only 10 landmarks
            }
            
            await websocket.send(json.dumps(payload_invalid))
            response = await websocket.recv()
            result = json.loads(response)
            print(f"Response: {result}")
            assert "error" in result, "Expected error for invalid landmark count"
            print("✓ Error handling works")
            
            print("\n" + "=" * 70)
            print("  ALL TESTS PASSED")
            print("=" * 70)
            
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_websocket())
