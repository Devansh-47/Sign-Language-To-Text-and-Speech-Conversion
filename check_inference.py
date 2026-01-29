import onnxruntime as ort
import numpy as np
import os

def check_inference():
    model_path = "asl_model_v2.onnx"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
        return

    print(f"Loading model from {model_path}...")
    try:
        session = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Failed to create session: {e}")
        return

    print("Model loaded successfully.")
    
    # Check inputs
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Input Name: {input_name}, Shape: {input_shape}")
    
    # Create dummy input
    # Shape is likely ['batch', 30, 63] or similar. Dynamic axis usually shows as string or Nonetype/unknown.
    # Our export had dynamic batch. 
    # Let's try (1, 30, 63)
    dummy_input = np.random.randn(1, 30, 63).astype(np.float32)
    
    print("Running inference...")
    try:
        outputs = session.run(None, {input_name: dummy_input})
        logits = outputs[0]
        print(f"Output info: Shape {logits.shape}")
        
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        print("Inference Successful!")
        print(f"Top class index: {np.argmax(probs)}")
        print(f"Confidence: {np.max(probs)}")
        
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    check_inference()
