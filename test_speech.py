import pyttsx3

print("Testing pyttsx3...")

try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    
    text = "Hello, this is a test"
    print(f"Speaking: {text}")
    
    engine.say(text)
    engine.runAndWait()
    
    print("Speech completed successfully")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
