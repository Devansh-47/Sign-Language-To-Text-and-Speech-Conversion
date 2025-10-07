import subprocess
import time
import mss
from PIL import Image

def verify_changes():
    """
    Launches the application, waits for it to load, takes a screenshot,
    and saves it for verification.
    """
    # 1. Start the application in a separate process
    app_process = subprocess.Popen(["python", "final_pred.py"])
    print("Application started...")

    # 2. Wait for the application window to appear
    # This delay is a simple way to allow the GUI to initialize.
    # A more robust solution might involve checking for the window's existence.
    time.sleep(5)
    print("Capturing screenshot...")

    # 3. Capture the screen
    with mss.mss() as sct:
        # Get information of monitor 1
        monitor_number = 1
        mon = sct.monitors[monitor_number]

        # The screen part to capture
        monitor = {
            "top": mon["top"],
            "left": mon["left"],
            "width": mon["width"],
            "height": mon["height"],
            "mon": monitor_number,
        }
        # Grab the data
        sct_img = sct.grab(monitor)

        # Create an Image
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

        # Save to the verification directory
        screenshot_path = "jules-scratch/verification/verification.png"
        img.save(screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

    # 4. Terminate the application process
    app_process.terminate()
    app_process.wait()
    print("Application terminated.")

if __name__ == "__main__":
    verify_changes()