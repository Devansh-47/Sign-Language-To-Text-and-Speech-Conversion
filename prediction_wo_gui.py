import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk

# Load the model
model = tf.keras.models.load_model('check.h5')

# Hindi class labels with "झ" at the end
class_labels = ["क", "ख", "ग", "घ", "च", "छ", "ज", "झ"]  # "झ" is now the last label

# Define the coordinates and size of the larger ROI (Region of Interest)
roi_x, roi_y, roi_w, roi_h = 50, 50, 600, 600  # Adjust these values for a larger ROI

def preprocess_image(image):
    image = tf.image.resize(image, [64, 64])
    image = image / 255.0
    return image

def predict_image(image):
    preprocessed_image = preprocess_image(image)
    preprocessed_image = tf.expand_dims(preprocessed_image, axis=0)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label

def update_frame():
    """Capture frame from camera, make predictions, and update the Tkinter GUI."""
    ret, frame = cap.read()
    
    if ret:
        # Define the ROI (Region of Interest) in the frame
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        
        # Convert BGR image to RGB
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Make predictions on the ROI
        predicted_class = predict_image(tf.convert_to_tensor(rgb_roi, dtype=tf.float32))
        
        # Draw a rectangle around the larger ROI
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
        
        # Display the prediction on the frame
        cv2.putText(frame, f"Prediction: {predicted_class}", (roi_x + 10, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert the frame to an ImageTk object for displaying in Tkinter
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update the label with the new frame
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        prediction_label.config(text=f"Recognized Letter: {predicted_class}")
    
    # Schedule the next frame update
    root.after(10, update_frame)

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Set up the GUI
root = tk.Tk()
root.title("Sign Language Recognition")

# Label to display the video
video_label = tk.Label(root)
video_label.pack()

# Label to display the prediction
prediction_label = tk.Label(root, text="Recognized Letter: None", font=("Arial", 24))
prediction_label.pack(pady=20)

# Start the video capture and update loop
update_frame()

# Start the Tkinter main loop
root.mainloop()

# Release the video capture when the window is closed
cap.release()
cv2.destroyAllWindows()
