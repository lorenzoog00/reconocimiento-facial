import cv2
import numpy as np
from time import time

def initialize_camera(camera_id=0, width=640, height=480):
    """Initialize the camera with specified parameters."""
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def load_face_cascade():
    """Load the pre-trained face detection cascade classifier."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

def detect_faces(frame, face_cascade):
    """Detect faces in the frame and return the frame with rectangles drawn around faces."""
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add text showing "Face Detected"
        cv2.putText(frame, 'Face Detected', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, len(faces)

def main():
    # Initialize camera and face cascade
    cap = initialize_camera()
    face_cascade = load_face_cascade()
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Variables for FPS calculation
    fps_start_time = time()
    fps_counter = 0
    fps = 0
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Detect faces and draw rectangles
        frame, num_faces = detect_faces(frame, face_cascade)
        
        # Calculate and display FPS
        fps_counter += 1
        if time() - fps_start_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time()
            
        cv2.putText(frame, f'FPS: {fps}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Faces: {num_faces}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Face Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()