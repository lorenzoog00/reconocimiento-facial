# recognize_faces.py

import cv2
import numpy as np
from deepface import DeepFace
import os
import json
import time

class FaceRecognitionTest:
    def __init__(self, db_path="face_database"):
        """Initialize the face recognition system"""
        self.db_path = db_path
        self.model_name = "VGG-Face"  # The deep learning model we're using
        self.detector_backend = "retinaface"  # Face detection algorithm
        self.recognition_threshold = 0.4  # Confidence threshold for recognition
        self.registered_faces = {}
        
        # Ensure the database exists
        if not os.path.exists(db_path):
            os.makedirs(db_path)
            print(f"Created database directory at {db_path}")
            
        # Load registered faces
        self.load_registered_faces()
        
    def load_registered_faces(self):
        """Load information about registered faces"""
        info_file = os.path.join(self.db_path, "face_info.json")
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                self.registered_faces = json.load(f)
                print(f"Loaded {len(self.registered_faces)} registered faces")
        else:
            print("No registered faces found. Please register faces first.")

    def recognize_face(self, frame):
        """
        Recognize faces in a frame
        Returns:
            - name of the recognized person (or None)
            - location of the face in the frame (or None)
            - confidence score (0-1)
        """
        try:
            # Step 1: Detect faces in the frame
            faces = DeepFace.extract_faces(
                frame,
                detector_backend=self.detector_backend,
                enforce_detection=True
            )
            
            if not faces:
                return None, None, 0
                
            # Get the first detected face
            face_info = faces[0]
            facial_area = face_info["facial_area"]
            
            # If no registered faces, return only detection
            if not self.registered_faces:
                return None, facial_area, 0
                
            # Step 2: Try to recognize the detected face
            highest_confidence = 0
            best_match = None
            
            # Compare with each registered face
            for name in self.registered_faces.keys():
                person_dir = os.path.join(self.db_path, name)
                if not os.path.exists(person_dir):
                    continue
                    
                # Check each sample of this person
                for sample_file in os.listdir(person_dir):
                    if not sample_file.endswith('.jpg'):
                        continue
                        
                    sample_path = os.path.join(person_dir, sample_file)
                    try:
                        # Compare the detected face with this sample
                        result = DeepFace.verify(
                            frame,
                            sample_path,
                            model_name=self.model_name,
                            detector_backend=self.detector_backend,
                            distance_metric="cosine"
                        )
                        
                        # Calculate confidence (1 - distance)
                        confidence = 1 - result["distance"]
                        
                        # Update best match if this is better
                        if confidence > highest_confidence:
                            highest_confidence = confidence
                            best_match = name
                            
                    except Exception as e:
                        print(f"Error comparing with {sample_file}: {str(e)}")
                        continue
            
            # Return results if confidence is above threshold
            if highest_confidence > (1 - self.recognition_threshold):
                return best_match, facial_area, highest_confidence
                
            # If no good match found
            return None, facial_area, highest_confidence
                
        except Exception as e:
            print(f"Error in face recognition: {str(e)}")
            return None, None, 0

    def start_recognition(self):
        """Start real-time face recognition"""
        print("\nStarting face recognition...")
        print("Press 'q' to quit")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
            
        # Variables for FPS calculation
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # Create a copy for display
            display_frame = frame.copy()
            
            try:
                # Perform face recognition
                name, face_location, confidence = self.recognize_face(frame)
                
                # If a face was detected
                if face_location is not None:
                    # Extract face coordinates
                    x = face_location["left"]
                    y = face_location["top"]
                    w = face_location["right"] - face_location["left"]
                    h = face_location["bottom"] - face_location["top"]
                    
                    if name:
                        # Known face - green rectangle
                        color = (0, 255, 0)
                        text = f"{name} ({confidence:.2%})"
                    else:
                        # Unknown face - red rectangle
                        color = (0, 0, 255)
                        text = "Unknown"
                    
                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw name/status
                    cv2.putText(display_frame, text, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
            
            # Display FPS
            cv2.putText(display_frame, f"FPS: {fps}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow("Face Recognition", display_frame)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognition = FaceRecognitionTest()
    recognition.start_recognition()