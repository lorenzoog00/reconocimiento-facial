# face_register.py

import os
import cv2
from deepface import DeepFace
from datetime import datetime
import json

class FaceRegistration:
    def __init__(self, db_path="face_database"):
        """Initialize the face registration system"""
        self.db_path = db_path
        self.detector_backend = "retinaface"
        self.registered_faces = {}
        
        # Create database directory if it doesn't exist
        if not os.path.exists(db_path):
            os.makedirs(db_path)
            
        # Load existing registrations
        self.load_registered_faces()
        
    def load_registered_faces(self):
        """Load information about registered faces"""
        info_file = os.path.join(self.db_path, "face_info.json")
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                self.registered_faces = json.load(f)

    def save_registered_faces(self):
        """Save information about registered faces"""
        info_file = os.path.join(self.db_path, "face_info.json")
        with open(info_file, 'w') as f:
            json.dump(self.registered_faces, f)

    def register_face(self, name):
        """Register a new face with multiple samples"""
        print(f"\nRegistering new face for {name}")
        print("Please look at the camera and move your face slightly to capture different angles")
        print("Press 'c' to capture a sample (need 5 samples)")
        print("Press 'q' to quit registration")
        
        cap = cv2.VideoCapture(0)
        samples_taken = 0
        required_samples = 5
        person_dir = os.path.join(self.db_path, name)
        
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        while samples_taken < required_samples:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from camera")
                break
            
            # Display current frame with counter
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Samples: {samples_taken}/{required_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Registration", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                try:
                    # Verify face is detectable
                    face_info = DeepFace.extract_faces(
                        frame, 
                        detector_backend=self.detector_backend,
                        enforce_detection=True
                    )
                    
                    if face_info:
                        # Save the frame
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{name}_{timestamp}.jpg"
                        filepath = os.path.join(person_dir, filename)
                        cv2.imwrite(filepath, frame)
                        
                        samples_taken += 1
                        print(f"Sample {samples_taken} captured successfully")
                    
                except Exception as e:
                    print("No face detected clearly. Please try again.")
                    
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if samples_taken == required_samples:
            self.registered_faces[name] = {
                "samples": required_samples,
                "register_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.save_registered_faces()
            print(f"\nSuccessfully registered {name} with {samples_taken} samples")
            return True
        
        print("\nRegistration incomplete")
        return False

    def list_registered_faces(self):
        """Display all registered faces"""
        if not self.registered_faces:
            print("\nNo faces registered yet")
            return
            
        print("\nRegistered Faces:")
        for name, info in self.registered_faces.items():
            print(f"- {name} (Registered: {info['register_date']})")

if __name__ == "__main__":
    registration = FaceRegistration()
    
    while True:
        print("\nFace Registration System")
        print("1. Register new face")
        print("2. List registered faces")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            name = input("Enter name for the new face: ")
            registration.register_face(name)
        elif choice == '2':
            registration.list_registered_faces()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")