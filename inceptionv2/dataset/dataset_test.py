from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN  # Assuming you're using the facenet_pytorch library for MTCNN

class YourFaceProcessingClass:
    def __init__(self, required_size=(112, 112)):
        self.required_size = required_size
        self.mtcnn = MTCNN(keep_all=True)  # Initialize MTCNN with keep_all=True

    def extract_face(self, image_path):
        image = Image.open(image_path).convert('RGB')
        
        # Use MTCNN for face detection and cropping
        boxes, probs = self.mtcnn.detect(image)
        
        if boxes is not None:
            # Assume there is only one face in the image for simplicity
            box = boxes[0]
            
            # Convert face region to NumPy array
            face_region = np.array(image.crop(box.astype(int)))

            # Convert face region to PIL Image
            face = Image.fromarray(face_region)

            # Resize to required size
            face = face.resize(self.required_size, Image.BILINEAR)
            
            return face_region
        else:
            # If no face is detected, return a placeholder image or handle it as needed
            return np.zeros((self.required_size[1], self.required_size[0], 3), dtype=np.uint8)