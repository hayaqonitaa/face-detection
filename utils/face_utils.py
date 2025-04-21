import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.nn.functional import cosine_similarity

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize face detector and embedding model
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_face_embedding(image_path):
    """
    Extract face from image and get embedding
    
    Args:
        image_path: Path to image file
        
    Returns:
        torch.Tensor: Face embedding vector (512-dim) or None if no face detected
    """
    img = Image.open(image_path)
    face = mtcnn(img)
    if face is not None:
        face = face.to(device)
        embedding = model(face.unsqueeze(0))  # Output: [1, 512]
        return embedding
    else:
        print(f"No face detected in image: {image_path}")
        return None

def compare_embeddings(emb1, emb2):
    """
    Calculate cosine similarity between two face embeddings
    
    Args:
        emb1: First face embedding
        emb2: Second face embedding
        
    Returns:
        float: Similarity score between 0 and 1
    """
    return cosine_similarity(emb1, emb2).item()

def get_face_with_box(image_path):
    """
    Detect face in image and return the image with a bounding box
    
    Args:
        image_path: Path to image file
        
    Returns:
        tuple: (image with box, face detected)
    """
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Detect faces
    boxes, _ = mtcnn.detect(img_array)
    
    # If no faces detected, return original image
    if boxes is None:
        return img, None
    
    # Get the first face
    box = boxes[0]
    
    # Get face tensor
    face = mtcnn(img)
    
    return img, face