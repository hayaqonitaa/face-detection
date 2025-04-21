import streamlit as st
import os
import torch
import numpy as np
from PIL import Image
import io
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.face_utils import get_face_embedding, compare_embeddings

# Set page config
st.set_page_config(
    page_title="Face Similarity Comparison",
    page_icon="👤",
    layout="wide"
)

# Apply custom CSS
try:
    with open('assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    pass  # Silently continue if CSS file is not found

# Main content
st.title("Face Similarity Comparison")
st.markdown("Upload two face images to compare their similarity.")

# Create two columns for the images
col1, col2 = st.columns(2)

with col1:
    st.subheader("First Image")
    uploaded_file1 = st.file_uploader("Upload first face", type=["jpg", "jpeg", "png"], key="face1")

with col2:
    st.subheader("Second Image")
    uploaded_file2 = st.file_uploader("Upload second face", type=["jpg", "jpeg", "png"], key="face2")

# Threshold setting
threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.01)

# Process button
process_button = st.button("Compare Faces")

# Initialize session state variables if they don't exist
if 'result_generated' not in st.session_state:
    st.session_state.result_generated = False
if 'similarity_score' not in st.session_state:
    st.session_state.similarity_score = None

# Function to display image with face detection box
def display_image(file, container):
    if file is not None:
        img = Image.open(file)
        container.image(img)
        return img
    return None

# Display uploaded images
img1 = display_image(uploaded_file1, col1) if uploaded_file1 else None
img2 = display_image(uploaded_file2, col2) if uploaded_file2 else None

# Process images when button is clicked
if process_button and uploaded_file1 is not None and uploaded_file2 is not None:
    with st.spinner("Processing faces..."):
        # We need to save the uploaded files temporarily
        temp_img1 = "temp_img1.jpg"
        temp_img2 = "temp_img2.jpg"
        
        with open(temp_img1, "wb") as f:
            f.write(uploaded_file1.getbuffer())
        with open(temp_img2, "wb") as f:
            f.write(uploaded_file2.getbuffer())
        
        # Get face embeddings
        embedding1 = get_face_embedding(temp_img1)
        embedding2 = get_face_embedding(temp_img2)
        
        # Compare if both faces are detected
        if embedding1 is not None and embedding2 is not None:
            score = compare_embeddings(embedding1, embedding2)
            st.session_state.similarity_score = score
            st.session_state.result_generated = True
        else:
            if embedding1 is None:
                st.error("No face detected in the first image.")
            if embedding2 is None:
                st.error("No face detected in the second image.")
            st.session_state.result_generated = False
        
        # Clean up temp files
        os.remove(temp_img1)
        os.remove(temp_img2)

# Display results if they exist
if st.session_state.result_generated and st.session_state.similarity_score is not None:
    score = st.session_state.similarity_score
    match = score >= threshold
    
    st.markdown("---")
    st.subheader("Results")
    
    # Create a results container
    results_col1, results_col2, results_col3 = st.columns([1, 1, 1])
    
    with results_col1:
        st.metric("Similarity Score", f"{score:.4f}")
    
    with results_col2:
        if match:
            st.success("MATCH ✅")
        else:
            st.error("NOT MATCH ❌")
    
    with results_col3:
        st.metric("Threshold", f"{threshold:.2f}")
    
    # Create a visual representation of the score
    st.progress(score)