# Face Similarity Detection App

A Streamlit-based application for comparing facial similarity between two images using deep learning.

## Overview

This application uses facial recognition technology to detect faces in images and compute their similarity. It's built with Streamlit for the interface and uses MTCNN for face detection and FaceNet (InceptionResnetV1) for face embedding extraction.

![Face Similarity App Screenshot](https://via.placeholder.com/800x400.png?text=Face+Similarity+App)

## Features

- **Face Detection**: Automatically detects faces in uploaded images
- **Similarity Comparison**: Calculates and displays similarity scores between faces
- **Adjustable Threshold**: Customize the matching threshold for different use cases
- **Simple Navigation**: Text-only sidebar for easy access to different features
- **Settings Management**: Configure application behavior through a dedicated settings page

## Project Structure

```
face_similarity_app/
├── main.py               # Main app with navigation sidebar
├── pages/
│   ├── 1_face_similarity.py  # First page - Face similarity
├── utils/
│   └── face_utils.py     # Utility functions for face processing
├── assets/
│   └── style.css         # Custom CSS styling
└── requirements.txt      # Dependencies
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/face-similarity-app.git
   cd face-similarity-app
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run main.py
   ```

2. The application will open in your default web browser at `http://localhost:8501`

3. Navigation:
   - Use the sidebar to navigate between different pages
   - Upload images on the "Face Similarity" page
   - Adjust the threshold as needed
   - Click "Compare Faces" to process the images

## Technical Details

- **Face Detection**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Face Embedding**: InceptionResnetV1 (FaceNet) pre-trained on VGGFace2
- **Similarity Metric**: Cosine Similarity
- **Default Threshold**: 0.7 (70%)


## License

[MIT License](LICENSE)

## Acknowledgments

- FaceNet: [A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
- MTCNN: [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878)
- Streamlit: [The fastest way to build data apps](https://streamlit.io/)
