import streamlit as st

# Page config
st.set_page_config(
    page_title="Facenalyze",
    page_icon="👤",
    layout="wide"
)

# Custom CSS (optional)
try:
    with open('assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Style file not found. Please create the assets/style.css file in the assets folder.")

# Header
st.title("👤 Face Similarity and Ethnic Detection System")
st.markdown("---")

# Welcome text
st.markdown("""
## 👋 Welcome!

Welcome to the **Face Similarity and Ethnicity Detection App**!  
This web application lets you **compare faces** and **predict ethnicity** with powerful, deep learning-based detection algorithms.  
It's designed to be **easy to use**, **accurate**, and **privacy-friendly**.

### Features:
- **Face Similarity**  
  Compare two face images and get a similarity score based on facial embeddings. The app uses **FaceNet** (InceptionResnetV1) for face embedding extraction and computes the similarity between the images.

- **Ethnicity Detection**  
  Predict the likely **ethnicity** of a person based on their facial features. This feature uses a **custom-trained ResNet model** that was fine-tuned on a specific facial ethnicity dataset. The model classifies faces into five ethnic groups: **Javanese**, **Sundanese**, **Chinese**, **Minahasan**, and **Betawi**.

Use the **sidebar** to navigate between features and explore both **Face Similarity** and **Ethnicity Detection**.

""")


# Footer
st.markdown("---")
st.markdown("Made with ❤️ by 008, 013, 017")

