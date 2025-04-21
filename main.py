import streamlit as st

# Set page config
st.set_page_config(
    page_title="Face Similarity App",
    page_icon="👤",
    layout="wide"
)

# Apply custom CSS
try:
    with open('assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Style file not found. Please create the assets/style.css file.")

# Main page content
st.title("Face Similarity Detection System")
st.markdown("""
## Welcome to Face Similarity Detection App

This application allows you to compare two faces and determine their similarity.

Use the sidebar to navigate to different features:
- **Face Similarity**: Upload and compare two face images
- **About**: Learn more about the application
- **Settings**: Configure application settings

To get started, click on "Face Similarity" in the sidebar.
""")

# Show a sample image
st.image("https://via.placeholder.com/800x400.png?text=Face+Similarity+Detection", use_container_width=True)

# Note: The sidebar is handled automatically by Streamlit
# when using the pages/ directory structure