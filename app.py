import streamlit as st
from src.preprocessing import load_image
from src.model import run_style_transfer

st.title("Neural Style Transfer")

#Sidebar inputs
content_img = st.sidebar.file_uploader("Choose a content image")
style_img = st.sidebar.file_uploader("Choose a style image")
steps = st.sidebar.slider("Number of optimization steps", 300, 900, 300, 100)
style_weight = st.sidebar.slider("Style weight", 1e6, 1e7, 1e6, 1e6)

if "output_image" not in st.session_state:
    st.session_state.output_image = None

if st.button("Run Style Transfer"):
    if content_img and style_img:
        #Reset session state for output image
        st.session_state.output_image = None
        #Save uploaded images locally for processing
        with open("content.jpg", "wb") as f:
            f.write(content_img.getbuffer())
        with open("style.jpg", "wb") as f:
            f.write(style_img.getbuffer())

        #Create a progress bar
        progress_bar = st.progress(0)  # Initialize progress bar

        #Run style transfer
        st.text("Running Style Transfer...")
        output_path = "output.jpg"
        for progress in run_style_transfer(content_img, style_img, output_path, num_steps=steps, style_weight=style_weight):
            progress_bar.progress(progress)

        #Save output image in session state
        with open(output_path, "rb") as f:
            st.session_state.output_image = f.read()
            
    else:
        st.error("Please upload both content and style images.")

#Show the image if it exists
if st.session_state.output_image:
    st.image(st.session_state.output_image, caption="Generated Image", use_container_width=True)


