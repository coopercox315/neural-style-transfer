import streamlit as st
from src.preprocessing import load_image
from src.model import run_style_transfer
from streamlit_image_select import image_select

st.title("Neural Style Transfer")

#Sidebar inputs
#Content image selection
st.sidebar.subheader("Select Content Image")
uploaded_content = st.sidebar.file_uploader("Upload a content image", type=["jpg", "png"])
with st.sidebar:
    if uploaded_content:
        #Save the uploaded image locally for further use
        with open("uploaded_content.jpg", "wb") as f:
            f.write(uploaded_content.getbuffer())
        content_img = "uploaded_content.jpg"
        st.sidebar.image(content_img, caption="Uploaded Content Image", use_container_width=False)
    else:
        content_img = image_select(
            label="or choose an example content image",
            images=[
                "examples/content/dancing.jpg",
                "examples/content/tiger.jpg",
                "examples/content/neckarfront.jpg",
                "examples/content/mountains.jpeg",
            ],
            captions=["Dancing", "Tiger", "Neckarfront", "Mountains"],
            use_container_width=False
        )

#Style image selection
st.sidebar.subheader("Select Style Image")
uploaded_style = st.sidebar.file_uploader("Upload a style image", type=["jpg", "png"])
with st.sidebar:
    if uploaded_style:
        #Save the uploaded image locally for further use
        with open("uploaded_style.jpg", "wb") as f:
            f.write(uploaded_style.getbuffer())
        style_img = "uploaded_style.jpg"
        style_img = uploaded_style
        st.sidebar.image(style_img, caption="Uploaded Style Image", use_container_width=False)
    else:
        style_img = image_select(
            label="or choose an example style image",
            images=[
                "examples/style/picasso.jpg",
                "examples/style/starry.jpg",
                "examples/style/shipwreck.jpg",
                "examples/style/abstractswirls.jpeg",
            ],
            captions=["Picasso", "Starry Night", "Shipwreck", "Abstract Swirls"],
            use_container_width=False
        )
        
steps = st.sidebar.slider("Number of optimization steps", 300, 2000, 300, 100)
style_weight = st.sidebar.slider("Style weight", 1e6, 1e7, 1e6, 1e6)
max_size = st.sidebar.number_input("Maximum image size (pixels)", 128, 2160, 512, 64)

if "output_image" not in st.session_state:
    st.session_state.output_image = None
    
#Main content
if st.button("Run Style Transfer"):
    if content_img and style_img:
        #Reset session state for output image
        st.session_state.output_image = None

        #Create a progress bar and initialize it
        progress_bar = st.progress(0)

        #Run style transfer
        st.text("Running Style Transfer...")
        output_path = "output.jpg"
        for progress in run_style_transfer(content_img, style_img, output_path, max_size, num_steps=steps, style_weight=style_weight):
            progress_bar.progress(progress)

        #Save output image in session state
        with open(output_path, "rb") as f:
            st.session_state.output_image = f.read()
    else:
        st.error("Please upload both content and style images.")

#Show the image if it exists
if st.session_state.output_image:
    st.image(st.session_state.output_image, caption="Generated Image", use_container_width=True)

#Example outputs
example_container = st.container()
with example_container:
    st.subheader("Output Examples")
    example1 = st.columns(3, vertical_alignment="bottom")
    with example1[0]:
        st.image("examples/content/tiger.jpg", caption="Content Image", use_container_width=True)
    with example1[1]:
        st.image("examples/style/starry.jpg", caption="Style Image", use_container_width=True)
    with example1[2]:
        st.image("examples/output/tiger_starry.jpg", caption="Generated Image", use_container_width=True)

    example2 = st.columns(3, vertical_alignment="bottom")
    with example2[0]:
        st.image("examples/content/cityscape.jpeg", caption="Content Image", use_container_width=True)
    with example2[1]:
        st.image("examples/style/compositionvii.jpg", caption="Style Image", use_container_width=True)
    with example2[2]:
        st.image("examples/output/cityscape_composition.jpg", caption="Generated Image", use_container_width=True)

    example3 = st.columns(3, vertical_alignment="bottom")
    with example3[0]:
        st.image("examples/content/neckarfront.jpg", caption="Content Image", use_container_width=True)
    with example3[1]:
        st.image("examples/style/shipwreck.jpg", caption="Style Image", use_container_width=True)
    with example3[2]:
        st.image("examples/output/neckarfront_shipwreck.jpg", caption="Generated Image", use_container_width=True)

    example4 = st.columns(3, vertical_alignment="bottom")
    with example4[0]:
        st.image("examples/content/bwportrait.jpg", caption="Content Image", use_container_width=True)
    with example4[1]:
        st.image("examples/style/abstractswirls.jpeg", caption="Style Image", use_container_width=True)
    with example4[2]:
        st.image("examples/output/bwportrait_swirls.jpg", caption="Generated Image", use_container_width=True)

    example5 = st.columns(3, vertical_alignment="bottom")
    with example5[0]:
        st.image("examples/content/mountains.jpeg", caption="Content Image", use_container_width=True)
    with example5[1]:
        st.image("examples/style/waterlillies.jpg", caption="Style Image", use_container_width=True)
    with example5[2]:
        st.image("examples/output/mountain_lillies.jpg", caption="Generated Image", use_container_width=True)