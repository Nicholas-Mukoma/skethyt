import streamlit as st
import cv2
from PIL import Image
import numpy as np
import io

st.markdown(
    """
        <style>
            .text {
                color: red
            }
        </style>
    """,
    unsafe_allow_html=True
)

def sketch_image(image, k_size, scale):
    
    # Convert to gray image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert Image
    invert_img = cv2.bitwise_not(gray_image)
    
    # Blur image
    blur_img = cv2.GaussianBlur(invert_img, (k_size, k_size), 0)
    
    # Invert Blurred Image
    invblur_img = cv2.bitwise_not(blur_img)
    
    # Sketch Image
    sketch_img = cv2.divide(gray_image, blur_img, scale=scale)
    
    rgb_sketch = cv2.cvtColor(sketch_img, cv2.COLOR_BGR2RGB)
    
    return rgb_sketch

col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    st.title('SketchyT')
    # st.markdown("<h1 class='text'>SketchyT</h1>", unsafe_allow_html=True)

uploaded_image = st.file_uploader('Upload an image to sketch', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    img = np.array(img)
    
    # st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img_array = np.array(img)
    
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    scale = st.slider("Adjust Sketch Intensity", min_value=50.0, max_value=1000.0, value=256.0)
    
    sketch = sketch_image(image=img_array, k_size=111, scale=scale)
    
    sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption='Original Image', use_column_width=True)
        
        
    with col2:
        st.image(sketch, caption="Sketch", use_column_width=True)
    
    pil_sketch = Image.fromarray(sketch)
    
    buffer = io.BytesIO()
    pil_sketch.save(buffer, format='JPEG')
    buffer.seek(0)

    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col3:
        st.download_button(
        label="Download Sketch",
        data=buffer,
        file_name='pencil_sketch.jpg',
        mime='image/jpeg',
    )
    
    
   