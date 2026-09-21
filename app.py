import io
import numpy as np
import streamlit as st
from PIL import Image
import processor

st.set_page_config(
    page_title="Computer Vision Image Toolkit",
    page_icon="📷",
    layout="wide"
)

st.title("📷 Computer Vision Image Processing Toolkit")
st.markdown("Upload an image, select an OpenCV operation, adjust settings, and download the result.")

# Sidebar - Image Upload & Operations Selection
st.sidebar.header("1. Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV RGB array format
    pil_image = Image.open(uploaded_file).convert("RGB")
    original_np = np.array(pil_image)

    st.sidebar.header("2. Choose Operation")
    operation = st.sidebar.selectbox(
        "Select Image Transformation",
        [
            "Original",
            "Grayscale",
            "Resize",
            "Blur",
            "Edge Detection",
            "Threshold",
            "Sharpen",
            "HSV Visualization",
            "Contour Detection"
        ]
    )

    # Dynamic Parameters in Sidebar based on selected operation
    processed_np = original_np.copy()

    if operation == "Resize":
        h, w, _ = original_np.shape
        new_w = st.sidebar.slider("Width", 10, w * 2, w)
        new_h = st.sidebar.slider("Height", 10, h * 2, h)
        processed_np = processor.process_resize(original_np, new_w, new_h)

    elif operation == "Grayscale":
        processed_np = processor.process_grayscale(original_np)

    elif operation == "Blur":
        kernel = st.sidebar.slider("Blur Intensity (Kernel Size)", 1, 31, 5, step=2)
        processed_np = processor.process_blur(original_np, kernel)

    elif operation == "Edge Detection":
        t1 = st.sidebar.slider("Threshold 1", 0, 255, 100)
        t2 = st.sidebar.slider("Threshold 2", 0, 255, 200)
        processed_np = processor.process_edge_detection(original_np, t1, t2)

    elif operation == "Threshold":
        thresh_val = st.sidebar.slider("Binary Threshold Value", 0, 255, 127)
        processed_np = processor.process_threshold(original_np, thresh_val)

    elif operation == "Sharpen":
        strength = st.sidebar.slider("Sharpening Strength", 0.5, 5.0, 1.0, step=0.5)
        processed_np = processor.process_sharpen(original_np, strength)

    elif operation == "HSV Visualization":
        processed_np = processor.process_hsv(original_np)

    elif operation == "Contour Detection":
        thresh_val = st.sidebar.slider("Threshold Value for Contours", 0, 255, 127)
        processed_np = processor.process_contours(original_np, thresh_val)

    # Main Display Area: Side-by-Side Comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(original_np, use_container_width=True)
        st.caption(f"Dimensions: {original_np.shape[1]}x{original_np.shape[0]} px")

    with col2:
        st.subheader(f"Processed ({operation})")
        st.image(processed_np, use_container_width=True)
        st.caption(f"Dimensions: {processed_np.shape[1]}x{processed_np.shape[0]} px")

    # Download Button
    result_pil = Image.fromarray(processed_np)
    buffer = io.BytesIO()
    result_pil.save(buffer, format="PNG")
    byte_im = buffer.getvalue()

    st.sidebar.markdown("---")
    st.sidebar.header("3. Download Result")
    st.sidebar.download_button(
        label="📥 Download Processed Image",
        data=byte_im,
        file_name=f"processed_{operation.lower().replace(' ', '_')}.png",
        mime="image/png"
    )

else:
    st.info("👈 Please upload an image using the sidebar to begin.")