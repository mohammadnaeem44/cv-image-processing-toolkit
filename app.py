import io
import os
import numpy as np
import streamlit as st
from PIL import Image
import processor

# Configuration
st.set_page_config(
    page_title="Computer Vision Image Toolkit",
    page_icon="📷",
    layout="wide"
)

# Header & Tech Stack Badges
st.title("📷 Computer Vision Image Processing Toolkit")
st.markdown("Upload an image or choose a sample to explore real-time OpenCV operations.")

# Tooltips and explanations for operations
OPERATION_INFO = {
    "Original": "Displays the unedited input image in its raw RGB color format.",
    "Grayscale": "Converts RGB channels into a single luminance channel ($Y = 0.299R + 0.587G + 0.114B$).",
    "Resize": "Rescales the image dimensions using bilinear/area interpolation.",
    "Blur": "Applies a 2D Gaussian Kernel filter to smooth image features and reduce high-frequency noise.",
    "Edge Detection": "Uses Canny Edge Detection to calculate intensity gradients and locate strong structural boundaries.",
    "Threshold": "Applies Binary Thresholding to segment pixels into background (0) or foreground (255) based on brightness.",
    "Sharpen": "Enhances edge contrast by applying a custom spatial convolution filter kernel.",
    "HSV Visualization": "Converts color space from RGB (Red-Green-Blue) to HSV (Hue-Saturation-Value) representation.",
    "Contour Detection": "Detects connected structural boundaries from thresholded binary masks and overlays vector outlines."
}

# Sidebar Setup
st.sidebar.header("1. Image Selection")

# Sample Images Setup (Idea 1)
sample_choice = st.sidebar.selectbox(
    "Choose Input Method",
    ["Upload Custom Image", "Sample: Synthetic Test Pattern", "Sample: High Contrast Shapes"]
)

uploaded_file = None
if sample_choice == "Upload Custom Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Load image based on choice
original_np = None

if sample_choice == "Sample: Synthetic Test Pattern":
    # Generate a sample gradient image programmatically
    x, y = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 1, 400))
    pattern = np.uint8(255 * np.sin(10 * x) * np.cos(10 * y))
    original_np = np.stack([pattern, pattern, pattern], axis=-1)

elif sample_choice == "Sample: High Contrast Shapes":
    # Generate geometric shapes sample programmatically
    canvas = np.zeros((400, 400, 3), dtype=np.uint8)
    import cv2
    cv2.circle(canvas, (200, 200), 80, (255, 100, 50), -1)
    cv2.rectangle(canvas, (50, 50), (150, 150), (50, 255, 100), -1)
    cv2.line(canvas, (100, 300), (300, 350), (255, 255, 255), 8)
    original_np = canvas

elif uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    original_np = np.array(pil_image)

# Main Application Logic
if original_np is not None:
    st.sidebar.header("2. Choose Operation")
    operation = st.sidebar.selectbox("Select Image Transformation", list(OPERATION_INFO.keys()))

    # Algorithm Tooltip / Description Display (Idea 3)
    st.sidebar.info(f"💡 **Algorithm Concept:** {OPERATION_INFO[operation]}")

    processed_np = original_np.copy()

    # Dynamic Controls
    if operation == "Resize":
        h, w, _ = original_np.shape
        new_w = st.sidebar.slider("Width", 10, max(w * 2, 500), w)
        new_h = st.sidebar.slider("Height", 10, max(h * 2, 500), h)
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

    # Image Display Area
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(original_np, use_container_width=True)
        st.caption(f"Dimensions: {original_np.shape[1]}x{original_np.shape[0]} px")

    with col2:
        st.subheader(f"Processed: {operation}")
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
    # Enhanced Empty State Dashboard (Idea 4)
    st.markdown("---")
    st.subheader("Welcome to the CV Processing Sandbox 👋")
    st.write("Get started immediately by selecting a sample image or uploading your own from the left sidebar.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1️⃣ Spatial Filters")
        st.markdown("Apply Gaussian blurs, custom matrix sharpening filters, and area rescaling.")

    with col2:
        st.markdown("### 2️⃣ Edge & Structure")
        st.markdown("Run Canny Edge detection algorithms and extract binary object contours.")

    with col3:
        st.markdown("### 3️⃣ Color Spaces")
        st.markdown("Transform native RGB representations into single-channel Grayscale or 3D HSV space.")