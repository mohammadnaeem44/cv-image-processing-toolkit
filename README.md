# Computer Vision Image Processing Toolkit

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A modular, web-based Computer Vision exploration platform built with **Python**, **OpenCV**, **NumPy**, and **Streamlit**. Designed following software engineering best practices, this toolkit provides an interactive environment to apply, parameterize, and analyze classic spatial transformations, color transformations, thresholding algorithms, and structural boundary detection methods.

---

## Key Features & Computer Vision Capabilities

The application abstracts complex matrix transformations and computer vision pipelines into an intuitive UI with real-time parametric feedback:

* **Spatial Transformations & Filtering**
  * **Gaussian Blur**: Implements 2D Gaussian Kernel smoothing with user-configurable kernel sizes to filter out high-frequency noise.
  * **Image Sharpening**: Applies custom spatial convolution matrices (laplacian-based kernel operations) to boost localized high-frequency edge details.
  * **Bilinear Rescaling**: Performs image width and height spatial adjustments using area interpolation to prevent aliasing artifacts.

* **Color Space Transformations**
  * **Grayscale Conversion**: Calculates localized single-channel luminance conversion ($Y = 0.299R + 0.587G + 0.114B$).
  * **HSV Mapping**: Deconstructs additive RGB color representations into Hue, Saturation, and Value cylindrical geometries for domain analysis.

* **Segmentation & Edge Analysis**
  * **Canny Edge Detection**: Calculates multi-stage intensity gradients with dual hysteresis thresholds to isolate structural boundaries.
  * **Binary Thresholding**: Segments image pixels into binary partitions based on dynamic cutoff intensities.
  * **Contour Extraction**: Identifies closed structural vector paths using binary thresholding and OpenCV's topological contour extraction algorithm (`cv2.findContours`).

* **Developer Experience & UI Utilities**
  * **Synthetic Test Bench**: Includes programmatically generated synthetic test patterns and geometric targets for instant algorithm testing without requiring local image uploads.
  * **Side-by-Side Spatial Inspection**: Displays aligned dual-canvas rendering for direct quantitative comparison.
  * **In-Memory Export Engine**: Encodes array buffer calculations into downloadable high-quality PNG formats.

---

## Technical Architecture & Design Decisions

To ensure production-grade separation of concerns, the project architecture divides presentation logic from numerical image processing calculations:

```text
cv-image-processing-toolkit/
├── app.py              # Presentation Layer: Streamlit UI state, reactive sliders, layout rendering
├── processor.py        # Core Domain Logic: Pure OpenCV & NumPy image manipulation functions
├── requirements.txt    # Project dependencies
├── .gitignore          # Environment & binary exclusion patterns
└── README.md           # Technical documentation
```

### Key Engineering Decisions:
1. **Decoupled Architecture**: `processor.py` contains side-effect-free pure functions operating strictly on `numpy.ndarray` inputs. This guarantees that image operations can be easily extracted for CLI pipelines, backend web services, or automated unit test runners without Streamlit UI dependencies.
2. **Headless Execution Environment**: Configured `opencv-python-headless` in `requirements.txt` to eliminate system-level X11/GUI display server bindings, preventing deployment conflicts in CI/CD pipelines or cloud hosting platforms (e.g., Streamlit Community Cloud, AWS App Runner).
3. **RGB/BGR Memory Standardization**: Explicit color space conversions ensure seamless memory passing between OpenCV’s native BGR matrix format and Streamlit’s RGB web graphics renderer.

---

## Getting Started

### Prerequisites

* Python `3.9` or higher
* `pip` package manager

### Local Setup & Execution

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mohammadnaeem44/cv-image-processing-toolkit.git
   cd cv-image-processing-toolkit
   ```

2. **Create and Activate a Virtual Environment**
   * **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   * **macOS / Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Application**
   ```bash
   streamlit run app.py
   ```
   Open your browser and navigate to `http://localhost:8501`.

---

## Technology Stack

* **Language**: Python
* **Computer Vision & Array Processing**: OpenCV (`opencv-python-headless`), NumPy
* **Frontend Interface**: Streamlit
* **Image I/O**: Pillow (PIL)

---

## Author & Contact

**Mohammad Naeem**
* GitHub: [@mohammadnaeem44](https://github.com/mohammadnaeem44)

---

## License

This project is open-source and available under the [MIT License](LICENSE).