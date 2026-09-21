import cv2
import numpy as np


def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """Converts a BGR or Grayscale image to RGB format for web rendering."""
    if len(image.shape) == 2:  # Grayscale image
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def process_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    """Converts RGB image to Grayscale."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def process_resize(image_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resizes image to specified width and height."""
    return cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)


def process_blur(image_rgb: np.ndarray, kernel_size: int) -> np.ndarray:
    """Applies Gaussian Blur. Kernel size must be an odd integer."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image_rgb, (kernel_size, kernel_size), 0)


def process_edge_detection(image_rgb: np.ndarray, threshold1: int, threshold2: int) -> np.ndarray:
    """Applies Canny Edge Detection."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def process_threshold(image_rgb: np.ndarray, threshold_val: int) -> np.ndarray:
    """Applies Binary Thresholding."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)


def process_sharpen(image_rgb: np.ndarray, strength: float) -> np.ndarray:
    """Sharpens the image using a custom kernel matrix."""
    # Standard sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 4 + strength, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image_rgb, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def process_hsv(image_rgb: np.ndarray) -> np.ndarray:
    """Converts RGB to HSV and maps back to RGB for visualization."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)


def process_contours(image_rgb: np.ndarray, threshold_val: int) -> np.ndarray:
    """Finds and draws external object contours on top of the original image."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_image = image_rgb.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)  # Draw green contours
    return result_image