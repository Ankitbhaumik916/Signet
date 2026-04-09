"""
Image Preprocessing Pipeline for Signature Authentication
Handles image loading, normalization, thresholding, and tensor conversion
"""

import cv2
import numpy as np
import torch
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Load image from bytes
    
    Args:
        image_bytes: Image data as bytes
    
    Returns:
        OpenCV image array or None if loading fails
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.warning("Failed to decode image from bytes")
            return None
        
        return img
    except Exception as e:
        logger.error(f"Error loading image from bytes: {str(e)}")
        return None


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale
    
    Args:
        image: Input image (can be BGR or grayscale)
    
    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        # Convert BGR to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Already grayscale
        gray = image
    
    return gray


def apply_otsu_threshold(image: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's thresholding to isolate signature strokes
    
    Args:
        image: Grayscale image
    
    Returns:
        Binary image with signature strokes isolated
    """
    # Apply Gaussian blur to reduce noise before thresholding
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply Otsu thresholding
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    return binary


def find_signature_region(image: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find bounding box of the signature region using contour detection
    
    Args:
        image: Binary image
    
    Returns:
        Tuple of (x, y, width, height) for the bounding box
    """
    # Find contours
    contours, _ = cv2.findContours(
        image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        # No signature found, return full image dimensions
        h, w = image.shape
        return 0, 0, w, h
    
    # Find bounding rectangle for all contours
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    # Return bounding box with some padding
    padding = 10
    h, w = image.shape
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    return x_min, y_min, x_max - x_min, y_max - y_min


def crop_and_center(image: np.ndarray) -> np.ndarray:
    """
    Crop signature region and center it
    
    Args:
        image: Binary image
    
    Returns:
        Cropped and centered signature image
    """
    x, y, w, h = find_signature_region(image)
    cropped = image[y:y+h, x:x+w]
    
    return cropped


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resize image to target dimensions
    
    Args:
        image: Input image
        target_size: Target (height, width) dimensions
    
    Returns:
        Resized image
    """
    h, w = target_size
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 1]
    
    Args:
        image: Input image (uint8)
    
    Returns:
        Normalized image (float32)
    """
    normalized = image.astype(np.float32) / 255.0
    return normalized


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply Gaussian blur for noise reduction
    
    Args:
        image: Input image
        kernel_size: Kernel size (must be odd)
    
    Returns:
        Blurred image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd size
    
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred


def preprocess_signature(
    image_bytes: bytes,
    target_size: Tuple[int, int] = (128, 128),
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Optional[torch.Tensor]:
    """
    Complete preprocessing pipeline for signature images
    
    Steps:
    1. Load image from bytes
    2. Convert to grayscale
    3. Apply Gaussian blur
    4. Apply Otsu thresholding
    5. Crop and center signature
    6. Resize to target size
    7. Normalize to [0, 1]
    8. Convert to tensor
    
    Args:
        image_bytes: Raw image data as bytes
        target_size: Target image dimensions (height, width)
        device: Device to place tensor on ("cuda" or "cpu")
    
    Returns:
        Preprocessed image tensor of shape (1, 1, H, W) or None if preprocessing fails
    """
    try:
        # Step 1: Load image
        image = load_image_from_bytes(image_bytes)
        if image is None:
            logger.error("Failed to load image from bytes")
            return None
        
        # Step 2: Convert to grayscale
        gray = convert_to_grayscale(image)
        
        # Step 3: Apply Gaussian blur
        blurred = apply_gaussian_blur(gray, kernel_size=5)
        
        # Step 4: Apply Otsu thresholding
        binary = apply_otsu_threshold(blurred)
        
        # Step 5: Crop and center signature
        cropped = crop_and_center(binary)
        
        # If cropped region is empty, use full image
        if cropped.size == 0:
            cropped = binary
        
        # Step 6: Resize to target size
        resized = resize_image(cropped, target_size)
        
        # Step 7: Normalize to [0, 1]
        # Invert binary image so signature strokes (white) become 1.0
        if binary.max() > 0:
            inverted = 255 - resized
            normalized = normalize_image(inverted)
        else:
            normalized = normalize_image(resized)
        
        # Optional quality gate: reject near-empty or near-solid inputs.
        signal_ratio = float((normalized > 0.1).mean())
        if signal_ratio < 0.003 or signal_ratio > 0.90:
            logger.warning("Input rejected by quality gate (signal_ratio=%.4f)", signal_ratio)
            return None

        # Step 8: Convert to tensor
        # Shape: (H, W) -> (1, H, W) -> (1, 1, H, W)
        tensor = torch.from_numpy(normalized).float()
        tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device)

        logger.info(f"Image preprocessed successfully. Shape: {tensor.shape}")
        return tensor
    
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}", exc_info=True)
        return None


def preprocess_signature_with_inversion(
    image_bytes: bytes,
    target_size: Tuple[int, int] = (224, 224),
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Optional[torch.Tensor]:
    """
    Alternative preprocessing that keeps signature strokes as black (0)
    
    Args:
        image_bytes: Raw image data as bytes
        target_size: Target image dimensions
        device: Device to place tensor on
    
    Returns:
        Preprocessed tensor or None if fails
    """
    try:
        image = load_image_from_bytes(image_bytes)
        if image is None:
            return None
        
        gray = convert_to_grayscale(image)
        blurred = apply_gaussian_blur(gray, kernel_size=5)
        binary = apply_otsu_threshold(blurred)
        
        # Don't invert - keep signature as black (low values)
        cropped = crop_and_center(binary)
        if cropped.size == 0:
            cropped = binary
        
        resized = resize_image(cropped, target_size)
        normalized = normalize_image(resized)
        
        tensor = torch.from_numpy(normalized).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(device)

        return tensor
    
    except Exception as e:
        logger.error(f"Error in alternative preprocessing: {str(e)}")
        return None
