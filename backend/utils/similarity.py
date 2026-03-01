"""
Similarity Computation and Heatmap Generation
Handles similarity scoring between signature embeddings and visualization
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import logging
import base64
from io import BytesIO
from typing import Tuple

logger = logging.getLogger(__name__)


def compute_euclidean_distance(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor
) -> torch.Tensor:
    """
    Compute Euclidean distance between two embeddings
    
    Args:
        embedding1: First embedding tensor
        embedding2: Second embedding tensor
    
    Returns:
        Euclidean distance as scalar tensor
    """
    distance = torch.norm(embedding1 - embedding2, p=2, dim=1)
    return distance


def compute_cosine_similarity(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor
) -> torch.Tensor:
    """
    Compute cosine similarity between two embeddings
    
    Args:
        embedding1: First embedding tensor
        embedding2: Second embedding tensor
    
    Returns:
        Cosine similarity score (0-1, where 1 is identical)
    """
    similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
    # Scale from [-1, 1] to [0, 1]
    similarity = (similarity + 1) / 2
    return similarity


def compute_similarity(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
    method: str = "euclidean"
) -> Tuple[float, str]:
    """
    Compute similarity score between two embeddings
    
    Methods:
    - euclidean: Inverse Euclidean distance (lower distance = higher similarity)
    - cosine: Direct cosine similarity
    
    Args:
        embedding1: First embedding tensor
        embedding2: Second embedding tensor
        method: Similarity computation method
    
    Returns:
        Tuple of (similarity_score, confidence_level)
        - similarity_score: float between 0.0 and 1.0
        - confidence_level: "High" / "Medium" / "Low"
    """
    try:
        with torch.no_grad():
            if method == "euclidean":
                # Euclidean distance approach
                # Normalize embeddings to [0, 1] range
                distance = compute_euclidean_distance(embedding1, embedding2)
                
                # Convert distance to similarity
                # Assuming max distance is ~2.0 for normalized embeddings
                similarity = torch.exp(-distance / 0.5).item()
                similarity = max(0.0, min(1.0, similarity))
            
            elif method == "cosine":
                # Cosine similarity approach (already in [0, 1])
                similarity = compute_cosine_similarity(embedding1, embedding2).item()
                similarity = max(0.0, min(1.0, similarity))
            
            else:
                raise ValueError(f"Unknown similarity method: {method}")
        
        # Determine confidence level based on similarity score
        if similarity >= 0.85:
            confidence = "High"
        elif similarity >= 0.70:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        logger.info(f"Similarity: {similarity:.4f}, Confidence: {confidence}")
        return similarity, confidence
    
    except Exception as e:
        logger.error(f"Error computing similarity: {str(e)}")
        return 0.0, "Low"


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert torch tensor to numpy array
    
    Args:
        tensor: Input tensor of shape (1, 1, H, W) or (1, H, W)
    
    Returns:
        Numpy array of shape (H, W)
    """
    tensor = tensor.cpu().detach()
    
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # (1, 1, H, W) -> (1, H, W)
    
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)  # (1, H, W) -> (H, W)
    
    array = tensor.numpy()
    return array


def compute_pixel_difference(
    image1_tensor: torch.Tensor,
    image2_tensor: torch.Tensor
) -> np.ndarray:
    """
    Compute pixel-wise difference between two signature images
    
    Args:
        image1_tensor: First image tensor (1, 1, H, W)
        image2_tensor: Second image tensor (1, 1, H, W)
    
    Returns:
        Difference map as numpy array (H, W), values in [0, 255]
    """
    try:
        # Convert to numpy
        img1 = tensor_to_numpy(image1_tensor)
        img2 = tensor_to_numpy(image2_tensor)
        
        # Ensure same dimensions
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Compute absolute difference
        diff = np.abs(img1 - img2)
        
        # Scale to [0, 255]
        diff_uint8 = (diff * 255).astype(np.uint8)
        
        return diff_uint8
    
    except Exception as e:
        logger.error(f"Error computing pixel difference: {str(e)}")
        h, w = tensor_to_numpy(image1_tensor).shape
        return np.zeros((h, w), dtype=np.uint8)


def create_heatmap(
    difference_map: np.ndarray,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Create colored heatmap from difference map
    
    Args:
        difference_map: Grayscale difference map (H, W)
        colormap: OpenCV colormap ID
    
    Returns:
        Colored heatmap (H, W, 3) in BGR format
    """
    # Ensure difference map is uint8
    if difference_map.dtype != np.uint8:
        difference_map = (difference_map * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(difference_map, colormap)
    
    return heatmap


def overlay_heatmap(
    original_image: torch.Tensor,
    heatmap: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay heatmap on original image
    
    Args:
        original_image: Original image tensor (1, 1, H, W)
        heatmap: Colored heatmap (H, W, 3)
        alpha: Blending factor for heatmap
    
    Returns:
        Blended image (H, W, 3) in BGR format
    """
    try:
        # Convert original to uint8
        original_array = tensor_to_numpy(original_image)
        original_uint8 = (original_array * 255).astype(np.uint8)
        
        # Convert grayscale to BGR for blending
        original_bgr = cv2.cvtColor(original_uint8, cv2.COLOR_GRAY2BGR)
        
        # Blend
        blended = cv2.addWeighted(
            original_bgr,
            1 - alpha,
            heatmap,
            alpha,
            0
        )
        
        return blended
    
    except Exception as e:
        logger.error(f"Error overlaying heatmap: {str(e)}")
        return heatmap


def compute_heatmap(
    image1_tensor: torch.Tensor,
    image2_tensor: torch.Tensor,
    overlay: bool = True,
    alpha: float = 0.5
) -> str:
    """
    Compute difference heatmap between two signature images
    
    Args:
        image1_tensor: First image tensor (reference)
        image2_tensor: Second image tensor (test)
        overlay: Whether to overlay heatmap on original image
        alpha: Transparency of heatmap overlay
    
    Returns:
        Base64-encoded PNG image
    """
    try:
        # Compute pixel difference
        diff_map = compute_pixel_difference(image1_tensor, image2_tensor)
        
        # Create colored heatmap
        heatmap_colored = create_heatmap(diff_map)
        
        # Optionally overlay on original
        if overlay:
            result = overlay_heatmap(image2_tensor, heatmap_colored, alpha)
        else:
            result = heatmap_colored
        
        # Convert to PIL Image
        result_bgr = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) if len(result.shape) == 3 else result
        
        # For uint8 BGR images
        if len(result.shape) == 3:
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        else:
            result_rgb = result
        
        pil_image = Image.fromarray(result_rgb)
        
        # Encode to base64
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        logger.info("Heatmap generated successfully")
        return img_b64
    
    except Exception as e:
        logger.error(f"Error computing heatmap: {str(e)}")
        # Return placeholder base64 image
        return ""


def compute_grad_cam_heatmap(
    image_tensor: torch.Tensor,
    model_encoder: torch.nn.Module,
    target_layer: torch.nn.Module
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for visualization of CNN attention
    
    Args:
        image_tensor: Input image tensor (1, 1, H, W)
        model_encoder: CNN encoder model
        target_layer: Layer to compute gradients for
    
    Returns:
        Grad-CAM heatmap (H, W) with values in [0, 1]
    """
    try:
        image_tensor.requires_grad_(True)
        
        # Forward pass
        output = model_encoder(image_tensor)
        
        # Backward pass to compute gradients
        loss = output.sum()
        loss.backward()
        
        # Get gradients from target layer
        # This is a simplified implementation
        gradients = image_tensor.grad
        
        # Compute CAM
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * image_tensor).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        
        # Convert to numpy
        cam_np = tensor_to_numpy(cam)
        
        return cam_np
    
    except Exception as e:
        logger.error(f"Error computing Grad-CAM: {str(e)}")
        return np.zeros((224, 224))


# Import PIL Image for base64 conversion
try:
    from PIL import Image
except ImportError:
    logger.warning("PIL not available, some heatmap features may not work")
