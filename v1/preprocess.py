"""Binary mask boundary extraction."""

import cv2
import numpy as np


def extract_boundary(label_np):
    """
    Extracts the binary boundary from a binary mask.
    label_np is a (H, W) numpy array of 0s and 1s.
    Returns a (H, W) boundary mask of 0s and 1s.
    """
    # Convert to uint8 for cv2
    label_uint8 = (label_np * 255).astype(np.uint8)
    # Morphological gradient gives the boundary
    kernel = np.ones((3,3), np.uint8)
    gradient = cv2.morphologyEx(label_uint8, cv2.MORPH_GRADIENT, kernel)
    return (gradient > 127).astype(np.float32)
