"""
Image preprocessing: normalization, resizing, color conversions
"""
import numpy as np
import cv2
from utils import normalize_image, rgb_to_grayscale, rgb_to_ycbcr


class ImagePreprocessor:
    """Handle all image preprocessing operations"""
    
    def __init__(self, config):
        self.config = config
        self.normalize = config['preprocessing']['normalize']
        self.resize = config['preprocessing']['resize']
    
    def preprocess(self, image_array):
        """
        Apply all preprocessing steps to image
        
        Args:
            image_array (np.array): Raw image (H, W, 3) in [0, 255]
            
        Returns:
            np.array: Preprocessed image
        """
        # Resize if specified
        if self.resize is not None:
            image_array = cv2.resize(image_array, (self.resize, self.resize))
        
        # Normalize to [0, 1]
        if self.normalize:
            image_array = normalize_image(image_array)
        
        return image_array
    
    def get_grayscale(self, image_array):
        """Get grayscale version of image"""
        if self.normalize:
            # Already in [0, 1], convert back to [0, 255] for OpenCV
            image_uint8 = (image_array * 255).astype(np.uint8)
        else:
            image_uint8 = image_array.astype(np.uint8)
        
        gray = rgb_to_grayscale(image_uint8)
        
        # Return in same scale as input
        if self.normalize:
            return gray
        else:
            return gray * 255
    
    def get_ycbcr(self, image_array):
        """Get YCbCr version of image"""
        if self.normalize:
            image_uint8 = (image_array * 255).astype(np.uint8)
        else:
            image_uint8 = image_array.astype(np.uint8)
        
        ycbcr = rgb_to_ycbcr(image_uint8)
        
        if self.normalize:
            return ycbcr / 255.0
        else:
            return ycbcr