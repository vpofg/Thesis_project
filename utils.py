"""
Utility functions for image loading, saving, and general operations
"""
import os
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
from download_kaggle_data import download_cifake_kaggle



def load_image(image_path):
    """
    Load image and return as numpy array (RGB)
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        np.array: Image in RGB format (H, W, 3) with values in [0, 255]
    """
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    return image_array


def save_image(image_array, output_path):
    """
    Save numpy array as image file
    
    Args:
        image_array (np.array): Image array
        output_path (str): Output file path
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to uint8 if needed
    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)
    
    image = Image.fromarray(image_array)
    image.save(output_path)


def normalize_image(image_array):
    """
    Normalize image to [0, 1] range
    
    Args:
        image_array (np.array): Image in [0, 255] range
        
    Returns:
        np.array: Normalized image in [0, 1] range
    """
    return image_array.astype(np.float32) / 255.0


def denormalize_image(image_array):
    """
    Denormalize image from [0, 1] to [0, 255] range
    
    Args:
        image_array (np.array): Image in [0, 1] range
        
    Returns:
        np.array: Image in [0, 255] range
    """
    return np.clip(image_array * 255, 0, 255).astype(np.uint8)


def rgb_to_grayscale(image_array):
    """
    Convert RGB image to grayscale
    
    Args:
        image_array (np.array): RGB image (H, W, 3)
        
    Returns:
        np.array: Grayscale image (H, W)
    """
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Use standard RGB to grayscale conversion
        gray = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return gray.astype(np.float32) / 255.0
    return image_array


def rgb_to_ycbcr(image_array):
    """
    Convert RGB image to YCbCr color space
    
    Args:
        image_array (np.array): RGB image (H, W, 3) with values in [0, 255]
        
    Returns:
        np.array: YCbCr image (H, W, 3) with values in [0, 255]
    """
    ycbcr = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    return ycbcr.astype(np.float32)


def load_dataset(config):
    """
    Load all images from dataset directory
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (real_images, fake_images, real_paths, fake_paths)
               Each: list of (image_array, filename)
    """
    data_dir = config['data']['image_dir']
    real_dir = os.path.join(data_dir, config['data']['real_dir'])
    fake_dir = os.path.join(data_dir, config['data']['fake_dir'])
    extensions = config['data']['image_extensions']
    
    real_images = []
    fake_images = []
    
    # Load real images
    print("Loading real images...")
    if os.path.exists(real_dir):
        for filename in tqdm(os.listdir(real_dir)):
            if any(filename.lower().endswith(ext) for ext in extensions):
                path = os.path.join(real_dir, filename)
                try:
                    img = load_image(path)
                    real_images.append((img, filename))
                except Exception as e:
                    print(f"Error loading {path}: {e}")
    
    # Load fake images
    print("Loading fake images...")
    if os.path.exists(fake_dir):
        for filename in tqdm(os.listdir(fake_dir)):
            if any(filename.lower().endswith(ext) for ext in extensions):
                path = os.path.join(fake_dir, filename)
                try:
                    img = load_image(path)
                    fake_images.append((img, filename))
                except Exception as e:
                    print(f"Error loading {path}: {e}")
    
    print(f"Loaded {len(real_images)} real images")
    print(f"Loaded {len(fake_images)} fake images")
    
    return real_images, fake_images


def create_output_dirs(config):
    """Create output directories for results"""
    os.makedirs(os.path.join(config['project']['output_dir'], 'features'), exist_ok=True)
    os.makedirs(os.path.join(config['project']['output_dir'], 'visualizations'), exist_ok=True)