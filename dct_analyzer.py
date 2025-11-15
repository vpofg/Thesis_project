"""
Discrete Cosine Transform (DCT) analysis for synthetic image detection
"""
import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from preprocessing import ImagePreprocessor


class DCTAnalyzer:
    """Analyze Discrete Cosine Transform of images"""
    
    def __init__(self, config, preprocessor):
        self.config = config
        self.preprocessor = preprocessor
        self.block_size = config['methods']['dct']['block_size']
    
    def compute_dct_coefficients(self, image_array):
        """
        Compute 2D DCT (entire image or first block)
        
        Args:
            image_array (np.array): Input image (H, W) or (H, W, 3)
            
        Returns:
            np.array: 2D DCT coefficients (normalized)
        """
        # Convert to grayscale
        gray = self.preprocessor.get_grayscale(image_array)
        
        # Normalize to [0, 1]
        gray = gray / (np.max(gray) + 1e-10)
        
        # Apply 2D DCT using scipy (norm='ortho' for orthonormal)
        dct_2d = dct(dct(gray, axis=0, norm='ortho'), axis=1, norm='ortho')
        
        return dct_2d
    
    def extract_block_features(self, image_array):
        """
        Extract DCT features from upper-left block (JPEG-style)
        
        Args:
            image_array (np.array): Input image
            
        Returns:
            dict: Dictionary of extracted features from DCT block
        """
        dct_2d = self.compute_dct_coefficients(image_array)
        
        # Extract upper-left block
        block = dct_2d[:self.block_size, :self.block_size]
        dct_flat = block.flatten()
        
        # DC component (average, first coefficient)
        dc_component = dct_2d[0, 0]
        
        # AC components (rest)
        ac_components = dct_flat[1:]
        
        # Statistical features
        features = {
            'dct_mean': float(np.mean(dct_flat)),
            'dct_std': float(np.std(dct_flat)),
            'dct_skewness': float(self._compute_skewness(dct_flat)),
            'dct_kurtosis': float(self._compute_kurtosis(dct_flat)),
            'dc_component': float(dc_component),
            'ac_energy': float(np.sum(ac_components**2)),
            'ac_mean': float(np.mean(np.abs(ac_components))),
            'dct_block': block
        }
        
        return features
    
    @staticmethod
    def _compute_skewness(x):
        """Compute skewness of distribution"""
        mean = np.mean(x)
        std = np.std(x)
        if std > 1e-10:
            return float(np.mean(((x - mean) / std)**3))
        return 0.0
    
    @staticmethod
    def _compute_kurtosis(x):
        """Compute excess kurtosis of distribution"""
        mean = np.mean(x)
        std = np.std(x)
        if std > 1e-10:
            return float(np.mean(((x - mean) / std)**4) - 3)
        return 0.0
    
    def visualize(self, image_array, features, title="", save_path=None):
        """
        Visualize DCT block and coefficients
        
        Args:
            image_array (np.array): Input image
            features (dict): Features dictionary from extract_block_features()
            title (str): Title for plot
            save_path (str): Path to save figure (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"DCT Analysis {title}", fontsize=14, fontweight='bold')
        
        # Original image
        if len(image_array.shape) == 3:
            display_img = image_array / 255.0 if image_array.max() > 1 else image_array
            axes[0, 0].imshow(display_img)
        else:
            axes[0, 0].imshow(image_array, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # DCT block heatmap
        im = axes[0, 1].imshow(features['dct_block'], cmap='RdBu_r')
        axes[0, 1].set_title(f'{self.block_size}x{self.block_size} DCT Block')
        axes[0, 1].set_xlabel('Frequency X')
        axes[0, 1].set_ylabel('Frequency Y')
        plt.colorbar(im, ax=axes[0, 1])
        
        # DCT coefficients histogram
        dct_flat = features['dct_block'].flatten()
        axes[1, 0].hist(dct_flat, bins=30, color='blue', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('DCT Coefficient Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('DCT Coefficient Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Feature summary
        feature_text = (
            f"Mean: {features['dct_mean']:.4f}\n"
            f"Std Dev: {features['dct_std']:.4f}\n"
            f"Skewness: {features['dct_skewness']:.4f}\n"
            f"Kurtosis: {features['dct_kurtosis']:.4f}\n"
            f"DC Component: {features['dc_component']:.4f}\n"
            f"AC Energy: {features['ac_energy']:.4f}"
        )
        axes[1, 1].text(0.1, 0.5, feature_text, fontsize=11, verticalalignment='center',
                       family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved DCT visualization to {save_path}")
        
        return fig