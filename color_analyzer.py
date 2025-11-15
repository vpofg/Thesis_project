"""
Color spectrum and RGB channel analysis for synthetic image detection
"""
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import ImagePreprocessor


class ColorAnalyzer:
    """Analyze color distributions in RGB and YCbCr color spaces"""
    
    def __init__(self, config, preprocessor):
        self.config = config
        self.preprocessor = preprocessor
    
    def extract_rgb_features(self, image_array):
        """
        Extract per-channel statistics from RGB image
        
        Args:
            image_array (np.array): Input image (H, W, 3)
            
        Returns:
            dict: Dictionary of RGB channel features
        """
        # Normalize if needed
        if image_array.max() > 1:
            image_norm = image_array / 255.0
        else:
            image_norm = image_array
        
        # Extract channels
        r, g, b = image_norm[:,:,0], image_norm[:,:,1], image_norm[:,:,2]
        
        features = {
            'r_mean': float(np.mean(r)),
            'g_mean': float(np.mean(g)),
            'b_mean': float(np.mean(b)),
            'r_std': float(np.std(r)),
            'g_std': float(np.std(g)),
            'b_std': float(np.std(b)),
            'r_skewness': float(self._compute_skewness(r.flatten())),
            'g_skewness': float(self._compute_skewness(g.flatten())),
            'b_skewness': float(self._compute_skewness(b.flatten())),
        }
        
        # Channel correlations
        features['rg_correlation'] = float(np.corrcoef(r.flatten(), g.flatten())[0, 1])
        features['rb_correlation'] = float(np.corrcoef(r.flatten(), b.flatten())[0, 1])
        features['gb_correlation'] = float(np.corrcoef(g.flatten(), b.flatten())[0, 1])
        
        return features
    
    def extract_ycbcr_features(self, image_array):
        """
        Extract per-channel statistics from YCbCr color space
        
        Args:
            image_array (np.array): Input image (H, W, 3) in RGB
            
        Returns:
            dict: Dictionary of YCbCr channel features
        """
        ycbcr = self.preprocessor.get_ycbcr(image_array)
        
        # Normalize to [0, 1]
        ycbcr_norm = ycbcr / 255.0
        
        y, cb, cr = ycbcr_norm[:,:,0], ycbcr_norm[:,:,1], ycbcr_norm[:,:,2]
        
        features = {
            'y_mean': float(np.mean(y)),
            'cb_mean': float(np.mean(cb)),
            'cr_mean': float(np.mean(cr)),
            'y_std': float(np.std(y)),
            'cb_std': float(np.std(cb)),
            'cr_std': float(np.std(cr)),
            'y_skewness': float(self._compute_skewness(y.flatten())),
            'cb_skewness': float(self._compute_skewness(cb.flatten())),
            'cr_skewness': float(self._compute_skewness(cr.flatten())),
        }
        
        return features
    
    def extract_all_color_features(self, image_array):
        """
        Extract all color features (RGB and YCbCr)
        
        Args:
            image_array (np.array): Input image
            
        Returns:
            dict: Combined RGB and YCbCr features
        """
        features = {}
        features.update(self.extract_rgb_features(image_array))
        features.update(self.extract_ycbcr_features(image_array))
        return features
    
    @staticmethod
    def _compute_skewness(x):
        """Compute skewness"""
        mean = np.mean(x)
        std = np.std(x)
        if std > 1e-10:
            return float(np.mean(((x - mean) / std)**3))
        return 0.0
    
    def visualize(self, image_array, features, title="", save_path=None):
        """
        Visualize color histograms
        
        Args:
            image_array (np.array): Input image
            features (dict): Features dictionary from extract_all_color_features()
            title (str): Title for plot
            save_path (str): Path to save figure (optional)
        """
        # Normalize for display
        if image_array.max() > 1:
            image_display = image_array / 255.0
        else:
            image_display = image_array
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Color Analysis {title}", fontsize=14, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(image_display)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # RGB histograms
        r, g, b = image_display[:,:,0], image_display[:,:,1], image_display[:,:,2]
        axes[0, 1].hist(r.flatten(), bins=50, color='red', alpha=0.5, label='R')
        axes[0, 1].hist(g.flatten(), bins=50, color='green', alpha=0.5, label='G')
        axes[0, 1].hist(b.flatten(), bins=50, color='blue', alpha=0.5, label='B')
        axes[0, 1].set_xlabel('Pixel Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('RGB Histograms')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature statistics for RGB
        rgb_text = (
            f"RGB Mean: ({features['r_mean']:.3f}, {features['g_mean']:.3f}, {features['b_mean']:.3f})\n"
            f"RGB Std: ({features['r_std']:.3f}, {features['g_std']:.3f}, {features['b_std']:.3f})\n"
            f"RGB Corr: RG={features['rg_correlation']:.3f}, "
            f"RB={features['rb_correlation']:.3f}, GB={features['gb_correlation']:.3f}\n"
            f"\nYCbCr Mean: ({features['y_mean']:.3f}, {features['cb_mean']:.3f}, {features['cr_mean']:.3f})\n"
            f"YCbCr Std: ({features['y_std']:.3f}, {features['cb_std']:.3f}, {features['cr_std']:.3f})"
        )
        axes[1, 0].text(0.1, 0.5, rgb_text, fontsize=10, verticalalignment='center',
                       family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        axes[1, 0].axis('off')
        
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved Color visualization to {save_path}")
        
        return fig