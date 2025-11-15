"""
Local variance and noise pattern analysis for synthetic image detection
"""
import numpy as np
from scipy.ndimage import median_filter
from scipy.fftpack import fft2, fftshift
import matplotlib.pyplot as plt
from preprocessing import ImagePreprocessor


class VarianceAnalyzer:
    """Analyze variance and noise patterns in images"""
    
    def __init__(self, config, preprocessor):
        self.config = config
        self.preprocessor = preprocessor
        self.median_size = config['methods']['variance']['median_filter_size']
    
    def extract_noise_residual(self, image_array):
        """
        Extract noise residual via median filtering
        
        Principle: 
        - Median filter removes high-frequency details (edges, noise)
        - Residual = original - denoised contains mostly noise
        - Real and fake images have different noise characteristics
        
        Args:
            image_array (np.array): Input image
            
        Returns:
            np.array: Noise residual image
        """
        gray = self.preprocessor.get_grayscale(image_array)
        
        # Ensure proper range for median_filter
        if gray.max() <= 1:
            gray_for_filter = (gray * 255).astype(np.uint8)
        else:
            gray_for_filter = gray.astype(np.uint8)
        
        # Apply median filter
        denoised = median_filter(gray_for_filter.astype(float), size=self.median_size)
        
        # Compute residual
        noise = gray_for_filter.astype(float) - denoised
        
        return noise, denoised
    
    def extract_features(self, image_array):
        """
        Extract variance and noise-based features
        
        Args:
            image_array (np.array): Input image
            
        Returns:
            dict: Dictionary of extracted features
        """
        noise, denoised = self.extract_noise_residual(image_array)
        
        # Feature 1: Overall noise variance
        overall_variance = float(np.var(noise))
        
        # Feature 2: Local variance map
        h, w = noise.shape
        window_size = 5
        local_var_list = []
        
        for i in range(h - window_size):
            for j in range(w - window_size):
                patch = noise[i:i+window_size, j:j+window_size]
                local_var_list.append(np.var(patch))
        
        local_var_array = np.array(local_var_list)
        mean_local_var = float(np.mean(local_var_array))
        variance_of_vars = float(np.var(local_var_array))
        
        # Feature 3: Frequency spectrum of noise
        noise_fft = fft2(noise)
        noise_fft_shift = fftshift(noise_fft)
        noise_magnitude = np.abs(noise_fft_shift)
        
        # Energy in high frequencies
        h_fft, w_fft = noise_magnitude.shape
        center_y, center_x = h_fft // 2, w_fft // 2
        
        # Define outer region as high frequencies
        outer_y_start = center_y // 2
        outer_y_end = center_y + center_y // 2
        outer_x_start = center_x // 2
        outer_x_end = center_x + center_x // 2
        
        if outer_y_start < outer_y_end and outer_x_start < outer_x_end:
            outer_region = noise_magnitude[outer_y_start:outer_y_end, outer_x_start:outer_x_end]
            high_freq_energy = float(np.sum(outer_region))
        else:
            high_freq_energy = 0.0
        
        total_energy = float(np.sum(noise_magnitude)) + 1e-10
        noise_high_freq_ratio = high_freq_energy / total_energy
        
        features = {
            'overall_variance': overall_variance,
            'mean_local_variance': mean_local_var,
            'variance_of_variances': variance_of_vars,
            'noise_high_freq_ratio': noise_high_freq_ratio,
            'noise_residual': noise,
            'denoised': denoised
        }
        
        return features
    
    def visualize(self, image_array, features, title="", save_path=None):
        """
        Visualize variance analysis
        
        Args:
            image_array (np.array): Input image
            features (dict): Features dictionary from extract_features()
            title (str): Title for plot
            save_path (str): Path to save figure (optional)
        """
        # Normalize for display
        if image_array.max() > 1:
            image_display = image_array / 255.0
        else:
            image_display = image_array
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Variance Analysis {title}", fontsize=14, fontweight='bold')
        
        # Original image
        if len(image_array.shape) == 3:
            axes[0, 0].imshow(image_display)
        else:
            axes[0, 0].imshow(image_display, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Denoised image
        denoised_display = features['denoised'] / 255.0
        axes[0, 1].imshow(denoised_display, cmap='gray')
        axes[0, 1].set_title('Denoised (Median Filter)')
        axes[0, 1].axis('off')
        
        # Noise residual
        noise_display = features['noise_residual']
        noise_display_norm = (noise_display - noise_display.min()) / (noise_display.max() - noise_display.min() + 1e-10)
        axes[1, 0].imshow(noise_display_norm, cmap='gray')
        axes[1, 0].set_title('Noise Residual')
        axes[1, 0].axis('off')
        
        # Feature summary
        feature_text = (
            f"Overall Variance: {features['overall_variance']:.4f}\n"
            f"Mean Local Variance: {features['mean_local_variance']:.4f}\n"
            f"Variance of Variances: {features['variance_of_variances']:.4f}\n"
            f"Noise High-Freq Ratio: {features['noise_high_freq_ratio']:.4f}"
        )
        axes[1, 1].text(0.1, 0.5, feature_text, fontsize=11, verticalalignment='center',
                       family='monospace', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved Variance visualization to {save_path}")
        
        return fig