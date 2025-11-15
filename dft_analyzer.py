"""
Discrete Fourier Transform (DFT) analysis for synthetic image detection
"""
import numpy as np
from scipy.fftpack import fft2, fftshift
import matplotlib.pyplot as plt
from preprocessing import ImagePreprocessor


class DFTAnalyzer:
    """Analyze Discrete Fourier Transform of images"""
    
    def __init__(self, config, preprocessor):
        self.config = config
        self.preprocessor = preprocessor
        self.num_freq_bins = config['methods']['dft']['num_freq_bins']
    
    def compute_spectrum(self, image_array):
        """
        Compute 2D FFT magnitude spectrum (log scale)
        
        Args:
            image_array (np.array): Input image (H, W) or (H, W, 3)
            
        Returns:
            tuple: (magnitude_log, magnitude_linear)
                   Both are centered frequency representations
        """
        # Convert to grayscale
        gray = self.preprocessor.get_grayscale(image_array)
        
        # Normalize to [0, 1]
        gray = gray / (np.max(gray) + 1e-10)
        
        # Apply Hann window to reduce edge effects
        h, w = gray.shape
        window = np.outer(np.hanning(h), np.hanning(w))
        gray_windowed = gray * window
        
        # Compute 2D FFT
        f_transform = fft2(gray_windowed)
        f_shift = fftshift(f_transform)
        
        # Magnitude spectrum
        magnitude = np.abs(f_shift)
        magnitude_log = np.log1p(magnitude)  # log(1 + magnitude) for visualization
        
        return magnitude_log, magnitude
    
    def compute_radial_profile(self, magnitude_spectrum):
        """
        Compute 1D radial (azimuthally-averaged) frequency profile
        
        Args:
            magnitude_spectrum (np.array): 2D magnitude spectrum (centered)
            
        Returns:
            tuple: (radial_profile, frequencies)
                   radial_profile: 1D array of average magnitude vs frequency
                   frequencies: 1D array of frequency bins
        """
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # Create distance map from center
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
        max_dist = int(np.sqrt(center_x**2 + center_y**2))
        
        # Compute radial average
        radial_profile = np.zeros(max_dist)
        bin_count = np.zeros(max_dist)
        
        for i in range(h):
            for j in range(w):
                d = distance[i, j]
                if d < max_dist:
                    radial_profile[d] += magnitude_spectrum[i, j]
                    bin_count[d] += 1
        
        # Normalize by bin count
        radial_profile /= (bin_count + 1e-10)
        frequencies = np.arange(max_dist)
        
        return radial_profile, frequencies
    
    def extract_features(self, image_array):
        """
        Extract quantitative features from DFT spectrum
        
        Args:
            image_array (np.array): Input image
            
        Returns:
            dict: Dictionary of extracted features
        """
        magnitude_log, magnitude = self.compute_spectrum(image_array)
        radial_profile, freqs = self.compute_radial_profile(magnitude)
        
        # Feature 1: High-frequency energy ratio
        high_freq_threshold = len(freqs) // 3  # Top 1/3 of frequencies
        energy_high = np.sum(radial_profile[high_freq_threshold:])
        energy_total = np.sum(radial_profile) + 1e-10
        high_freq_ratio = energy_high / energy_total
        
        # Feature 2: Peak detection in high frequencies
        high_freq_section = radial_profile[high_freq_threshold:]
        mean_hf = np.mean(high_freq_section)
        std_hf = np.std(high_freq_section)
        threshold = mean_hf + 1.5 * std_hf
        num_peaks = np.sum(high_freq_section > threshold)
        
        # Feature 3: Spectral slope (power law: magnitude ~ frequency^-slope)
        # Fit in log space: log(mag) = a - slope * log(freq)
        valid_idx = (freqs > 0) & (freqs < 100)
        log_freqs = np.log10(freqs[valid_idx] + 1e-10)
        log_magnitude = np.log10(radial_profile[valid_idx] + 1e-10)
        coeffs = np.polyfit(log_freqs, log_magnitude, 1)
        spectral_slope = -coeffs[0]  # Negative because magnitude decreases
        
        # Feature 4: Energy concentration in low vs mid frequencies
        freq_25 = len(freqs) // 4
        freq_50 = len(freqs) // 2
        energy_low = np.sum(radial_profile[:freq_25])
        energy_mid = np.sum(radial_profile[freq_25:freq_50])
        energy_high_alt = np.sum(radial_profile[freq_50:])
        
        energy_norm = energy_total
        energy_low_ratio = energy_low / energy_norm
        energy_mid_ratio = energy_mid / energy_norm
        energy_high_ratio_alt = energy_high_alt / energy_norm
        
        features = {
            'high_freq_energy_ratio': float(high_freq_ratio),
            'peak_count': int(num_peaks),
            'spectral_slope': float(spectral_slope),
            'energy_low_ratio': float(energy_low_ratio),
            'energy_mid_ratio': float(energy_mid_ratio),
            'energy_high_ratio': float(energy_high_ratio_alt),
            'radial_profile': radial_profile,
            'frequencies': freqs,
            'magnitude_log': magnitude_log
        }
        
        return features
    
    def visualize(self, image_array, features, title="", save_path=None):
        """
        Visualize 2D spectrum and radial profile
        
        Args:
            image_array (np.array): Input image
            features (dict): Features dictionary from extract_features()
            title (str): Title for plot
            save_path (str): Path to save figure (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"DFT Analysis {title}", fontsize=14, fontweight='bold')
        
        # Original image
        if len(image_array.shape) == 3:
            display_img = image_array / 255.0 if image_array.max() > 1 else image_array
            axes[0, 0].imshow(display_img)
        else:
            axes[0, 0].imshow(image_array, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 2D Magnitude Spectrum
        im = axes[0, 1].imshow(features['magnitude_log'], cmap='hot')
        axes[0, 1].set_title('2D Magnitude Spectrum (log)')
        axes[0, 1].set_xlabel('Frequency X')
        axes[0, 1].set_ylabel('Frequency Y')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Radial profile
        axes[1, 0].plot(features['frequencies'], features['radial_profile'], 
                       linewidth=2, color='blue')
        axes[1, 0].set_xlabel('Frequency (bins)')
        axes[1, 0].set_ylabel('Average Magnitude')
        axes[1, 0].set_title('Radial Frequency Profile')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Feature summary
        feature_text = (
            f"High-Freq Energy Ratio: {features['high_freq_energy_ratio']:.4f}\n"
            f"Peak Count: {features['peak_count']}\n"
            f"Spectral Slope: {features['spectral_slope']:.4f}\n"
            f"Energy Low/Mid/High: "
            f"{features['energy_low_ratio']:.3f}/"
            f"{features['energy_mid_ratio']:.3f}/"
            f"{features['energy_high_ratio']:.3f}"
        )
        axes[1, 1].text(0.1, 0.5, feature_text, fontsize=11, verticalalignment='center',
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved DFT visualization to {save_path}")
        
        return fig