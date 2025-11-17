"""
Main execution script for manual detection methods
"""

from download_kaggle_data import download_cifake_kaggle
download_cifake_kaggle("./data")
import os
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from utils import load_image, load_dataset, create_output_dirs
from preprocessing import ImagePreprocessor
from dft_analyzer import DFTAnalyzer
from dct_analyzer import DCTAnalyzer
from color_analyzer import ColorAnalyzer
from variance_analyzer import VarianceAnalyzer
import matplotlib.pyplot as plt




def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def analyze_image(image_array, filename, config, preprocessor, label):
    """
    Analyze single image using all methods
    
    Args:
        image_array (np.array): Image data
        filename (str): Image filename
        config (dict): Configuration
        preprocessor (ImagePreprocessor): Preprocessor instance
        label (str): 'real' or 'fake'
        
    Returns:
        dict: All features for this image
    """
    # Preprocess
    image_preprocessed = preprocessor.preprocess(image_array)
    
    # Initialize analyzers
    dft_analyzer = DFTAnalyzer(config, preprocessor)
    dct_analyzer = DCTAnalyzer(config, preprocessor)
    color_analyzer = ColorAnalyzer(config, preprocessor)
    variance_analyzer = VarianceAnalyzer(config, preprocessor)
    
    result = {
        'filename': filename,
        'label': label,
        'image_shape': image_array.shape,
    }
    
    # DFT Analysis
    if config['methods']['dft']['enabled']:
        dft_features = dft_analyzer.extract_features(image_preprocessed)
        result.update({f'dft_{k}': v for k, v in dft_features.items() 
                      if k not in ['radial_profile', 'frequencies', 'magnitude_log']})
    
    # DCT Analysis
    if config['methods']['dct']['enabled']:
        dct_features = dct_analyzer.extract_block_features(image_preprocessed)
        result.update({f'dct_{k}': v for k, v in dct_features.items() 
                      if k != 'dct_block'})
    
    # Color Analysis
    if config['methods']['color']['enabled']:
        color_features = color_analyzer.extract_all_color_features(image_preprocessed)
        result.update({f'color_{k}': v for k, v in color_features.items()})
    
    # Variance Analysis
    if config['methods']['variance']['enabled']:
        var_features = variance_analyzer.extract_features(image_preprocessed)
        result.update({f'var_{k}': v for k, v in var_features.items() 
                      if k not in ['noise_residual', 'denoised']})
    
    return result


def process_dataset(config):
    """
    Process entire dataset and extract all features
    
    Args:
        config (dict): Configuration
        
    Returns:
        pd.DataFrame: Features dataframe with columns for all extracted features
    """
    print("="*70)
    print("MANUAL IMAGE DETECTION - FEATURE EXTRACTION")
    print("="*70)
    
    # Setup
    create_output_dirs(config)
    preprocessor = ImagePreprocessor(config)
    np.random.seed(config['project']['seed'])
    
    # Load dataset
    print("\nLoading dataset...")
    real_images, fake_images = load_dataset(config)
    
    if len(real_images) == 0 or len(fake_images) == 0:
        print("Error: Dataset not found or empty. Please check data directory.")
        return None
    
    # Extract features
    all_results = []
    
    print("\nExtracting features from REAL images...")
    for image_array, filename in tqdm(real_images):
        try:
            result = analyze_image(image_array, filename, config, preprocessor, 'real')
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print("\nExtracting features from FAKE images...")
    for image_array, filename in tqdm(fake_images):
        try:
            result = analyze_image(image_array, filename, config, preprocessor, 'fake')
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Create dataframe
    df = pd.DataFrame(all_results)
    
    return df


def save_results(df, config):
    """Save features dataframe to CSV"""
    output_path = os.path.join(config['project']['output_dir'], 'features', 
                               'all_features.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nFeatures saved to: {output_path}")


def print_statistics(df):
    """Print statistical comparison between real and fake"""
    print("\n" + "="*70)
    print("FEATURE STATISTICS COMPARISON")
    print("="*70)
    
    real_df = df[df['label'] == 'real']
    fake_df = df[df['label'] == 'fake']
    
    # Get numeric columns (exclude filename, label, shape)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\nDataset: {len(real_df)} real images, {len(fake_df)} fake images")
    print("\nTop distinguishing features (highest variance between real and fake):\n")
    
    stats = []
    for col in numeric_cols:
        real_mean = real_df[col].mean()
        fake_mean = fake_df[col].mean()
        
        if pd.notna(real_mean) and pd.notna(fake_mean):
            # Compute effect size (simple difference ratio)
            max_val = max(abs(real_mean), abs(fake_mean))
            if max_val > 1e-10:
                diff_ratio = abs(real_mean - fake_mean) / max_val
                stats.append((col, real_mean, fake_mean, diff_ratio))
    
    # Sort by difference
    stats.sort(key=lambda x: x[3], reverse=True)
    
    # Print top 15
    for i, (col, real_mean, fake_mean, diff) in enumerate(stats[:15]):
        print(f"{i+1:2d}. {col:40s} | Real: {real_mean:8.4f} | Fake: {fake_mean:8.4f} | Diff: {diff:.4f}")


def visualize_sample_images(df, config, real_images, fake_images):
    """Create visualizations for sample images - 10 real and 10 fake"""
    print("\n" + "="*70)
    print("CREATING SAMPLE VISUALIZATIONS (10 REAL + 10 FAKE)")
    print("="*70)
    
    preprocessor = ImagePreprocessor(config)
    dft_analyzer = DFTAnalyzer(config, preprocessor)
    dct_analyzer = DCTAnalyzer(config, preprocessor)
    color_analyzer = ColorAnalyzer(config, preprocessor)
    variance_analyzer = VarianceAnalyzer(config, preprocessor)
    
    # Select 10 samples from each group
    num_samples = min(10, len(real_images), len(fake_images))
    sample_real_images = real_images[:num_samples]
    sample_fake_images = fake_images[:num_samples]
    
    print(f"Processing {num_samples} real and {num_samples} fake images...")
    
    # Extract features for all samples
    real_features = {'dft': [], 'dct': [], 'color': [], 'variance': []}
    fake_features = {'dft': [], 'dct': [], 'color': [], 'variance': []}
    
    # Process real images
    print("Extracting features from real images...")
    for img_array, filename in tqdm(sample_real_images):
        img_preprocessed = preprocessor.preprocess(img_array)
        
        if config['methods']['dft']['enabled']:
            real_features['dft'].append(dft_analyzer.extract_features(img_preprocessed))
        if config['methods']['dct']['enabled']:
            real_features['dct'].append(dct_analyzer.extract_block_features(img_preprocessed))
        if config['methods']['color']['enabled']:
            real_features['color'].append(color_analyzer.extract_all_color_features(img_preprocessed))
        if config['methods']['variance']['enabled']:
            real_features['variance'].append(variance_analyzer.extract_features(img_preprocessed))
    
    # Process fake images
    print("Extracting features from fake images...")
    for img_array, filename in tqdm(sample_fake_images):
        img_preprocessed = preprocessor.preprocess(img_array)
        
        if config['methods']['dft']['enabled']:
            fake_features['dft'].append(dft_analyzer.extract_features(img_preprocessed))
        if config['methods']['dct']['enabled']:
            fake_features['dct'].append(dct_analyzer.extract_block_features(img_preprocessed))
        if config['methods']['color']['enabled']:
            fake_features['color'].append(color_analyzer.extract_all_color_features(img_preprocessed))
        if config['methods']['variance']['enabled']:
            fake_features['variance'].append(variance_analyzer.extract_features(img_preprocessed))
    
    # Create aggregate visualizations
    viz_dir = os.path.join(config['project']['output_dir'], 'visualizations')
    
    # DFT comparison
    if config['methods']['dft']['enabled']:
        create_dft_comparison(real_features['dft'], fake_features['dft'], 
                             os.path.join(viz_dir, 'dft_comparison.png'))
    
    # DCT comparison
    if config['methods']['dct']['enabled']:
        create_dct_comparison(real_features['dct'], fake_features['dct'],
                             os.path.join(viz_dir, 'dct_comparison.png'))
    
    # Color comparison
    if config['methods']['color']['enabled']:
        create_color_comparison(real_features['color'], fake_features['color'],
                               os.path.join(viz_dir, 'color_comparison.png'))
    
    # Variance comparison
    if config['methods']['variance']['enabled']:
        create_variance_comparison(real_features['variance'], fake_features['variance'],
                                  os.path.join(viz_dir, 'variance_comparison.png'))
    
    print("Comparison visualizations saved!")


def create_dft_comparison(real_features, fake_features, save_path):
    """Create side-by-side DFT comparison for real vs fake"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('DFT Analysis: Real vs Fake Images (10 samples each)', 
                 fontsize=16, fontweight='bold')
    
    # Extract radial profiles
    real_profiles = [f['radial_profile'] for f in real_features]
    fake_profiles = [f['radial_profile'] for f in fake_features]
    
    # Get frequencies (same for all)
    freqs = real_features[0]['frequencies']
    
    # Plot 1: All real radial profiles
    for profile in real_profiles:
        axes[0, 0].plot(freqs, profile, alpha=0.3, color='blue')
    axes[0, 0].plot(freqs, np.mean(real_profiles, axis=0), 
                   color='darkblue', linewidth=3, label='Mean')
    axes[0, 0].set_title('Real Images - Radial Profiles')
    axes[0, 0].set_xlabel('Frequency (bins)')
    axes[0, 0].set_ylabel('Magnitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: All fake radial profiles
    for profile in fake_profiles:
        axes[1, 0].plot(freqs, profile, alpha=0.3, color='red')
    axes[1, 0].plot(freqs, np.mean(fake_profiles, axis=0), 
                   color='darkred', linewidth=3, label='Mean')
    axes[1, 0].set_title('Fake Images - Radial Profiles')
    axes[1, 0].set_xlabel('Frequency (bins)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 3: Mean comparison
    axes[0, 1].plot(freqs, np.mean(real_profiles, axis=0), 
                   color='blue', linewidth=2, label='Real (mean)')
    axes[0, 1].plot(freqs, np.mean(fake_profiles, axis=0), 
                   color='red', linewidth=2, label='Fake (mean)')
    axes[0, 1].set_title('Mean Radial Profile Comparison')
    axes[0, 1].set_xlabel('Frequency (bins)')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 4: High-freq energy ratio box plot
    real_hf_ratios = [f['high_freq_energy_ratio'] for f in real_features]
    fake_hf_ratios = [f['high_freq_energy_ratio'] for f in fake_features]
    axes[1, 1].boxplot([real_hf_ratios, fake_hf_ratios], 
                       labels=['Real', 'Fake'])
    axes[1, 1].set_title('High-Frequency Energy Ratio')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Peak count comparison
    real_peaks = [f['peak_count'] for f in real_features]
    fake_peaks = [f['peak_count'] for f in fake_features]
    axes[0, 2].boxplot([real_peaks, fake_peaks], 
                       labels=['Real', 'Fake'])
    axes[0, 2].set_title('Peak Count')
    axes[0, 2].set_ylabel('Number of Peaks')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Spectral slope comparison
    real_slopes = [f['spectral_slope'] for f in real_features]
    fake_slopes = [f['spectral_slope'] for f in fake_features]
    axes[1, 2].boxplot([real_slopes, fake_slopes], 
                       labels=['Real', 'Fake'])
    axes[1, 2].set_title('Spectral Slope')
    axes[1, 2].set_ylabel('Slope Value')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved DFT comparison to {save_path}")


def create_dct_comparison(real_features, fake_features, save_path):
    """Create side-by-side DCT comparison"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('DCT Analysis: Real vs Fake Images (10 samples each)', 
                 fontsize=16, fontweight='bold')
    
    # Mean DCT blocks
    real_blocks = [f['dct_block'] for f in real_features]
    fake_blocks = [f['dct_block'] for f in fake_features]
    
    real_mean_block = np.mean(real_blocks, axis=0)
    fake_mean_block = np.mean(fake_blocks, axis=0)
    
    # Plot 1: Mean real DCT block
    im1 = axes[0, 0].imshow(real_mean_block, cmap='RdBu_r')
    axes[0, 0].set_title('Real Images - Mean DCT Block')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Mean fake DCT block
    im2 = axes[1, 0].imshow(fake_mean_block, cmap='RdBu_r')
    axes[1, 0].set_title('Fake Images - Mean DCT Block')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Plot 3: DCT mean comparison
    real_means = [f['dct_mean'] for f in real_features]
    fake_means = [f['dct_mean'] for f in fake_features]
    axes[0, 1].boxplot([real_means, fake_means], labels=['Real', 'Fake'])
    axes[0, 1].set_title('DCT Mean')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: DCT std comparison
    real_stds = [f['dct_std'] for f in real_features]
    fake_stds = [f['dct_std'] for f in fake_features]
    axes[1, 1].boxplot([real_stds, fake_stds], labels=['Real', 'Fake'])
    axes[1, 1].set_title('DCT Standard Deviation')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 5: AC Energy comparison
    real_ac = [f['ac_energy'] for f in real_features]
    fake_ac = [f['ac_energy'] for f in fake_features]
    axes[0, 2].boxplot([real_ac, fake_ac], labels=['Real', 'Fake'])
    axes[0, 2].set_title('AC Energy')
    axes[0, 2].set_ylabel('Value')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Skewness comparison
    real_skew = [f['dct_skewness'] for f in real_features]
    fake_skew = [f['dct_skewness'] for f in fake_features]
    axes[1, 2].boxplot([real_skew, fake_skew], labels=['Real', 'Fake'])
    axes[1, 2].set_title('DCT Skewness')
    axes[1, 2].set_ylabel('Value')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved DCT comparison to {save_path}")


def create_color_comparison(real_features, fake_features, save_path):
    """Create side-by-side color comparison"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Color Analysis: Real vs Fake Images (10 samples each)', 
                 fontsize=16, fontweight='bold')
    
    # RGB means
    real_r = [f['r_mean'] for f in real_features]
    real_g = [f['g_mean'] for f in real_features]
    real_b = [f['b_mean'] for f in real_features]
    
    fake_r = [f['r_mean'] for f in fake_features]
    fake_g = [f['g_mean'] for f in fake_features]
    fake_b = [f['b_mean'] for f in fake_features]
    
    # Plot 1: RGB means for real
    axes[0, 0].boxplot([real_r, real_g, real_b], 
                       labels=['R', 'G', 'B'])
    axes[0, 0].set_title('Real Images - RGB Channel Means')
    axes[0, 0].set_ylabel('Mean Value')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: RGB means for fake
    axes[1, 0].boxplot([fake_r, fake_g, fake_b], 
                       labels=['R', 'G', 'B'])
    axes[1, 0].set_title('Fake Images - RGB Channel Means')
    axes[1, 0].set_ylabel('Mean Value')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: R channel comparison
    axes[0, 1].boxplot([real_r, fake_r], labels=['Real', 'Fake'])
    axes[0, 1].set_title('Red Channel Mean')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: G channel comparison
    axes[1, 1].boxplot([real_g, fake_g], labels=['Real', 'Fake'])
    axes[1, 1].set_title('Green Channel Mean')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 5: B channel comparison
    axes[0, 2].boxplot([real_b, fake_b], labels=['Real', 'Fake'])
    axes[0, 2].set_title('Blue Channel Mean')
    axes[0, 2].set_ylabel('Value')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # Plot 6: RG correlation comparison
    real_rg_corr = [f['rg_correlation'] for f in real_features]
    fake_rg_corr = [f['rg_correlation'] for f in fake_features]
    axes[1, 2].boxplot([real_rg_corr, fake_rg_corr], labels=['Real', 'Fake'])
    axes[1, 2].set_title('R-G Channel Correlation')
    axes[1, 2].set_ylabel('Correlation')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Color comparison to {save_path}")


def create_variance_comparison(real_features, fake_features, save_path):
    """Create side-by-side variance comparison"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Variance Analysis: Real vs Fake Images (10 samples each)', 
                 fontsize=16, fontweight='bold')
    
    # Overall variance
    real_var = [f['overall_variance'] for f in real_features]
    fake_var = [f['overall_variance'] for f in fake_features]
    axes[0, 0].boxplot([real_var, fake_var], labels=['Real', 'Fake'])
    axes[0, 0].set_title('Overall Noise Variance')
    axes[0, 0].set_ylabel('Variance')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Mean local variance
    real_local = [f['mean_local_variance'] for f in real_features]
    fake_local = [f['mean_local_variance'] for f in fake_features]
    axes[0, 1].boxplot([real_local, fake_local], labels=['Real', 'Fake'])
    axes[0, 1].set_title('Mean Local Variance')
    axes[0, 1].set_ylabel('Variance')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Variance of variances
    real_var_of_var = [f['variance_of_variances'] for f in real_features]
    fake_var_of_var = [f['variance_of_variances'] for f in fake_features]
    axes[1, 0].boxplot([real_var_of_var, fake_var_of_var], labels=['Real', 'Fake'])
    axes[1, 0].set_title('Variance of Local Variances')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Noise high-freq ratio
    real_noise_hf = [f['noise_high_freq_ratio'] for f in real_features]
    fake_noise_hf = [f['noise_high_freq_ratio'] for f in fake_features]
    axes[1, 1].boxplot([real_noise_hf, fake_noise_hf], labels=['Real', 'Fake'])
    axes[1, 1].set_title('Noise High-Frequency Ratio')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Variance comparison to {save_path}")

def visualize_individual_images(df, config, real_images, fake_images):
    """
    Create individual detailed plots for each image analyzed.
    Each image gets 4 plots: one for each method (DFT, DCT, Color, Variance)
    """
    print("\n" + "="*70)
    print("CREATING INDIVIDUAL IMAGE VISUALIZATIONS")
    print("="*70)
    
    preprocessor = ImagePreprocessor(config)
    dft_analyzer = DFTAnalyzer(config, preprocessor)
    dct_analyzer = DCTAnalyzer(config, preprocessor)
    color_analyzer = ColorAnalyzer(config, preprocessor)
    variance_analyzer = VarianceAnalyzer(config, preprocessor)
    
    # Select 10 samples from each group
    num_samples = min(10, len(real_images), len(fake_images))
    sample_real_images = real_images[:num_samples]
    sample_fake_images = fake_images[:num_samples]
    
    viz_dir = os.path.join(config['project']['output_dir'], 'visualizations', 'individual')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Process real images
    print(f"\nGenerating individual plots for {num_samples} real images...")
    for idx, (img_array, filename) in enumerate(tqdm(sample_real_images)):
        img_preprocessed = preprocessor.preprocess(img_array)
        base_name = f"real_{idx+1:02d}_{filename.split('.')[0]}"
        
        # DFT individual plot
        if config['methods']['dft']['enabled']:
            dft_feat = dft_analyzer.extract_features(img_preprocessed)
            save_path = os.path.join(viz_dir, f"{base_name}_dft.png")
            dft_analyzer.visualize(img_preprocessed, dft_feat, 
                                  title=f"(Real #{idx+1})", save_path=save_path)
            plt.close()
        
        # DCT individual plot
        if config['methods']['dct']['enabled']:
            dct_feat = dct_analyzer.extract_block_features(img_preprocessed)
            save_path = os.path.join(viz_dir, f"{base_name}_dct.png")
            dct_analyzer.visualize(img_preprocessed, dct_feat,
                                  title=f"(Real #{idx+1})", save_path=save_path)
            plt.close()
        
        # Color individual plot
        if config['methods']['color']['enabled']:
            color_feat = color_analyzer.extract_all_color_features(img_preprocessed)
            save_path = os.path.join(viz_dir, f"{base_name}_color.png")
            color_analyzer.visualize(img_preprocessed, color_feat,
                                    title=f"(Real #{idx+1})", save_path=save_path)
            plt.close()
        
        # Variance individual plot
        if config['methods']['variance']['enabled']:
            var_feat = variance_analyzer.extract_features(img_preprocessed)
            save_path = os.path.join(viz_dir, f"{base_name}_variance.png")
            variance_analyzer.visualize(img_preprocessed, var_feat,
                                       title=f"(Real #{idx+1})", save_path=save_path)
            plt.close()
    
    # Process fake images
    print(f"\nGenerating individual plots for {num_samples} fake images...")
    for idx, (img_array, filename) in enumerate(tqdm(sample_fake_images)):
        img_preprocessed = preprocessor.preprocess(img_array)
        base_name = f"fake_{idx+1:02d}_{filename.split('.')[0]}"
        
        # DFT individual plot
        if config['methods']['dft']['enabled']:
            dft_feat = dft_analyzer.extract_features(img_preprocessed)
            save_path = os.path.join(viz_dir, f"{base_name}_dft.png")
            dft_analyzer.visualize(img_preprocessed, dft_feat, 
                                  title=f"(Fake #{idx+1})", save_path=save_path)
            plt.close()
        
        # DCT individual plot
        if config['methods']['dct']['enabled']:
            dct_feat = dct_analyzer.extract_block_features(img_preprocessed)
            save_path = os.path.join(viz_dir, f"{base_name}_dct.png")
            dct_analyzer.visualize(img_preprocessed, dct_feat,
                                  title=f"(Fake #{idx+1})", save_path=save_path)
            plt.close()
        
        # Color individual plot
        if config['methods']['color']['enabled']:
            color_feat = color_analyzer.extract_all_color_features(img_preprocessed)
            save_path = os.path.join(viz_dir, f"{base_name}_color.png")
            color_analyzer.visualize(img_preprocessed, color_feat,
                                    title=f"(Fake #{idx+1})", save_path=save_path)
            plt.close()
        
        # Variance individual plot
        if config['methods']['variance']['enabled']:
            var_feat = variance_analyzer.extract_features(img_preprocessed)
            save_path = os.path.join(viz_dir, f"{base_name}_variance.png")
            variance_analyzer.visualize(img_preprocessed, var_feat,
                                       title=f"(Fake #{idx+1})", save_path=save_path)
            plt.close()
    
    print(f"\nIndividual image plots saved to: {viz_dir}")


def create_summary_report(config, df):
    """
    Create a summary report with key statistics
    Useful for quick reference when writing thesis
    """
    print("\n" + "="*70)
    print("GENERATING SUMMARY REPORT")
    print("="*70)
    
    report_path = os.path.join(config['project']['output_dir'], 'analysis_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("AI-GENERATED IMAGE DETECTION - MANUAL ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Dataset: CIFAKE\n")
        f.write(f"Total Images Analyzed: {len(df)}\n")
        f.write(f"Real Images: {len(df[df['label'] == 'real'])}\n")
        f.write(f"Fake Images: {len(df[df['label'] == 'fake'])}\n\n")
        
        real_df = df[df['label'] == 'real']
        fake_df = df[df['label'] == 'fake']
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        f.write("="*70 + "\n")
        f.write("KEY FEATURES COMPARISON\n")
        f.write("="*70 + "\n\n")
        
        stats = []
        for col in numeric_cols:
            real_mean = real_df[col].mean()
            fake_mean = fake_df[col].mean()
            real_std = real_df[col].std()
            fake_std = fake_df[col].std()
            
            if pd.notna(real_mean) and pd.notna(fake_mean):
                max_val = max(abs(real_mean), abs(fake_mean))
                if max_val > 1e-10:
                    diff_ratio = abs(real_mean - fake_mean) / max_val
                    stats.append((col, real_mean, real_std, fake_mean, fake_std, diff_ratio))
        
        # Sort by difference
        stats.sort(key=lambda x: x[5], reverse=True)
        
        # Print top 20 distinguishing features
        f.write("Top 20 Features that Distinguish Real from Fake:\n\n")
        f.write(f"{'Rank':<5} {'Feature':<50} {'Real Mean':<12} {'Fake Mean':<12} {'Difference':<12}\n")
        f.write("-" * 91 + "\n")
        
        for i, (col, real_mean, real_std, fake_mean, fake_std, diff) in enumerate(stats[:20]):
            f.write(f"{i+1:<5} {col:<50} {real_mean:<12.6f} {fake_mean:<12.6f} {diff:<12.6f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("DETAILED STATISTICS BY METHOD\n")
        f.write("="*70 + "\n\n")
        
        # DFT statistics
        dft_cols = [col for col in numeric_cols if col.startswith('dft_')]
        if dft_cols:
            f.write("DFT ANALYSIS:\n")
            f.write("-" * 70 + "\n")
            for col in dft_cols:
                real_val = real_df[col].mean()
                fake_val = fake_df[col].mean()
                f.write(f"  {col:<45} | Real: {real_val:8.4f} | Fake: {fake_val:8.4f}\n")
            f.write("\n")
        
        # DCT statistics
        dct_cols = [col for col in numeric_cols if col.startswith('dct_')]
        if dct_cols:
            f.write("DCT ANALYSIS:\n")
            f.write("-" * 70 + "\n")
            for col in dct_cols:
                real_val = real_df[col].mean()
                fake_val = fake_df[col].mean()
                f.write(f"  {col:<45} | Real: {real_val:8.4f} | Fake: {fake_val:8.4f}\n")
            f.write("\n")
        
        # Color statistics
        color_cols = [col for col in numeric_cols if col.startswith('color_')]
        if color_cols:
            f.write("COLOR ANALYSIS:\n")
            f.write("-" * 70 + "\n")
            for col in color_cols:
                real_val = real_df[col].mean()
                fake_val = fake_df[col].mean()
                f.write(f"  {col:<45} | Real: {real_val:8.4f} | Fake: {fake_val:8.4f}\n")
            f.write("\n")
        
        # Variance statistics
        var_cols = [col for col in numeric_cols if col.startswith('var_')]
        if var_cols:
            f.write("VARIANCE ANALYSIS:\n")
            f.write("-" * 70 + "\n")
            for col in var_cols:
                real_val = real_df[col].mean()
                fake_val = fake_df[col].mean()
                f.write(f"  {col:<45} | Real: {real_val:8.4f} | Fake: {fake_val:8.4f}\n")
            f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*70 + "\n\n")

def main():
    """Main execution"""
    # Load configuration
    config = load_config('config.yaml')
    
    # Process dataset
    df = process_dataset(config)
    
    if df is None or len(df) == 0:
        print("No data processed. Exiting.")
        return
    
    # Save results
    save_results(df, config)
    
    # Print statistics
    print_statistics(df)
    
    # Create visualizations
    real_images, fake_images = load_dataset(config)
    
    # Create comparison plots (10 real + 10 fake aggregated)
    visualize_sample_images(df, config, real_images, fake_images)
    
    # Create individual plots for each image
    visualize_individual_images(df, config, real_images, fake_images)
    
    # Create summary report
    create_summary_report(config, df)
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Results saved to: {config['project']['output_dir']}")
    print(f"\nOutput structure:")
    print(f"  - features/all_features.csv (all extracted features)")
    print(f"  - visualizations/dft_comparison.png (aggregate comparison)")
    print(f"  - visualizations/dct_comparison.png")
    print(f"  - visualizations/color_comparison.png")
    print(f"  - visualizations/variance_comparison.png")
    print(f"  - visualizations/individual/ (individual image plots)")
    print(f"  - analysis_summary.txt (statistical summary)")


if __name__ == '__main__':
    main()
