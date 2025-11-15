"""
Main execution script for manual detection methods
"""
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
    """Create visualizations for sample images"""
    print("\n" + "="*70)
    print("CREATING SAMPLE VISUALIZATIONS")
    print("="*70)
    
    preprocessor = ImagePreprocessor(config)
    dft_analyzer = DFTAnalyzer(config, preprocessor)
    dct_analyzer = DCTAnalyzer(config, preprocessor)
    color_analyzer = ColorAnalyzer(config, preprocessor)
    variance_analyzer = VarianceAnalyzer(config, preprocessor)
    
    # Select one sample real and one fake
    if len(real_images) > 0:
        sample_real = real_images[0]
        img_real = sample_real[0]
        filename_real = sample_real[1]
        
        img_preprocessed = preprocessor.preprocess(img_real)
        
        # DFT
        if config['methods']['dft']['enabled']:
            dft_feat = dft_analyzer.extract_features(img_preprocessed)
            save_path = os.path.join(config['project']['output_dir'], 'visualizations', 
                                    f'dft_real_{filename_real.split(".")[0]}.png')
            dft_analyzer.visualize(img_preprocessed, dft_feat, 
                                  title="(Real Sample)", save_path=save_path)
        
        # DCT
        if config['methods']['dct']['enabled']:
            dct_feat = dct_analyzer.extract_block_features(img_preprocessed)
            save_path = os.path.join(config['project']['output_dir'], 'visualizations',
                                    f'dct_real_{filename_real.split(".")[0]}.png')
            dct_analyzer.visualize(img_preprocessed, dct_feat,
                                  title="(Real Sample)", save_path=save_path)
        
        # Color
        if config['methods']['color']['enabled']:
            color_feat = color_analyzer.extract_all_color_features(img_preprocessed)
            save_path = os.path.join(config['project']['output_dir'], 'visualizations',
                                    f'color_real_{filename_real.split(".")[0]}.png')
            color_analyzer.visualize(img_preprocessed, color_feat,
                                    title="(Real Sample)", save_path=save_path)
        
        # Variance
        if config['methods']['variance']['enabled']:
            var_feat = variance_analyzer.extract_features(img_preprocessed)
            save_path = os.path.join(config['project']['output_dir'], 'visualizations',
                                    f'variance_real_{filename_real.split(".")[0]}.png')
            variance_analyzer.visualize(img_preprocessed, var_feat,
                                       title="(Real Sample)", save_path=save_path)
    
    # Fake samples
    if len(fake_images) > 0:
        sample_fake = fake_images[0]
        img_fake = sample_fake[0]
        filename_fake = sample_fake[1]
        
        img_preprocessed = preprocessor.preprocess(img_fake)
        
        # DFT
        if config['methods']['dft']['enabled']:
            dft_feat = dft_analyzer.extract_features(img_preprocessed)
            save_path = os.path.join(config['project']['output_dir'], 'visualizations',
                                    f'dft_fake_{filename_fake.split(".")[0]}.png')
            dft_analyzer.visualize(img_preprocessed, dft_feat,
                                  title="(Fake Sample)", save_path=save_path)
        
        # DCT
        if config['methods']['dct']['enabled']:
            dct_feat = dct_analyzer.extract_block_features(img_preprocessed)
            save_path = os.path.join(config['project']['output_dir'], 'visualizations',
                                    f'dct_fake_{filename_fake.split(".")[0]}.png')
            dct_analyzer.visualize(img_preprocessed, dct_feat,
                                  title="(Fake Sample)", save_path=save_path)
        
        # Color
        if config['methods']['color']['enabled']:
            color_feat = color_analyzer.extract_all_color_features(img_preprocessed)
            save_path = os.path.join(config['project']['output_dir'], 'visualizations',
                                    f'color_fake_{filename_fake.split(".")[0]}.png')
            color_analyzer.visualize(img_preprocessed, color_feat,
                                    title="(Fake Sample)", save_path=save_path)
        
        # Variance
        if config['methods']['variance']['enabled']:
            var_feat = variance_analyzer.extract_features(img_preprocessed)
            save_path = os.path.join(config['project']['output_dir'], 'visualizations',
                                    f'variance_fake_{filename_fake.split(".")[0]}.png')
            variance_analyzer.visualize(img_preprocessed, var_feat,
                                       title="(Fake Sample)", save_path=save_path)
    
    print("Visualizations saved!")


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
    visualize_sample_images(df, config, real_images, fake_images)
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Results saved to: {config['project']['output_dir']}")


if __name__ == '__main__':
    main()