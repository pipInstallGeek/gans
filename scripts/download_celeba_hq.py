#!/usr/bin/env python
# scripts/download_celeba_hq.py
"""
Download script for CelebA-HQ dataset

This script downloads the CelebA-HQ dataset and prepares it for use with the GAN models.
CelebA-HQ is a high-quality version of CelebA with 30,000 images at 1024x1024 resolution.
"""

import os
import argparse
import urllib.request
import zipfile
import glob
from tqdm import tqdm
import subprocess
import sys
from PIL import Image
import numpy as np

# Add progress bar for downloads
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download URL with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, 
                                  reporthook=t.update_to)

def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = ['tqdm', 'pillow', 'numpy']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def download_celeba_hq(output_dir="data/celeba-hq", resolution=256):
    """Download and prepare CelebA-HQ dataset"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if dataset already exists
    existing_files = glob.glob(os.path.join(output_dir, "*.jpg")) + glob.glob(os.path.join(output_dir, "*.png"))
    if len(existing_files) > 0:
        print(f"Found {len(existing_files)} existing images in {output_dir}")
        user_input = input("Dataset already exists. Do you want to re-download? (y/n): ")
        if user_input.lower() != 'y':
            print("Skipping download.")
            return
    
    print("Downloading CelebA-HQ dataset...")
    
    # Option 1: Download from Kaggle (requires Kaggle account)
    try:
        print("Attempting to download from Kaggle...")
        # Check if kaggle is installed
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        except Exception:
            print("Failed to install Kaggle API. Skipping Kaggle download method.")
            raise ImportError("Kaggle API installation failed")
        
        # Import after installation
        try:
            import kaggle
        except ImportError:
            raise ImportError("Kaggle API import failed after installation")
        
        # Download from Kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('lamsimon/celebahq', path='./temp_download', unzip=True)
        print("Download successful!")
        
        # Move files to output directory
        files = glob.glob('./temp_download/*.jpg') + glob.glob('./temp_download/*.png')
        print(f"Moving {len(files)} files to {output_dir}...")
        for i, file in enumerate(files):
            img = Image.open(file)
            # Resize if needed
            if resolution != 1024:
                img = img.resize((resolution, resolution), Image.Resampling.BICUBIC)
            # Save with consistent naming
            img.save(os.path.join(output_dir, f"{i}.jpg"))
        
        # Cleanup
        import shutil
        shutil.rmtree('./temp_download')
        
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("Falling back to alternative download source...")
        
        # Option 2: Alternative direct download
        # This is a smaller subset (~5000 images) from an academic source
        try:
            zip_path = "./celeba_hq_download.zip"
            # URL for a smaller subset of CelebA-HQ (for educational purposes)
            download_url("https://drive.google.com/uc?export=download&id=1R72NB79CX0MuMMc-n3KezuvFVKnW5iJD", zip_path)
            
            # Extract zip file
            print("Extracting ZIP file...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("./temp_extract")
            
            # Process and resize images
            files = glob.glob('./temp_extract/**/*.jpg', recursive=True) + glob.glob('./temp_extract/**/*.png', recursive=True)
            print(f"Processing {len(files)} images...")
            for i, file in enumerate(tqdm(files)):
                try:
                    img = Image.open(file)
                    # Resize to requested resolution
                    img = img.resize((resolution, resolution), Image.Resampling.BICUBIC)
                    # Save with consistent naming
                    img.save(os.path.join(output_dir, f"{i}.jpg"))
                except Exception as img_error:
                    print(f"Error processing {file}: {img_error}")
            
            # Cleanup
            import shutil
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if os.path.exists("./temp_extract"):
                shutil.rmtree("./temp_extract")
            
            print(f"Successfully processed {len(glob.glob(os.path.join(output_dir, '*.jpg')))} images.")
            
        except Exception as direct_error:
            print(f"Alternative download failed: {direct_error}")
            print("\nAutomatic download failed. Please manually download CelebA-HQ dataset:")
            print("1. Visit: https://github.com/tkarras/progressive_growing_of_gans")
            print("2. Download the dataset from the link provided there")
            print(f"3. Extract and place the images in {output_dir}")
            print("4. Ensure the images are in jpg format with names like 0.jpg, 1.jpg, etc.")
    
    # Verify the dataset
    files = glob.glob(os.path.join(output_dir, "*.jpg"))
    if len(files) > 0:
        print(f"✅ Successfully downloaded {len(files)} CelebA-HQ images to {output_dir}")
        print(f"Images resized to {resolution}x{resolution}")
    else:
        print("❌ Download failed. No images found in the output directory.")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download CelebA-HQ dataset')
    parser.add_argument('--output_dir', type=str, default="data/celeba-hq",
                        help='Output directory for the dataset')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Resolution to resize images to (default: 256)')
    
    args = parser.parse_args()
    
    # Check and install dependencies
    check_and_install_dependencies()
    
    # Download the dataset
    download_celeba_hq(args.output_dir, args.resolution)
