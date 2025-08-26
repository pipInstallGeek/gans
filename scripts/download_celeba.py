#!/usr/bin/env python
# scripts/download_celeba.py
"""
Dedicated script for downloading CelebA dataset
This script provides more robust download functionality for CelebA
"""

import os
import sys
import time
import requests
import zipfile
import tarfile
import shutil
import subprocess
from tqdm import tqdm
import torchvision

def download_file(url, destination, chunk_size=8192):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def download_celeba(output_dir='/tmp/data/celeba', method='torchvision'):
    """
    Download CelebA dataset using multiple methods for better reliability
    
    Args:
        output_dir: Directory to save the dataset
        method: Download method ('torchvision', 'direct', or 'all')
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading CelebA dataset to {output_dir}")
    
    if method in ['torchvision', 'all']:
        print("\nMethod 1: Using torchvision.datasets.CelebA")
        try:
            print("Downloading CelebA using torchvision...")
            # Try to download with torchvision
            torchvision.datasets.CelebA(
                root='/tmp/data',
                split='train',
                download=True,
                transform=None
            )
            print("✅ CelebA dataset downloaded successfully using torchvision")
            return True
        except Exception as e:
            print(f"❌ Error downloading CelebA via torchvision: {e}")
            if method == 'torchvision':
                return False
    
    if method in ['direct', 'all']:
        print("\nMethod 2: Direct download from official source")
        try:
            # Direct download from official URL
            img_url = "https://mmlab.ie.cuhk.edu.hk/projects/CelebA/img_align_celeba.zip"
            zip_path = os.path.join(output_dir, "img_align_celeba.zip")
            
            print(f"Downloading from {img_url}")
            print("This may take a while (>1GB download)...")
            
            try:
                download_file(img_url, zip_path)
            except Exception as e:
                print(f"❌ Direct download failed: {e}")
                return False
            
            # Extract zip file
            print("Extracting zip file...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            # Clean up
            os.remove(zip_path)
            print("✅ CelebA dataset downloaded and extracted successfully")
            return True
        except Exception as e:
            print(f"❌ Error in direct download: {e}")
            return False
    
    return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Download CelebA dataset')
    parser.add_argument('--method', type=str, default='all',
                        choices=['torchvision', 'direct', 'all'],
                        help='Download method to use')
    parser.add_argument('--output', type=str, default='/tmp/data/celeba',
                        help='Directory to save the dataset')
    
    args = parser.parse_args()
    
    success = download_celeba(args.output, args.method)
    
    if success:
        print("\n" + "="*80)
        print("✅ CelebA download completed successfully!")
        print("="*80)
        print(f"Dataset location: {args.output}")
        print(f"You can now train your GAN models on CelebA.")
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("❌ CelebA download failed")
        print("="*80)
        print("Please try the following:")
        print("1. Check your internet connection")
        print("2. Try downloading with a different method:")
        print("   python scripts/download_celeba.py --method direct")
        print("3. Download manually from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
        print("   and extract to /tmp/data/celeba")
        sys.exit(1)
