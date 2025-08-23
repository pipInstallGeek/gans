# scripts/download_data.py
"""
Script to pre-download datasets
"""
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import subprocess
import sys

def download_datasets(dataset=None):
    """Download all required datasets"""
    data_root = "data"
    os.makedirs(data_root, exist_ok=True)
    
    if dataset is None or dataset == "mnist":
        print("Downloading MNIST...")
        torchvision.datasets.MNIST(
            root=data_root, train=True, download=True,
            transform=transforms.ToTensor()
        )
    
    if dataset is None or dataset == "cifar10":
        print("Downloading CIFAR-10...")
        torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True,
            transform=transforms.ToTensor()
        )
    
    if dataset is None or dataset == "celeba":
        print("Attempting to download CelebA...")
        try:
            torchvision.datasets.CelebA(
                root=data_root, split='train', download=True,
                transform=transforms.ToTensor()
            )
            print("CelebA downloaded successfully")
        except Exception as e:
            print(f"CelebA download failed: {e}")
            print("You may need to manually download CelebA from the official source")
    
    if dataset is None or dataset == "celeba-hq":
        print("For CelebA-HQ dataset, please use the dedicated download script:")
        print("python scripts/download_celeba_hq.py --resolution 256")
        
        # Ask if user wants to download CelebA-HQ now
        user_input = input("Do you want to download CelebA-HQ now? (y/n): ")
        if user_input.lower() == 'y':
            # Run the CelebA-HQ download script
            try:
                resolution = input("Enter resolution for CelebA-HQ images (default: 256): ") or "256"
                print(f"Downloading CelebA-HQ at {resolution}x{resolution} resolution...")
                download_script = os.path.join(os.path.dirname(__file__), "download_celeba_hq.py")
                subprocess.call([sys.executable, download_script, "--resolution", resolution])
            except Exception as e:
                print(f"Error running CelebA-HQ download script: {e}")
    
    print("Dataset download complete!")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download datasets for GAN training')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['mnist', 'cifar10', 'celeba', 'celeba-hq'],
                        help='Specific dataset to download (default: all)')
    
    args = parser.parse_args()
    download_datasets(args.dataset)