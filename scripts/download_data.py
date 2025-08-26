# scripts/download_data.py
"""
Script to pre-download datasets
"""
import torchvision
import torchvision.transforms as transforms
import os
import sys
import argparse

# Add the scripts directory to the path for import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def download_datasets(dataset=None):
    """Download all required datasets to ephemeral storage"""
    data_root = "/tmp/data"
    os.makedirs(data_root, exist_ok=True)
    
    if dataset is None or dataset == "mnist":
        print("Downloading MNIST to ephemeral storage...")
        torchvision.datasets.MNIST(
            root=data_root, train=True, download=True,
            transform=transforms.ToTensor()
        )
    
    if dataset is None or dataset == "cifar10":
        print("Downloading CIFAR-10 to ephemeral storage...")
        torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True,
            transform=transforms.ToTensor()
        )
    
    if dataset is None or dataset == "celeba":
        print("Attempting to download CelebA to ephemeral storage...")
        try:
            # Use our dedicated CelebA downloader for better reliability
            from download_celeba import download_celeba
            success = download_celeba('/tmp/data/celeba', 'all')
            
            if not success:
                print("\nAutomatic download failed. For more reliable download, run:")
                print("python scripts/download_celeba.py")
            else:
                print("CelebA downloaded successfully to /tmp/data")
        except Exception as e:
            print(f"CelebA download failed: {e}")
            print("For more reliable download, run: python scripts/download_celeba.py")
    
    print("Dataset download complete! All datasets stored in ephemeral storage (/tmp/data)")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download datasets for GAN training')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['mnist', 'cifar10', 'celeba'],
                        help='Specific dataset to download (default: all)')
    
    args = parser.parse_args()
    download_datasets(args.dataset)