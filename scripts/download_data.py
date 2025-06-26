# scripts/download_data.py
"""
Script to pre-download datasets
"""
import torchvision
import torchvision.transforms as transforms
import os

def download_datasets():
    """Download all required datasets"""
    data_root = "data"
    os.makedirs(data_root, exist_ok=True)
    
    print("Downloading MNIST...")
    torchvision.datasets.MNIST(
        root=data_root, train=True, download=True,
        transform=transforms.ToTensor()
    )
    
    print("Downloading CIFAR-10...")
    torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True,
        transform=transforms.ToTensor()
    )
    
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
    
    print("Dataset download complete!")

if __name__ == "__main__":
    download_datasets()