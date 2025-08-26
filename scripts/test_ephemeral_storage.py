#!/usr/bin/env python
# scripts/test_ephemeral_storage.py
"""
Script to test dataset loading from ephemeral storage
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import argparse

def test_dataset(dataset_name):
    """Test loading a dataset from ephemeral storage"""
    data_root = "/tmp/data"
    
    print(f"Testing {dataset_name} dataset in ephemeral storage (/tmp/data)...")
    
    # Check if directory exists
    if not os.path.exists(data_root):
        print(f"Creating data directory: {data_root}")
        os.makedirs(data_root, exist_ok=True)
    
    # Configure transforms
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Try to load dataset
        try:
            start_time = time.time()
            dataset = torchvision.datasets.MNIST(
                root=data_root,
                train=True,
                download=True,
                transform=transform
            )
            load_time = time.time() - start_time
            print(f"✅ Successfully loaded MNIST with {len(dataset)} images in {load_time:.2f} seconds")
        except Exception as e:
            print(f"❌ Error loading MNIST: {e}")
            return False
            
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Try to load dataset
        try:
            start_time = time.time()
            dataset = torchvision.datasets.CIFAR10(
                root=data_root,
                train=True,
                download=True,
                transform=transform
            )
            load_time = time.time() - start_time
            print(f"✅ Successfully loaded CIFAR-10 with {len(dataset)} images in {load_time:.2f} seconds")
        except Exception as e:
            print(f"❌ Error loading CIFAR-10: {e}")
            return False
            
    elif dataset_name == 'celeba':
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Try to load dataset
        try:
            start_time = time.time()
            dataset = torchvision.datasets.CelebA(
                root=data_root,
                split='train',
                download=True,
                transform=transform
            )
            load_time = time.time() - start_time
            print(f"✅ Successfully loaded CelebA with {len(dataset)} images in {load_time:.2f} seconds")
        except Exception as e:
            print(f"❌ Error loading CelebA: {e}")
            print("Note: CelebA may take a long time to download (~1.4GB)")
            return False
    else:
        print(f"❌ Unknown dataset: {dataset_name}")
        return False
    
    # Test batch loading
    batch_size = 32
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Load a batch
    start_time = time.time()
    batch = next(iter(dataloader))
    batch_load_time = time.time() - start_time
    
    if dataset_name == 'mnist':
        batch_data, batch_labels = batch
        print(f"✅ Successfully loaded a batch with shape {batch_data.shape} in {batch_load_time:.4f} seconds")
    elif dataset_name == 'cifar10':
        batch_data, batch_labels = batch
        print(f"✅ Successfully loaded a batch with shape {batch_data.shape} in {batch_load_time:.4f} seconds")
    elif dataset_name == 'celeba':
        print(f"✅ Successfully loaded a CelebA batch in {batch_load_time:.4f} seconds")
    
    # Check storage location
    print(f"\nVerifying storage location:")
    if dataset_name == 'mnist':
        dataset_dir = os.path.join(data_root, 'MNIST')
    elif dataset_name == 'cifar10':
        dataset_dir = os.path.join(data_root, 'cifar-10-batches-py')
    elif dataset_name == 'celeba':
        dataset_dir = os.path.join(data_root, 'celeba')
    
    if os.path.exists(dataset_dir):
        print(f"✅ Dataset stored in ephemeral storage at {dataset_dir}")
        # Get total size
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(dataset_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        print(f"Total size: {total_size / (1024 * 1024):.2f} MB")
    else:
        print(f"❌ Dataset not found at expected location: {dataset_dir}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test dataset loading from ephemeral storage')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['mnist', 'cifar10', 'celeba', 'all'],
                        help='Dataset to test')
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        for dataset in ['mnist', 'cifar10', 'celeba']:
            print(f"\n{'='*50}")
            print(f"Testing {dataset.upper()}")
            print(f"{'='*50}")
            test_dataset(dataset)
    else:
        test_dataset(args.dataset)
