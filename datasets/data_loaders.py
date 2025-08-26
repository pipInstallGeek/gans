import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import glob
from PIL import Image 

class DatasetLoader:
    """GPU-optimized dataset loader"""
    
    def __init__(self, config):
        self.config = config
        self.data_root = "/tmp/data"
        os.makedirs(self.data_root, exist_ok=True)
        
        # GPU-OPTIMIZED DATA LOADING SETTINGS
        if torch.cuda.is_available():
            self.num_workers = 4          # MORE WORKERS for GPU (was 2)
            self.pin_memory = True        # ENABLE PIN MEMORY for faster GPU transfer
            self.persistent_workers = True # KEEP WORKERS ALIVE
            self.prefetch_factor = 2      # PREFETCH BATCHES
        else:
            # CPU fallback
            self.num_workers = 2
            self.pin_memory = False
            self.persistent_workers = False
            self.prefetch_factor = 2
        
        print(f"📊 Data loading: {self.num_workers} workers, pin_memory={self.pin_memory}")
    
    def get_dataloader(self, dataset_name, batch_size=None):
        """Get GPU-optimized dataloader"""
        if batch_size is None:
            batch_size = self.config.batch_size
            
        if dataset_name == 'mnist':
            return self._get_mnist_loader(batch_size)
        elif dataset_name == 'cifar10':
            return self._get_cifar10_loader(batch_size)
        elif dataset_name == 'celeba':
            return self._get_celeba_loader(batch_size)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _get_mnist_loader(self, batch_size):
        """GPU-optimized MNIST dataloader"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
        
        print("Loading MNIST dataset...")
        dataset = torchvision.datasets.MNIST(
            root=self.data_root,
            train=True,
            download=True,
            transform=transform
        )
        print("MNIST dataset loaded successfully.")

        # GPU-OPTIMIZED DATALOADER
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,              # More workers for GPU
            pin_memory=self.pin_memory,                # Faster GPU transfer
            persistent_workers=self.persistent_workers, # Keep workers alive
            prefetch_factor=self.prefetch_factor,      # Prefetch batches
            drop_last=True
        )
        
        print(f"✅ DataLoader created: batch_size={batch_size}, workers={self.num_workers}")
        return dataloader
    
    def _get_cifar10_loader(self, batch_size):
        """GPU-optimized CIFAR-10 dataloader"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        print("Loading CIFAR-10 dataset...")
        dataset = torchvision.datasets.CIFAR10(
            root=self.data_root,
            train=True,
            download=True,
            transform=transform
        )
        print("CIFAR-10 dataset loaded successfully.")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True
        )
        
        return dataloader
    
    def _get_celeba_loader(self, batch_size):
        """GPU-optimized CelebA dataloader with robust downloading"""
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        print("Loading CelebA dataset...")
        
        # Check if dataset is already downloaded
        celeba_dir = os.path.join(self.data_root, 'celeba')
        img_dir = os.path.join(celeba_dir, 'img_align_celeba')
        
        if os.path.exists(img_dir) and len(os.listdir(img_dir)) > 0:
            print(f"Found existing CelebA images in {img_dir}")
            print(f"Number of images found: {len(os.listdir(img_dir))}")
            
            # Always use the direct file loading method to avoid hanging
            print("Using direct file loading method instead of torchvision loader")
            try:
                # Create a custom dataset from local files
                from PIL import Image
                
                class DirectCelebADataset(torch.utils.data.Dataset):
                    def __init__(self, img_dir, transform=None):
                        self.img_dir = img_dir
                        self.transform = transform
                        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
                        print(f"DirectCelebADataset: Found {len(self.image_files)} images")
                        
                    def __len__(self):
                        return len(self.image_files)
                        
                    def __getitem__(self, idx):
                        img_name = os.path.join(self.img_dir, self.image_files[idx])
                        image = Image.open(img_name).convert('RGB')
                        
                        if self.transform:
                            image = self.transform(image)
                            
                        return image
                
                dataset = DirectCelebADataset(img_dir, transform=transform)
                print(f"Created direct file dataset with {len(dataset)} images")
            except Exception as e:
                print(f"Error creating direct dataset: {e}")
                print("Falling back to default method...")
                # Continue with default method as fallback
                try:
                    dataset = torchvision.datasets.CelebA(
                        root=self.data_root,
                        split='train',
                        download=False,  # Don't try to download again
                        transform=transform
                    )
                    print("CelebA dataset loaded via torchvision successfully.")
                except Exception as e2:
                    print(f"Error loading CelebA via torchvision: {e2}")
                    print("\n" + "=" * 80)
                    print("ERROR: Failed to load CelebA dataset")
                    print("=" * 80)
                    print("CelebA is required for face generation. Please follow these steps:")
                    print("1. Make sure you have internet access")
                    print("2. Run: python scripts/download_data.py --dataset celeba")
                    print("3. If that fails, download CelebA manually from:")
                    print("   https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
                    print("   and extract it to /tmp/data/celeba")
                    print("=" * 80 + "\n")
                    raise RuntimeError("CelebA dataset is required for face generation")
        else:
            print("CelebA images not found. Will attempt to download.")
            try:
                # Try to download with torchvision
                dataset = torchvision.datasets.CelebA(
                    root=self.data_root,
                    split='train',
                    download=True,
                    transform=transform
                )
                print("CelebA dataset downloaded and loaded successfully.")
            except Exception as e:
                print(f"Error downloading/loading CelebA: {e}")
                print("\n" + "=" * 80)
                print("ERROR: Failed to download CelebA dataset")
                print("=" * 80)
                print("CelebA is required for face generation. Please follow these steps:")
                print("1. Make sure you have internet access")
                print("2. Try running again with: python scripts/download_data.py --dataset celeba")
                print("3. If that fails, download CelebA manually from:")
                print("   https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
                print("   and extract it to /tmp/data/celeba")
                print("=" * 80 + "\n")
                raise RuntimeError("CelebA dataset is required for face generation")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True
        )
        
        return dataloader
    
    def get_eval_dataloader(self, dataset_name, batch_size=64):  # Smaller eval batch
        """Get dataloader for evaluation with smaller batch size"""
        return self.get_dataloader(dataset_name, batch_size)
    
# Include the get_model_class function
def get_model_class(model_name):
    """Get model class by name"""
    if model_name == 'vanilla':
        from models.vanilla_gan import VanillaGAN
        return VanillaGAN
    elif model_name == 'dcgan':
        from models.dcgan import DCGAN
        return DCGAN
    elif model_name == 'wgan':
        from models.wgan import WGAN
        return WGAN
    elif model_name == 'sn_gan':
        from models.sn_gan import SNGAN
        return SNGAN
    else:
        raise ValueError(f"Unknown model: {model_name}")
    