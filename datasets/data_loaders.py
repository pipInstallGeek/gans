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
        self.data_root = "data"
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
        elif dataset_name == 'celeba-hq':
            return self._get_celeba_hq_loader(batch_size)
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
        """GPU-optimized CelebA dataloader"""
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        print("Loading CelebA dataset...")
        
        dataset = torchvision.datasets.CelebA(
                root=self.data_root,
                split='train',
                download=True,
                transform=transform
            )
        print("CelebA dataset loaded successfully.")
        
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
    
    def _get_celeba_hq_loader(self, batch_size):
        """GPU-optimized CelebA-HQ dataloader"""
        # CelebA-HQ images are typically 1024x1024, but we'll resize them
        # to a more manageable size for GAN training
        image_size = 256  # Can be adjusted to 128, 256, 512 depending on your GPU memory
        
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        print(f"Loading CelebA-HQ dataset (size: {image_size}x{image_size})...")
        
        # Create a custom dataset since CelebA-HQ isn't directly available in torchvision
        dataset = CelebAHQDataset(
            root=os.path.join(self.data_root, 'celeba-hq'),
            transform=transform
        )
        print("CelebA-HQ dataset loaded successfully.")
        
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
    
# Custom dataset class for CelebA-HQ
class CelebAHQDataset(Dataset):
    """
    Custom dataset for CelebA-HQ images
    
    This dataset assumes that CelebA-HQ images are stored in a directory
    with each image named by its index (e.g., 0.jpg, 1.jpg, etc.)
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        # Create directory if it doesn't exist
        os.makedirs(root, exist_ok=True)
        
        # Check if dataset exists, and provide download instructions if not
        self.image_paths = sorted(glob.glob(os.path.join(root, '*.jpg')) + 
                                 glob.glob(os.path.join(root, '*.png')))
        
        if len(self.image_paths) == 0:
            print("⚠️ CelebA-HQ dataset not found.")
            print("Please download CelebA-HQ dataset manually from:")
            print("https://github.com/tkarras/progressive_growing_of_gans")
            print("Or download and extract the dataset using:")
            print("$ python scripts/download_data.py --dataset celeba-hq --download_dir ./data")
            print(f"Then place the images in {root}")
            
            # Create a dummy dataset with a warning message
            self.is_empty = True
        else:
            print(f"Found {len(self.image_paths)} images in CelebA-HQ dataset.")
            self.is_empty = False
    
    def __len__(self):
        if self.is_empty:
            return 1  # Return 1 for the dummy dataset
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.is_empty:
            # Return a dummy sample if dataset is empty
            dummy_img = torch.zeros(3, 256, 256)
            return dummy_img
        
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image
    
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
    