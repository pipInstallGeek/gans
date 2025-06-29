import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os

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
        elif dataset_name == 'ffhq':
            return self._get_ffhq_loader(batch_size)
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
    
    def _get_ffhq_loader(self, batch_size):
        """FFHQ dataloader - works with Kaggle downloaded data"""
        
        # Transform for FFHQ - resize to 64x64 for your GAN
        transform = transforms.Compose([
            transforms.Resize(64),          # Resize to 64x64 
            transforms.CenterCrop(64),      # Center crop to ensure square
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
        print("Loading FFHQ dataset...")
        
        ffhq_dir = os.path("/kaggle/input/flickrfaceshq-dataset-ffhq")  
        # Create dataset
        dataset = FFHQDataset(ffhq_dir, transform=transform)
        
        # Create dataloader with same settings as other datasets
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
    elif model_name == 'wgan_gp':
        from models.wgan_gp import WGAN_GP
        return WGAN_GP
    else:
        raise ValueError(f"Unknown model: {model_name}")
    

class FFHQDataset(Dataset):
    """FFHQ Dataset for PyTorch - works with Kaggle download"""
    
    def __init__(self, root_dir, transform=None, image_size=64):
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        
        # Find all image files in the directory and subdirectories
        image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        self.image_paths = []
        
        for pattern in image_patterns:
            # Search in root and all subdirectories
            self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', pattern), recursive=True))
            self.image_paths.extend(glob.glob(os.path.join(root_dir, pattern)))
        
        # Remove duplicates and sort
        self.image_paths = sorted(list(set(self.image_paths)))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}. Please check the directory structure.")
        
        print(f"📊 Found {len(self.image_paths)} FFHQ images in {root_dir}")
        
        # Show some example paths for debugging
        if len(self.image_paths) > 0:
            print(f"Example image path: {self.image_paths[0]}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            
            # Load image and convert to RGB
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Return image and dummy label (0) for GAN training
            return image, 0
            
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return next image if this one fails
            return self.__getitem__((idx + 1) % len(self.image_paths))
