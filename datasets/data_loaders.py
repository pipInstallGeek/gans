# datasets/data_loaders.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

class DatasetLoader:
    """Unified dataset loader for different datasets"""
    
    def __init__(self, config):
        self.config = config
        self.data_root = "data"
        os.makedirs(self.data_root, exist_ok=True)
    
    def get_dataloader(self, dataset_name, batch_size=None):
        """Get dataloader for specified dataset"""
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
        """Get MNIST dataloader"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
        
        dataset = torchvision.datasets.MNIST(
            root=self.data_root,
            train=True,
            download=True,
            transform=transform
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        
        return dataloader
    
    def _get_cifar10_loader(self, batch_size):
        """Get CIFAR-10 dataloader"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
        
        dataset = torchvision.datasets.CIFAR10(
            root=self.data_root,
            train=True,
            download=True,
            transform=transform
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )
        
        return dataloader
    
    def _get_celeba_loader(self, batch_size):
        """Get CelebA dataloader"""
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
        
        try:
            dataset = torchvision.datasets.CelebA(
                root=self.data_root,
                split='train',
                download=True,
                transform=transform
            )
        except:
            # Fallback to a smaller synthetic dataset if CelebA is not available
            print("Warning: CelebA not available, using CIFAR-10 as substitute")
            return self._get_cifar10_loader(batch_size)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )
        
        return dataloader
    
    def get_eval_dataloader(self, dataset_name, batch_size=256):
        """Get dataloader for evaluation (can use different batch size)"""
        return self.get_dataloader(dataset_name, batch_size)

# Synthetic datasets for visualization
class Synthetic2DDataset:
    """Generate 2D synthetic datasets for mode collapse visualization"""
    
    @staticmethod
    def generate_gaussian_mixture(n_samples=10000, n_modes=8):
        """Generate 2D Gaussian mixture with specified number of modes"""
        import numpy as np
        
        # Create modes in a circle
        angles = np.linspace(0, 2*np.pi, n_modes, endpoint=False)
        centers = [(2*np.cos(angle), 2*np.sin(angle)) for angle in angles]
        
        # Generate samples
        samples = []
        samples_per_mode = n_samples // n_modes
        
        for center in centers:
            mode_samples = np.random.multivariate_normal(
                center, 
                [[0.1, 0], [0, 0.1]], 
                samples_per_mode
            )
            samples.append(mode_samples)
        
        return np.vstack(samples)
    
    @staticmethod
    def generate_swiss_roll(n_samples=10000):
        """Generate Swiss roll dataset"""
        import numpy as np
        
        t = 3 * np.pi / 2 * (1 + 2 * np.random.rand(n_samples))
        x = t * np.cos(t)
        y = t * np.sin(t)
        
        return np.column_stack([x, y])

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