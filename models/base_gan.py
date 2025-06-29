# models/base_gan.py
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
import os

class BaseGAN(ABC):
    """Abstract base class for all GAN implementations"""
    
    def __init__(self, config, dataset_config):
        self.config = config
        self.dataset_config = dataset_config
        self.device = config.device
        
        # Initialize networks
        self.generator = self.build_generator().to(self.device)
        self.discriminator = self.build_discriminator().to(self.device)
        
        # Initialize optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate_g,
            betas=(config.beta1, config.beta2)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate_d,
            betas=(config.beta1, config.beta2)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training history
        self.g_losses = []
        self.d_losses = []
        self.training_history = {}
        
    @abstractmethod
    def build_generator(self):
        """Build and return the generator network"""
        pass
    
    @abstractmethod
    def build_discriminator(self):
        """Build and return the discriminator network"""
        pass
    
    @abstractmethod
    def train_step(self, real_data):
        """Perform one training step and return losses"""
        pass
    
    def generate_samples(self, n_samples=64, return_tensor=False):
        """GPU-optimized sample generation with memory management"""
        self.generator.eval()
        
        # CLEAR GPU MEMORY before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Generate in smaller batches to avoid memory issues
        batch_size = 32  # Smaller batches for GPU memory
        samples = []
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                current_batch_size = min(batch_size, n_samples - i)
                z = torch.randn(current_batch_size, self.config.z_dim, device=self.device)
                batch_samples = self.generator(z)
                
                if return_tensor:
                    samples.append(batch_samples.cpu())  # Move to CPU to save GPU memory
                else:
                    samples.append(batch_samples.cpu())
                
                # Clear GPU memory after each batch
                del z, batch_samples
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Concatenate all samples
        all_samples = torch.cat(samples, dim=0)
        
        if return_tensor:
            return all_samples
        else:
            return all_samples

    def save_models(self, epoch, model_name, dataset_name):
        """GPU-optimized model saving with memory management"""
        save_dir = os.path.join(self.config.models_dir, f"{model_name}_{dataset_name}")
        os.makedirs(save_dir, exist_ok=True)
        
        # MOVE MODELS TO CPU before saving to reduce GPU memory usage
        generator_state = {k: v.cpu() for k, v in self.generator.state_dict().items()}
        discriminator_state = {k: v.cpu() for k, v in self.discriminator.state_dict().items()}
        
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': generator_state,
            'discriminator_state_dict': discriminator_state,
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses
        }
        
        torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        # CLEAR GPU MEMORY after saving
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"💾 Model saved: epoch_{epoch}.pth")

    def load_models(self, checkpoint_path):
        """GPU-optimized model loading with memory management"""
        print(f"📥 Loading checkpoint: {os.path.basename(checkpoint_path)}")
        
        # CLEAR GPU MEMORY before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load to CPU first
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load state dicts
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        # Move models to GPU
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Load optimizers
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        
        # Load training history
        self.g_losses = checkpoint['g_losses']
        self.d_losses = checkpoint['d_losses']
        
        epoch = checkpoint['epoch']
        print(f"✅ Checkpoint loaded successfully! Epoch: {epoch}")
        
        # CLEAR GPU MEMORY after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return epoch

    def weights_init(m):
        """Initialize network weights"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if hasattr(m, 'weight'):  # ADD THIS CHECK
                nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            if hasattr(m, 'weight'):  # ADD THIS CHECK
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if hasattr(m, 'bias'):    # ADD THIS CHECK
                nn.init.constant_(m.bias.data, 0)

class ConvGenerator(nn.Module):
    """Convolutional Generator for DCGAN-style architectures"""
    
    def __init__(self, z_dim, num_channels, ngf, image_size):
        super(ConvGenerator, self).__init__()
        self.z_dim = z_dim
        self.num_channels = num_channels
        self.ngf = ngf
        
        # Calculate the size of the first layer
        if image_size == 28:  # MNIST
            self.init_size = 7
        elif image_size == 32:  # CIFAR-10
            self.init_size = 4
        elif image_size == 64:  # CelebA
            self.init_size = 4
        else:
            self.init_size = 4
        
        self.l1 = nn.Sequential(
            nn.Linear(z_dim, ngf * 8 * self.init_size * self.init_size),
            nn.BatchNorm1d(ngf * 8 * self.init_size * self.init_size),
            nn.ReLU(True)
        )
        
        layers = []
        
        if image_size == 28:  # MNIST - FIXED VERSION
            layers.extend([
                # 7×7 → 14×14
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # 14×14 → 28×28
                nn.ConvTranspose2d(ngf * 4, num_channels, 4, 2, 1, bias=False),
                nn.Tanh()
            ])
        else:  # CIFAR-10 and CelebA
            layers.extend([
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True)
            ])
            
            if image_size == 64:  # Additional layer for CelebA
                layers.extend([
                    nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False),
                    nn.Tanh()
                ])
            else:  # CIFAR-10
                layers.extend([
                    nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False),
                    nn.Tanh()
                ])
        
        self.conv_layers = nn.Sequential(*layers)
        
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        out = self.conv_layers(out)
        return out  # FIXED: Return out instead of undefined variables

class ConvDiscriminator(nn.Module):
    """Convolutional Discriminator for DCGAN-style architectures"""
    
    def __init__(self, num_channels, ndf, image_size):
        super(ConvDiscriminator, self).__init__()
        self.num_channels = num_channels
        self.ndf = ndf
        
        layers = []
        
        if image_size == 28:  # MNIST - FIXED VERSION
            layers.extend([
                nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # 28×28 → 14×14
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # 14×14 → 7×7
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # 7×7 → 3×3
                nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # 3×3 → 1×1 (single value per sample)
                nn.Conv2d(ndf * 8, 1, 3, 1, 0, bias=False),
                nn.Sigmoid()
            ])
        else:  # CIFAR-10 and CelebA
            layers.extend([
                nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            
            if image_size == 64:  # Additional layer for CelebA
                layers.extend([
                    nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                    nn.Sigmoid()
                ])
            else:  # CIFAR-10
                layers.extend([
                    nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
                    nn.Sigmoid()
                ])
        
        self.main = nn.Sequential(*layers)
        
    def forward(self, input):
        output = self.main(input)
        # FIXED: Properly flatten to [batch_size] shape
        return output.view(input.size(0)).squeeze()