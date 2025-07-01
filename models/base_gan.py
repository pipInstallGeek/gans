# models/base_gan.py - UPDATED WITH GENERIC ARCHITECTURE
# =====================================================

import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
import os
import math

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
                
                # ✅ ADD RESHAPING LOGIC FOR VANILLA GAN
                # Check if output is flattened (2D tensor) and needs reshaping
                if len(batch_samples.shape) == 2:  # [batch, flattened_pixels]
                    # This is likely Vanilla GAN - reshape to image format
                    c = self.dataset_config['num_channels']
                    h = w = self.dataset_config['image_size']
                    
                    # Verify the dimensions match
                    expected_pixels = c * h * w
                    actual_pixels = batch_samples.shape[1]
                    
                    if actual_pixels == expected_pixels:
                        # Reshape from [batch, pixels] to [batch, channels, height, width]
                        batch_samples = batch_samples.view(current_batch_size, c, h, w)
                        print(f"🔍 Reshaped {actual_pixels} pixels to [{current_batch_size}, {c}, {h}, {w}]")
                    else:
                        print(f"⚠️ Warning: Expected {expected_pixels} pixels, got {actual_pixels}")
                
                samples.append(batch_samples.cpu())  # Move to CPU to save GPU memory
                
                # Clear GPU memory after each batch
                del z, batch_samples
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Concatenate all samples
        all_samples = torch.cat(samples, dim=0)
        
        print(f"🔍 Final generate_samples output shape: {all_samples.shape}")
        
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
        
        print(f"Model saved: epoch_{epoch}.pth")
    
    def load_models(self, checkpoint_path):
        """GPU-optimized model loading with memory management"""
        print(f"Loading checkpoint: {os.path.basename(checkpoint_path)}")
        
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
        print(f"Checkpoint loaded successfully! Epoch: {epoch}")
        
        # CLEAR GPU MEMORY after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return epoch

def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias'):
            nn.init.constant_(m.bias.data, 0)

# ============================================================================
# GENERIC CONVOLUTIONAL GENERATOR - WORKS FOR ANY IMAGE SIZE
# ============================================================================

class ConvGenerator(nn.Module):
    """Generic Generator that adapts to ANY image size"""
    
    def __init__(self, z_dim, num_channels, ngf, target_size):
        super(ConvGenerator, self).__init__()
        self.z_dim = z_dim
        self.num_channels = num_channels
        self.ngf = ngf
        self.target_size = target_size
        
        # Calculate the architecture automatically
        self.layers_info = self._calculate_architecture()
        
        # Build the initial linear layer
        self.initial_layer = self._build_initial_layer()
        
        # Build the convolutional layers
        self.conv_layers = self._build_conv_layers()
        
        print(f"Generic Generator: {target_size}×{target_size}, layers: {self.layers_info['layers_needed']}")
    
    def _calculate_architecture(self):
        """Calculate the required architecture for target image size"""
        
        # Find the best starting size (power of 2, <= 8)
        possible_starts = [4, 8]
        best_architecture = None
        
        for start_size in possible_starts:
            layers_needed = 0
            current_size = start_size
            
            # Count upsampling layers needed
            while current_size < self.target_size:
                current_size *= 2
                layers_needed += 1
            
            # Check if we can reach target exactly or get close
            if current_size == self.target_size:
                best_architecture = {
                    'start_size': start_size,
                    'layers_needed': layers_needed,
                    'final_size': current_size,
                    'exact_match': True
                }
                break
            elif best_architecture is None or current_size < best_architecture['final_size']:
                best_architecture = {
                    'start_size': start_size,
                    'layers_needed': layers_needed,
                    'final_size': current_size,
                    'exact_match': False
                }
        
        return best_architecture
    
    def _build_initial_layer(self):
        """Build the initial linear layer"""
        start_size = self.layers_info['start_size']
        initial_features = self.ngf * 8 * start_size * start_size
        
        return nn.Sequential(
            nn.Linear(self.z_dim, initial_features),
            nn.BatchNorm1d(initial_features),
            nn.ReLU(True)
        )
    
    def _build_conv_layers(self):
        """Build convolutional layers dynamically"""
        layers = []
        
        start_size = self.layers_info['start_size']
        layers_needed = self.layers_info['layers_needed']
        
        # Calculate channel progression
        current_channels = self.ngf * 8
        current_size = start_size
        
        for layer_idx in range(layers_needed):
            # Calculate next layer parameters
            next_size = current_size * 2
            
            # Channel reduction as we go up
            if layer_idx == layers_needed - 1:
                # Final layer outputs target channels
                next_channels = self.num_channels
            else:
                # Reduce channels as we go up
                next_channels = max(self.ngf, current_channels // 2)
            
            # Build layer
            if layer_idx == layers_needed - 1:
                # Final layer (no batch norm, use Tanh)
                layers.extend([
                    nn.ConvTranspose2d(current_channels, next_channels, 4, 2, 1, bias=False),
                    nn.Tanh()
                ])
            else:
                # Intermediate layer
                layers.extend([
                    nn.ConvTranspose2d(current_channels, next_channels, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(next_channels),
                    nn.ReLU(True)
                ])
            
            current_channels = next_channels
            current_size = next_size
        
        return nn.Sequential(*layers)
    
    def forward(self, z):
        # Linear layer
        out = self.initial_layer(z)
        
        # Reshape for conv layers
        start_size = self.layers_info['start_size']
        out = out.view(out.shape[0], -1, start_size, start_size)
        
        # Conv layers
        out = self.conv_layers(out)
        
        # Handle final size adjustment if needed
        if out.shape[-1] != self.target_size:
            out = nn.functional.interpolate(out, size=(self.target_size, self.target_size), 
                                          mode='bilinear', align_corners=False)
        
        return out

# ============================================================================
# GENERIC CONVOLUTIONAL DISCRIMINATOR - WORKS FOR ANY IMAGE SIZE  
# ============================================================================

class ConvDiscriminator(nn.Module):
    """Generic Discriminator that adapts to ANY image size"""
    
    def __init__(self, num_channels, ndf, image_size):
        super(ConvDiscriminator, self).__init__()
        self.num_channels = num_channels
        self.ndf = ndf
        self.image_size = image_size
        
        # Calculate architecture automatically
        self.layers_info = self._calculate_architecture()
        
        # Build layers
        self.main = self._build_layers()
        
        print(f"Generic Discriminator: {image_size}×{image_size}, layers: {self.layers_info['layers_needed']}")
    
    def _calculate_architecture(self):
        """Calculate required layers to reduce image to 1×1"""
        
        layers_needed = 0
        current_size = self.image_size
        size_progression = [current_size]
        
        # Calculate layer sequence
        while current_size > 1:
            if current_size <= 4:
                # For small sizes, go directly to 1×1
                layers_needed += 1
                size_progression.append(1)
                break
            else:
                # Regular halving
                current_size = current_size // 2
                layers_needed += 1
                size_progression.append(current_size)
        
        return {
            'layers_needed': layers_needed,
            'size_progression': size_progression
        }
    
    def _build_layers(self):
        """Build discriminator layers dynamically"""
        layers = []
        
        size_progression = self.layers_info['size_progression']
        layers_needed = self.layers_info['layers_needed']
        
        current_channels = self.num_channels
        
        for layer_idx in range(layers_needed):
            current_size = size_progression[layer_idx]
            next_size = size_progression[layer_idx + 1]
            
            # Calculate output channels
            if layer_idx == 0:
                # First layer
                next_channels = self.ndf
            elif layer_idx == layers_needed - 1:
                # Final layer outputs 1 channel
                next_channels = 1
            else:
                # Intermediate layers - double channels up to limit
                next_channels = min(current_channels * 2, self.ndf * 16)
            
            # Choose layer configuration based on size transition
            if next_size == 1:
                # Final reduction to 1×1
                if current_size <= 4:
                    # Direct convolution
                    layers.extend([
                        nn.Conv2d(current_channels, next_channels, current_size, 1, 0, bias=False),
                        nn.Sigmoid()
                    ])
                else:
                    # Use adaptive pooling for robustness
                    layers.extend([
                        nn.Conv2d(current_channels, next_channels, 4, 2, 1, bias=False),
                        nn.AdaptiveAvgPool2d(1),
                        nn.Sigmoid()
                    ])
            else:
                # Regular conv layer with halving
                if layer_idx == 0:
                    # First layer (no batch norm)
                    layers.extend([
                        nn.Conv2d(current_channels, next_channels, 4, 2, 1, bias=False),
                        nn.LeakyReLU(0.2, inplace=True)
                    ])
                else:
                    # Intermediate layer
                    layers.extend([
                        nn.Conv2d(current_channels, next_channels, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(next_channels),
                        nn.LeakyReLU(0.2, inplace=True)
                    ])
            
            current_channels = next_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, input):
        output = self.main(input)
        # Ensure output is [batch_size] regardless of input size
        return output.view(input.size(0))
