# models/wgan.py
import torch
import torch.nn as nn
import torch.optim as optim
from .base_gan import BaseGAN, ConvGenerator, ConvDiscriminator, weights_init

class WGANDiscriminator(ConvDiscriminator):
    """WGAN Discriminator (Critic) without sigmoid activation"""
    
    def __init__(self, num_channels, ndf, image_size):
        super().__init__(num_channels, ndf, image_size)
        # Remove sigmoid activation for WGAN
        layers = list(self.main.children())[:-1]  # Remove last sigmoid
        self.main = nn.Sequential(*layers)
    
    def forward(self, input):
        return self.main(input).view(-1)

class WGAN(BaseGAN):
    """Wasserstein GAN implementation"""
    
    def __init__(self, config, dataset_config):
        super().__init__(config, dataset_config)
        
        # Use RMSprop optimizer for WGAN
        self.optimizer_g = optim.RMSprop(
            self.generator.parameters(),
            lr=config.learning_rate_g
        )
        self.optimizer_d = optim.RMSprop(
            self.discriminator.parameters(),
            lr=config.learning_rate_d
        )
        
    def build_generator(self):
        generator = ConvGenerator(
            self.config.z_dim,
            self.dataset_config['num_channels'],
            self.config.ngf,
            self.dataset_config['image_size']
        )
        generator.apply(weights_init)
        return generator
    
    def build_discriminator(self):
        discriminator = WGANDiscriminator(
            self.dataset_config['num_channels'],
            self.config.ndf,
            self.dataset_config['image_size']
        )
        discriminator.apply(weights_init)
        return discriminator
    
    def train_step(self, real_data):
        batch_size = real_data.size(0)
        
        # Train Critic multiple times
        for _ in range(self.config.n_critic):
            self.optimizer_d.zero_grad()
            
            # Real data
            output_real = self.discriminator(real_data)
            loss_d_real = -torch.mean(output_real)
            
            # Fake data
            z = torch.randn(batch_size, self.config.z_dim, device=self.device)
            fake_data = self.generator(z)
            output_fake = self.discriminator(fake_data.detach())
            loss_d_fake = torch.mean(output_fake)
            
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            self.optimizer_d.step()
            
            # Clip weights
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.config.clip_value, self.config.clip_value)
        
        # Train Generator
        self.optimizer_g.zero_grad()
        z = torch.randn(batch_size, self.config.z_dim, device=self.device)
        fake_data = self.generator(z)
        output_fake = self.discriminator(fake_data)
        loss_g = -torch.mean(output_fake)
        loss_g.backward()
        self.optimizer_g.step()
        
        return loss_g.item(), loss_d.item()
