# models/dcgan.py
import torch
import torch.nn as nn
from .base_gan import BaseGAN, ConvGenerator, ConvDiscriminator, weights_init

class DCGAN(BaseGAN):
    """Deep Convolutional GAN implementation"""
    
    def __init__(self, config, dataset_config):
        super().__init__(config, dataset_config)
        
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
        discriminator = ConvDiscriminator(
            self.dataset_config['num_channels'],
            self.config.ndf,
            self.dataset_config['image_size']
        )
        discriminator.apply(weights_init)
        return discriminator
    
    def train_step(self, real_data):
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, device=self.device)
        fake_labels = torch.zeros(batch_size, device=self.device)
        
        # Train Discriminator
        self.optimizer_d.zero_grad()
        
        # Real data
        output_real = self.discriminator(real_data)
        loss_d_real = self.criterion(output_real, real_labels)
        
        # Fake data
        z = torch.randn(batch_size, self.config.z_dim, device=self.device)
        fake_data = self.generator(z)
        output_fake = self.discriminator(fake_data.detach())
        loss_d_fake = self.criterion(output_fake, fake_labels)
        
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        self.optimizer_d.step()
        
        # Train Generator
        self.optimizer_g.zero_grad()
        
        output_fake = self.discriminator(fake_data)
        loss_g = self.criterion(output_fake, real_labels)
        loss_g.backward()
        self.optimizer_g.step()
        
        return loss_g.item(), loss_d.item()