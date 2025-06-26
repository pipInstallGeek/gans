
# models/wgan_gp.py
import torch
import torch.nn as nn
import torch.optim as optim
from .base_gan import BaseGAN, ConvGenerator, weights_init
from .wgan import WGANDiscriminator

class WGAN_GP(BaseGAN):
    """WGAN with Gradient Penalty implementation"""
    
    def __init__(self, config, dataset_config):
        super().__init__(config, dataset_config)
        
        # Use Adam optimizer for WGAN-GP
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate_g,
            betas=(0, 0.9)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate_d,
            betas=(0, 0.9)
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
    
    def gradient_penalty(self, real_data, fake_data):
        """Calculate gradient penalty for WGAN-GP"""
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        alpha = alpha.expand_as(real_data)
        
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated = interpolated.requires_grad_(True)
        
        prob_interpolated = self.discriminator(interpolated)
        
        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size(), device=self.device),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
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
            
            # Gradient penalty
            gp = self.gradient_penalty(real_data, fake_data)
            
            loss_d = loss_d_real + loss_d_fake + self.config.lambda_gp * gp
            loss_d.backward()
            self.optimizer_d.step()
        
        # Train Generator
        self.optimizer_g.zero_grad()
        z = torch.randn(batch_size, self.config.z_dim, device=self.device)
        fake_data = self.generator(z)
        output_fake = self.discriminator(fake_data)
        loss_g = -torch.mean(output_fake)
        loss_g.backward()
        self.optimizer_g.step()
        
        return loss_g.item(), loss_d.item()