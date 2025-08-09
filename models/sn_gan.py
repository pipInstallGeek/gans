# models/sn_gan.py
"""
Simple SN-GAN Implementation
Just adds spectral normalization to discriminator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_gan import BaseGAN, ConvGenerator, ConvDiscriminator, weights_init


class SpectralNorm(nn.Module):
    """Spectral Normalization"""

    def __init__(self, module, power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.power_iterations = power_iterations

        # Get weight dimensions
        w = self.module.weight.data
        height = w.shape[0]
        width = w.view(height, -1).shape[1]

        # Initialize u and v vectors
        u = nn.Parameter(w.new_empty(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.new_empty(width).normal_(0, 1), requires_grad=False)

        # Normalize
        u.data = F.normalize(u.data, dim=0)
        v.data = F.normalize(v.data, dim=0)

        # Register buffers
        self.module.register_buffer('u', u)
        self.module.register_buffer('v', v)

    def forward(self, x):
        # Get weight
        w = self.module.weight
        height = w.shape[0]

        # Power iteration to find largest singular value
        for _ in range(self.power_iterations):
            # v = W^T u / ||W^T u||
            v = F.normalize(torch.mv(w.view(height, -1).t(), self.module.u), dim=0)
            # u = W v / ||W v||
            u = F.normalize(torch.mv(w.view(height, -1), v), dim=0)

        # Update u (v will be computed next iteration)
        self.module.u.data = u.data

        # Compute spectral norm
        sigma = torch.dot(u, torch.mv(w.view(height, -1), v))

        # Normalize weight and compute output
        self.module.weight.data = w.data / sigma

        return self.module.forward(x)


class SNGAN(BaseGAN):
    """SN-GAN: DCGAN with Spectral Normalization on Discriminator"""

    def __init__(self, config, dataset_config):
        super().__init__(config, dataset_config)

    def build_generator(self):
        """Standard generator - no changes"""
        generator = ConvGenerator(
            self.config.z_dim,
            self.dataset_config['num_channels'],
            self.config.ngf,
            self.dataset_config['image_size']
        )
        generator.apply(weights_init)
        return generator

    def build_discriminator(self):
        """Discriminator with spectral normalization"""
        # Create standard discriminator
        discriminator = ConvDiscriminator(
            self.dataset_config['num_channels'],
            self.config.ndf,
            self.dataset_config['image_size']
        )

        # Apply spectral normalization to all Conv2d layers
        for name, module in discriminator.named_modules():
            if isinstance(module, nn.Conv2d):
                # Replace conv layer with spectral normalized version
                sn_module = SpectralNorm(module)
                # Set it back in the parent
                parent = discriminator
                parts = name.split('.')
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                if len(parts) > 0:
                    setattr(parent, parts[-1], sn_module)

        return discriminator

    def train_step(self, real_data):
        """Standard GAN training with BCE loss"""
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, device=self.device)
        fake_labels = torch.zeros(batch_size, device=self.device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        self.optimizer_d.zero_grad()

        # Real data
        output_real = self.discriminator(real_data)
        loss_d_real = self.criterion(output_real, real_labels)

        # Fake data
        z = torch.randn(batch_size, self.config.z_dim, device=self.device)
        fake_data = self.generator(z)
        output_fake = self.discriminator(fake_data.detach())
        loss_d_fake = self.criterion(output_fake, fake_labels)

        # Total discriminator loss
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        self.optimizer_d.step()

        # -----------------
        # Train Generator
        # -----------------
        self.optimizer_g.zero_grad()

        # Generate fake data again (gradients need to flow)
        output_fake = self.discriminator(fake_data)
        loss_g = self.criterion(output_fake, real_labels)

        loss_g.backward()
        self.optimizer_g.step()

        return loss_g.item(), loss_d.item()