# models/vanilla_gan.py
import torch
import torch.nn as nn
from .base_gan import BaseGAN, weights_init


class VanillaGenerator(nn.Module):
    """Simple fully-connected generator"""

    def __init__(self, z_dim, output_dim):
        super(VanillaGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class VanillaDiscriminator(nn.Module):
    """Simple fully-connected discriminator"""

    def __init__(self, input_dim):
        super(VanillaDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x).squeeze()


class VanillaGAN(BaseGAN):
    """Original GAN implementation with fully-connected networks"""

    def __init__(self, config, dataset_config):
        self.output_dim = (dataset_config['num_channels'] *
                           dataset_config['image_size'] *
                           dataset_config['image_size'])
        super().__init__(config, dataset_config)

    def build_generator(self):
        generator = VanillaGenerator(self.config.z_dim, self.output_dim)
        generator.apply(weights_init)
        return generator

    def build_discriminator(self):
        discriminator = VanillaDiscriminator(self.output_dim)
        discriminator.apply(weights_init)
        return discriminator

    def train_step(self, real_data):
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, device=self.device)
        fake_labels = torch.zeros(batch_size, device=self.device)

        # Train Discriminator
        self.optimizer_d.zero_grad()

        # Real data
        real_data = real_data.view(batch_size, -1)
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
