# test_image_generation.py
import torch
import os
from config import Config
from models.dcgan import DCGAN
from torchvision.utils import make_grid
import torchvision.transforms as transforms

print("Testing image generation...")

# Setup
config = Config()
dataset_config = config.get_dataset_config('mnist')
model = DCGAN(config, dataset_config)

print("Generating samples...")
samples = model.generate_samples(64)  # Generate 64 images

print(f"Generated samples shape: {samples.shape}")

# Create samples directory
os.makedirs('results/samples/test', exist_ok=True)

# Denormalize samples (from [-1, 1] to [0, 1])
samples = (samples + 1) / 2.0
samples = torch.clamp(samples, 0, 1)

# Create grid
grid = make_grid(samples, nrow=8, padding=2, normalize=False)

# Save image
transforms.ToPILImage()(grid).save('results/samples/test/generated_images.png')

print("✅ Images saved to: results/samples/test/generated_images.png")
print("Check this file to see the generated MNIST digits!")

# Also save individual samples
for i in range(min(16, samples.size(0))):
    transforms.ToPILImage()(samples[i]).save(f'results/samples/test/digit_{i}.png')

print("✅ Individual digits saved as digit_0.png to digit_15.png")