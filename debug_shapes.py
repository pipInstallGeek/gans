# debug_shapes.py
from config import Config
from datasets.data_loaders import DatasetLoader
from models.dcgan import DCGAN
import torch

config = Config()
dataset_config = config.get_dataset_config('mnist')
model = DCGAN(config, dataset_config)

loader = DatasetLoader(config)
dataloader = loader.get_dataloader('mnist')

# Get one batch
for batch in dataloader:
    real_data, _ = batch
    print(f"Real data shape: {real_data.shape}")
    
    # Test discriminator
    output = model.discriminator(real_data)
    print(f"Discriminator output shape: {output.shape}")
    print(f"Expected shape: torch.Size([{real_data.size(0)}])")
    
    # Test generator
    z = torch.randn(real_data.size(0), config.z_dim)
    fake_data = model.generator(z)
    print(f"Generated data shape: {fake_data.shape}")
    break