# simple_train_test.py
from config import Config
from datasets.data_loaders import DatasetLoader
from models.dcgan import DCGAN
from training.trainer import GANTrainer

print("Starting simple training test...")

config = Config()
config.epochs = 2  # Very short test

dataset_config = config.get_dataset_config('mnist')
print(f"Dataset config: {dataset_config}")

loader = DatasetLoader(config)
dataloader = loader.get_dataloader('mnist')
print(f"Dataloader created: {len(dataloader)} batches")

model = DCGAN(config, dataset_config)
print("Model created")

trainer = GANTrainer(config)
print("Trainer created")

print("Starting training...")
trainer.train_model(model, dataloader, 'dcgan', 'mnist')
print("Training completed!")