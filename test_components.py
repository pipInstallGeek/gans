# test_components.py
print("Testing imports...")

try:
    from config import Config
    print("✓ Config imported")
except Exception as e:
    print(f"✗ Config failed: {e}")

try:
    from datasets.data_loaders import DatasetLoader
    print("✓ DatasetLoader imported")
except Exception as e:
    print(f"✗ DatasetLoader failed: {e}")

try:
    from models.dcgan import DCGAN
    print("✓ DCGAN imported")
except Exception as e:
    print(f"✗ DCGAN failed: {e}")

try:
    from training.trainer import GANTrainer
    print("✓ GANTrainer imported")
except Exception as e:
    print(f"✗ GANTrainer failed: {e}")

try:
    from experiments.run_experiments import ExperimentRunner
    print("✓ ExperimentRunner imported")
except Exception as e:
    print(f"✗ ExperimentRunner failed: {e}")

print("\nTesting basic functionality...")

try:
    config = Config()
    print(f"✓ Config created, device: {config.device}")
    
    dataset_config = config.get_dataset_config('mnist')
    print(f"✓ Dataset config: {dataset_config}")
    
    model = DCGAN(config, dataset_config)
    print("✓ DCGAN model created")
    
    samples = model.generate_samples(4)
    print(f"✓ Sample generation works: {samples.shape}")
    
except Exception as e:
    print(f"✗ Basic test failed: {e}")
    import traceback
    traceback.print_exc()