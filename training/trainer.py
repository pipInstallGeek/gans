# training/trainer.py
import os
import time
import gc
from typing import Any, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler   # ✅ correct import
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms as transforms

from utils.device_manager import DeviceManager

# If you have helpers, wire them here; otherwise these tiny shims are safe.
def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1

def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0

def _to_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        # detach to be safe, then pull scalar
        return x.detach().float().item()
    return float(x)

class GANTrainer:
    """GPU-optimized training utilities with optional DDP."""

    def __init__(self, config):
        self.config = config
        self.device_manager = DeviceManager(getattr(config, 'device', None))
        self.device = self.device_manager.device

        self.device_manager.empty_cache()
        if self.device_manager.is_cuda():
            print("🔧 GPU memory initialized")

    def _maybe_set_epoch(self, dataloader, epoch: int):
        # ✅ Works with Pylance: use the imported class for isinstance checks
        sampler = getattr(dataloader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

    def _ddp_mean(self, value_f: float) -> float:
        """Average a python float across ranks when DDP is active."""
        if not is_distributed():
            return value_f
        t = torch.tensor(value_f, device=self.device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= get_world_size()
        return t.item()

    def train_model(self, model, dataloader, model_name: str, dataset_name: str) -> float:
        """GPU-optimized training loop"""
        print(f"🚀 Training {model_name} on {dataset_name}")

        if self.device_manager.is_cuda():
            print(f"📊 GPU Memory before training: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

        # Make sure these are Python lists – avoids “Tensor has no append” in Pylance
        if not hasattr(model, 'g_losses_iter') or not isinstance(model.g_losses_iter, list):
            model.g_losses_iter = []
        if not hasattr(model, 'd_losses_iter') or not isinstance(model.d_losses_iter, list):
            model.d_losses_iter = []
        if not hasattr(model, 'g_losses') or not isinstance(model.g_losses, list):
            model.g_losses = []
        if not hasattr(model, 'd_losses') or not isinstance(model.d_losses, list):
            model.d_losses = []

        start_time = time.time()

        for epoch in range(self.config.epochs):
            self._maybe_set_epoch(dataloader, epoch)

            epoch_g_losses = []
            epoch_d_losses = []

            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.config.epochs}')
            for i, batch in enumerate(pbar):
                # Move batch to device (MNIST/CIFAR return (x, y); CelebA returns images only)
                if isinstance(batch, (list, tuple)):
                    real_data = batch[0]
                else:  # batch already a tensor
                    real_data = batch
                
                real_data = real_data.to(self.config.device, non_blocking=True)

                # One step
                g_loss, d_loss = model.train_step(real_data)  # typically tensors

                # → Python floats for logging/append
                g_float = _to_float(g_loss)
                d_float = _to_float(d_loss)

                # Average across ranks if DDP
                g_float = self._ddp_mean(g_float)
                d_float = self._ddp_mean(d_float)

                # Store iteration-level losses (lists, not tensors)
                model.g_losses_iter.append(g_float)
                model.d_losses_iter.append(d_float)
                epoch_g_losses.append(g_float)
                epoch_d_losses.append(d_float)

                # Progress box (strings expect numbers, not tensors)
                if self.device_manager.is_cuda() and i % 100 == 0:
                    gpu_mem = torch.cuda.memory_allocated(0) / 1e9
                    pbar.set_postfix({'G_loss': f'{g_float:.4f}',
                                      'D_loss': f'{d_float:.4f}',
                                      'GPU_GB': f'{gpu_mem:.1f}'})
                else:
                    pbar.set_postfix({'G_loss': f'{g_float:.4f}',
                                      'D_loss': f'{d_float:.4f}'})

            # Epoch aggregates
            avg_g_loss = sum(epoch_g_losses) / max(1, len(epoch_g_losses))
            avg_d_loss = sum(epoch_d_losses) / max(1, len(epoch_d_losses))

            # Only rank-0 writes global lists to avoid duplication
            if get_rank() == 0:
                model.g_losses.append(avg_g_loss)
                model.d_losses.append(avg_d_loss)

            # Periodic samples/checkpoints on rank-0
            if get_rank() == 0:
                if (epoch + 1) % self.config.sample_interval == 0:
                    self.save_samples(model, epoch + 1, model_name, dataset_name)
                if (epoch + 1) % 10 == 0:
                    model.save_models(epoch + 1, model_name, dataset_name)
                    self.device_manager.empty_cache()
                    gc.collect()
                    if self.device_manager.is_cuda():
                        print(f"GPU memory cleaned at epoch {epoch + 1}")

            print(f'Epoch [{epoch+1}/{self.config.epochs}] G_loss: {avg_g_loss:.4f} D_loss: {avg_d_loss:.4f}')

        training_time = time.time() - start_time

        if get_rank() == 0:
            if self.device_manager.is_cuda():
                print(f"Final GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
                print(f"Max GPU Memory: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
            print(f"Training completed in {training_time:.2f} seconds")
            print(f"Average time per epoch: {training_time/self.config.epochs:.1f} seconds")
            model.save_models(self.config.epochs, model_name, dataset_name)
            self.save_training_curves(model, model_name, dataset_name)

        return training_time

    def save_samples(self, model, epoch, model_name, dataset_name):
        """Save a grid of generated samples (rank-0 only)."""
        if get_rank() != 0:
            return

        self.device_manager.empty_cache()
        samples = model.generate_samples(64)  # tensor in [-1, 1]
        samples = (samples + 1) / 2.0
        samples = torch.clamp(samples, 0, 1)

        grid = make_grid(samples, nrow=8, padding=2, normalize=False)
        save_dir = os.path.join(self.config.samples_dir, f"{model_name}_{dataset_name}")
        os.makedirs(save_dir, exist_ok=True)
        transforms.ToPILImage()(grid).save(os.path.join(save_dir, f'epoch_{epoch}.png'))
        print(f"✅ Samples saved: {save_dir}/epoch_{epoch}.png")

    def save_training_curves(self, model, model_name, dataset_name):
        """Save training loss curves (rank-0 only)."""
        if get_rank() != 0:
            return

        plt.figure(figsize=(10, 5))

        # Left: raw losses
        plt.subplot(1, 2, 1)
        plt.plot(model.g_losses, label='Generator Loss')
        plt.plot(model.d_losses, label='Discriminator Loss')
        plt.title(f'{model_name} Training Losses on {dataset_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Right: smoothed losses
        plt.subplot(1, 2, 2)
        window = 5
        if len(model.g_losses) > window:
            g_smooth = [sum(model.g_losses[i:i+window])/window for i in range(len(model.g_losses)-window+1)]
            d_smooth = [sum(model.d_losses[i:i+window])/window for i in range(len(model.d_losses)-window+1)]
            plt.plot(range(window-1, len(model.g_losses)), g_smooth, label='Generator (Smoothed)')
            plt.plot(range(window-1, len(model.d_losses)), d_smooth, label='Discriminator (Smoothed)')

        plt.title(f'{model_name} Smoothed Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_dir = os.path.join(self.config.plots_dir, f"{model_name}_{dataset_name}")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
