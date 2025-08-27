import os
import time
import gc

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch.distributed as dist

from utils.device_manager import DeviceManager
from utils.accelerator import rank as get_rank, world_size as get_world_size, local_rank, init_distributed


class GANTrainer:
    """GPU-optimized training utilities with optional distributed support"""

    def __init__(self, config):
        self.config = config
        # Initialise distributed process group if needed
        init_distributed()

        # If distributed, adjust device to local rank; otherwise fallback to provided device
        if get_world_size() > 1 and torch.cuda.is_available():
            self.config.device = torch.device(f'cuda:{local_rank()}')
        self.device_manager = DeviceManager(getattr(self.config, 'device', None))

        # Clear cache on start
        self.device_manager.empty_cache()
        if self.device_manager.is_cuda():
            if get_rank() == 0:
                print("🔧 GPU memory initialized")

    def train_model(self, model, dataloader, model_name, dataset_name):
        """GPU-optimized training loop with optional distributed support"""
        if get_rank() == 0:
            print(f"🚀 Training {model_name} on {dataset_name}")

        # Move model to appropriate device and wrap with DDP when distributed
        model = model.to(self.config.device)
        is_distributed = dist.is_available() and dist.is_initialized()
        if is_distributed:
            # Wrap model for multi-GPU training.  The `find_unused_parameters` flag
            # avoids errors in case some modules don't receive gradients every step.
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank()] if torch.cuda.is_available() else None,
                output_device=local_rank() if torch.cuda.is_available() else None,
                find_unused_parameters=True
            )

        # GPU memory info before training (only rank 0)
        if self.device_manager.is_cuda() and get_rank() == 0:
            print(f"📊 GPU Memory before training: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

        start_time = time.time()

        for epoch in range(self.config.epochs):
            epoch_g_losses = []
            epoch_d_losses = []

            # For distributed training, shuffle differently every epoch
            if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
                dataloader.sampler.set_epoch(epoch)

            # Progress bar (only rank 0 shows progress)
            if get_rank() == 0:
                pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.config.epochs}')
            else:
                # When not rank 0, iterate without tqdm
                pbar = dataloader

            for i, batch in enumerate(pbar):
                # GPU-OPTIMIZED DATA TRANSFER
                if dataset_name == 'celeba':
                    real_data = batch[0].to(self.config.device, non_blocking=True)  # NON_BLOCKING transfer
                else:
                    real_data, _ = batch
                    real_data = real_data.to(self.config.device, non_blocking=True)  # NON_BLOCKING transfer

                # Training step
                g_loss, d_loss = model.module.train_step(real_data) if is_distributed else model.train_step(real_data)

                # Store iteration-level losses
                model.module.g_losses_iter.append(g_loss) if is_distributed else model.g_losses_iter.append(g_loss)
                model.module.d_losses_iter.append(d_loss) if is_distributed else model.d_losses_iter.append(d_loss)

                epoch_g_losses.append(g_loss)
                epoch_d_losses.append(d_loss)

                # UPDATE PROGRESS with GPU memory info
                if self.device_manager.is_cuda() and i % 100 == 0:
                    if get_rank() == 0:
                        gpu_mem = torch.cuda.memory_allocated(local_rank()) / 1e9
                        pbar.set_postfix({
                            'G_loss': f'{g_loss:.4f}',
                            'D_loss': f'{d_loss:.4f}',
                            'GPU_GB': f'{gpu_mem:.1f}'
                        })
                    # If not rank 0, we don't update progress bar
                else:
                    if get_rank() == 0:
                        pbar.set_postfix({
                            'G_loss': f'{g_loss:.4f}',
                            'D_loss': f'{d_loss:.4f}'
                        })

            # Store epoch losses
            avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)
            avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)

            if is_distributed:
                # Sync average losses across processes (take mean over all ranks)
                tensor_g = torch.tensor(avg_g_loss, device=self.config.device)
                tensor_d = torch.tensor(avg_d_loss, device=self.config.device)
                dist.all_reduce(tensor_g, op=dist.ReduceOp.SUM)
                dist.all_reduce(tensor_d, op=dist.ReduceOp.SUM)
                avg_g_loss = tensor_g.item() / get_world_size()
                avg_d_loss = tensor_d.item() / get_world_size()

            # Append to history on each replica
            if is_distributed:
                model.module.g_losses.append(avg_g_loss)
                model.module.d_losses.append(avg_d_loss)
            else:
                model.g_losses.append(avg_g_loss)
                model.d_losses.append(avg_d_loss)

            # Save samples periodically (only on rank 0)
            if (epoch + 1) % self.config.sample_interval == 0 and get_rank() == 0:
                self.save_samples(model.module if is_distributed else model, epoch + 1, model_name, dataset_name)

            # Save model checkpoint and CLEAR GPU MEMORY every few epochs (only rank 0)
            if (epoch + 1) % 10 == 0 and get_rank() == 0:
                (model.module if is_distributed else model).save_models(epoch + 1, model_name, dataset_name)
                # GPU MEMORY CLEANUP
                self.device_manager.empty_cache()
                gc.collect()
                if self.device_manager.is_cuda():
                    print(f"GPU memory cleaned at epoch {epoch + 1}")

            if get_rank() == 0:
                print(f'Epoch [{epoch+1}/{self.config.epochs}] '
                      f'G_loss: {avg_g_loss:.4f} D_loss: {avg_d_loss:.4f}')

        training_time = time.time() - start_time

        # FINAL PERFORMANCE STATS (only rank 0)
        if self.device_manager.is_cuda() and get_rank() == 0:
            print(f"Final GPU Memory: {torch.cuda.memory_allocated(local_rank()) / 1e9:.2f} GB")
            print(f"Max GPU Memory: {torch.cuda.max_memory_allocated(local_rank()) / 1e9:.2f} GB")

        if get_rank() == 0:
            print(f"Training completed in {training_time:.2f} seconds")
            print(f"Average time per epoch: {training_time/self.config.epochs:.1f} seconds")
            # Save final model and training curves on rank 0 only
            (model.module if is_distributed else model).save_models(self.config.epochs, model_name, dataset_name)
            self.save_training_curves(model.module if is_distributed else model, model_name, dataset_name)

        return training_time

    def save_samples(self, model, epoch, model_name, dataset_name):
        """GPU-optimized sample generation"""
        if get_rank() == 0:
            print(f"Saving samples for epoch {epoch}...")

        # CLEAR MEMORY before sample generation
        self.device_manager.empty_cache()

        samples = model.generate_samples(64)

        # Create sample grid
        from torchvision.utils import make_grid
        import torchvision.transforms as transforms

        # Denormalize samples (from [-1, 1] to [0, 1])
        samples = (samples + 1) / 2.0
        samples = torch.clamp(samples, 0, 1)

        # Create grid
        grid = make_grid(samples, nrow=8, padding=2, normalize=False)

        # Save image
        save_dir = os.path.join(self.config.samples_dir, f"{model_name}_{dataset_name}")
        os.makedirs(save_dir, exist_ok=True)

        transforms.ToPILImage()(grid).save(
            os.path.join(save_dir, f'epoch_{epoch}.png')
        )

        if get_rank() == 0:
            print(f"✅ Samples saved: {save_dir}/epoch_{epoch}.png")

    def save_training_curves(self, model, model_name, dataset_name):
        """Save training loss curves"""
        if get_rank() != 0:
            return
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(model.g_losses, label='Generator Loss')
        plt.plot(model.d_losses, label='Discriminator Loss')
        plt.title(f'{model_name} Training Losses on {dataset_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        # Moving average for smoother curves
        window = 5
        if len(model.g_losses) > window:
            g_smooth = [sum(model.g_losses[i:i+window])/window
                       for i in range(len(model.g_losses)-window+1)]
            d_smooth = [sum(model.d_losses[i:i+window])/window
                       for i in range(len(model.d_losses)-window+1)]

            plt.plot(range(window-1, len(model.g_losses)), g_smooth,
                    label='Generator (Smoothed)')
            plt.plot(range(window-1, len(model.d_losses)), d_smooth,
                    label='Discriminator (Smoothed)')

        plt.title(f'{model_name} Smoothed Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # Save plot
        save_dir = os.path.join(self.config.plots_dir, f"{model_name}_{dataset_name}")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')