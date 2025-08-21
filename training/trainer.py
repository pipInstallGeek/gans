import torch
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
from utils.device_manager import DeviceManager

class GANTrainer:
    """GPU-optimized training utilities"""

    def __init__(self, config):
        self.config = config
        self.device_manager = DeviceManager(getattr(config, 'device', None))

        self.device_manager.empty_cache()
        if self.device_manager.is_cuda():
            print("🔧 GPU memory initialized")
    
    def train_model(self, model, dataloader, model_name, dataset_name):
        """GPU-optimized training loop"""
        print(f"🚀 Training {model_name} on {dataset_name}")

        if self.device_manager.is_cuda():
            print(f"📊 GPU Memory before training: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_g_losses = []
            epoch_d_losses = []
            
            # Progress bar
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.config.epochs}')
            
            for i, batch in enumerate(pbar):
                # GPU-OPTIMIZED DATA TRANSFER
                if dataset_name == 'celeba':
                    real_data = batch[0].to(self.config.device, non_blocking=True)  # NON_BLOCKING transfer
                else:
                    real_data, _ = batch
                    real_data = real_data.to(self.config.device, non_blocking=True)  # NON_BLOCKING transfer
                
                # Training step
                g_loss, d_loss = model.train_step(real_data)
                
                # Store iteration-level losses
                model.g_losses_iter.append(g_loss)
                model.d_losses_iter.append(d_loss)
                
                epoch_g_losses.append(g_loss)
                epoch_d_losses.append(d_loss)
                
                # UPDATE PROGRESS with GPU memory info
                if self.device_manager.is_cuda() and i % 100 == 0:  # Show GPU memory less frequently
                    gpu_mem = torch.cuda.memory_allocated(0) / 1e9
                    pbar.set_postfix({
                        'G_loss': f'{g_loss:.4f}',
                        'D_loss': f'{d_loss:.4f}',
                        'GPU_GB': f'{gpu_mem:.1f}'
                    })
                else:
                    pbar.set_postfix({
                        'G_loss': f'{g_loss:.4f}',
                        'D_loss': f'{d_loss:.4f}'
                    })
            
            # Store epoch losses
            avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)
            avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
            
            model.g_losses.append(avg_g_loss)
            model.d_losses.append(avg_d_loss)
            
            # Save samples periodically
            if (epoch + 1) % self.config.sample_interval == 0:
                self.save_samples(model, epoch + 1, model_name, dataset_name)
            
            # Save model checkpoint and CLEAR GPU MEMORY
            if (epoch + 1) % 10 == 0:  # More frequent saves for GPU
                model.save_models(epoch + 1, model_name, dataset_name)

                # GPU MEMORY CLEANUP
                self.device_manager.empty_cache()
                gc.collect()
                if self.device_manager.is_cuda():
                    print(f"GPU memory cleaned at epoch {epoch + 1}")
            
            print(f'Epoch [{epoch+1}/{self.config.epochs}] '
                  f'G_loss: {avg_g_loss:.4f} D_loss: {avg_d_loss:.4f}')
        
        training_time = time.time() - start_time
        
        # FINAL PERFORMANCE STATS
        if self.device_manager.is_cuda():
            print(f"Final GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"Max GPU Memory: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Average time per epoch: {training_time/self.config.epochs:.1f} seconds")        
        # Save final model
        model.save_models(self.config.epochs, model_name, dataset_name)
        self.save_training_curves(model, model_name, dataset_name)
        
        return training_time
    
    def save_samples(self, model, epoch, model_name, dataset_name):
        """GPU-optimized sample generation"""
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

        print(f"✅ Samples saved: {save_dir}/epoch_{epoch}.png")
    
    def save_training_curves(self, model, model_name, dataset_name):
        """Save training loss curves"""
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
        plt.close()