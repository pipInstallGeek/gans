# visualization/sample_grid.py
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os

class SampleGridGenerator:
    """Generate and save sample grids for visualization"""
    
    def __init__(self, config):
        self.config = config
    
    def generate_comparison_grid(self, models_dict, dataset_name, n_samples=64):
        """Generate comparison grid for multiple models"""
        
        fig, axes = plt.subplots(1, len(models_dict), figsize=(5*len(models_dict), 5))
        if len(models_dict) == 1:
            axes = [axes]
        
        fig.suptitle(f'Generated Samples Comparison - {dataset_name.upper()}', 
                    fontsize=16, fontweight='bold')
        
        for i, (model_name, model) in enumerate(models_dict.items()):
            # Generate samples
            samples = model.generate_samples(n_samples)
            
            # Denormalize
            samples = (samples + 1) / 2.0
            samples = torch.clamp(samples, 0, 1)
            
            # Create grid
            grid = vutils.make_grid(samples, nrow=8, padding=2, normalize=False)
            
            # Convert to numpy and plot
            grid_np = grid.permute(1, 2, 0).cpu().numpy()
            if grid_np.shape[2] == 1:  # Grayscale
                grid_np = grid_np.squeeze(2)
                axes[i].imshow(grid_np, cmap='gray')
            else:  # RGB
                axes[i].imshow(grid_np)
            
            axes[i].set_title(f'{model_name.upper()}', fontweight='bold')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.config.plots_dir, f'comparison_grid_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_interpolation_grid(self, model, dataset_name, n_interpolations=10):
        """Generate interpolation between random points"""
        
        model.generator.eval()
        
        with torch.no_grad():
            # Generate two random points
            z1 = torch.randn(1, self.config.z_dim, device=self.config.device)
            z2 = torch.randn(1, self.config.z_dim, device=self.config.device)
            
            # Interpolate
            interpolations = []
            for i in range(n_interpolations):
                alpha = i / (n_interpolations - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                sample = model.generator(z_interp)
                interpolations.append(sample)
            
            # Stack samples
            interpolations = torch.cat(interpolations, dim=0)
            
            # Denormalize
            interpolations = (interpolations + 1) / 2.0
            interpolations = torch.clamp(interpolations, 0, 1)
            
            # Create grid
            grid = vutils.make_grid(interpolations, nrow=n_interpolations, padding=2, normalize=False)
            
            # Plot
            plt.figure(figsize=(15, 3))
            grid_np = grid.permute(1, 2, 0).cpu().numpy()
            
            if grid_np.shape[2] == 1:  # Grayscale
                grid_np = grid_np.squeeze(2)
                plt.imshow(grid_np, cmap='gray')
            else:  # RGB
                plt.imshow(grid_np)
            
            plt.title(f'Latent Space Interpolation - {dataset_name.upper()}', 
                     fontsize=14, fontweight='bold')
            plt.axis('off')
            
            # Save
            save_path = os.path.join(self.config.plots_dir, f'interpolation_{dataset_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return save_path