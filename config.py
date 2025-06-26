# config.py
import torch

class Config:
    """Configuration class for GAN comparison experiments"""
    
    def __init__(self):
        # Training parameters
        self.batch_size = 64
        self.epochs = 100
        self.learning_rate_g = 0.0002
        self.learning_rate_d = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.z_dim = 100
        
        # Model specific parameters
        self.ngf = 64  # Generator feature maps
        self.ndf = 64  # Discriminator feature maps
        
        # WGAN specific
        self.n_critic = 5
        self.clip_value = 0.01
        
        # WGAN-GP specific
        self.lambda_gp = 10
        
        # Evaluation parameters
        self.sample_interval = 10  # epochs
        self.eval_interval = 10
        self.n_eval_samples = 10000
        self.fid_batch_size = 256
        
        # Dataset parameters
        self.image_size = 32  # Will be adjusted per dataset
        self.num_channels = 3  # Will be adjusted per dataset
        
        # Paths
        self.results_dir = "results"
        self.models_dir = "results/models"
        self.samples_dir = "results/samples"
        self.metrics_dir = "results/metrics"
        self.plots_dir = "results/plots"
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        import os
        for dir_path in [self.results_dir, self.models_dir, self.samples_dir, 
                        self.metrics_dir, self.plots_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_dataset_config(self, dataset_name):
        """Get dataset-specific configuration"""
        configs = {
            'mnist': {
                'image_size': 28,
                'num_channels': 1,
                'num_classes': 10
            },
            'cifar10': {
                'image_size': 32,
                'num_channels': 3,
                'num_classes': 10
            },
            'celeba': {
                'image_size': 64,
                'num_channels': 3,
                'num_classes': 1
            }
        }
        return configs.get(dataset_name, {})
    
    def get_model_config(self, model_name):
        """Get model-specific configuration"""
        configs = {
            'vanilla': {
                'use_batch_norm': False,
                'use_conv': False
            },
            'dcgan': {
                'use_batch_norm': True,
                'use_conv': True
            },
            'wgan': {
                'use_batch_norm': True,
                'use_conv': True,
                'use_wasserstein': True
            },
            'wgan_gp': {
                'use_batch_norm': True,
                'use_conv': True,
                'use_wasserstein': True,
                'use_gradient_penalty': True
            }
        }
        return configs.get(model_name, {})