import torch
import os

# Set GPU memory optimization flags
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class Config:
    """GPU-Optimized Configuration for GAN training"""
    
    def __init__(self):
        # GPU-OPTIMIZED TRAINING PARAMETERS
        self.batch_size = 128         
        self.epochs = 20              
        self.learning_rate_g = 0.0002 
        self.learning_rate_d = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.z_dim = 100              
        
        # FULL MODEL SIZE - GPU can handle complexity
        self.ngf = 64                 
        self.ndf = 64                 
        
        # WGAN specific
        self.n_critic = 5
        self.clip_value = 0.01
        self.lambda_gp = 10
        
        # GPU-OPTIMIZED EVALUATION (prevent memory issues)
        self.sample_interval = 5      
        self.eval_interval = 10
        self.n_eval_samples = 10000
        self.fid_batch_size = 32      
        
        # Dataset parameters
        self.image_size = 32
        self.num_channels = 3
        self.calculate_mode_metrics = True
        # Paths
        self.results_dir = "results"
        self.models_dir = "results/models"
        self.samples_dir = "results/samples"
        self.metrics_dir = "results/metrics"
        self.plots_dir = "results/plots"
        
        # GPU DEVICE WITH OPTIMIZATION
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # GPU-specific optimizations
        if torch.cuda.is_available():
            # Enable cuDNN optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Clear GPU cache on initialization
            torch.cuda.empty_cache()
            
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"Batch Size: {self.batch_size} (GPU optimized)")
            print(f"Model Size: ngf={self.ngf}, ndf={self.ndf} (Full complexity)")
        else:
            print("⚠️ No GPU available - falling back to CPU settings")
            # Emergency CPU fallback
            self.batch_size = 32
            self.ngf = 32
            self.ndf = 32
        
        # Create directories
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
            },
            'ffhq': {                    
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