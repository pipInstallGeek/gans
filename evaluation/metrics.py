import torch
import numpy as np
import warnings
import os
import tempfile
import traceback
from typing import Dict, Optional
from torch_fidelity import calculate_metrics
from torchvision.utils import save_image
from tqdm import tqdm
from sklearn.cluster import KMeans
import prdc
from utils.device_manager import DeviceManager

warnings.filterwarnings("ignore")

def clear_cache():
    """Helper function to clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class StandardGANMetrics:
    """
    GAN Metrics using established, well-tested libraries
    Ensures reproducibility and matches academic standards
    """
    
    def __init__(self, device_manager):
        self.device_manager = device_manager
        self.device = device_manager.device
        
        print("="*80)
        print("STANDARD GAN METRICS INITIALIZED")
        print("Using torch-fidelity for all primary metrics (FID, IS, KID)")
        print("="*80)
    
    def calculate_metrics(self, real_images, fake_images):
        """
        Calculate all primary metrics (FID, IS, KID) using torch-fidelity
        """
        print("📊 Calculating metrics with torch-fidelity...")
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                real_dir = os.path.join(tmpdir, 'real')
                fake_dir = os.path.join(tmpdir, 'fake')
                os.makedirs(real_dir)
                os.makedirs(fake_dir)
                
                # Save images with progress bar
                print("Saving images for metrics calculation...")
                with tqdm(total=min(1000, len(real_images)), desc="Saving real images") as pbar:
                    for i in range(min(1000, len(real_images))):
                        save_image(real_images[i], os.path.join(real_dir, f'{i}.png'))
                        pbar.update(1)
                        
                with tqdm(total=min(1000, len(fake_images)), desc="Saving fake images") as pbar:
                    for i in range(min(1000, len(fake_images))):
                        save_image(fake_images[i], os.path.join(fake_dir, f'{i}.png'))
                        pbar.update(1)
                
                # Clear memory
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Calculate all metrics at once
                print("Computing metrics (this may take a while)...")
                metrics = calculate_metrics(
                    input1=real_dir,
                    input2=fake_dir,
                    cuda=torch.cuda.is_available(),
                    isc=True,  # Inception Score
                    fid=True,  # FID
                    kid=True,  # Kernel Inception Distance
                    verbose=True
                )
                
                # Print results
                print(f"\n✅ Metrics calculated:")
                print(f"  ⊛ FID: {metrics.get('frechet_inception_distance', 'N/A'):.4f}")
                print(f"  ⊛ IS: {metrics.get('inception_score_mean', 'N/A'):.4f} ± {metrics.get('inception_score_std', 'N/A'):.4f}")
                print(f"  ⊛ KID: {metrics.get('kernel_inception_distance', 'N/A'):.6f}")
                
                return metrics
                
        except Exception as e:
            print(f"❌ Metrics calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def calculate_prdc_metrics(self, real_features, fake_features):
        """
        Calculate Precision, Recall, Density, Coverage using prdc library
        These metrics are mentioned in recent GAN papers for better evaluation
        """
        print("📊 Calculating Precision, Recall, Density, Coverage...")
        
        try:
            # Limit samples for computational efficiency
            n_samples = min(5000, len(real_features), len(fake_features))
            real_features = real_features[:n_samples]
            fake_features = fake_features[:n_samples]
            
            # Calculate metrics
            metrics = prdc.compute_prdc(
                real_features=real_features,
                fake_features=fake_features,
                nearest_k=5
            )
            
            print(f"✅ Precision: {metrics['precision']:.3f}")
            print(f"✅ Recall: {metrics['recall']:.3f}")
            print(f"✅ Density: {metrics['density']:.3f}")
            print(f"✅ Coverage: {metrics['coverage']:.3f}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ PRDC metrics failed: {e}")
            return {}
    
    def calculate_mode_metrics(self, real_features, fake_features, n_modes=10):
        """
        PROPER mode coverage and collapse detection
        Using established clustering methods
        """
        print("📊 Calculating Mode Coverage and Collapse...")
        
        try:
            # Use KMeans to find modes in real data
            kmeans = KMeans(n_clusters=n_modes, n_init=10, random_state=42)
            real_labels = kmeans.fit_predict(real_features)
            real_centers = kmeans.cluster_centers_
            
            # Calculate coverage
            from scipy.spatial.distance import cdist
            
            # For each fake sample, find nearest real mode
            distances = cdist(fake_features, real_centers)
            nearest_modes = np.argmin(distances, axis=1)
            min_distances = np.min(distances, axis=1)
            
            # Calculate threshold based on real data statistics
            real_distances = cdist(real_features, real_centers)
            real_min_distances = np.min(real_distances, axis=1)
            threshold = np.percentile(real_min_distances, 95)
            
            # Count covered modes
            covered_samples = min_distances < threshold
            modes_covered = len(np.unique(nearest_modes[covered_samples]))
            mode_coverage = modes_covered / n_modes
            
            # Calculate mode distribution entropy (for collapse detection)
            mode_counts = np.bincount(nearest_modes[covered_samples], minlength=n_modes)
            mode_probs = mode_counts / (np.sum(mode_counts) + 1e-10)
            mode_probs = mode_probs[mode_probs > 0]
            
            if len(mode_probs) > 0:
                entropy = -np.sum(mode_probs * np.log(mode_probs + 1e-10))
                max_entropy = np.log(n_modes)
                mode_collapse_score = 1.0 - (entropy / max_entropy)
            else:
                mode_collapse_score = 1.0
            
            print(f"✅ Mode Coverage: {mode_coverage:.2%} ({modes_covered}/{n_modes})")
            print(f"✅ Mode Collapse Score: {mode_collapse_score:.3f}")
            
            return {
                'mode_coverage': mode_coverage,
                'modes_covered': modes_covered,
                'total_modes': n_modes,
                'mode_collapse_score': mode_collapse_score
            }
            
        except Exception as e:
            print(f"❌ Mode metrics failed: {e}")
            return {
                'mode_coverage': 0.0,
                'modes_covered': 0,
                'total_modes': n_modes,
                'mode_collapse_score': 1.0
            }


class GANEvaluator:
    """
    Main evaluator using torch-fidelity for all primary metrics
    """
    
    def __init__(self, config):
        self.config = config
        self.device_manager = DeviceManager(getattr(config, 'device', None))
        self.metrics = StandardGANMetrics(self.device_manager)
        
        print("\n" + "="*80)
        print(" GAN EVALUATOR WITH TORCH-FIDELITY")
        print("Using research-standard implementation for all metrics")
        print("="*80 + "\n")
    
    def evaluate_model(self, model, dataloader, dataset_name, model_name):
        """
        Comprehensive evaluation using torch-fidelity for all primary metrics
        """
        results = {}
        
        print(f"\n{'='*80}")
        print(f"EVALUATING {model_name} on {dataset_name}")
        print(f"Using torch-fidelity for all primary metrics")
        print(f"{'='*80}")
        
        try:
            # Generate samples
            n_samples = min(getattr(self.config, 'n_eval_samples', 5000), 10000)
            print(f"\n📦 Generating {n_samples} samples...")
            
            fake_samples = model.generate_samples(n_samples, return_tensor=True).cpu()
            
            # Collect real samples
            real_samples = []
            for i, batch in enumerate(dataloader):
                if dataset_name == 'celeba':
                    data = batch[0].cpu()
                else:
                    data = batch[0].cpu()
                real_samples.append(data)
                if len(real_samples) * data.size(0) >= n_samples:
                    break
            
            real_samples = torch.cat(real_samples)[:n_samples]
            
            print(f"✅ Collected {real_samples.size(0)} real, {fake_samples.size(0)} fake samples")
            
            # ========== CALCULATE ALL PRIMARY METRICS AT ONCE ==========
            print(f"\n{'='*50}")
            print("CALCULATING PRIMARY METRICS (FID, IS, KID)")
            print(f"{'='*50}")
            
            # Calculate all metrics using torch-fidelity
            metrics_results = self.metrics.calculate_metrics(real_samples, fake_samples)
            
            # Store metrics in results
            if metrics_results:
                results['fid'] = metrics_results.get('frechet_inception_distance', float('inf'))
                results['is_mean'] = metrics_results.get('inception_score_mean', 0.0)
                results['is_std'] = metrics_results.get('inception_score_std', 0.0)
                results['kid'] = metrics_results.get('kernel_inception_distance', 0.0)
            # ========== MODE METRICS ==========
            if getattr(self.config, 'calculate_mode_metrics', True):
                print(f"\n{'='*50}")
                print("MODE COVERAGE & COLLAPSE METRICS")
                print(f"{'='*50}")
                
                # Extract features for advanced metrics
                # Use InceptionV3 features
                from torchvision.models import inception_v3
                import torch.nn as nn
                
                inception = inception_v3(pretrained=True)
                
                # Properly replace the fc layer with a compatible layer
                in_features = inception.fc.in_features
                inception.fc = nn.Linear(in_features, in_features)
                # Initialize with identity-like weights
                with torch.no_grad():
                    inception.fc.weight.fill_(0)
                    # Set diagonal to 1 to mimic identity function
                    for i in range(in_features):
                        inception.fc.weight[i, i] = 1
                    inception.fc.bias.fill_(0)
                    
                inception.eval().to(self.device_manager.device)
                
                with torch.no_grad():
                    # Get features with progress bar
                    from tqdm import tqdm
                    
                    real_features = []
                    fake_features = []
                    
                    print("Extracting inception features for real images...")
                    for i in tqdm(range(0, len(real_samples), 32), desc="Real Features"):
                        batch = real_samples[i:i+32].to(self.device_manager.device)
                        if batch.size(1) == 1:
                            batch = batch.repeat(1, 3, 1, 1)
                        batch = torch.nn.functional.interpolate(batch, size=(299, 299))
                        features = inception(batch).cpu().numpy()
                        real_features.append(features)
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    print("Extracting inception features for fake images...")
                    for i in tqdm(range(0, len(fake_samples), 32), desc="Fake Features"):
                        batch = fake_samples[i:i+32].to(self.device_manager.device)
                        if batch.size(1) == 1:
                            batch = batch.repeat(1, 3, 1, 1)
                        batch = torch.nn.functional.interpolate(batch, size=(299, 299))
                        features = inception(batch).cpu().numpy()
                        fake_features.append(features)
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    real_features = np.concatenate(real_features)
                    fake_features = np.concatenate(fake_features)
                
                # Calculate mode metrics
                mode_metrics = self.metrics.calculate_mode_metrics(
                    real_features, fake_features, 
                    n_modes=getattr(self.config, 'n_modes', 10)
                )
                results.update(mode_metrics)
                
                # PRDC metrics if requested
                if getattr(self.config, 'calculate_prdc', True):
                    prdc_metrics = self.metrics.calculate_prdc_metrics(
                        real_features, fake_features
                    )
                    results.update(prdc_metrics)
            
            # ========== SUMMARY ==========
            print(f"\n{'='*80}")
            print(f"EVALUATION SUMMARY for {model_name}")
            print(f"{'='*80}")
            print(f"📊 FID: {results.get('fid', float('inf')):.4f} (lower is better)")
            print(f"📊 IS: {results.get('is_mean', 0):.4f} ± {results.get('is_std', 0):.4f} (higher is better)")
            print(f"📊 KID: {results.get('kid', 0):.6f} (lower is better)")
            
            if 'mode_coverage' in results:
                print(f"📊 Mode Coverage: {results.get('mode_coverage', 0):.2%}")
                print(f"📊 Mode Collapse Score: {results.get('mode_collapse_score', 1):.4f}")
            
            if 'precision' in results:
                print(f"📊 Precision: {results['precision']:.4f}")
                print(f"📊 Recall: {results['recall']:.4f}")
                print(f"📊 Density: {results['density']:.4f}")
                print(f"📊 Coverage: {results['coverage']:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'fid': float('inf'),
                'is_mean': 0.0,
                'is_std': 0.0,
                'kid': 0.0,
                'error': str(e)
            }




