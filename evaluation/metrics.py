import torch
import numpy as np
import warnings
from typing import Dict, Tuple, Optional, List
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torch_fidelity import calculate_metrics
from cleanfid import fid as cleanfid
import prdc
from pytorch_gan_metrics import get_inception_score, get_fid, get_inception_score_and_fid
from pytorch_gan_metrics.core import get_inception_feature, calculate_frechet_distance
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.device_manager import DeviceManager

warnings.filterwarnings("ignore")


class StandardGANMetrics:
    """
    GAN Metrics using established, well-tested libraries
    Ensures reproducibility and matches academic standards
    """
    
    def __init__(self, device_manager):
        self.device_manager = device_manager
        self.device = device_manager.device
        
        # Initialize standard metric calculators
        self.metric_calculators = {}
        
        # torchmetrics FID - most reliable implementation
        self.fid_metric = FrechetInceptionDistance(
            feature=2048,  # Inception feature dimension
            reset_real_features=True,
            normalize=True
        ).to(self.device)
        
        # torchmetrics IS
        self.is_metric = InceptionScore(
            feature=2048,
            splits=10,
            normalize=True
        ).to(self.device)
        
        print("✅ Using torchmetrics for FID and IS (recommended)")
        
        print("="*80)
        print("STANDARD GAN METRICS INITIALIZED")
        print("✅ All metric libraries loaded successfully")
        print("="*80)
    
    def calculate_fid_torchmetrics(self, real_images, fake_images):
        """
        Calculate FID using torchmetrics (most reliable)
        """
        print("📊 Calculating FID with torchmetrics...")
        
        try:
            # Reset metric
            self.fid_metric.reset()
            
            # Process in batches to avoid memory issues
            batch_size = 50
            
            # Update with real images
            for i in range(0, len(real_images), batch_size):
                batch = real_images[i:i+batch_size].to(self.device)
                if batch.size(1) == 1:  # Convert grayscale to RGB
                    batch = batch.repeat(1, 3, 1, 1)
                self.fid_metric.update(batch, real=True)
            
            # Update with fake images
            for i in range(0, len(fake_images), batch_size):
                batch = fake_images[i:i+batch_size].to(self.device)
                if batch.size(1) == 1:  # Convert grayscale to RGB
                    batch = batch.repeat(1, 3, 1, 1)
                self.fid_metric.update(batch, real=False)
            
            # Compute FID
            fid_score = self.fid_metric.compute().item()
            print(f"✅ FID (torchmetrics): {fid_score:.2f}")
            
            return fid_score
            
        except Exception as e:
            print(f"❌ FID calculation failed: {e}")
            return None
    
    def calculate_is_torchmetrics(self, fake_images):
        """
        Calculate IS using torchmetrics
        """
        print("📊 Calculating IS with torchmetrics...")
        
        try:
            # Reset metric
            self.is_metric.reset()
            
            # Process in batches
            batch_size = 50
            
            for i in range(0, len(fake_images), batch_size):
                batch = fake_images[i:i+batch_size].to(self.device)
                if batch.size(1) == 1:  # Convert grayscale to RGB
                    batch = batch.repeat(1, 3, 1, 1)
                self.is_metric.update(batch)
            
            # Compute IS
            is_mean, is_std = self.is_metric.compute()
            print(f"✅ IS (torchmetrics): {is_mean:.2f} ± {is_std:.2f}")
            
            return is_mean.item(), is_std.item()
            
        except Exception as e:
            print(f"❌ IS calculation failed: {e}")
            return None, None
    
    def calculate_metrics_torch_fidelity(self, real_images, fake_images):
        """
        Calculate multiple metrics using torch-fidelity
        Very reliable library used in many papers
        """
        print("📊 Calculating metrics with torch-fidelity...")
        
        try:
            # Save images temporarily for torch-fidelity
            import tempfile
            import os
            from torchvision.utils import save_image
            
            with tempfile.TemporaryDirectory() as tmpdir:
                real_dir = os.path.join(tmpdir, 'real')
                fake_dir = os.path.join(tmpdir, 'fake')
                os.makedirs(real_dir)
                os.makedirs(fake_dir)
                
                # Save images
                for i in range(min(1000, len(real_images))):
                    save_image(real_images[i], os.path.join(real_dir, f'{i}.png'))
                
                for i in range(min(1000, len(fake_images))):
                    save_image(fake_images[i], os.path.join(fake_dir, f'{i}.png'))
                
                # Calculate metrics
                metrics = calculate_metrics(
                    input1=real_dir,
                    input2=fake_dir,
                    cuda=torch.cuda.is_available(),
                    isc=True,  # Inception Score
                    fid=True,  # FID
                    kid=True,  # Kernel Inception Distance
                    verbose=False
                )
                
                print(f"✅ torch-fidelity metrics calculated")
                return metrics
                
        except Exception as e:
            print(f"❌ torch-fidelity failed: {e}")
            return {}
    
    def calculate_cleanfid(self, real_images, fake_images):
        """
        Calculate FID using clean-fid (addresses limitations of standard FID)
        """
        print("📊 Calculating Clean-FID...")
        
        try:
            # clean-fid requires numpy arrays
            real_np = real_images.cpu().numpy()
            fake_np = fake_images.cpu().numpy()
            
            # Transpose to (N, H, W, C) format
            if real_np.shape[1] in [1, 3]:
                real_np = np.transpose(real_np, (0, 2, 3, 1))
                fake_np = np.transpose(fake_np, (0, 2, 3, 1))
            
            # Calculate clean-fid
            fid_score = cleanfid.compute_fid(fake_np, real_np)
            print(f"✅ Clean-FID: {fid_score:.2f}")
            
            return fid_score
            
        except Exception as e:
            print(f"❌ Clean-FID failed: {e}")
            return None
    
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
    Main evaluator using standard libraries
    Combines multiple established implementations for robustness
    """
    
    def __init__(self, config):
        self.config = config
        self.device_manager = DeviceManager(getattr(config, 'device', None))
        self.metrics = StandardGANMetrics(self.device_manager)
        
        print("\n" + "="*80)
        print("ENHANCED GAN EVALUATOR WITH STANDARD LIBRARIES")
        print("Using established, peer-reviewed implementations")
        print("="*80 + "\n")
    
    def evaluate_model(self, model, dataloader, dataset_name, model_name):
        """
        Comprehensive evaluation using standard libraries
        """
        results = {}
        
        print(f"\n{'='*80}")
        print(f"EVALUATING {model_name} on {dataset_name}")
        print(f"Using standard library implementations")
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
            
            # ========== PRIMARY METRICS (YOUR PAPER) ==========
            print(f"\n{'='*50}")
            print("PRIMARY METRICS (FID & IS from your paper)")
            print(f"{'='*50}")
            
            # 1. FID - Try multiple implementations for robustness
            fid_scores = {}
            
            # Calculate FID with torchmetrics
            fid_tm = self.metrics.calculate_fid_torchmetrics(real_samples, fake_samples)
            if fid_tm is not None:
                fid_scores['torchmetrics'] = fid_tm
            
            # Calculate FID with torch-fidelity
            tf_metrics = self.metrics.calculate_metrics_torch_fidelity(real_samples, fake_samples)
            if 'frechet_inception_distance' in tf_metrics:
                fid_scores['torch_fidelity'] = tf_metrics['frechet_inception_distance']
            
            # Calculate FID with clean-fid
            fid_clean = self.metrics.calculate_cleanfid(real_samples, fake_samples)
            if fid_clean is not None:
                fid_scores['clean_fid'] = fid_clean
            
            # Calculate FID with pytorch-gan-metrics
            try:
                fid_pgm = get_fid(fake_samples, real_samples)
                fid_scores['pytorch_gan_metrics'] = fid_pgm
            except Exception as e:
                print(f"⚠️ pytorch-gan-metrics FID failed: {e}")
            
            # Use mean of available FID scores for robustness
            if fid_scores:
                results['fid'] = np.mean(list(fid_scores.values()))
                results['fid_scores'] = fid_scores
                print(f"\n📊 FID (averaged): {results['fid']:.2f}")
                print(f"   Individual scores: {fid_scores}")
            else:
                results['fid'] = float('inf')
            
            # 2. IS - Inception Score
            # Calculate IS with torchmetrics
            is_mean, is_std = self.metrics.calculate_is_torchmetrics(fake_samples)
            if is_mean is not None:
                results['is_mean'] = is_mean
                results['is_std'] = is_std
            else:
                # Try with pytorch-gan-metrics as backup
                try:
                    is_score = get_inception_score(fake_samples)
                    results['is_mean'] = is_score
                    results['is_std'] = 0.0
                except Exception as e:
                    print(f"⚠️ pytorch-gan-metrics IS failed: {e}")
                    results['is_mean'] = 0.0
                    results['is_std'] = 0.0
            
            # ========== MODE METRICS ==========
            print(f"\n{'='*50}")
            print("MODE COVERAGE & COLLAPSE METRICS")
            print(f"{'='*50}")
            
            # Extract features for advanced metrics
            # Use InceptionV3 features
            from torchvision.models import inception_v3
            import torch.nn as nn
            
            inception = inception_v3(pretrained=True)
            inception.fc = nn.Identity()  # type: ignore
            inception.eval().to(self.device_manager.device)
            
            with torch.no_grad():
                # Get features
                real_features = []
                fake_features = []
                
                for i in range(0, len(real_samples), 32):
                    batch = real_samples[i:i+32].to(self.device_manager.device)
                    if batch.size(1) == 1:
                        batch = batch.repeat(1, 3, 1, 1)
                    batch = torch.nn.functional.interpolate(batch, size=(299, 299))
                    features = inception(batch).cpu().numpy()
                    real_features.append(features)
                
                for i in range(0, len(fake_samples), 32):
                    batch = fake_samples[i:i+32].to(self.device_manager.device)
                    if batch.size(1) == 1:
                        batch = batch.repeat(1, 3, 1, 1)
                    batch = torch.nn.functional.interpolate(batch, size=(299, 299))
                    features = inception(batch).cpu().numpy()
                    fake_features.append(features)
                
                real_features = np.concatenate(real_features)
                fake_features = np.concatenate(fake_features)
            
            # Calculate mode metrics
            mode_metrics = self.metrics.calculate_mode_metrics(
                real_features, fake_features, 
                n_modes=getattr(self.config, 'n_modes', 10)
            )
            results.update(mode_metrics)
            
            # PRDC metrics
            prdc_metrics = self.metrics.calculate_prdc_metrics(
                real_features, fake_features
            )
            results.update(prdc_metrics)
            
            # ========== SUMMARY ==========
            print(f"\n{'='*80}")
            print(f"EVALUATION SUMMARY for {model_name}")
            print(f"{'='*80}")
            print(f"📊 FID: {results.get('fid', 'N/A'):.2f} (lower is better)")
            print(f"📊 IS: {results.get('is_mean', 0):.2f} ± {results.get('is_std', 0):.2f} (higher is better)")
            print(f"📊 Mode Coverage: {results.get('mode_coverage', 0):.2%}")
            print(f"📊 Mode Collapse Score: {results.get('mode_collapse_score', 1):.3f}")
            
            if 'precision' in results:
                print(f"📊 Precision: {results['precision']:.3f}")
                print(f"📊 Recall: {results['recall']:.3f}")
                print(f"📊 Density: {results['density']:.3f}")
                print(f"📊 Coverage: {results['coverage']:.3f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return {
                'fid': float('inf'),
                'is_mean': 0.0,
                'is_std': 0.0,
                'error': str(e)
            }




