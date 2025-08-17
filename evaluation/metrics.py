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

    def calculate_metrics(self, real_images, fake_images):
        """
        Calculate all primary metrics (FID, IS, KID) using torch-fidelity
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                real_dir = os.path.join(tmpdir, 'real')
                fake_dir = os.path.join(tmpdir, 'fake')
                os.makedirs(real_dir)
                os.makedirs(fake_dir)

                # Save images with progress bar
                with tqdm(total=len(real_images), desc="Saving real images") as pbar:
                    for i in range(len(real_images)):
                        save_image(real_images[i], os.path.join(real_dir, f'{i}.png'))
                        pbar.update(1)

                with tqdm(total=len(fake_images), desc="Saving fake images") as pbar:
                    for i in range(len(fake_images)):
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

                return metrics

        except Exception as e:
            print(f"❌ Metrics calculation failed: {e}")
            traceback.print_exc()
            return {}

    def calculate_prdc_metrics(self, real_features, fake_features):
        """
        Calculate Precision, Recall, Density, Coverage using prdc library
        """
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

            return metrics

        except Exception as e:
            print(f"❌ PRDC metrics failed: {e}")
            return {}

    def calculate_mode_metrics(self, real_features, fake_features, n_modes=10):
        """
        Mode coverage and collapse detection using established clustering methods
        """
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

        print("=" * 80)
        print(" GAN EVALUATOR WITH TORCH-FIDELITY")
        print("Using research-standard implementation for all metrics")
        print("=" * 80)

    def evaluate_model(self, model, dataloader, dataset_name, model_name):
        """
        Comprehensive evaluation using torch-fidelity for all primary metrics
        """
        results = {}

        print(f"\n{'=' * 80}")
        print(f"EVALUATING {model_name} on {dataset_name}")
        print(f"Using torch-fidelity for all primary metrics")
        print(f"{'=' * 80}")

        try:
            # Generate samples
            n_samples = getattr(self.config, 'n_eval_samples', 5000)
            print(f"\n📦 Generating {n_samples} samples...")

            fake_samples = model.generate_samples(n_samples, return_tensor=True).cpu()

            # Collect real samples
            real_samples = []
            for i, batch in enumerate(dataloader):
                data = batch[0].cpu()
                real_samples.append(data)
                if len(real_samples) * data.size(0) >= n_samples:
                    break

            real_samples = torch.cat(real_samples)[:n_samples]

            print(f"✅ Collected {real_samples.size(0)} real, {fake_samples.size(0)} fake samples")

            # ========== CALCULATE ALL PRIMARY METRICS AT ONCE ==========
            print(f"\n{'=' * 50}")
            print("CALCULATING PRIMARY METRICS (FID, IS, KID)")
            print(f"{'=' * 50}")

            # Calculate all metrics using torch-fidelity
            metrics_results = self.metrics.calculate_metrics(real_samples, fake_samples)

            # Store metrics in results with proper handling of 'N/A' values
            if metrics_results:
                results['fid'] = metrics_results.get('frechet_inception_distance', float('inf'))
                results['is_mean'] = metrics_results.get('inception_score_mean', 0.0)
                results['is_std'] = metrics_results.get('inception_score_std', 0.0)
                results['kid_mean'] = metrics_results.get('kernel_inception_distance_mean', 0.0)
                results['kid_std'] = metrics_results.get('kernel_inception_distance_std', 0.0)

                print(f"✅ Metrics calculated:")
                print(f"  ⊛ FID: {results['fid']:.4f}")
                print(f"  ⊛ IS: {results['is_mean']:.4f} ± {results['is_std']:.4f}")
                print(f"  ⊛ KID: {results['kid_mean']:.6f} ± {results['kid_std']:.6f}")

            # ========== MODE METRICS ==========
            if getattr(self.config, 'calculate_mode_metrics', True):
                print(f"\n{'=' * 50}")
                print("MODE COVERAGE & COLLAPSE METRICS")
                print(f"{'=' * 50}")

                # Extract features for advanced metrics
                from torchvision.models import inception_v3
                import torch.nn as nn

                inception = inception_v3(pretrained=True)

                # Properly replace the fc layer with a compatible layer
                in_features = inception.fc.in_features
                inception.fc = nn.Linear(in_features, in_features)
                # Initialize with identity-like weights
                with torch.no_grad():
                    inception.fc.weight.fill_(0)
                    for i in range(in_features):
                        inception.fc.weight[i, i] = 1
                    inception.fc.bias.fill_(0)

                inception.eval().to(self.device_manager.device)

                with torch.no_grad():
                    real_features = []
                    fake_features = []

                    print("Extracting inception features for real images...")
                    for i in tqdm(range(0, len(real_samples), 32), desc="Real Features"):
                        batch = real_samples[i:i + 32].to(self.device_manager.device)
                        if batch.size(1) == 1:
                            batch = batch.repeat(1, 3, 1, 1)
                        batch = torch.nn.functional.interpolate(batch, size=(299, 299))
                        features = inception(batch).cpu().numpy()
                        real_features.append(features)
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None

                    print("Extracting inception features for fake images...")
                    for i in tqdm(range(0, len(fake_samples), 32), desc="Fake Features"):
                        batch = fake_samples[i:i + 32].to(self.device_manager.device)
                        if batch.size(1) == 1:
                            batch = batch.repeat(1, 3, 1, 1)
                        batch = torch.nn.functional.interpolate(batch, size=(299, 299))
                        features = inception(batch).cpu().numpy()
                        fake_features.append(features)
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None

                    real_features = np.concatenate(real_features)
                    fake_features = np.concatenate(fake_features)

                # Calculate mode metrics
                print("📊 Calculating Mode Coverage and Collapse...")
                mode_metrics = self.metrics.calculate_mode_metrics(
                    real_features, fake_features,
                    n_modes=getattr(self.config, 'n_modes', 10)
                )
                results.update(mode_metrics)

                # Print mode metrics results
                print(
                    f"✅ Mode Coverage: {results.get('mode_coverage', 0):.2%} ({results.get('modes_covered', 0)}/{results.get('total_modes', 10)})")
                print(f"✅ Mode Collapse Score: {results.get('mode_collapse_score', 1):.3f}")

                # PRDC metrics if requested
                if getattr(self.config, 'calculate_prdc', True):
                    print("📊 Calculating Precision, Recall, Density, Coverage...")
                    prdc_metrics = self.metrics.calculate_prdc_metrics(
                        real_features, fake_features
                    )
                    results.update(prdc_metrics)

                    # Print PRDC results
                    if prdc_metrics:
                        print(f"✅ Precision: {prdc_metrics['precision']:.3f}")
                        print(f"✅ Recall: {prdc_metrics['recall']:.3f}")
                        print(f"✅ Density: {prdc_metrics['density']:.3f}")
                        print(f"✅ Coverage: {prdc_metrics['coverage']:.3f}")

            # ========== FINAL SUMMARY (CONSOLIDATED) ==========
            print(f"\n{'=' * 80}")
            print(f"EVALUATION SUMMARY for {model_name}")
            print(f"{'=' * 80}")
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

            print(f"Evaluation completed for {model_name} on {dataset_name}")
            return results

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            traceback.print_exc()
            return {
                'fid': float('inf'),
                'is_mean': 0.0,
                'is_std': 0.0,
                'kid': 0.0,
                'error': str(e)
            }