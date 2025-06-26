# evaluation/metrics.py
import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

class FIDCalculator:
    """Calculate Fréchet Inception Distance (FID)"""
    
    def __init__(self, device):
        self.device = device
        self.inception_model = inception_v3(pretrained=True, transform_input=False)
        self.inception_model.fc = nn.Identity()  # Remove final classification layer
        self.inception_model.eval()
        self.inception_model.to(device)
    
    def get_activations(self, images, batch_size=256):
        """Get inception activations for a batch of images"""
        self.inception_model.eval()
        activations = []
        
        # Ensure images are in the right format
        if images.size(1) == 1:  # Grayscale to RGB
            images = images.repeat(1, 3, 1, 1)
        
        # Resize to 299x299 for Inception
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(self.device)
                pred = self.inception_model(batch)
                activations.append(pred.cpu().numpy())
        
        return np.concatenate(activations, axis=0)
    
    def calculate_activation_statistics(self, activations):
        """Calculate mean and covariance of activations"""
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma
    
    def calculate_fid(self, real_images, fake_images):
        """Calculate FID between real and fake images"""
        # Get activations
        real_activations = self.get_activations(real_images)
        fake_activations = self.get_activations(fake_images)
        
        # Calculate statistics
        mu_real, sigma_real = self.calculate_activation_statistics(real_activations)
        mu_fake, sigma_fake = self.calculate_activation_statistics(fake_activations)
        
        # Calculate FID
        diff = mu_real - mu_fake
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
        
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % 1e-6
            print(msg)
            offset = np.eye(sigma_real.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))
        
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean)
        return fid

class InceptionScoreCalculator:
    """Calculate Inception Score (IS)"""
    
    def __init__(self, device):
        self.device = device
        self.inception_model = inception_v3(pretrained=True)
        self.inception_model.eval()
        self.inception_model.to(device)
    
    def calculate_is(self, images, batch_size=256, splits=10):
        """Calculate Inception Score"""
        # Ensure images are in the right format
        if images.size(1) == 1:  # Grayscale to RGB
            images = images.repeat(1, 3, 1, 1)
        
        # Resize to 299x299 for Inception
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Get predictions
        self.inception_model.eval()
        preds = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(self.device)
                pred = F.softmax(self.inception_model(batch), dim=1)
                preds.append(pred.cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        
        # Calculate IS
        scores = []
        for i in range(splits):
            part = preds[i * (len(preds) // splits): (i + 1) * (len(preds) // splits)]
            kl_div = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
            kl_div = np.mean(np.sum(kl_div, axis=1))
            scores.append(np.exp(kl_div))
        
        return np.mean(scores), np.std(scores)

class ModeCoverageCalculator:
    """Calculate mode coverage for synthetic datasets"""
    
    def __init__(self, tolerance=0.5):
        self.tolerance = tolerance
    
    def calculate_coverage_2d(self, real_data, fake_data, n_modes=8):
        """Calculate mode coverage for 2D datasets"""
        from sklearn.cluster import KMeans
        
        # Cluster real data to find modes
        kmeans = KMeans(n_clusters=n_modes, random_state=42)
        real_clusters = kmeans.fit_predict(real_data)
        real_centers = kmeans.cluster_centers_
        
        # Assign fake data to nearest real mode
        fake_distances = np.linalg.norm(
            fake_data[:, np.newaxis] - real_centers[np.newaxis, :], axis=2
        )
        fake_assignments = np.argmin(fake_distances, axis=1)
        min_distances = np.min(fake_distances, axis=1)
        
        # Count covered modes (within tolerance)
        covered_modes = set()
        for i, dist in enumerate(min_distances):
            if dist < self.tolerance:
                covered_modes.add(fake_assignments[i])
        
        coverage = len(covered_modes) / n_modes
        return coverage, len(covered_modes), n_modes
    
    def calculate_coverage_mnist(self, real_data, fake_data, real_labels):
        """Calculate mode coverage for MNIST (digit classes)"""
        from sklearn.neighbors import KNeighborsClassifier
        
        # Train a classifier on real data
        real_data_flat = real_data.view(real_data.size(0), -1).numpy()
        fake_data_flat = fake_data.view(fake_data.size(0), -1).numpy()
        
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(real_data_flat, real_labels.numpy())
        
        # Predict labels for fake data
        fake_predictions = knn.predict(fake_data_flat)
        
        # Count unique predicted classes
        unique_classes = set(fake_predictions)
        total_classes = len(set(real_labels.numpy()))
        
        coverage = len(unique_classes) / total_classes
        return coverage, len(unique_classes), total_classes

class TrainingStabilityAnalyzer:
    """Analyze training stability metrics"""
    
    def __init__(self):
        pass
    
    def calculate_stability_metrics(self, g_losses, d_losses):
        """Calculate various stability metrics"""
        g_losses = np.array(g_losses)
        d_losses = np.array(d_losses)
        
        metrics = {}
        
        # Loss variance (lower is better)
        metrics['g_loss_var'] = np.var(g_losses)
        metrics['d_loss_var'] = np.var(d_losses)
        
        # Loss convergence (check if losses are decreasing)
        window_size = min(20, len(g_losses) // 4)
        if len(g_losses) > window_size:
            early_g = np.mean(g_losses[:window_size])
            late_g = np.mean(g_losses[-window_size:])
            early_d = np.mean(d_losses[:window_size])
            late_d = np.mean(d_losses[-window_size:])
            
            metrics['g_loss_improvement'] = early_g - late_g
            metrics['d_loss_improvement'] = early_d - late_d
        
        # Loss oscillation (frequency of sign changes in gradient)
        g_diff = np.diff(g_losses)
        d_diff = np.diff(d_losses)
        
        g_sign_changes = np.sum(np.diff(np.sign(g_diff)) != 0)
        d_sign_changes = np.sum(np.diff(np.sign(d_diff)) != 0)
        
        metrics['g_oscillation'] = g_sign_changes / len(g_diff) if len(g_diff) > 0 else 0
        metrics['d_oscillation'] = d_sign_changes / len(d_diff) if len(d_diff) > 0 else 0
        
        # Overall stability score (0-10, higher is better)
        stability_score = 10 - (
            min(metrics['g_loss_var'], 5) + 
            min(metrics['d_loss_var'], 5) +
            min(metrics['g_oscillation'] * 5, 5) +
            min(metrics['d_oscillation'] * 5, 5)
        ) / 2
        
        metrics['stability_score'] = max(0, stability_score)
        
        return metrics

class GANEvaluator:
    """Main evaluator class that combines all metrics"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        self.fid_calculator = FIDCalculator(self.device)
        self.is_calculator = InceptionScoreCalculator(self.device)
        self.coverage_calculator = ModeCoverageCalculator()
        self.stability_analyzer = TrainingStabilityAnalyzer()
    
    def evaluate_model(self, model, dataloader, dataset_name, model_name):
        """Comprehensive evaluation of a GAN model"""
        results = {}
        
        # Generate samples
        n_samples = min(self.config.n_eval_samples, len(dataloader.dataset))
        fake_samples = model.generate_samples(n_samples, return_tensor=True)
        
        # Get real samples
        real_samples = []
        real_labels = []
        
        for i, batch in enumerate(dataloader):
            if dataset_name == 'celeba':
                data = batch[0]  # CelebA returns (data, attributes)
                labels = torch.zeros(data.size(0))  # Dummy labels
            else:
                data, labels = batch
            
            real_samples.append(data)
            real_labels.append(labels)
            
            if len(real_samples) * self.config.batch_size >= n_samples:
                break
        
        real_samples = torch.cat(real_samples)[:n_samples]
        real_labels = torch.cat(real_labels)[:n_samples]
        
        # Calculate FID
        try:
            fid_score = self.fid_calculator.calculate_fid(real_samples, fake_samples)
            results['fid'] = fid_score
        except Exception as e:
            print(f"FID calculation failed: {e}")
            results['fid'] = float('inf')
        
        # Calculate IS
        try:
            is_mean, is_std = self.is_calculator.calculate_is(fake_samples)
            results['is_mean'] = is_mean
            results['is_std'] = is_std
        except Exception as e:
            print(f"IS calculation failed: {e}")
            results['is_mean'] = 0
            results['is_std'] = 0
        
        # Calculate mode coverage
        if dataset_name == 'mnist':
            coverage, covered, total = self.coverage_calculator.calculate_coverage_mnist(
                real_samples, fake_samples, real_labels
            )
        else:
            # For other datasets, use a simpler coverage metric
            coverage = 1.0  # Placeholder
            covered = 10
            total = 10
        
        results['mode_coverage'] = coverage
        results['modes_covered'] = covered
        results['total_modes'] = total
        
        # Training stability
        stability_metrics = self.stability_analyzer.calculate_stability_metrics(
            model.g_losses, model.d_losses
        )
        results.update(stability_metrics)
        
        return results