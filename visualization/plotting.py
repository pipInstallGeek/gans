# visualization/plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json
import torch
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d

class ResultsVisualizer:
    """Create visualizations for GAN comparison results"""
    
    def __init__(self, config):
        self.config = config
        self.results_dir = config.results_dir
        self.plots_dir = config.plots_dir
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_results(self):
        """Load results from JSON files"""
        training_file = os.path.join(self.results_dir, 'training_results.json')
        evaluation_file = os.path.join(self.results_dir, 'evaluation_results.json')
        
        training_results = {}
        evaluation_results = {}
        
        if os.path.exists(training_file):
            with open(training_file, 'r') as f:
                training_results = json.load(f)
        
        if os.path.exists(evaluation_file):
            with open(evaluation_file, 'r') as f:
                evaluation_results = json.load(f)
        
        return training_results, evaluation_results
    
    def create_all_visualizations(self, model_names, dataset_names):
        """Create all visualization plots"""
        training_results, evaluation_results = self.load_results()
        
        if not training_results and not evaluation_results:
            print("No results found to visualize")
            return
        
        print("Creating visualizations...")
        
        # 1. Training curves comparison
        self.plot_training_curves_comparison(model_names, dataset_names)
        
        # 2. Metrics comparison
        self.plot_metrics_comparison(evaluation_results, model_names, dataset_names)
        
        # 3. Performance radar charts
        self.plot_performance_radar(evaluation_results, model_names, dataset_names)
        
        # 4. Training stability analysis
        self.plot_stability_analysis(evaluation_results, model_names, dataset_names)
        
        # 5. Sample quality progression
        self.plot_sample_progression(model_names, dataset_names)
        
        # 6. Comprehensive comparison dashboard
        self.create_comparison_dashboard(training_results, evaluation_results, 
                                      model_names, dataset_names)
        
        # 7. Loss landscape visualization
        self.plot_loss_landscape(model_names, dataset_names)
        
        # 8. Quality-diversity tradeoff
        self.plot_quality_diversity_tradeoff(evaluation_results, model_names, dataset_names)
        
        # 9. Mode collapse visualization
        self.visualize_mode_collapse(evaluation_results, model_names, dataset_names)
        
        # 10. Metrics correlation analysis
        self.plot_metric_correlations(evaluation_results, model_names, dataset_names)
        
        # 11. Convergence analysis
        self.plot_convergence_analysis(model_names, dataset_names)
        
        # 12. Output diversity comparison
        self.plot_output_diversity(model_names, dataset_names)
        
        self.plot_iteration_losses(model_names, dataset_names)
        print(f"All visualizations saved to {self.plots_dir}")
    
    def plot_training_curves_comparison(self, model_names, dataset_names):
        """Plot training curves for all models side by side"""
        
        for dataset_name in dataset_names:
            fig, axes = plt.subplots(2, len(model_names), figsize=(4*len(model_names), 8))
            if len(model_names) == 1:
                axes = axes.reshape(-1, 1)
            
            fig.suptitle(f'Training Curves Comparison - {dataset_name.upper()}', 
                        fontsize=16, fontweight='bold')
            
            for i, model_name in enumerate(model_names):
                # Try to load individual model training history
                model_dir = os.path.join(self.config.models_dir, f"{model_name}_{dataset_name}")
                
                if os.path.exists(model_dir):
                    # Find latest checkpoint
                    checkpoint_files = [f for f in os.listdir(model_dir) 
                                      if f.startswith('checkpoint_')]
                    if checkpoint_files:
                        latest_checkpoint = sorted(checkpoint_files)[-1]
                        checkpoint_path = os.path.join(model_dir, latest_checkpoint)
                        
                        try:
                            checkpoint = torch.load(checkpoint_path, map_location='cpu')
                            g_losses = checkpoint.get('g_losses', [])
                            d_losses = checkpoint.get('d_losses', [])
                            
                            # Plot generator loss
                            axes[0, i].plot(g_losses, label='Generator', color='blue', alpha=0.7)
                            axes[0, i].set_title(f'{model_name.upper()}\nGenerator Loss')
                            axes[0, i].set_xlabel('Epoch')
                            axes[0, i].set_ylabel('Loss')
                            axes[0, i].grid(True, alpha=0.3)
                            
                            # Plot discriminator loss
                            axes[1, i].plot(d_losses, label='Discriminator', color='red', alpha=0.7)
                            axes[1, i].set_title(f'Discriminator Loss')
                            axes[1, i].set_xlabel('Epoch')
                            axes[1, i].set_ylabel('Loss')
                            axes[1, i].grid(True, alpha=0.3)
                            
                        except Exception as e:
                            print(f"Could not load training curves for {model_name}: {e}")
                            axes[0, i].text(0.5, 0.5, 'No Data', ha='center', va='center')
                            axes[1, i].text(0.5, 0.5, 'No Data', ha='center', va='center')
                else:
                    axes[0, i].text(0.5, 0.5, 'No Data', ha='center', va='center')
                    axes[1, i].text(0.5, 0.5, 'No Data', ha='center', va='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'training_curves_{dataset_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_metrics_comparison(self, evaluation_results, model_names, dataset_names):
        """Plot metrics comparison bar charts"""
        
        # Updated metrics to match your evaluation system
        metrics = ['fid', 'is_mean', 'mode_coverage', 'mode_collapse_score']
        metric_names = ['FID (↓)', 'Inception Score (↑)', 'Mode Coverage (↑)', 'Mode Collapse Score (↓)']
        
        for dataset_name in dataset_names:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            fig.suptitle(f'Performance Metrics Comparison - {dataset_name.upper()}', 
                        fontsize=16, fontweight='bold')
            
            for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
                values = []
                labels = []
                
                for model_name in model_names:
                    key = f"{model_name}_{dataset_name}"
                    if key in evaluation_results and metric in evaluation_results[key]:
                        value = evaluation_results[key][metric]
                        if not np.isnan(value) and np.isfinite(value):
                            values.append(value)
                            labels.append(model_name.upper())
                
                if values:
                    colors = sns.color_palette("husl", len(values))
                    bars = axes[i].bar(labels, values, color=colors, alpha=0.8)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
                    
                    axes[i].set_title(metric_name, fontweight='bold')
                    axes[i].set_ylabel('Score')
                    axes[i].grid(True, alpha=0.3, axis='y')
                    
                    # Rotate x-axis labels if needed
                    if len(labels) > 3:
                        axes[i].tick_params(axis='x', rotation=45)
                else:
                    axes[i].text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
                               transform=axes[i].transAxes)
                    axes[i].set_title(metric_name, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'metrics_comparison_{dataset_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_performance_radar(self, evaluation_results, model_names, dataset_names):
        """Create radar charts for performance comparison"""
        
        for dataset_name in dataset_names:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # Define metrics and their ranges
            metrics = ['fid', 'is_mean', 'mode_coverage', 'precision', 'recall']
            metric_labels = ['FID\n(Lower Better)', 'Inception Score\n(Higher Better)', 
                           'Mode Coverage\n(Higher Better)', 'Precision\n(Higher Better)',
                           'Recall\n(Higher Better)']
            
            # Collect all values for normalization
            all_values = {metric: [] for metric in metrics}
            for model_name in model_names:
                key = f"{model_name}_{dataset_name}"
                if key in evaluation_results:
                    for metric in metrics:
                        if metric in evaluation_results[key]:
                            value = evaluation_results[key][metric]
                            if np.isfinite(value):
                                all_values[metric].append(value)
            
            # Calculate angles for radar chart
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            colors = sns.color_palette("husl", len(model_names))
            
            for i, model_name in enumerate(model_names):
                key = f"{model_name}_{dataset_name}"
                if key not in evaluation_results:
                    continue
                
                values = []
                for metric in metrics:
                    if metric in evaluation_results[key]:
                        value = evaluation_results[key][metric]
                        if np.isfinite(value):
                            # Normalize values to 0-1 scale
                            if metric == 'fid':  # Lower is better for FID
                                min_val = min(all_values[metric]) if all_values[metric] else 0
                                max_val = max(all_values[metric]) if all_values[metric] else 1
                                normalized = 1 - (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                            else:  # Higher is better for others
                                min_val = min(all_values[metric]) if all_values[metric] else 0
                                max_val = max(all_values[metric]) if all_values[metric] else 1
                                normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                            values.append(normalized)
                        else:
                            values.append(0)
                    else:
                        values.append(0)
                
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=model_name.upper(), 
                       color=colors[i], alpha=0.8)
                ax.fill(angles, values, alpha=0.1, color=colors[i])
            
            # Customize the chart
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
            ax.grid(True)
            
            plt.title(f'Performance Radar Chart - {dataset_name.upper()}', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'radar_chart_{dataset_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_stability_analysis(self, evaluation_results, model_names, dataset_names):
        """Plot detailed stability analysis based on available metrics"""
        
        # Use actual metrics that exist in your evaluation results
        stability_metrics = ['mode_collapse_score', 'precision', 'recall', 'coverage']
        metric_names = ['Mode Collapse Score (↓)', 'Precision (↑)', 'Recall (↑)', 'Coverage (↑)']
        
        for dataset_name in dataset_names:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            fig.suptitle(f'Training Stability Analysis - {dataset_name.upper()}', 
                        fontsize=16, fontweight='bold')
            
            for i, (metric, metric_name) in enumerate(zip(stability_metrics, metric_names)):
                values = []
                labels = []
                
                for model_name in model_names:
                    key = f"{model_name}_{dataset_name}"
                    if key in evaluation_results and metric in evaluation_results[key]:
                        value = evaluation_results[key][metric]
                        if np.isfinite(value):
                            values.append(value)
                            labels.append(model_name.upper())
                
                if values:
                    colors = sns.color_palette("husl", len(values))
                    bars = axes[i].bar(labels, values, color=colors, alpha=0.8)
                    
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
                    
                    axes[i].set_title(metric_name, fontweight='bold')
                    axes[i].set_ylabel('Value')
                    axes[i].grid(True, alpha=0.3, axis='y')
                    
                    if len(labels) > 3:
                        axes[i].tick_params(axis='x', rotation=45)
                else:
                    axes[i].text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                               transform=axes[i].transAxes)
                    axes[i].set_title(metric_name, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'stability_analysis_{dataset_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_sample_progression(self, model_names, dataset_names):
        """Plot sample quality progression over training"""
        
        for dataset_name in dataset_names:
            for model_name in model_names:
                sample_dir = os.path.join(self.config.samples_dir, f"{model_name}_{dataset_name}")
                
                if not os.path.exists(sample_dir):
                    continue
                
                # Get all sample files
                sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
                sample_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                
                if len(sample_files) < 4:
                    continue
                
                # Select 4 evenly spaced samples
                indices = np.linspace(0, len(sample_files)-1, 4, dtype=int)
                selected_files = [sample_files[i] for i in indices]
                
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                fig.suptitle(f'Sample Quality Progression - {model_name.upper()} on {dataset_name.upper()}',
                           fontsize=14, fontweight='bold')
                
                for i, (ax, filename) in enumerate(zip(axes, selected_files)):
                    try:
                        from PIL import Image
                        img = Image.open(os.path.join(sample_dir, filename))
                        ax.imshow(img)
                        epoch = filename.split('_')[1].split('.')[0]
                        ax.set_title(f'Epoch {epoch}', fontweight='bold')
                        ax.axis('off')
                    except Exception as e:
                        ax.text(0.5, 0.5, 'Error\nLoading\nImage', ha='center', va='center')
                        ax.set_title(f'Epoch {filename.split("_")[1].split(".")[0]}')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, 
                                       f'sample_progression_{model_name}_{dataset_name}.png'),
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    def create_comparison_dashboard(self, training_results, evaluation_results, 
                                  model_names, dataset_names):
        """Create a comprehensive comparison dashboard"""
        
        for dataset_name in dataset_names:
            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
            
            fig.suptitle(f'GAN Comparison Dashboard - {dataset_name.upper()}', 
                        fontsize=20, fontweight='bold')
            
            # 1. FID Comparison (top-left)
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_single_metric(ax1, evaluation_results, model_names, dataset_name, 
                                   'fid', 'FID Score (↓)', invert=True)
            
            # 2. IS Comparison (top-center-left)
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_single_metric(ax2, evaluation_results, model_names, dataset_name, 
                                   'is_mean', 'Inception Score (↑)')
            
            # 3. Mode Coverage (top-center-right)
            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_single_metric(ax3, evaluation_results, model_names, dataset_name, 
                                   'mode_coverage', 'Mode Coverage (↑)')
            
            # 4. Mode Collapse Score (top-right)
            ax4 = fig.add_subplot(gs[0, 3])
            self._plot_single_metric(ax4, evaluation_results, model_names, dataset_name, 
                                   'mode_collapse_score', 'Mode Collapse (↓)')
            
            # 5. Training Time Comparison (middle-left)
            ax5 = fig.add_subplot(gs[1, 0])
            self._plot_training_time(ax5, training_results, model_names, dataset_name)
            
            # 6. Loss Comparison (middle-center)
            ax6 = fig.add_subplot(gs[1, 1:3])
            self._plot_final_losses(ax6, training_results, model_names, dataset_name)
            
            # 7. Overall Ranking (middle-right)
            ax7 = fig.add_subplot(gs[1, 3])
            self._plot_overall_ranking(ax7, evaluation_results, model_names, dataset_name)
            
            # 8. Summary Table (bottom)
            ax8 = fig.add_subplot(gs[2, :])
            self._create_summary_table(ax8, training_results, evaluation_results, 
                                     model_names, dataset_name)
            
            plt.savefig(os.path.join(self.plots_dir, f'dashboard_{dataset_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_single_metric(self, ax, evaluation_results, model_names, dataset_name, 
                           metric, title, invert=False):
        """Plot a single metric comparison"""
        values = []
        labels = []
        
        for model_name in model_names:
            key = f"{model_name}_{dataset_name}"
            if key in evaluation_results and metric in evaluation_results[key]:
                value = evaluation_results[key][metric]
                if np.isfinite(value):
                    values.append(value)
                    labels.append(model_name.upper())
        
        if values:
            colors = sns.color_palette("husl", len(values))
            bars = ax.bar(labels, values, color=colors, alpha=0.8)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            ax.set_title(title, fontweight='bold', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontweight='bold', fontsize=10)
    
    def _plot_training_time(self, ax, training_results, model_names, dataset_name):
        """Plot training time comparison"""
        times = []
        labels = []
        
        for model_name in model_names:
            key = f"{model_name}_{dataset_name}"
            if key in training_results and 'training_time' in training_results[key]:
                time_minutes = training_results[key]['training_time'] / 60
                times.append(time_minutes)
                labels.append(model_name.upper())
        
        if times:
            colors = sns.color_palette("husl", len(times))
            bars = ax.bar(labels, times, color=colors, alpha=0.8)
            
            for bar, time in zip(bars, times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{time:.1f}m', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            ax.set_title('Training Time', fontweight='bold', fontsize=10)
            ax.set_ylabel('Minutes')
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Time', fontweight='bold', fontsize=10)
    
    def _plot_final_losses(self, ax, training_results, model_names, dataset_name):
        """Plot final loss comparison"""
        g_losses = []
        d_losses = []
        labels = []
        
        for model_name in model_names:
            key = f"{model_name}_{dataset_name}"
            if key in training_results:
                g_loss = training_results[key].get('final_g_loss', float('inf'))
                d_loss = training_results[key].get('final_d_loss', float('inf'))
                
                if np.isfinite(g_loss) and np.isfinite(d_loss):
                    g_losses.append(g_loss)
                    d_losses.append(d_loss)
                    labels.append(model_name.upper())
        
        if g_losses and d_losses:
            x = np.arange(len(labels))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, g_losses, width, label='Generator', alpha=0.8)
            bars2 = ax.bar(x + width/2, d_losses, width, label='Discriminator', alpha=0.8)
            
            ax.set_title('Final Training Losses', fontweight='bold', fontsize=10)
            ax.set_ylabel('Loss')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='y', labelsize=8)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Final Training Losses', fontweight='bold', fontsize=10)
    
    def _plot_overall_ranking(self, ax, evaluation_results, model_names, dataset_name):
        """Plot overall ranking based on multiple metrics"""
        scores = []
        labels = []
        
        for model_name in model_names:
            key = f"{model_name}_{dataset_name}"
            if key in evaluation_results:
                # Calculate composite score (normalized)
                fid = evaluation_results[key].get('fid', float('inf'))
                is_score = evaluation_results[key].get('is_mean', 0)
                coverage = evaluation_results[key].get('mode_coverage', 0)
                mode_collapse = evaluation_results[key].get('mode_collapse_score', 1)
                
                if np.isfinite(fid):
                    # Normalize and combine metrics (higher is better)
                    composite = (
                        (1 / (1 + fid)) * 0.3 +         # FID (inverted)
                        (is_score / 10) * 0.3 +          # IS (normalized to ~0-1)
                        coverage * 0.2 +                 # Mode coverage
                        (1 - mode_collapse) * 0.2        # Mode collapse (inverted)
                    )
                    scores.append(composite)
                    labels.append(model_name.upper())
        
        if scores:
            colors = sns.color_palette("viridis", len(scores))
            bars = ax.bar(labels, scores, color=colors, alpha=0.8)
            
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            ax.set_title('Overall Ranking\n(Composite Score)', fontweight='bold', fontsize=10)
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Overall Ranking', fontweight='bold', fontsize=10)
    
    def _create_summary_table(self, ax, training_results, evaluation_results, model_names, dataset_name):
        """Create a summary table for the dashboard"""
        # Create a table with model names and key metrics
        table_data = []
        for model_name in model_names:
            key = f"{model_name}_{dataset_name}"
            
            # Initialize row with model name
            row = [model_name.upper()]
            
            # Add FID score if available
            if key in evaluation_results and 'fid' in evaluation_results[key]:
                fid = evaluation_results[key]['fid']
                row.append(f"{fid:.2f}")
            else:
                row.append("N/A")
                
            # Add IS score if available
            if key in evaluation_results and 'is_mean' in evaluation_results[key]:
                is_mean = evaluation_results[key]['is_mean']
                is_std = evaluation_results[key].get('is_std', 0)
                row.append(f"{is_mean:.2f} ± {is_std:.2f}")
            else:
                row.append("N/A")
            
            # Add KID score if available (updated to use kid_mean/kid_std)
            if key in evaluation_results and 'kid_mean' in evaluation_results[key]:
                kid_mean = evaluation_results[key]['kid_mean']
                kid_std = evaluation_results[key].get('kid_std', 0)
                row.append(f"{kid_mean:.4f} ± {kid_std:.4f}")
            else:
                row.append("N/A")
            
            # Add training time if available
            if key in training_results and 'training_time' in training_results[key]:
                time_hours = training_results[key]['training_time'] / 3600
                row.append(f"{time_hours:.2f}h")
            else:
                row.append("N/A")
            
            table_data.append(row)
        
        # Create the table
        columns = ['Model', 'FID ↓', 'IS ↑', 'KID ↓', 'Train Time']
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=table_data, colLabels=columns, 
                       loc='center', cellLoc='center')
        
        # Set table properties
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Highlight header
        for i, key in enumerate(columns):
            cell = table[(0, i)]
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#D6EAF8')
        
        # Return the table object
        return table
    
    def plot_loss_landscape(self, model_names, dataset_names):
        """Plot loss landscape visualization from training history"""
        
        for dataset_name in dataset_names:
            fig, axes = plt.subplots(len(model_names), 2, figsize=(12, 5*len(model_names)))
            
            if len(model_names) == 1:
                axes = axes.reshape(1, 2)
            
            fig.suptitle(f'Loss Landscape Analysis - {dataset_name.upper()}', 
                        fontsize=16, fontweight='bold')
            
            for i, model_name in enumerate(model_names):
                model_dir = os.path.join(self.config.models_dir, f"{model_name}_{dataset_name}")
                
                if os.path.exists(model_dir):
                    # Find latest checkpoint
                    checkpoint_files = [f for f in os.listdir(model_dir) 
                                      if f.startswith('checkpoint_')]
                    if checkpoint_files:
                        latest_checkpoint = sorted(checkpoint_files)[-1]
                        checkpoint_path = os.path.join(model_dir, latest_checkpoint)
                        
                        try:
                            checkpoint = torch.load(checkpoint_path, map_location='cpu')
                            g_losses = checkpoint.get('g_losses', [])
                            d_losses = checkpoint.get('d_losses', [])
                            
                            if len(g_losses) > 0 and len(d_losses) > 0:
                                # Calculate differentials (rate of change)
                                g_diff = np.diff(g_losses)
                                d_diff = np.diff(d_losses)
                                
                                # Plot generator loss landscape
                                x = np.arange(len(g_diff))
                                axes[i, 0].plot(x, g_diff, color='blue', alpha=0.7)
                                axes[i, 0].set_title(f'{model_name.upper()} - Generator Loss Gradient', fontweight='bold')
                                axes[i, 0].set_xlabel('Epoch')
                                axes[i, 0].set_ylabel('Loss Gradient')
                                axes[i, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
                                axes[i, 0].grid(True, alpha=0.3)
                                
                                # Calculate smoothed gradient for visualization
                                if len(g_diff) > 3:
                                    smoothed_grad = gaussian_filter1d(g_diff, sigma=3)
                                    axes[i, 0].plot(x, smoothed_grad, color='green', alpha=0.5, 
                                                  linewidth=2, label='Smoothed')
                                    axes[i, 0].legend()
                                
                                # Plot discriminator loss landscape
                                axes[i, 1].plot(x, d_diff, color='red', alpha=0.7)
                                axes[i, 1].set_title(f'{model_name.upper()} - Discriminator Loss Gradient', fontweight='bold')
                                axes[i, 1].set_xlabel('Epoch')
                                axes[i, 1].set_ylabel('Loss Gradient')
                                axes[i, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
                                axes[i, 1].grid(True, alpha=0.3)
                                
                                # Calculate smoothed gradient for visualization
                                if len(d_diff) > 3:
                                    smoothed_grad = gaussian_filter1d(d_diff, sigma=3)
                                    axes[i, 1].plot(x, smoothed_grad, color='green', alpha=0.5, 
                                                  linewidth=2, label='Smoothed')
                                    axes[i, 1].legend()
                            else:
                                axes[i, 0].text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=axes[i, 0].transAxes)
                                axes[i, 1].text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=axes[i, 1].transAxes)
                        
                        except Exception as e:
                            print(f"Could not analyze loss landscape for {model_name}: {e}")
                            axes[i, 0].text(0.5, 0.5, 'Error Loading Data', ha='center', va='center', transform=axes[i, 0].transAxes)
                            axes[i, 1].text(0.5, 0.5, 'Error Loading Data', ha='center', va='center', transform=axes[i, 1].transAxes)
                    else:
                        axes[i, 0].text(0.5, 0.5, 'No Checkpoint Found', ha='center', va='center', transform=axes[i, 0].transAxes)
                        axes[i, 1].text(0.5, 0.5, 'No Checkpoint Found', ha='center', va='center', transform=axes[i, 1].transAxes)
                else:
                    axes[i, 0].text(0.5, 0.5, 'No Model Directory', ha='center', va='center', transform=axes[i, 0].transAxes)
                    axes[i, 1].text(0.5, 0.5, 'No Model Directory', ha='center', va='center', transform=axes[i, 1].transAxes)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'loss_landscape_{dataset_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def plot_quality_diversity_tradeoff(self, evaluation_results, model_names, dataset_names):
        """Plot quality vs diversity tradeoff for each model"""
        
        for dataset_name in dataset_names:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Quality metric: IS (higher is better)
            # Diversity metrics: Mode coverage (higher is better)
            
            quality_values = []
            diversity_values = []
            labels = []
            sizes = []
            
            for model_name in model_names:
                key = f"{model_name}_{dataset_name}"
                if key in evaluation_results:
                    if 'is_mean' in evaluation_results[key] and 'mode_coverage' in evaluation_results[key]:
                        is_value = evaluation_results[key]['is_mean']
                        coverage = evaluation_results[key]['mode_coverage']
                        
                        if np.isfinite(is_value) and np.isfinite(coverage):
                            quality_values.append(is_value)
                            diversity_values.append(coverage)
                            labels.append(model_name.upper())
                            
                            # Size based on inverse FID (better FID = bigger marker)
                            fid = evaluation_results[key].get('fid', 100)
                            if np.isfinite(fid) and fid > 0:
                                size = 500 / fid  # Inverse relation
                                sizes.append(min(1000, max(100, size * 100)))  # Constrain size
                            else:
                                sizes.append(200)  # Default size
            
            if quality_values and diversity_values:
                colors = sns.color_palette("husl", len(quality_values))
                
                scatter = ax.scatter(quality_values, diversity_values, s=sizes, c=colors, alpha=0.7)
                
                # Add labels
                for i, label in enumerate(labels):
                    ax.annotate(label, (quality_values[i], diversity_values[i]),
                               xytext=(5, 5), textcoords='offset points',
                               fontweight='bold', fontsize=10)
                
                # Plot quadrants for quality-diversity analysis
                if len(quality_values) > 1:
                    x_mid = np.mean(quality_values)
                    y_mid = np.mean(diversity_values)
                    
                    ax.axhline(y=float(y_mid), color='gray', linestyle='--', alpha=0.5)
                    ax.axvline(x=float(x_mid), color='gray', linestyle='--', alpha=0.5)
                    
                    # Add quadrant labels
                    ax.text(ax.get_xlim()[1]*0.95, ax.get_ylim()[1]*0.95, 'High Quality\nHigh Diversity', 
                           ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
                    
                    ax.text(ax.get_xlim()[0]*1.05, ax.get_ylim()[1]*0.95, 'Low Quality\nHigh Diversity', 
                           ha='left', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
                    
                    ax.text(ax.get_xlim()[1]*0.95, ax.get_ylim()[0]*1.05, 'High Quality\nLow Diversity', 
                           ha='right', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
                    
                    ax.text(ax.get_xlim()[0]*1.05, ax.get_ylim()[0]*1.05, 'Low Quality\nLow Diversity', 
                           ha='left', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
                
                # Set labels and title
                ax.set_xlabel('Image Quality (Inception Score)', fontweight='bold')
                ax.set_ylabel('Diversity (Mode Coverage)', fontweight='bold')
                ax.set_title(f'Quality-Diversity Analysis - {dataset_name.upper()}', 
                           fontsize=16, fontweight='bold')
                
                ax.grid(True, alpha=0.3)
                
                # Add a note about marker size
                ax.text(0.01, 0.01, 'Note: Marker size inversely proportional to FID score (larger = better)', 
                       transform=ax.transAxes, fontsize=8, va='bottom', ha='left',
                       bbox=dict(facecolor='white', alpha=0.5))
            else:
                ax.text(0.5, 0.5, 'Insufficient Data for Quality-Diversity Analysis', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Quality-Diversity Analysis - {dataset_name.upper()}', 
                           fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'quality_diversity_{dataset_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def visualize_mode_collapse(self, evaluation_results, model_names, dataset_names):
        """Visualize mode collapse indicators"""
        
        for dataset_name in dataset_names:
            # Prepare data for visualization
            models = []
            mode_coverage = []
            mode_collapse_scores = []
            prdc_recall = []
            
            for model_name in model_names:
                key = f"{model_name}_{dataset_name}"
                if key in evaluation_results:
                    models.append(model_name.upper())
                    
                    # Get mode coverage (higher is better)
                    mode_cov = evaluation_results[key].get('mode_coverage', 0)
                    mode_coverage.append(mode_cov if np.isfinite(mode_cov) else 0)
                    
                    # Get mode collapse score (lower is better)
                    collapse_score = evaluation_results[key].get('mode_collapse_score', 1)
                    mode_collapse_scores.append(collapse_score if np.isfinite(collapse_score) else 1)
                    
                    # Get PRDC recall (higher is better)
                    recall = evaluation_results[key].get('recall', 0)
                    prdc_recall.append(recall if np.isfinite(recall) else 0)
            
            if not models:
                continue
                
            # Create figure
            fig = plt.figure(figsize=(14, 10))
            
            # 1. Mode coverage radar chart
            ax1 = fig.add_subplot(221, polar=True)
            
            # Create radar chart
            N = len(models)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # close the loop
            
            # If we have data for multiple models
            if N > 0:
                # Normalize mode coverage to a 0-1 scale for radar display
                max_coverage = max(mode_coverage) if max(mode_coverage) > 0 else 1
                normalized_coverage = [m/max_coverage for m in mode_coverage]
                
                # Add the data
                ax1.plot(angles, normalized_coverage + normalized_coverage[:1], 'o-', linewidth=2)
                ax1.fill(angles, normalized_coverage + normalized_coverage[:1], alpha=0.25)
                
                # Set radar chart attributes
                ax1.set_xticks(angles[:-1])
                ax1.set_xticklabels(models)
                ax1.set_title('Mode Coverage', size=14, fontweight='bold')
                
                # Add percentage labels at each point
                for i, (angle, coverage) in enumerate(zip(angles[:-1], mode_coverage)):
                    ax1.text(angle, normalized_coverage[i] + 0.1, 
                           f"{coverage:.2f}", 
                           horizontalalignment='center',
                           verticalalignment='center')
                           
            else:
                ax1.text(0, 0, 'No Data Available', ha='center')
            
            # 2. Bar chart for mode collapse indicators
            ax2 = fig.add_subplot(222)
            
            x = np.arange(len(models))
            width = 0.35
            
            # Mode collapse scores (inverse for visualization since lower is better)
            inverse_collapse = [1 - score for score in mode_collapse_scores]
            ax2.bar(x - width/2, inverse_collapse, width, label='1 - Mode Collapse Score', color='skyblue')
            
            # PRDC Recall
            ax2.bar(x + width/2, prdc_recall, width, label='PRDC Recall', color='lightcoral')
            
            ax2.set_xlabel('Models')
            ax2.set_ylabel('Score')
            ax2.set_title('Mode Collapse Indicators', fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(models)
            ax2.legend()
            
            # 3. Scatter plot of Mode Coverage vs Mode Collapse Score
            ax3 = fig.add_subplot(223)
            
            if len(models) > 1:
                # Create a colormap
                colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(models)))
                
                for i, model in enumerate(models):
                    ax3.scatter(mode_collapse_scores[i], mode_coverage[i], color=colors[i], s=100, label=model)
                    ax3.annotate(model, (mode_collapse_scores[i], mode_coverage[i]), 
                               xytext=(5, 5), textcoords='offset points')
                
                ax3.set_xlabel('Mode Collapse Score (lower is better)')
                ax3.set_ylabel('Mode Coverage (higher is better)')
                ax3.set_title('Mode Coverage vs Mode Collapse Score', fontweight='bold')
                
                # Add quadrant lines
                if mode_collapse_scores and mode_coverage:
                    collapse_mid = np.median(mode_collapse_scores)
                    coverage_mid = np.median(mode_coverage)
                    
                    ax3.axhline(y=float(coverage_mid), color='gray', linestyle='--', alpha=0.5)
                    ax3.axvline(x=float(collapse_mid), color='gray', linestyle='--', alpha=0.5)
                    
                    # Add quadrant labels
                    ax3.text(ax3.get_xlim()[1]*0.95, ax3.get_ylim()[1]*0.95, 'High Coverage\nLow Collapse\n(Best)', 
                           ha='right', va='top', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
                    
                    ax3.set_xlim(left=0)  # Mode collapse score cannot be negative
                    ax3.set_ylim(bottom=0)  # Coverage cannot be negative
                
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'Insufficient Data\nNeed Multiple Models', ha='center', va='center', transform=ax3.transAxes)
            
            # 4. Text summary of mode collapse analysis
            ax4 = fig.add_subplot(224)
            ax4.axis('off')
            
            if models:
                # Find the best and worst models
                best_idx = np.argmax(mode_coverage)
                worst_idx = np.argmin(mode_coverage)
                
                # Create analysis text
                text = "MODE COLLAPSE ANALYSIS\n\n"
                text += f"Dataset: {dataset_name.upper()}\n\n"
                text += f"Best Mode Coverage: {models[best_idx]} ({mode_coverage[best_idx]:.2f})\n"
                text += f"Lowest Mode Coverage: {models[worst_idx]} ({mode_coverage[worst_idx]:.2f})\n\n"
                
                text += "Mode Collapse Interpretation:\n"
                text += "- Higher mode coverage indicates better diversity\n"
                text += "- Lower mode collapse score indicates less collapse\n"
                text += "- Higher recall indicates better coverage of real data modes\n\n"
                
                text += "Recommendations:\n"
                if mode_coverage[best_idx] < 0.5:
                    text += "- All models show signs of mode collapse\n"
                    text += "- Consider architectural changes or regularization\n"
                else:
                    text += f"- {models[best_idx]} shows best mode coverage\n"
                    text += f"- Consider analyzing {models[best_idx]} architecture for insights\n"
                
                ax4.text(0, 1, text, va='top', fontsize=10)
            else:
                ax4.text(0.5, 0.5, 'No Data Available for Analysis', ha='center', va='center')
            
            plt.tight_layout()
            plt.suptitle(f'Mode Collapse Analysis - {dataset_name.upper()}', fontsize=16, fontweight='bold', y=1.02)
            plt.subplots_adjust(top=0.92)
            
            plt.savefig(os.path.join(self.plots_dir, f'mode_collapse_{dataset_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def plot_metric_correlations(self, evaluation_results, model_names, dataset_names):
        """Plot correlation heatmap between different metrics"""
        
        for dataset_name in dataset_names:
            # Collect metrics for all models on this dataset
            model_data = []
            
            for model_name in model_names:
                key = f"{model_name}_{dataset_name}"
                if key in evaluation_results:
                    # Extract standard metrics that should be available for most models
                    model_metrics = {
                        'fid': evaluation_results[key].get('fid', np.nan),
                        'kid_mean': evaluation_results[key].get('kid_mean', np.nan),
                        'is_mean': evaluation_results[key].get('is_mean', np.nan),
                        'is_std': evaluation_results[key].get('is_std', np.nan),
                        'mode_coverage': evaluation_results[key].get('mode_coverage', np.nan),
                        'mode_collapse_score': evaluation_results[key].get('mode_collapse_score', np.nan),
                        'precision': evaluation_results[key].get('precision', np.nan),
                        'recall': evaluation_results[key].get('recall', np.nan),
                        'density': evaluation_results[key].get('density', np.nan),
                        'coverage': evaluation_results[key].get('coverage', np.nan)
                    }
                    
                    # Only include metrics that have valid values
                    model_metrics = {k: v for k, v in model_metrics.items() 
                                   if np.isfinite(v) and v is not None}
                    
                    if model_metrics:
                        model_metrics['model'] = model_name
                        model_data.append(model_metrics)
            
            if not model_data:
                continue
                
            # Convert to DataFrame for easier correlation analysis
            df = pd.DataFrame(model_data)
            
            # Set model as index but keep it in the DataFrame for plotting
            df.set_index('model', inplace=True, drop=False)
            
            # Calculate correlations between metrics
            # First, get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'model']
            
            if len(numeric_cols) < 2:
                # Not enough metrics for correlation
                continue
                
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                      square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
            
            plt.title(f'Metric Correlation Heatmap - {dataset_name.upper()}', 
                    fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'metric_correlations_{dataset_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def plot_convergence_analysis(self, model_names, dataset_names):
        """Analyze and visualize convergence properties of different models"""
        
        for dataset_name in dataset_names:
            # Create a figure with subplots for different convergence metrics
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Flatten axes for easier indexing
            axes = axes.flatten()
            
            # Colors for different models
            colors = plt.cm.get_cmap('tab10')(range(10))
            
            # Store convergence data for all models
            all_g_losses = {}
            all_d_losses = {}
            all_convergence_rates = {}
            
            # Collect training data for all models
            for i, model_name in enumerate(model_names):
                model_dir = os.path.join(self.config.models_dir, f"{model_name}_{dataset_name}")
                
                if os.path.exists(model_dir):
                    # Find latest checkpoint
                    checkpoint_files = [f for f in os.listdir(model_dir) 
                                      if f.startswith('checkpoint_')]
                    if checkpoint_files:
                        latest_checkpoint = sorted(checkpoint_files)[-1]
                        checkpoint_path = os.path.join(model_dir, latest_checkpoint)
                        
                        try:
                            checkpoint = torch.load(checkpoint_path, map_location='cpu')
                            g_losses = checkpoint.get('g_losses', [])
                            d_losses = checkpoint.get('d_losses', [])
                            
                            if len(g_losses) > 0 and len(d_losses) > 0:
                                all_g_losses[model_name] = g_losses
                                all_d_losses[model_name] = d_losses
                                
                                # Calculate convergence rate
                                # (how quickly loss values stabilize)
                                if len(g_losses) > 10:
                                    # Use rolling standard deviation as a measure of stability
                                    # Lower std dev means more stable training
                                    window_size = min(10, len(g_losses) // 4)
                                    rolling_std = []
                                    
                                    for j in range(window_size, len(g_losses)):
                                        window = g_losses[j-window_size:j]
                                        rolling_std.append(np.std(window))
                                    
                                    all_convergence_rates[model_name] = rolling_std
                        
                        except Exception as e:
                            print(f"Could not analyze convergence for {model_name}: {e}")
            
            # 1. Plot loss curves (both G and D)
            for i, model_name in enumerate(all_g_losses.keys()):
                color = colors[i % len(colors)]
                
                # Generator loss
                g_losses = all_g_losses[model_name]
                epochs = range(1, len(g_losses) + 1)
                
                # Plot with slight transparency to help distinguish overlapping lines
                axes[0].plot(epochs, g_losses, color=color, linestyle='-', alpha=0.7,
                          label=f"{model_name.upper()} - G")
                
                # Discriminator loss
                d_losses = all_d_losses[model_name]
                
                # Ensure same length by truncating if necessary
                min_length = min(len(epochs), len(d_losses))
                axes[0].plot(epochs[:min_length], d_losses[:min_length], color=color, 
                          linestyle='--', alpha=0.7, label=f"{model_name.upper()} - D")
            
            axes[0].set_title('Loss Curves', fontweight='bold')
            axes[0].set_xlabel('Epochs')
            axes[0].set_ylabel('Loss Value')
            axes[0].legend(loc='upper right')
            axes[0].grid(True, alpha=0.3)
            
            # 2. Plot convergence rates
            for i, model_name in enumerate(all_convergence_rates.keys()):
                color = colors[i % len(colors)]
                rates = all_convergence_rates[model_name]
                epochs = range(len(rates))
                
                axes[1].plot(epochs, rates, color=color, alpha=0.8, label=model_name.upper())
            
            axes[1].set_title('Convergence Stability (Rolling StdDev)', fontweight='bold')
            axes[1].set_xlabel('Epochs')
            axes[1].set_ylabel('Standard Deviation (Lower is more stable)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # 3. Plot loss ratio (G/D) - indicator of balance
            for i, model_name in enumerate(all_g_losses.keys()):
                color = colors[i % len(colors)]
                
                g_losses = np.array(all_g_losses[model_name])
                d_losses = np.array(all_d_losses[model_name])
                
                # Ensure same length
                min_length = min(len(g_losses), len(d_losses))
                g_losses = g_losses[:min_length]
                d_losses = d_losses[:min_length]
                
                # Avoid division by zero
                d_losses = np.where(d_losses == 0, 1e-10, d_losses)
                
                # Calculate ratio and apply smoothing
                ratios = g_losses / d_losses
                
                # Apply smoothing using moving average
                window = min(5, min_length // 5) if min_length > 5 else 1
                if window > 1:
                    kernel = np.ones(window) / window
                    smoothed_ratios = np.convolve(ratios, kernel, mode='valid')
                    
                    # Plot smoothed ratio
                    valid_epochs = range(window//2, min_length - window//2)
                    axes[2].plot(valid_epochs, smoothed_ratios, color=color, 
                              label=model_name.upper())
                else:
                    axes[2].plot(range(min_length), ratios, color=color, 
                              label=model_name.upper())
            
            axes[2].set_title('Generator/Discriminator Loss Ratio', fontweight='bold')
            axes[2].set_xlabel('Epochs')
            axes[2].set_ylabel('G/D Loss Ratio')
            axes[2].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, 
                         label='Optimal Balance')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            # 4. Plot training dynamics (phase plot of G vs D loss)
            for i, model_name in enumerate(all_g_losses.keys()):
                color = colors[i % len(colors)]
                
                g_losses = all_g_losses[model_name]
                d_losses = all_d_losses[model_name]
                
                # Ensure same length
                min_length = min(len(g_losses), len(d_losses))
                
                if min_length > 10:  # Only plot if we have enough data
                    # Downsample if too many points
                    step = max(1, min_length // 100)
                    
                    # Plot with markers that get darker to show progression
                    for j in range(0, min_length, step):
                        frac = j / min_length
                        alpha = 0.3 + 0.7 * frac  # Start light, end dark
                        axes[3].scatter(d_losses[j], g_losses[j], color=color, 
                                     s=30, alpha=alpha)
                    
                    # Add connecting lines with alpha gradient
                    for j in range(0, min_length-step, step):
                        axes[3].plot([d_losses[j], d_losses[j+step]], 
                                  [g_losses[j], g_losses[j+step]], 
                                  color=color, alpha=0.3)
                    
                    # Mark start and end
                    axes[3].scatter(d_losses[0], g_losses[0], color='none', 
                                 s=80, edgecolors=color, linewidth=2, 
                                 label=f"{model_name.upper()} Start")
                    
                    axes[3].scatter(d_losses[min_length-1], g_losses[min_length-1], 
                                 s=100, color=color, marker='*', 
                                 label=f"{model_name.upper()} End")
            
            axes[3].set_title('Training Dynamics (Phase Portrait)', fontweight='bold')
            axes[3].set_xlabel('Discriminator Loss')
            axes[3].set_ylabel('Generator Loss')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
            
            # Overall figure settings
            plt.suptitle(f'Convergence Analysis - {dataset_name.upper()}', 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            plt.savefig(os.path.join(self.plots_dir, f'convergence_analysis_{dataset_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
    def plot_iteration_losses(self, model_names, dataset_names):
        """Plot detailed iteration-level losses for each model"""
        print("Plotting iteration-level losses...")
        
        for dataset_name in dataset_names:
            for model_name in model_names:
                # Create folder for this model-dataset combination
                save_dir = os.path.join(self.plots_dir, f"{model_name}_{dataset_name}_iter_losses")
                os.makedirs(save_dir, exist_ok=True)
                
                # Try to load model checkpoint
                model_dir = os.path.join(self.config.models_dir, f"{model_name}_{dataset_name}")
                if not os.path.exists(model_dir):
                    print(f"No model directory found for {model_name} on {dataset_name}")
                    continue
                    
                # Find latest checkpoint
                checkpoint_files = [f for f in os.listdir(model_dir) if f.startswith('checkpoint_')]
                if not checkpoint_files:
                    print(f"No checkpoints found for {model_name} on {dataset_name}")
                    continue
                    
                latest_checkpoint = sorted(checkpoint_files)[-1]
                checkpoint_path = os.path.join(model_dir, latest_checkpoint)
                
                try:
                    # Load checkpoint data
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    
                    # Check if iteration-level losses exist
                    if 'g_losses_iter' not in checkpoint or 'd_losses_iter' not in checkpoint:
                        print(f"No iteration-level losses found in checkpoint for {model_name} on {dataset_name}")
                        continue
                    
                    g_losses_iter = checkpoint['g_losses_iter']
                    d_losses_iter = checkpoint['d_losses_iter']
                    
                    if len(g_losses_iter) == 0 or len(d_losses_iter) == 0:
                        print(f"Empty iteration losses for {model_name} on {dataset_name}")
                        continue
                    
                    print(f"Plotting iteration losses for {model_name} on {dataset_name} ({len(g_losses_iter)} iterations)")
                    
                    # 1. Plot all iteration losses
                    plt.figure(figsize=(12, 6))
                    iterations = np.arange(1, len(g_losses_iter) + 1)
                    plt.plot(iterations, g_losses_iter, label='Generator Loss', alpha=0.7)
                    plt.plot(iterations, d_losses_iter, label='Discriminator Loss', alpha=0.7)
                    plt.title(f'{model_name.upper()} - Iteration-Level Training Losses on {dataset_name.upper()}')
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # Add moving average for smoother visualization
                    window = min(500, len(g_losses_iter) // 10)  # Use 10% of data or 500 points, whichever is smaller
                    if window > 10:  # Only smooth if we have enough data points
                        g_smooth = np.convolve(g_losses_iter, np.ones(window)/window, mode='valid')
                        d_smooth = np.convolve(d_losses_iter, np.ones(window)/window, mode='valid')
                        # Align the smoothed data with the original iterations
                        smooth_iterations = iterations[window-1:]
                        
                        plt.plot(smooth_iterations, g_smooth, 'r-', 
                                 label=f'Generator (MA{window})', linewidth=2)
                        plt.plot(smooth_iterations, d_smooth, 'b-', 
                                 label=f'Discriminator (MA{window})', linewidth=2)
                        plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'all_iterations.png'), dpi=300)
                    plt.close()
                    
                    # 2. Plot detailed sections
                    # Divide iterations into multiple chunks for detailed analysis
                    total_iterations = len(g_losses_iter)
                    chunk_size = 1000  # Show 1000 iterations per plot
                    num_chunks = (total_iterations + chunk_size - 1) // chunk_size  # Ceiling division
                    
                    for i in range(num_chunks):
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, total_iterations)
                        
                        plt.figure(figsize=(12, 6))
                        chunk_iterations = iterations[start_idx:end_idx]
                        plt.plot(chunk_iterations, g_losses_iter[start_idx:end_idx], 
                                 label='Generator Loss', alpha=0.8)
                        plt.plot(chunk_iterations, d_losses_iter[start_idx:end_idx], 
                                 label='Discriminator Loss', alpha=0.8)
                        
                        plt.title(f'{model_name.upper()} - Detailed Losses (Iterations {start_idx+1} to {end_idx})')
                        plt.xlabel('Iteration')
                        plt.ylabel('Loss')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        
                        # Save the detailed chunk plot
                        plt.savefig(os.path.join(save_dir, f'iterations_{start_idx+1}_to_{end_idx}.png'), dpi=300)
                        plt.close()
                    
                    # 3. Create loss landscape visualization (if epoch info is available)
                    if 'g_losses' in checkpoint and 'd_losses' in checkpoint and len(checkpoint['g_losses']) > 1:
                        epoch_g_losses = checkpoint['g_losses']
                        iterations_per_epoch = len(g_losses_iter) / len(epoch_g_losses)
                        
                        try:
                            # Create a heatmap visualization of loss landscape
                            iterations_per_epoch_int = int(iterations_per_epoch)
                            num_epochs = len(epoch_g_losses)
                            
                            # Only create heatmap if we have enough iterations per epoch
                            if iterations_per_epoch_int > 10:
                                # Create matrix of correct size (limit to reasonable size)
                                max_iters_per_epoch = min(iterations_per_epoch_int, 500)
                                g_loss_matrix = np.zeros((num_epochs, max_iters_per_epoch))
                                d_loss_matrix = np.zeros((num_epochs, max_iters_per_epoch))
                                
                                # Fill the matrices with available data
                                for iter_idx in range(min(len(g_losses_iter), num_epochs * iterations_per_epoch_int)):
                                    epoch_idx = int(iter_idx / iterations_per_epoch_int)
                                    iter_in_epoch = int(iter_idx % iterations_per_epoch_int)
                                    
                                    if epoch_idx < num_epochs and iter_in_epoch < max_iters_per_epoch:
                                        g_loss_matrix[epoch_idx, iter_in_epoch] = g_losses_iter[iter_idx]
                                        d_loss_matrix[epoch_idx, iter_in_epoch] = d_losses_iter[iter_idx]
                                
                                # Create heatmap for generator losses
                                plt.figure(figsize=(15, 10))
                                sns.heatmap(g_loss_matrix, cmap="viridis", 
                                          xticklabels=max_iters_per_epoch//10, 
                                          yticklabels=1)
                                plt.title(f'{model_name.upper()} - Generator Loss Landscape on {dataset_name.upper()}')
                                plt.xlabel('Iteration within Epoch')
                                plt.ylabel('Epoch')
                                plt.tight_layout()
                                plt.savefig(os.path.join(save_dir, 'generator_loss_heatmap.png'), dpi=300)
                                plt.close()
                                
                                # Create heatmap for discriminator losses
                                plt.figure(figsize=(15, 10))
                                sns.heatmap(d_loss_matrix, cmap="magma", 
                                          xticklabels=max_iters_per_epoch//10, 
                                          yticklabels=1)
                                plt.title(f'{model_name.upper()} - Discriminator Loss Landscape on {dataset_name.upper()}')
                                plt.xlabel('Iteration within Epoch')
                                plt.ylabel('Epoch')
                                plt.tight_layout()
                                plt.savefig(os.path.join(save_dir, 'discriminator_loss_heatmap.png'), dpi=300)
                                plt.close()
                                
                                # Create a loss ratio visualization (D/G ratio)
                                ratio_matrix = np.zeros_like(g_loss_matrix)
                                for i in range(g_loss_matrix.shape[0]):
                                    for j in range(g_loss_matrix.shape[1]):
                                        if g_loss_matrix[i,j] > 0:
                                            ratio_matrix[i,j] = d_loss_matrix[i,j] / g_loss_matrix[i,j]
                                
                                plt.figure(figsize=(15, 10))
                                sns.heatmap(ratio_matrix, cmap="coolwarm", center=1.0,
                                          xticklabels=max_iters_per_epoch//10, 
                                          yticklabels=1)
                                plt.title(f'{model_name.upper()} - D/G Loss Ratio on {dataset_name.upper()}')
                                plt.xlabel('Iteration within Epoch')
                                plt.ylabel('Epoch')
                                plt.tight_layout()
                                plt.savefig(os.path.join(save_dir, 'dg_ratio_heatmap.png'), dpi=300)
                                plt.close()
                        except Exception as e:
                            print(f"Error creating heatmap for {model_name} on {dataset_name}: {e}")
                            
                    # 4. Export to CSV for further analysis
                    try:
                        import pandas as pd
                        
                        # Create dataframe with iteration losses
                        iter_data = {
                            'iteration': list(range(1, len(g_losses_iter) + 1)),
                            'generator_loss': g_losses_iter,
                            'discriminator_loss': d_losses_iter
                        }
                        
                        # Add epoch information if available
                        if 'g_losses' in checkpoint and len(checkpoint['g_losses']) > 1:
                            # Estimate which epoch each iteration belongs to
                            epochs = []
                            for i in range(len(g_losses_iter)):
                                epoch = int(i / iterations_per_epoch) + 1
                                epochs.append(epoch)
                            iter_data['epoch'] = epochs
                        
                        # Create and save dataframe
                        df = pd.DataFrame(iter_data)
                        csv_path = os.path.join(save_dir, 'iteration_losses.csv')
                        df.to_csv(csv_path, index=False)
                        print(f"Iteration losses exported to CSV: {csv_path}")
                    except Exception as e:
                        print(f"Error exporting to CSV: {e}")
                    
                    print(f"✅ Iteration loss visualizations for {model_name} on {dataset_name} saved to {save_dir}")
                    
                except Exception as e:
                    print(f"Error processing checkpoint for {model_name} on {dataset_name}: {e}")
        
        print("Iteration loss plotting completed.")

    def plot_output_diversity(self, model_names, dataset_names):
        """Visualize diversity of generated samples from different models"""
        
        for dataset_name in dataset_names:
            # Determine number of models to visualize
            num_models = len(model_names)
            if num_models == 0:
                continue
                
            # Create figure: one row per model, plus one row for real data
            fig, axes = plt.subplots(num_models + 1, 1, figsize=(14, 5 * (num_models + 1)))
            
            if num_models == 0:
                axes = np.array([axes])
            elif num_models == 1:
                axes = np.array([axes[0], axes[1]])
                
            # Set up real data samples using YOUR DATALOADER SYSTEM
            real_samples = []
            
            try:
                # Import YOUR dataset loader
                from datasets.data_loaders import DatasetLoader
                
                # Create dataloader instance using your config
                dataset_loader = DatasetLoader(self.config)
                
                # Get dataloader for real data
                real_dataloader = dataset_loader.get_eval_dataloader(dataset_name, batch_size=32)
                
                # Extract samples from dataloader
                samples_collected = 0
                target_samples = 8
                
                for batch_idx, batch in enumerate(real_dataloader):
                    if samples_collected >= target_samples:
                        break
                    
                    # Handle different dataset formats based on your implementation
                    if dataset_name == 'celeba':
                        # CelebA returns batch[0] directly (data only)
                        data = batch[0]
                    else:
                        # MNIST/CIFAR10 return (data, labels)
                        data, _ = batch
                    
                    # Convert to numpy and process each image in batch
                    data = data.cpu().numpy()
                    
                    for i in range(data.shape[0]):
                        if samples_collected >= target_samples:
                            break
                        
                        img = data[i]
                        
                        # Handle different image formats based on your normalization
                        if img.shape[0] == 1:  # Grayscale (1, H, W) - MNIST
                            img = np.transpose(img, (1, 2, 0))  # (H, W, 1)
                            img = np.squeeze(img, axis=2)  # (H, W)
                        elif img.shape[0] == 3:  # RGB (3, H, W) - CIFAR10/CelebA
                            img = np.transpose(img, (1, 2, 0))  # (H, W, 3)
                        
                        # Denormalize from [-1, 1] to [0, 1] as per your transforms
                        img = (img + 1) / 2
                        
                        # Ensure values are in [0, 1] range
                        img = np.clip(img, 0, 1)
                        
                        real_samples.append(img)
                        samples_collected += 1
                        
            except Exception as e:
                print(f"Could not load real data using YOUR dataloader: {e}")
                print("FALLBACK: This should not happen with your dataloader implementation")
            
            # Plot real samples
            if real_samples:
                axes[0].set_title(f"Real Data Samples - {dataset_name.upper()}", 
                                fontweight='bold', fontsize=14)
                axes[0].axis('off')
                
                # Create a grid of images
                n_cols = min(8, len(real_samples))
                n_rows = (len(real_samples) + n_cols - 1) // n_cols
                
                # Create subplot grid within the main axes
                for idx, img in enumerate(real_samples):
                    row = idx // n_cols
                    col = idx % n_cols
                    
                    # Calculate position within the axes
                    left = col / n_cols
                    bottom = 1 - (row + 1) / n_rows
                    width = 1 / n_cols
                    height = 1 / n_rows
                    
                    # Create inset axes
                    inset_ax = axes[0].inset_axes([left, bottom, width, height])
                    
                    # Handle grayscale vs color images
                    if len(img.shape) == 2:  # Grayscale
                        inset_ax.imshow(img, cmap='gray')
                    else:  # Color
                        inset_ax.imshow(img)
                    
                    inset_ax.axis('off')
            else:
                axes[0].text(0.5, 0.5, "ERROR: Could not load real data samples", 
                           ha='center', va='center', fontsize=14, color='red')
                axes[0].set_title(f"Real Data Samples - {dataset_name.upper()}", 
                                fontweight='bold', fontsize=14)
                axes[0].axis('off')