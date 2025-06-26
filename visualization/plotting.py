# visualization/plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json
from matplotlib.gridspec import GridSpec

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
                            import torch
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
        
        metrics = ['fid', 'is_mean', 'mode_coverage', 'stability_score']
        metric_names = ['FID (↓)', 'Inception Score (↑)', 'Mode Coverage (↑)', 'Stability Score (↑)']
        
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
            metrics = ['fid', 'is_mean', 'mode_coverage', 'stability_score']
            metric_labels = ['FID\n(Lower Better)', 'Inception Score\n(Higher Better)', 
                           'Mode Coverage\n(Higher Better)', 'Stability Score\n(Higher Better)']
            
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
        """Plot detailed stability analysis"""
        
        stability_metrics = ['g_loss_var', 'd_loss_var', 'g_oscillation', 'd_oscillation']
        metric_names = ['Generator Loss Variance', 'Discriminator Loss Variance', 
                       'Generator Oscillation', 'Discriminator Oscillation']
        
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
                    
                    axes[i].set_title(f'{metric_name}\n(Lower = More Stable)', fontweight='bold')
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
            
            # 4. Stability Score (top-right)
            ax4 = fig.add_subplot(gs[0, 3])
            self._plot_single_metric(ax4, evaluation_results, model_names, dataset_name, 
                                   'stability_score', 'Stability Score (↑)')
            
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
                stability = evaluation_results[key].get('stability_score', 0)
                
                if np.isfinite(fid):
                    # Normalize and combine metrics (higher is better)
                    composite = (
                        (1 / (1 + fid)) * 0.3 +  # FID (inverted)
                        (is_score / 10) * 0.3 +   # IS (normalized to ~0-1)
                        coverage * 0.2 +          # Mode coverage
                        (stability / 10) * 0.2    # Stability
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
    
    def _create_summary_table(self, ax, training_results, evaluation_results, 
                             model_names, dataset_name):
        """Create a summary table"""
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        headers = ['Model', 'FID↓', 'IS↑', 'Coverage', 'Stability', 'Time(m)', 'Status']
        
        for model_name in model_names:
            key = f"{model_name}_{dataset_name}"
            
            # Get evaluation results
            fid = 'N/A'
            is_score = 'N/A'
            coverage = 'N/A'
            stability = 'N/A'
            
            if key in evaluation_results:
                eval_data = evaluation_results[key]
                fid = f"{eval_data.get('fid', float('inf')):.2f}" if np.isfinite(eval_data.get('fid', float('inf'))) else 'N/A'
                is_score = f"{eval_data.get('is_mean', 0):.2f}" if eval_data.get('is_mean', 0) > 0 else 'N/A'
                coverage = f"{eval_data.get('mode_coverage', 0):.3f}" if eval_data.get('mode_coverage', 0) > 0 else 'N/A'
                stability = f"{eval_data.get('stability_score', 0):.2f}" if eval_data.get('stability_score', 0) > 0 else 'N/A'
            
            # Get training results
            training_time = 'N/A'
            status = 'Failed'
            
            if key in training_results:
                train_data = training_results[key]
                if 'error' not in train_data:
                    training_time = f"{train_data.get('training_time', 0)/60:.1f}"
                    status = 'Success'
                else:
                    status = 'Failed'
            
            table_data.append([model_name.upper(), fid, is_score, coverage, stability, training_time, status])
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        
        # Color code the status column
        for i in range(len(table_data)):
            status_col = len(headers) - 1  # Last column index
            if table_data[i][-1] == 'Success':
                table[(i+1, status_col)].set_facecolor('#90EE90')  # Light green
            else:
                table[(i+1, status_col)].set_facecolor('#FFB6C1')  # Light red
        
        # Style header
        for j in range(len(headers)):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Summary Table', fontweight='bold', fontsize=12, pad=20)