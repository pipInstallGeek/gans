import os
import json
import pandas as pd
from datasets.data_loaders import DatasetLoader, get_model_class
from training.trainer import GANTrainer
from evaluation.metrics import GANEvaluator


class ExperimentRunner:
    """Main experiment runner"""

    def __init__(self, config):
        self.config = config
        self.dataset_loader = DatasetLoader(config)
        self.trainer = GANTrainer(config)
        self.evaluator = GANEvaluator(config)

        # Results storage
        self.training_results = {}
        self.evaluation_results = {}

    def run_training_experiments(self, model_names, dataset_names):
        """Run training experiments for all model-dataset combinations"""

        for dataset_name in dataset_names:
            print(f"\n{'=' * 50}")
            print(f"Training on {dataset_name.upper()}")
            print(f"{'=' * 50}")

            # Get dataset configuration
            dataset_config = self.config.get_dataset_config(dataset_name)
            dataloader = self.dataset_loader.get_dataloader(dataset_name)

            for model_name in model_names:
                print(f"\n{'-' * 30}")
                print(f"Training {model_name.upper()}")
                print(f"{'-' * 30}")

                try:
                    # Initialize model
                    model_class = get_model_class(model_name)
                    model = model_class(self.config, dataset_config)

                    # Train model
                    training_time = self.trainer.train_model(
                        model, dataloader, model_name, dataset_name
                    )

                    # Store results
                    key = f"{model_name}_{dataset_name}"
                    self.training_results[key] = {
                        'training_time': training_time,
                        'final_g_loss': model.g_losses[-1] if model.g_losses else float('inf'),
                        'final_d_loss': model.d_losses[-1] if model.d_losses else float('inf'),
                        'total_epochs': len(model.g_losses)
                    }

                    print(f"Training completed successfully for {model_name} on {dataset_name}")

                except Exception as e:
                    print(f"Training failed for {model_name} on {dataset_name}: {e}")
                    self.training_results[f"{model_name}_{dataset_name}"] = {
                        'training_time': 0,
                        'final_g_loss': float('inf'),
                        'final_d_loss': float('inf'),
                        'total_epochs': 0,
                        'error': str(e)
                    }

        # Save training results
        self.save_training_results()

    def run_evaluation_experiments(self, model_names, dataset_names):
        """Run evaluation experiments for all trained models"""

        for dataset_name in dataset_names:
            print(f"\n{'=' * 50}")
            print(f"Evaluating on {dataset_name.upper()}")
            print(f"{'=' * 50}")

            # Get dataset configuration
            dataset_config = self.config.get_dataset_config(dataset_name)
            dataloader = self.dataset_loader.get_eval_dataloader(dataset_name)

            for model_name in model_names:
                print(f"\n{'-' * 30}")
                print(f"Evaluating {model_name.upper()}")
                print(f"{'-' * 30}")

                try:
                    # Load trained model
                    model_class = get_model_class(model_name)
                    model = model_class(self.config, dataset_config)

                    # Find latest checkpoint
                    model_dir = os.path.join(
                        self.config.models_dir,
                        f"{model_name}_{dataset_name}"
                    )

                    if not os.path.exists(model_dir):
                        print(f"No trained model found for {model_name} on {dataset_name}")
                        continue

                    # Load the final checkpoint
                    checkpoint_files = [f for f in os.listdir(model_dir) if f.startswith('checkpoint_')]
                    if not checkpoint_files:
                        print(f"No checkpoint found for {model_name} on {dataset_name}")
                        continue

                    latest_checkpoint = sorted(checkpoint_files)[-1]
                    checkpoint_path = os.path.join(model_dir, latest_checkpoint)

                    model.load_models(checkpoint_path)

                    # Evaluate model
                    results = self.evaluator.evaluate_model(
                        model, dataloader, dataset_name, model_name
                    )

                    # Store results
                    key = f"{model_name}_{dataset_name}"
                    self.evaluation_results[key] = results

                    # Print actual metrics that exist
                    print(f"FID: {results.get('fid', 'N/A'):.2f}")
                    print(f"IS: {results.get('is_mean', 'N/A'):.2f} ± {results.get('is_std', 'N/A'):.2f}")
                    print(f"KID: {results.get('kid_mean', 'N/A'):.6f} ± {results.get('kid_std', 'N/A'):.6f}")
                    print(f"Mode Coverage: {results.get('mode_coverage', 'N/A'):.3f}")
                    print(f"Mode Collapse Score: {results.get('mode_collapse_score', 'N/A'):.3f}")

                    # PRDC metrics if available
                    if 'precision' in results:
                        print(f"Precision: {results['precision']:.3f}")
                        print(f"Recall: {results['recall']:.3f}")
                        print(f"Density: {results['density']:.3f}")
                        print(f"Coverage: {results['coverage']:.3f}")

                except Exception as e:
                    print(f"Evaluation failed for {model_name} on {dataset_name}: {e}")
                    self.evaluation_results[f"{model_name}_{dataset_name}"] = {
                        'error': str(e)
                    }

        # Save evaluation results
        self.save_evaluation_results()

        # Create comparison tables
        self.create_comparison_tables()

    def save_training_results(self):
        """Save training results to JSON"""
        results_file = os.path.join(self.config.results_dir, 'training_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.training_results, f, indent=2, cls=NumpyEncoder)
        print(f"Training results saved to {results_file}")

    def save_evaluation_results(self):
        """Save evaluation results to JSON"""
        results_file = os.path.join(self.config.results_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, cls=NumpyEncoder)
        print(f"Evaluation results saved to {results_file}")

    def create_comparison_tables(self):
        """Create comparison tables for thesis using ACTUAL metrics"""

        # Prepare data for DataFrame
        data = []

        for key, results in self.evaluation_results.items():
            if 'error' in results:
                continue

            model_name, dataset_name = key.split('_', 1)

            # Get training results
            training_info = self.training_results.get(key, {})

            row = {
                'Model': model_name.upper(),
                'Dataset': dataset_name.upper(),
                'FID ↓': results.get('fid', float('inf')),
                'IS ↑': results.get('is_mean', 0),
                'IS_std': results.get('is_std', 0),
                'KID_mean ↓': results.get('kid_mean', float('inf')),
                'KID_std': results.get('kid_std', 0),
                'Mode Coverage ↑': results.get('mode_coverage', 0),
                'Mode Collapse ↓': results.get('mode_collapse_score', 1),
                'Training Time (min)': training_info.get('training_time', 0) / 60,
                'Final G Loss': training_info.get('final_g_loss', float('inf')),
                'Final D Loss': training_info.get('final_d_loss', float('inf'))
            }

            # Add PRDC metrics if available
            if 'precision' in results:
                row['Precision ↑'] = results['precision']
                row['Recall ↑'] = results['recall']
                row['Density ↑'] = results['density']
                row['Coverage ↑'] = results['coverage']

            data.append(row)

        if not data:
            print("No valid results to create comparison table")
            return

        df = pd.DataFrame(data)

        # Save to CSV
        csv_file = os.path.join(self.config.results_dir, 'comparison_table.csv')
        df.to_csv(csv_file, index=False)

        # Create formatted table for each dataset
        for dataset in df['Dataset'].unique():
            dataset_df = df[df['Dataset'] == dataset].copy()
            dataset_df = dataset_df.drop('Dataset', axis=1)

            # Round numerical values
            numerical_cols = ['FID ↓', 'IS ↑', 'IS_std', 'KID_mean ↓', 'KID_std',
                              'Mode Coverage ↑', 'Mode Collapse ↓', 'Training Time (min)',
                              'Final G Loss', 'Final D Loss']

            # Add PRDC columns if they exist
            prdc_cols = ['Precision ↑', 'Recall ↑', 'Density ↑', 'Coverage ↑']
            for col in prdc_cols:
                if col in dataset_df.columns:
                    numerical_cols.append(col)

            for col in numerical_cols:
                if col in dataset_df.columns:
                    dataset_df[col] = dataset_df[col].round(3)

            # Save dataset-specific table
            dataset_file = os.path.join(
                self.config.results_dir,
                f'results_{dataset.lower()}.csv'
            )
            dataset_df.to_csv(dataset_file, index=False)

            # Print formatted table
            print(f"\n{'=' * 60}")
            print(f"RESULTS FOR {dataset}")
            print(f"{'=' * 60}")
            print(dataset_df.to_string(index=False))

        print(f"\nComparison tables saved to {self.config.results_dir}")

    def load_results(self):
        """Load previously saved results"""
        training_file = os.path.join(self.config.results_dir, 'training_results.json')
        evaluation_file = os.path.join(self.config.results_dir, 'evaluation_results.json')

        if os.path.exists(training_file):
            with open(training_file, 'r') as f:
                self.training_results = json.load(f)

        if os.path.exists(evaluation_file):
            with open(evaluation_file, 'r') as f:
                self.evaluation_results = json.load(f)

        print("Previous results loaded")


import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)