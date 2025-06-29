import os
import torch
import argparse
from config import Config
from experiments.run_experiments import ExperimentRunner
from visualization.plotting import ResultsVisualizer

def main():
    parser = argparse.ArgumentParser(description='GAN Comparison Framework')
    
    # ADD MODE ARGUMENT FIRST - BEFORE parse_args()
    parser.add_argument('--mode', choices=['train', 'evaluate', 'visualize', 'all'], 
                       default='all', help='Mode to run')
    
    parser.add_argument('--models', nargs='+', 
                       choices=['vanilla', 'dcgan', 'wgan', 'wgan_gp'], 
                       default=['dcgan'],          
                       help='Models to compare')
    parser.add_argument('--datasets', nargs='+',
                       choices=['mnist', 'cifar10', 'celeba', 'STL-10'],
                       default=['mnist'],          
                       help='Datasets to use')
    parser.add_argument('--epochs', type=int, default=25, help='Training epochs')
    
    # PARSE ARGUMENTS AFTER ALL ARE DEFINED
    args = parser.parse_args()
    
    print(f"Mode: {args.mode}")
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Epochs: {args.epochs}")
    
    # Initialize configuration
    config = Config()
    config.epochs = args.epochs
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Initialize experiment runner
    runner = ExperimentRunner(config)
    
    if args.mode in ['train', 'all']:
        print("Starting training phase...")
        runner.run_training_experiments(args.models, args.datasets)
    
    if args.mode in ['evaluate', 'all']:
        print("Starting evaluation phase...")
        runner.run_evaluation_experiments(args.models, args.datasets)
    
    if args.mode in ['visualize', 'all']:
        print("Starting visualization phase...")
        visualizer = ResultsVisualizer(config)
        visualizer.create_all_visualizations(args.models, args.datasets)
    
    print("Experiment complete! Check results/ directory for outputs.")

if __name__ == "__main__":
    main()