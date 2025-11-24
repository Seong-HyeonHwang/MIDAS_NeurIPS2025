"""
Unified training script for multimodal learning experiments
Supports CREMA-D, UCF101, Food101, and Kinetics datasets with various methods
"""
import os
import sys
import time
import random
import argparse
import numpy as np
import torch
from pytorch_lightning import seed_everything

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# Import utilities
from configs.default_config import get_config, get_method_config
from utils import create_data_splits, get_dataloader, get_model, get_available_methods

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)
torch.set_float32_matmul_precision('high')


def setup_seed(seed):
    """Set random seeds for reproducibility"""
    seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train multimodal models on CREMA-D, UCF101, Food101, or Kinetics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset arguments
    parser.add_argument(
        '--dataset',
        type=str,
        default='cremad',
        choices=['cremad', 'ucf101', 'food101', 'kinetics'],
        help='Dataset to use'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        required=True,
        help='Root directory of the dataset (required)'
    )

    # Model arguments
    parser.add_argument(
        '--method',
        type=str,
        default='Midas',
        help='Training method/model to use (currently only Midas is supported)'
    )

    # Training arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=2,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--devices',
        type=str,
        default='0',
        help='GPU device ids (comma-separated, e.g., "0,1")'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size. If not specified, uses default from config.'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=None,
        help='Maximum training epochs. If not specified, uses default from config.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate. If not specified, uses default from config.'
    )

    # Misc arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for model checkpoints and logs'
    )
    parser.add_argument(
        '--list_methods',
        action='store_true',
        help='List available methods for the specified dataset and exit'
    )

    args = parser.parse_args()
    return args


def main():
    """Main training function"""
    args = parse_args()

    # List available methods if requested
    if args.list_methods:
        methods = get_available_methods(args.dataset)
        print(f"\nAvailable methods for {args.dataset.upper()}:")
        for method in methods:
            print(f"  - {method}")
        return

    # Setup
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Method: {args.method}")
    print(f"Seed: {args.seed}")
    print(f"Devices: {args.devices}")
    print(f"{'='*60}\n")

    # Set random seed
    setup_seed(args.seed)

    # Load dataset configuration
    config = get_config(args.dataset, args.dataset_root)

    # Create data splits
    print("Creating data splits...")
    train_df, dev_df, test_df, split_paths = create_data_splits(
        args.dataset,
        config.dataset_root,
        seed=args.seed,
        save_splits=True
    )

    # Get method-specific configuration
    hparams = get_method_config(args.method, config)

    # Update with split paths
    hparams['train_path'] = split_paths['train']
    hparams['dev_path'] = split_paths['dev']
    hparams['test_path'] = split_paths['test']

    # Override with command line arguments if provided
    if args.batch_size is not None:
        hparams['batch_size'] = args.batch_size
    if args.max_epochs is not None:
        hparams['max_epochs'] = args.max_epochs
    if args.lr is not None:
        hparams['lr'] = args.lr
    if args.output_dir is not None:
        hparams['output_path'] = args.output_dir

    # Update devices
    hparams['devices'] = [int(device) for device in args.devices.split(',')]

    # Print hyperparameters
    print("\nHyperparameters:")
    print("-" * 60)
    for key, value in sorted(hparams.items()):
        if not key.endswith('_path') and not key.endswith('_dir'):
            print(f"  {key}: {value}")
    print("-" * 60)

    # Create model
    print(f"\nInitializing {args.method} model...")
    model = get_model(args.method, args.dataset, hparams)

    # Train model
    print("\nStarting training...")
    start_time = time.time()
    model.fit()
    end_time = time.time()

    training_time = end_time - start_time
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"{'='*60}\n")

    # Test model
    print("Testing model...")
    test_loader = get_dataloader(
        args.dataset,
        mode='test',
        hparams=hparams,
        split_path=split_paths['test'],
        model=model
    )

    model.trainer.test(ckpt_path="best", dataloaders=test_loader)

    # Save records if applicable
    if args.method in ['Midas', 'Baseline']:
        if hparams.get('recording', False):
            print(f"Saving records for seed {args.seed}...")
            model.save_records(args.seed)

    if args.method == 'Midas':
        if hasattr(model, 'save_conf_records'):
            print(f"Saving confidence records for seed {args.seed}...")
            model.save_conf_records(args.seed)

    print("\nDone!")


if __name__ == "__main__":
    main()
