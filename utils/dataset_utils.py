"""
Dataset utilities for loading and splitting datasets
Supports CREMA-D, UCF101, Food101, and Kinetics datasets
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def create_data_splits(dataset, dataset_root, seed=2, save_splits=True):
    """
    Create train/dev/test splits for the specified dataset

    Args:
        dataset (str): Dataset name ('cremad', 'ucf101', 'food101', 'kinetics')
        dataset_root (str): Root directory of the dataset
        seed (int): Random seed for reproducibility
        save_splits (bool): Whether to save split files

    Returns:
        tuple: (train_df, dev_df, test_df, split_paths)
            where split_paths is a dict with keys 'train', 'dev', 'test'
    """
    dataset = dataset.lower()

    if dataset == 'cremad':
        return _create_cremad_splits(dataset_root, seed, save_splits)
    elif dataset == 'ucf101':
        return _create_ucf101_splits(dataset_root, seed, save_splits)
    elif dataset == 'food101':
        return _create_food101_splits(dataset_root, seed, save_splits)
    elif dataset == 'kinetics':
        return _create_kinetics_splits(dataset_root, seed, save_splits)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Choose from: 'cremad', 'ucf101', 'food101', 'kinetics'"
        )


def _create_cremad_splits(dataset_root, seed, save_splits):
    """Create splits for CREMA-D dataset (70% train, 10% dev, 20% test)"""

    train_path = os.path.join(dataset_root, "train_split.csv")
    dev_path = os.path.join(dataset_root, "val.csv")
    test_path = os.path.join(dataset_root, "test.csv")

    # Load original splits
    train_df = pd.read_csv(train_path, sep=',')
    dev_df = pd.read_csv(dev_path, sep=',')
    test_df = pd.read_csv(test_path, sep=',')

    # Combine all data
    combined_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

    # Split: train (70%) / temp (30%)
    train_df, temp_df = train_test_split(
        combined_df,
        test_size=0.3,
        random_state=seed,
        stratify=combined_df['label']
    )

    # Split: dev (10%) / test (20%) from temp
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=2/3,  # 20% = 2/3 of 30%
        random_state=seed,
        stratify=temp_df['label']
    )

    print(f"CREMA-D splits - Train: {train_df.shape}, Dev: {dev_df.shape}, Test: {test_df.shape}")

    # Save splits
    split_paths = {}
    if save_splits:
        train_split_path = f'./train_split_cremad_{seed}.csv'
        dev_split_path = f'./dev_split_cremad_{seed}.csv'
        test_split_path = f'./test_split_cremad_{seed}.csv'

        train_df.to_csv(train_split_path, sep=',', index=False)
        dev_df.to_csv(dev_split_path, sep=',', index=False)
        test_df.to_csv(test_split_path, sep=',', index=False)

        split_paths = {
            'train': train_split_path,
            'dev': dev_split_path,
            'test': test_split_path
        }
    else:
        split_paths = {
            'train': None,
            'dev': None,
            'test': None
        }

    return train_df, dev_df, test_df, split_paths


def _create_ucf101_splits(dataset_root, seed, save_splits):
    """Create splits for UCF101 dataset (70% train, 10% dev, 20% test)"""

    train_path = os.path.join(dataset_root, "trainlist01_new.txt")
    test_path = os.path.join(dataset_root, "testlist01_new.txt")

    # Load original splits
    train_df = pd.read_csv(train_path, sep=' ')
    test_df = pd.read_csv(test_path, sep=' ')

    # Combine all data
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # Split: train (70%) / temp (30%)
    train_df, temp_df = train_test_split(
        combined_df,
        train_size=0.7,
        random_state=seed,
        stratify=combined_df['label']
    )

    # Split: dev (10%) / test (20%) from temp
    dev_df, test_df = train_test_split(
        temp_df,
        train_size=1/3,  # 10% = 1/3 of 30%
        random_state=seed,
        stratify=temp_df['label']
    )

    print(f"UCF101 splits - Train: {train_df.shape}, Dev: {dev_df.shape}, Test: {test_df.shape}")

    # Save splits
    split_paths = {}
    if save_splits:
        train_split_path = f'./train_split_ucf101_{seed}.csv'
        dev_split_path = f'./dev_split_ucf101_{seed}.csv'
        test_split_path = f'./test_split_ucf101_{seed}.csv'

        train_df.to_csv(train_split_path, sep=' ', index=False)
        dev_df.to_csv(dev_split_path, sep=' ', index=False)
        test_df.to_csv(test_split_path, sep=' ', index=False)

        split_paths = {
            'train': train_split_path,
            'dev': dev_split_path,
            'test': test_split_path
        }
    else:
        split_paths = {
            'train': None,
            'dev': None,
            'test': None
        }

    return train_df, dev_df, test_df, split_paths


def _create_food101_splits(dataset_root, seed, save_splits):
    """Create splits for Food101 dataset (70% train, 10% dev, 20% test)"""

    train_path = os.path.join(dataset_root, "train.jsonl")
    test_path = os.path.join(dataset_root, "test.jsonl")

    # Load original splits
    train_frame = pd.read_json(train_path, lines=True)
    test_frame = pd.read_json(test_path, lines=True)

    train_frame["original_split"] = "train"
    test_frame["original_split"] = "test"

    # Combine all data
    full_frame = pd.concat([train_frame, test_frame], ignore_index=True)

    print(f"Total Food101 data: {full_frame.shape}")

    # Split: train (70%) / temp (30%)
    train_split, val_test_split = train_test_split(
        full_frame,
        train_size=0.7,
        stratify=full_frame['label'],
        random_state=seed
    )

    # Split: dev (10%) / test (20%) from temp
    val_split, test_split = train_test_split(
        val_test_split,
        train_size=1/3,  # 10% = 1/3 of 30%
        stratify=val_test_split['label'],
        random_state=seed
    )

    print(f"Food101 splits - Train: {train_split.shape}, Dev: {val_split.shape}, Test: {test_split.shape}")

    # Save splits
    split_paths = {}
    if save_splits:
        train_split_path = f'./train_split_food101_{seed}.jsonl'
        dev_split_path = f'./dev_split_food101_{seed}.jsonl'
        test_split_path = f'./test_split_food101_{seed}.jsonl'

        train_split.to_json(train_split_path, orient='records', lines=True)
        val_split.to_json(dev_split_path, orient='records', lines=True)
        test_split.to_json(test_split_path, orient='records', lines=True)

        split_paths = {
            'train': train_split_path,
            'dev': dev_split_path,
            'test': test_split_path
        }
    else:
        split_paths = {
            'train': None,
            'dev': None,
            'test': None
        }

    return train_split, val_split, test_split, split_paths


def _create_kinetics_splits(dataset_root, seed, save_splits):
    """Create splits for Kinetics dataset (70% train, 10% dev, 20% test)"""

    video_dir = os.path.join(dataset_root, 'videos')
    train_path = os.path.join(video_dir, "train.txt")
    dev_path = os.path.join(video_dir, "val.txt")

    # Load original splits
    train_frame = pd.read_csv(train_path, sep=' ', header=0, names=['filename', 'start', 'end', 'label'])
    dev_frame = pd.read_csv(dev_path, sep=' ', header=0, names=['filename', 'start', 'end', 'label'])

    # Combine all data
    combined_frame = pd.concat([train_frame, dev_frame], ignore_index=True)

    # Split: train (70%) / temp (30%)
    train_split, temp_split = train_test_split(
        combined_frame,
        test_size=0.3,
        random_state=seed,
        stratify=combined_frame['label']
    )

    # Split: dev (10%) / test (20%) from temp
    dev_split, test_split = train_test_split(
        temp_split,
        test_size=2/3,  # 20% = 2/3 of 30%
        random_state=seed,
        stratify=temp_split['label']
    )

    print(f"Kinetics splits - Train: {train_split.shape}, Dev: {dev_split.shape}, Test: {test_split.shape}")

    # Save splits
    split_paths = {}
    if save_splits:
        train_split_path = f'./train_split_kinetics_{seed}.txt'
        dev_split_path = f'./dev_split_kinetics_{seed}.txt'
        test_split_path = f'./test_split_kinetics_{seed}.txt'

        train_split.to_csv(train_split_path, sep=' ', index=False)
        dev_split.to_csv(dev_split_path, sep=' ', index=False)
        test_split.to_csv(test_split_path, sep=' ', index=False)

        split_paths = {
            'train': train_split_path,
            'dev': dev_split_path,
            'test': test_split_path
        }
    else:
        split_paths = {
            'train': None,
            'dev': None,
            'test': None
        }

    return train_split, dev_split, test_split, split_paths


def get_dataloader(dataset, mode, hparams, split_path=None, model=None):
    """
    Get dataloader for specified dataset and mode

    Args:
        dataset (str): Dataset name ('cremad', 'ucf101', 'food101', 'kinetics')
        mode (str): 'train', 'val', or 'test'
        hparams (dict): Hyperparameters dictionary
        split_path (str): Path to split file
        model: Model object (needed for Food101 text transform)

    Returns:
        DataLoader object
    """
    from torch.utils.data import DataLoader
    dataset = dataset.lower()

    if dataset == 'cremad':
        from datasets.Cremad_dataset import CREMADDataset
        dataset_obj = CREMADDataset(
            mode=mode,
            audio_path=hparams['audio_path'],
            visual_path=hparams['visual_path'],
            stat_csv=hparams['stat_path'],
            train_csv=hparams.get('train_path') if mode == 'train' else None,
            val_csv=hparams.get('dev_path') if mode == 'val' else None,
            test_csv=split_path if mode == 'test' else hparams.get('test_path')
        )

    elif dataset == 'ucf101':
        from datasets.Ucf101_dataset import UCF101Dataset
        dataset_obj = UCF101Dataset(
            mode=mode,
            stat_path=hparams['stat_path'],
            visual_path=hparams['visual_path'],
            flow_path_u=hparams['flow_path_u'],
            flow_path_v=hparams['flow_path_v'],
            train_path=hparams.get('train_path') if mode == 'train' else None,
            val_path=hparams.get('dev_path') if mode == 'val' else None,
            test_path=split_path if mode == 'test' else hparams.get('test_path')
        )

    elif dataset == 'food101':
        from datasets.Food101_dataset import Food101Dataset
        from torchvision import transforms

        # Image transforms
        test_image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

        # Text transform (use model's text transform if available)
        text_transform = model.text_transform if model and hasattr(model, 'text_transform') else None

        dataset_obj = Food101Dataset(
            data_path=split_path if split_path else hparams.get('test_path'),
            img_dir=hparams['img_dir'],
            image_transform=test_image_transform,
            text_transform=text_transform
        )

    elif dataset == 'kinetics':
        from datasets.Kinetics_dataset import KineticsDataset

        # Audio and video transforms (use model's if available)
        audio_transform = model.audio_transform if model and hasattr(model, 'audio_transform') else None
        video_transform = model.video_transform if model and hasattr(model, 'video_transform') else None

        dataset_obj = KineticsDataset(
            mode=mode,
            audio_dir=hparams['audio_dir'],
            video_dir=hparams['video_dir'],
            audio_transform=audio_transform,
            video_transform=video_transform,
            train_path=hparams.get('train_path') if mode == 'train' else None,
            dev_path=hparams.get('dev_path') if mode == 'val' else None,
            test_path=split_path if mode == 'test' else hparams.get('test_path')
        )

    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Choose from: 'cremad', 'ucf101', 'food101', 'kinetics'"
        )

    shuffle = (mode == 'train')
    dataloader = DataLoader(
        dataset_obj,
        batch_size=hparams['batch_size'],
        shuffle=shuffle,
        num_workers=hparams.get('num_workers', 4),
        pin_memory=True
    )

    return dataloader
