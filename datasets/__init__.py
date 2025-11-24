"""
Dataset modules for multimodal learning
Supports CREMA-D, UCF101, Food101, and Kinetics datasets

Note: The actual dataset implementation files should be copied from:
- CREMA-D & UCF101: /home/soyoung/nas/MIDAS/datasets/
- Food101: /home/soyoung/BEAST/datasets/Food101_dataset.py
- Kinetics: /home/soyoung/BEAST/datasets/Kinetics_dataset.py
"""

__all__ = [
    'CREMADDataset',
    'CREMADDataset_sample_level',
    'UCF101Dataset',
    'UCF101Dataset_sample_level',
    'Food101Dataset',
    'Food101Dataset_sample_level',
    'KineticsDataset',
    'KineticsDataset_sample_level',
]
