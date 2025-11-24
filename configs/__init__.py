"""
Configuration modules for multimodal learning experiments
"""
from .default_config import (
    BaseConfig,
    CremadConfig,
    UCF101Config,
    Food101Config,
    KineticsConfig,
    get_config,
    get_method_config,
)

__all__ = [
    'BaseConfig',
    'CremadConfig',
    'UCF101Config',
    'Food101Config',
    'KineticsConfig',
    'get_config',
    'get_method_config',
]
