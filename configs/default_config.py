"""
Unified configuration for multimodal learning experiments
Supports CREMA-D, UCF101, Food101, and Kinetics datasets
"""
import os


class BaseConfig:
    """Base configuration class with common parameters"""

    # Random seed
    seed = 2

    # GPU settings
    devices = '0'

    # Training parameters
    batch_size = 64
    num_workers = 5
    accumulate_grad_batches = 1
    max_epochs = 70
    warmup_epochs = 5
    early_stop_patience = 100

    # Model parameters
    embedding_dim = 150
    fusion_output_size = 256
    dropout_p = 0.2
    hidden_dim = 1024

    # Optimizer parameters
    lr = 1e-3
    weight_decay = 1e-4

    # Augmentation
    use_augmentation = False

    # Method-specific parameters (AMCo, etc.)
    alpha = 1
    sigma = 0.5
    eps = 0.3
    modulation_starts = 0
    modulation_ends = 70

    # Misc
    dev_limit = None
    recording = False


class CremadConfig(BaseConfig):
    """Configuration for CREMA-D dataset (Audio-Video emotion recognition)"""

    # Dataset info
    dataset = 'cremad'
    num_classes = 6
    modality_names = ['audio', 'video']

    # Feature dimensions
    audio_feature_dim = 512
    video_feature_dim = 512

    # Data paths (relative to dataset root)
    train_file = 'train_split.csv'
    val_file = 'val.csv'
    test_file = 'test.csv'
    stat_file = 'stat.csv'
    audio_dir = 'AudioWAV'
    video_dir = 'visual_frames'

    # CREMA-D specific parameters
    fps = 3  # frames per second for video sampling

    def __init__(self, dataset_root=None):
        if dataset_root is None:
            raise ValueError("dataset_root must be specified for CREMA-D dataset")
        self.dataset_root = dataset_root
        self.train_path = os.path.join(dataset_root, self.train_file)
        self.dev_path = os.path.join(dataset_root, self.val_file)
        self.test_path = os.path.join(dataset_root, self.test_file)
        self.stat_path = os.path.join(dataset_root, self.stat_file)
        self.audio_path = os.path.join(dataset_root, self.audio_dir)
        self.visual_path = os.path.join(dataset_root, self.video_dir)


class UCF101Config(BaseConfig):
    """Configuration for UCF101 dataset (RGB-Flow action recognition)"""

    # Dataset info
    dataset = 'ucf101'
    num_classes = 101
    modality_names = ['visual', 'flow']

    # Feature dimensions
    flow_feature_dim = 512
    vision_feature_dim = 512

    # Data paths (relative to dataset root)
    train_file = 'trainlist01_new.txt'
    test_file = 'testlist01_new.txt'
    stat_file = 'classInd.txt'
    visual_dir = 'frames_rgb'
    flow_u_dir = 'flows_u'
    flow_v_dir = 'flows_v'

    # UCF101 specific learning rate
    lr = 1e-2

    def __init__(self, dataset_root=None):
        if dataset_root is None:
            raise ValueError("dataset_root must be specified for UCF101 dataset")
        self.dataset_root = dataset_root
        self.train_path = os.path.join(dataset_root, self.train_file)
        self.test_path = os.path.join(dataset_root, self.test_file)
        self.stat_path = os.path.join(dataset_root, self.stat_file)
        self.visual_path = os.path.join(dataset_root, self.visual_dir)
        self.flow_path_u = os.path.join(dataset_root, self.flow_u_dir)
        self.flow_path_v = os.path.join(dataset_root, self.flow_v_dir)


class Food101Config(BaseConfig):
    """Configuration for Food101 dataset (Image-Text food classification)"""

    # Dataset info
    dataset = 'food101'
    num_classes = 101
    modality_names = ['image', 'text']

    # Feature dimensions
    language_feature_dim = 256
    vision_feature_dim = 256

    # Data paths
    train_file = 'train.jsonl'
    test_file = 'test.jsonl'
    image_dir = 'images'

    # Food101 specific parameters
    lr = 1e-3
    warmup_epochs = 2
    max_epochs = 30
    dropout_p = 0.0

    def __init__(self, dataset_root=None):
        if dataset_root is None:
            raise ValueError("dataset_root must be specified for Food101 dataset")
        self.dataset_root = dataset_root
        self.train_path = os.path.join(dataset_root, self.train_file)
        self.test_path = os.path.join(dataset_root, self.test_file)
        self.img_dir = os.path.join(dataset_root, self.image_dir)


class KineticsConfig(BaseConfig):
    """Configuration for Kinetics dataset (Audio-Video action recognition)"""

    # Dataset info
    dataset = 'kinetics'
    num_classes = 400  # Kinetics-400
    modality_names = ['audio', 'video']

    # Feature dimensions
    audio_feature_dim = 512
    video_feature_dim = 512

    # Data paths
    train_file = 'train.txt'
    val_file = 'val.txt'
    audio_dir = 'audios'
    video_dir = 'videos'

    # Kinetics specific parameters
    lr = 1e-3
    warmup_epochs = 5
    max_epochs = 70
    dropout_p = 0.2
    hidden_dim = 1024

    def __init__(self, dataset_root=None):
        if dataset_root is None:
            raise ValueError("dataset_root must be specified for Kinetics dataset")
        self.dataset_root = dataset_root
        self.train_path = os.path.join(dataset_root, self.video_dir, self.train_file)
        self.dev_path = os.path.join(dataset_root, self.video_dir, self.val_file)
        self.audio_dir = os.path.join(dataset_root, 'audios')
        self.video_dir = os.path.join(dataset_root, 'videos')


def get_config(dataset='cremad', dataset_root=None):
    """
    Factory function to get dataset-specific configuration

    Args:
        dataset (str): Dataset name ('cremad', 'ucf101', 'food101', 'kinetics')
        dataset_root (str): Root directory of the dataset. If None, uses default.

    Returns:
        Config object for the specified dataset
    """
    dataset = dataset.lower()

    if dataset == 'cremad':
        return CremadConfig(dataset_root) if dataset_root else CremadConfig()
    elif dataset == 'ucf101':
        return UCF101Config(dataset_root) if dataset_root else UCF101Config()
    elif dataset == 'food101':
        return Food101Config(dataset_root) if dataset_root else Food101Config()
    elif dataset == 'kinetics':
        return KineticsConfig(dataset_root) if dataset_root else KineticsConfig()
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Choose from: 'cremad', 'ucf101', 'food101', 'kinetics'"
        )


def get_method_config(method, base_config):
    """
    Update config with method-specific parameters

    Args:
        method (str): Method name (e.g., 'Baseline', 'Midas', 'AMCo')
        base_config (BaseConfig): Base configuration object

    Returns:
        Updated config dictionary
    """
    # Get all attributes from config object
    config_dict = {k: v for k, v in vars(base_config).items()
                   if not k.startswith('_')}

    dataset_name = base_config.dataset

    # Method-specific adjustments
    if method == 'Baseline':
        config_dict['recording'] = False

    elif method == 'Midas':
        # Dataset-specific warmup epochs for Midas
        if dataset_name == 'cremad':
            config_dict['warmup_epochs'] = 10
        elif dataset_name == 'ucf101':
            config_dict['warmup_epochs'] = 0
        elif dataset_name == 'food101':
            config_dict['warmup_epochs'] = 2
        elif dataset_name == 'kinetics':
            config_dict['warmup_epochs'] = 10

        config_dict['use_augmentation'] = True
        config_dict['recording'] = False

    elif method == 'MidasInputLevel':
        # Same as Midas
        if dataset_name == 'cremad':
            config_dict['warmup_epochs'] = 10
        elif dataset_name == 'ucf101':
            config_dict['warmup_epochs'] = 0
        elif dataset_name == 'food101':
            config_dict['warmup_epochs'] = 2
        elif dataset_name == 'kinetics':
            config_dict['warmup_epochs'] = 10

        config_dict['use_augmentation'] = True
        config_dict['recording'] = False

    elif method == 'MidasL2':
        # Same as Midas
        if dataset_name == 'cremad':
            config_dict['warmup_epochs'] = 10
        elif dataset_name == 'ucf101':
            config_dict['warmup_epochs'] = 0
        elif dataset_name == 'food101':
            config_dict['warmup_epochs'] = 2
        elif dataset_name == 'kinetics':
            config_dict['warmup_epochs'] = 10

        config_dict['use_augmentation'] = True
        config_dict['recording'] = False

    elif method == 'Resampling':
        if dataset_name == 'cremad':
            config_dict['warmup_epochs'] = 5
        elif dataset_name == 'ucf101':
            config_dict['warmup_epochs'] = 10
        elif dataset_name == 'food101':
            config_dict['warmup_epochs'] = 2
        elif dataset_name == 'kinetics':
            config_dict['warmup_epochs'] = 5

    elif method == 'DynCIM':
        config_dict['max_epochs'] = 70

    elif method == 'CGGM':
        config_dict['warmup_epochs'] = 20
        if dataset_name == 'food101':
            config_dict['max_epochs'] = 70

    config_dict['method'] = method
    config_dict['output_path'] = f"model-outputs-{dataset_name}-{method}"

    return config_dict
