"""
Model factory for creating model instances based on method and dataset
Supports CREMA-D, UCF101, Food101, and Kinetics datasets
"""


def get_model(method, dataset, hparams):
    """
    Factory function to get model instance

    Args:
        method (str): Method name (currently only 'Midas' is supported)
        dataset (str): Dataset name ('cremad', 'ucf101', 'food101', 'kinetics')
        hparams (dict): Hyperparameters dictionary

    Returns:
        Model instance

    Raises:
        ValueError: If method or dataset is not supported
    """
    dataset = dataset.lower()

    if method != 'Midas':
        raise ValueError(f"Currently only 'Midas' method is supported, got: {method}")

    if dataset == 'cremad':
        from models.model_midas_cremad import MidasModel
        return MidasModel(params=hparams)
    elif dataset == 'ucf101':
        from models.model_midas_ucf101 import MidasModel
        return MidasModel(params=hparams)
    elif dataset == 'food101':
        from models.model_midas_food101 import MidasModel
        return MidasModel(params=hparams)
    elif dataset == 'kinetics':
        from models.model_midas_kinetics import MidasModel
        return MidasModel(params=hparams)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Choose from: 'cremad', 'ucf101', 'food101', 'kinetics'"
        )


def get_available_methods(dataset):
    """
    Get list of available methods for a dataset

    Args:
        dataset (str): Dataset name ('cremad', 'ucf101', 'food101', 'kinetics')

    Returns:
        list: List of available method names (currently only ['Midas'])
    """
    # Currently only Midas is supported for all datasets
    return ['Midas']
