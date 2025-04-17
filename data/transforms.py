from typing import Callable, List, Tuple

import torch
import torchvision.transforms.v2 as transforms
from torch import Tensor

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]

DEPTH_MEAN = [sum(RGB_MEAN) / 3]
DEPTH_STD = [sum(RGB_STD) / 3]


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, image: Tensor) -> Tensor:
        return image + torch.randn(image.shape) * self.std + self.mean


def get_train_transforms(image_size: Tuple[int, int] = (240, 320), modality: str = "rgb") -> Callable[[Tensor], Tensor]:
    """
    Get the transformations for training.

    Args:
        image_size (Tuple[int, int], optional): The desired image size.

    Returns:
        transformations (Callable[[Tensor], Tensor]): The transformations to apply during training.
    """

    mean, std = _get_norms(modality)

    geometric_transforms = (
        transforms.RandomChoice(
            [
                transforms.RandomRotation(degrees=5),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
            ]
        ),
    )

    photometric_transforms = (
        transforms.RandomChoice(
            [
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                AddGaussianNoise(std=0.05),
            ]
        ),
    )

    # Build the transformations
    transformations = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(geometric_transforms, p=0.5),
            transforms.RandomApply(photometric_transforms, p=0.5),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.15), ratio=(0.3, 3.0), value=0),
        ]
    )

    return transformations


def get_val_transforms(image_size: Tuple[int, int] = (240, 320), modality: str = "rgb") -> Callable[[Tensor], Tensor]:
    """
    Get the transformations for validation.

    Args:
        image_size (Tuple[int, int], optional): The desired image size.

    Returns:
        transformations (Callable[[Tensor], Tensor]): The transformations to apply during validation.
    """

    # Get the normalization values
    mean, std = _get_norms(modality)

    # Build the transformations
    transformations = transforms.Compose([transforms.Resize(image_size), transforms.Normalize(mean, std)])

    return transformations


def _get_norms(modality: str) -> Tuple[List[int], List[int]]:
    """
    Get the normalization values for the specified modality.

    Args:
        modality (str): The modality to use, must be one of 'rgb', 'depth', or 'both'.

    Returns:
        mean (List[int]): The mean values for normalization.
        std (List[int]): The standard deviation values for normalization.
    """

    # Validate the modality
    assert modality in ["rgb", "depth", "both"], f"Invalid modality {modality}, must be one of 'rgb', 'depth', or 'both'."

    if modality == "rgb":
        mean = RGB_MEAN
        std = RGB_STD
    elif modality == "depth":
        mean = DEPTH_MEAN
        std = DEPTH_STD
    elif modality == "both":
        mean = RGB_MEAN + DEPTH_MEAN
        std = RGB_STD + DEPTH_STD

    return mean, std
