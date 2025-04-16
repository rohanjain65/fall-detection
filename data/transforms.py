from typing import Callable, Tuple

import torch
import torchvision.transforms.v2 as transforms
from torch import Tensor

IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD = [0.229, 0.224, 0.225]


class AddGausianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, image: Tensor) -> Tensor:
        return image + torch.randn(image.shape) * self.std + self.mean


def get_train_transforms(image_size: Tuple[int, int] = (240, 320)) -> Callable[[Tensor], Tensor]:
    """
    Get the transformations for training.

    Args:
        image_size (Tuple[int, int], optional): The desired image size.

    Returns:
        transformations (Callable[[Tensor], Tensor]): The transformations to apply during training.
    """

    transformations = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            # transforms.RandomGrayscale(p=0.1),
            transforms.Normalize(IMNET_MEAN, IMNET_STD),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.0), value=0),
        ]
    )

    return transformations


def get_val_transforms(image_size: Tuple[int, int] = (240, 320)) -> Callable[[Tensor], Tensor]:
    """
    Get the transformations for validation.

    Args:
        image_size (Tuple[int, int], optional): The desired image size.

    Returns:
        transformations (Callable[[Tensor], Tensor]): The transformations to apply during validation.
    """

    transformations = transforms.Compose([transforms.Resize(image_size), transforms.Normalize(IMNET_MEAN, IMNET_STD)])

    return transformations
