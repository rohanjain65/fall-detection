from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils.misc import take_annotation_from


class Criterion(nn.Module):
    """
    Parent criterion class that defines the interface for all custom loss functions.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args) -> Dict[str, Tensor]:
        raise NotImplementedError

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs) -> Dict[str, Tensor]:
        return nn.Module.__call__(self, *args, **kwargs)

    def _check_for_nans(self, losses: Dict[str, Tensor]) -> None:
        nan_losses = [name for name, loss in losses.items() if torch.isnan(loss).any()]

        if nan_losses:
            raise ValueError(f"NaNs detected in losses: {nan_losses}")


class FallDetectionCriterion(Criterion):
    """

    Args:
        class_frequencies (Tensor): Frequencies of each class in the dataset.
    """

    def __init__(self, frequencies: Tensor = None) -> None:
        super().__init__()

        self.weight = self._calculate_weights(frequencies) if frequencies is not None else None

    def forward(self, predictions: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        mse_loss = F.cross_entropy(predictions, targets, weight=self.weight)

        losses = {"ce": mse_loss}

        losses["overall"] = losses["ce"]

        return losses

    def _calculate_weights(self, class_frequencies: Tensor) -> Tensor:
        """
        Assigns weights to each class, inversely proportional to its frequency in the dataset.

        Args:
            class_frequencies (Tensor): The frequency of each class in the dataset.

        Returns:
            class_weights (Tensor): The weight of each class.
        """

        class_weights = torch.ones_like(class_frequencies, dtype=torch.float32)

        class_weights[class_frequencies > 0] = class_frequencies.sum() / (len(class_frequencies) * class_frequencies[class_frequencies > 0])

        return class_weights
