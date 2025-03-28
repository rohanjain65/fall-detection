from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils.misc import take_annotation_from

EPSILON = 1e-8


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

    def __init__(self, focal_gamma: float = 1.0, frequencies: Tensor = None) -> None:
        super().__init__()

        self.focal_gamma = focal_gamma
        self.weight = self._calculate_weights(frequencies) if frequencies is not None else None

    def forward(self, predictions: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        # ce_loss = F.cross_entropy(predictions, targets, weight=self.weight)
        ce_loss = self._get_focal_loss(predictions, targets, weight=self.weight)

        losses = {"ce": ce_loss}

        losses["overall"] = losses["ce"]

        return losses

    def _get_focal_loss(self, prediction_logits: Tensor, target_labels: Tensor, weight: Tensor = None) -> Tensor:
        """
        Computes the focal loss for the classification head. Gives more weight to hard-to-classify examples.

        Inputs:
            prediction_logits (Tensor): The predicted class logits [num_objects, num_classes].
            target_labels (Tensor): The target class labels [num_objects].
            weight (Tensor): The class weights [num_classes].

        Returns:
            focal_loss (Tensor): The focal loss.
        """

        # Compute the class probabilities
        prediction_probabilities = F.softmax(prediction_logits, dim=1)

        # Calculate the cross-entropy loss
        log_probabilities = torch.log(prediction_probabilities.clamp_min(EPSILON))

        loss_ce = F.nll_loss(log_probabilities, target_labels, weight=weight, reduction="none")

        # Calculate the focal term
        probability_target = prediction_probabilities[torch.arange(len(prediction_probabilities)), target_labels]

        focal_term = (1 - probability_target) ** self.focal_gamma

        # Combine the terms
        focal_loss = focal_term * loss_ce

        # Average the loss
        focal_loss = focal_loss.mean()

        return focal_loss

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
