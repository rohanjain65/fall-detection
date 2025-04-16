from torch import Tensor, nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

from utils.misc import take_annotation_from


class FallDetectionModel(nn.Module):
    """
    Vision Transformer model for fall detection.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool, optional): Load pretrained weights, enabled by default.
    """

    def __init__(self, num_classes: int, *, pretrained: bool = True) -> None:
        super(FallDetectionModel, self).__init__()

        # Load pretrained ViT model
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone: nn.Module = vit_b_16(weights=weights)

        # Replace the head classifier
        num_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Identity()

        self.classifier = nn.Linear(num_features, num_classes)

        # Initialize weights
        self._initialize_weights()

    def forward_features(self, features: Tensor) -> Tensor:
        """
        Forward pass of the backbone model.

        Args:
            features (Tensor): Input tensor.

        Returns:
            features (Tensor): Image features, shape (batch_size, num_features).
        """
        features = self.backbone(features)
        return features

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            logits (Tensor): Model predictions, shape (batch_size, num_classes).
        """
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)

    def _initialize_weights(self) -> None:
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
