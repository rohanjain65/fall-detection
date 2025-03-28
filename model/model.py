from torch import Tensor, nn
from torchvision import models

from utils.data import NUM_CLASSES
from utils.misc import take_annotation_from


class FallDetectionModel(nn.Module):
    """
    Vision model for fall detection.

    Args:
        backbone_name (str): Name of the backbone model.
        num_classes (int, optional): Number of output classes, default is NUM_CLASSES.
        pretrained (bool, optional): Load pretrained weights, enabled by default.
    """

    def __init__(self, backbone: str, *, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> None:
        super(FallDetectionModel, self).__init__()

        # Validate the backbone name, TODO: Implement for ViT (isinstance)
        assert hasattr(models, backbone), f"Model {backbone} not found in torchvision.models"
        assert "convnext" in backbone, f"Model {backbone} is not a ConvNeXt model"

        # Load the backbone
        self.backbone: nn.Module = getattr(models, backbone)(weights="DEFAULT" if pretrained else None)

        # Replace the classifier
        num_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Identity()

        self.classifier = nn.Linear(num_features, num_classes)

        # Initialize the model weights
        self._initialize_weights()

    def forward_features(self, features: Tensor) -> Tensor:
        """
        Forward pass of the backbone model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            features (Tensor): Image features, with shape (batch_size, num_features).
        """
        features = self.backbone(features)

        return features

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            logits (Tensor): Model predictions, with shape (batch_size, num_classes).
        """

        # Extract the image features
        features = self.forward_features(x)

        # Pass the features through the classifier
        logits = self.classifier(features)

        return logits

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)

    def _initialize_weights(self) -> None:
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
