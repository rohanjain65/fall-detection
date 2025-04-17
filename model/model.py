from typing import Union

from torch import Tensor, nn
from torchvision import models

from utils.misc import take_annotation_from

Backbone = Union[models.ConvNeXt, models.SwinTransformer]

class FallDetectionModel(nn.Module):
    """
    Vision model for fall detection.

    Args:
        backbone_name (str): Name of the backbone model, currently only supports models from the ConvNeXt and SwinTransformer families.
        num_classes (int): Number of output classes.
        pretrained (bool, optional): Load pretrained weights, enabled by default.
    """

    def __init__(self, backbone: str, num_classes: int, *, num_channels: int = 3, pretrained: bool = True) -> None:
        super(FallDetectionModel, self).__init__()

        assert hasattr(models, backbone), f"Model {backbone} not found"

        # Load the backbone
        self.backbone: Backbone = getattr(models, backbone)(weights="DEFAULT" if pretrained else None)

        # Replace the first convolutional layer
        self._replace_first_conv(num_channels)

        # Replace the classifier
        if isinstance(self.backbone, models.ConvNeXt):
            self.classifier = self._replace_convnext_classifier(num_classes)
        elif isinstance(self.backbone, models.SwinTransformer):
            self.classifier = self._replace_swin_classifier(num_classes)
        else:
            raise ValueError(f"Unsupported backbone model: {backbone}") 

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

    def _replace_first_conv(self, num_channels: int) -> None:
        """
        In-place replacement of the first convolutional layer of the backbone.

        Args:
            num_channels (int): Number of input channels.
        """

        # Don't need to replace for RGB imagery
        if num_channels == 3:
            return

        out_channels = self.backbone.features[0][0].out_channels
        kernel_size = self.backbone.features[0][0].kernel_size
        stride = self.backbone.features[0][0].stride
        padding = self.backbone.features[0][0].padding

        self.backbone.features[0][0] = nn.Conv2d(
            num_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

    def _replace_convnext_classifier(self, num_classes: int) -> nn.Linear:
        """
        Removes the classifier of a ConvNeXt backbone and replaces it with a new one.

        Args:
            num_classes (int): Number of output classes.

        Returns:
            classifier (nn.Linear): The new classifier layer.
        """

        # Replace the classifier
        num_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Identity()

        # Create a new classifier
        classifier = nn.Linear(num_features, num_classes)

        return classifier

    def _replace_swin_classifier(self, num_classes: int) -> nn.Linear:
        """
        Removes the classifier of a Swin backbone and replaces it with a new one.

        Args:
            num_classes (int): Number of output classes.

        Returns:
            classifier (nn.Linear): The new classifier layer.
        """

        # Replace the classifier
        num_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        # Create a new classifier
        classifier = nn.Linear(num_features, num_classes)

        return classifier

    def _initialize_weights(self) -> None:
        """
        Initialize the weights of the classifier.
        """

        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
