from torch import Tensor, nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights, vit_b_16

from utils.misc import take_annotation_from


class FallDetectionModel(nn.Module):
    """
    Vision model for fall detection supporting ConvNeXt and ViT backbones.

    Args:
        backbone_name (str): Name of the backbone model.
        num_classes (int): Number of output classes.
        pretrained (bool, optional): Load pretrained weights, enabled by default.
    """

    def __init__(self, backbone: str, num_classes: int, *, num_channels: int = 3, pretrained: bool = True) -> None:
        super(FallDetectionModel, self).__init__()

        assert hasattr(models, backbone) or backbone == "vit_b_16", f"Model {backbone} not found"

        # Load the backbone
        self.backbone: models.ConvNeXt = getattr(models, backbone)(weights="DEFAULT" if pretrained else None)

        # Replace the first convolutional layer
        self._replace_convnext_first_conv(num_channels)

        # Replace the classifier
        self.classifier = self._replace_convnext_classifier(num_classes)
        if "convnext" in backbone:
            self.backbone: nn.Module = getattr(models, backbone)(weights="DEFAULT" if pretrained else None)
            num_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Identity()

        elif backbone == "vit_b_16":
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            self.backbone = vit_b_16(weights=weights)
            num_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()

        else:
            raise ValueError(f"Unsupported backbone type: {backbone}")

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

    def _replace_convnext_first_conv(self, num_channels: int) -> None:
        """
        In-place replacement of the first convolutional layer of the ConvNeXt model.

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
        Removes the classifier of the ConvNeXt model and replaces it with a new one.

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

    def _initialize_weights(self) -> None:
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
