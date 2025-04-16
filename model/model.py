from torch import Tensor, nn
from torchvision import models
from torchvision.models import vit_b_16, ViT_B_16_Weights

from utils.misc import take_annotation_from


class FallDetectionModel(nn.Module):
    """
    Vision model for fall detection supporting ConvNeXt and ViT backbones.

    Args:
        backbone_name (str): Name of the backbone model.
        num_classes (int): Number of output classes.
        pretrained (bool, optional): Load pretrained weights, enabled by default.
    """

    def __init__(self, backbone: str, num_classes: int, *, pretrained: bool = True) -> None:
        super(FallDetectionModel, self).__init__()

        assert hasattr(models, backbone) or backbone == "vit_b_16", f"Model {backbone} not found"

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

    def _initialize_weights(self) -> None:
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)