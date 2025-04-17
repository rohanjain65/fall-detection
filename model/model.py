from typing import Union

import torch
from torch import Tensor, nn
from torchvision import models

from utils.misc import take_annotation_from

Backbone = Union[models.ConvNeXt, models.SwinTransformer, models.VisionTransformer]

class ImageClassifier(nn.Module):
    """
    Vision model for fall detection.

    Args:
        backbone (str): Name of the backbone model, currently only supports models from the ConvNeXt, ViT, and SwinTransformer families.
        num_classes (int): Number of output classes.
        pretrained (bool, optional): Load pretrained weights, enabled by default.
    """

    def __init__(self, backbone: str, num_classes: int, *, num_channels: int = 3, pretrained: bool = True) -> None:
        super(ImageClassifier, self).__init__()

        assert hasattr(models, backbone), f"Model {backbone} not found"

        # Load the backbone
        self.backbone: Backbone = getattr(models, backbone)(weights="DEFAULT" if pretrained else None)

        # Replace the first convoluational layer and the classifier
        if isinstance(self.backbone, models.ConvNeXt):
            self._replace_first_conv(num_channels)

            self.classifier = self._replace_convnext_classifier(num_classes)
        elif isinstance(self.backbone, models.SwinTransformer):
            self._replace_first_conv(num_channels)

            self.classifier = self._replace_swin_classifier(num_classes)
        elif isinstance(self.backbone, models.VisionTransformer):
            self._replace_vit_first_conv(num_channels)

            self.classifier = self._replace_vit_classifier(num_classes)
        else:
            raise ValueError(f"Unsupported backbone model: {backbone}") 
        
        self.embed_dim = self.classifier.in_features

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

    def forward(self, images: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            images (Tensor): Input image with shape (batch_size, num_channels, height, width).

        Returns:
            logits (Tensor): Model predictions, shape (batch_size, num_classes).
        """

        features = self.forward_features(images)

        logits = self.classifier(features)

        return logits

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)

    def _replace_first_conv(self, num_channels: int) -> None:
        """
        In-place replacement of the first convolutional layer of the backbone for ConvNext and SwinTransformer models.

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

    def _replace_vit_first_conv(self, num_channels: int) -> None:
        """
        In-place replacement of the first convolutional layer of the backbone for ViT models.

        Args:
            num_channels (int): Number of input channels.
        """

        # Don't need to replace for RGB imagery
        if num_channels == 3:
            return
        out_channels = self.backbone.conv_proj.out_channels
        kernel_size = self.backbone.conv_proj.kernel_size
        stride = self.backbone.conv_proj.stride
        padding = self.backbone.conv_proj.padding

        self.backbone.conv_proj = nn.Conv2d(
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

    def _replace_vit_classifier(self, num_classes: int) -> nn.Linear:
        """
        Removes the classifier of a ViT backbone and replaces it with a new one.

        Args:
            num_classes (int): Number of output classes.

        Returns:
            classifier (nn.Linear): The new classifier layer.
        """

        # Replace the classifier
        num_features = self.backbone.heads[0].in_features
        self.backbone.heads[0] = nn.Identity()

        # Create a new classifier
        classifier = nn.Linear(num_features, num_classes)

        return classifier

    def _initialize_weights(self) -> None:
        """
        Initialize the weights of the classifier.
        """

        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)


class LateFusionImageClassifier(nn.Module):

    """
    Late fusion model for fall detection.

    Args:
        backbone (str): Name of the backbone model, currently only supports models from the ConvNeXt and SwinTransformer families.
        num_classes (int): Number of output classes.
        rgb_weights (str, optional): Path to the RGB model weights.
        depth_weights (str, optional): Path to the depth model weights.
    """

    def __init__(self, backbone: str, num_classes: int, rgb_weights: str = None, depth_weights: str = None) -> None:
        super(LateFusionImageClassifier, self).__init__()

        # Create the RGB and depth models
        self.rgb_model = ImageClassifier(backbone, num_classes, num_channels=3)
        self.depth_model = ImageClassifier(backbone, num_classes, num_channels=1)

        # Create the classifier
        self.classifier = nn.Linear(self.rgb_model.embed_dim + self.depth_model.embed_dim, num_classes)

        # If weights are provided, load them
        if rgb_weights is not None:
            self.rgb_model.load_state_dict(torch.load(rgb_weights, map_location="cpu"))

        if depth_weights is not None:
            self.depth_model.load_state_dict(torch.load(depth_weights, map_location="cpu"))

    def forward(self, images: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            images (Tensor): Input image with shape (batch_size, num_channels, height, width).

        Returns:
            logits (Tensor): Model predictions, shape (batch_size, num_classes).
        """

        # Separate RGB and depth images
        rgb_images = images[:, 0:3]
        depth_images = images[:, 3:4]

        # Extract features
        rgb_features = self.rgb_model.forward_features(rgb_images)
        depth_features = self.depth_model.forward_features(depth_images)
        features = torch.cat((rgb_features, depth_features), dim=1)

        # Pass through the classifier
        logits = self.classifier(features)

        return logits
    
    def freeze_backbones(self) -> None:
        """
        Freeze the RGB and depth models.
        """

        for param in self.rgb_model.parameters():
            param.requires_grad = False

        for param in self.depth_model.parameters():
            param.requires_grad = False

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)



        

