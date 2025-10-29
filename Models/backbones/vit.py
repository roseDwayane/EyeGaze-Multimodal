"""
Vision Transformer (ViT) backbone for image classification
Following Hugging Face model implementation standards
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor


class ViTClassifier(nn.Module):
    """
    Vision Transformer for image classification
    Supports concatenated dual-image inputs (player1 + player2)
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_labels: int = 3,
        image_size: int = 224,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize ViT classifier

        Args:
            model_name: Pretrained model name from Hugging Face
            num_labels: Number of classification classes
            image_size: Input image size
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone parameters
        """
        super().__init__()

        self.num_labels = num_labels
        self.image_size = image_size

        if pretrained:
            # Load pretrained model
            self.model = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
        else:
            # Create model from config
            config = ViTConfig.from_pretrained(model_name)
            config.num_labels = num_labels
            config.image_size = image_size
            self.model = ViTForImageClassification(config)

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.model.vit.parameters():
                param.requires_grad = False

    def forward(self, pixel_values, labels=None):
        """
        Forward pass

        Args:
            pixel_values: Input images [batch_size, 3, height, width]
            labels: Ground truth labels [batch_size]

        Returns:
            Dictionary containing loss and logits
        """
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        return outputs

    def get_image_processor(self):
        """Get the image processor for this model"""
        return ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224",
            size=self.image_size,
            do_resize=True,
            do_normalize=True
        )


class DualImageViTClassifier(nn.Module):
    """
    ViT classifier for dual-image inputs (player1 + player2)
    Concatenates two images horizontally or vertically before processing
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_labels: int = 3,
        image_size: int = 224,
        pretrained: bool = True,
        concat_mode: str = "horizontal",  # "horizontal" or "vertical"
        freeze_backbone: bool = False
    ):
        """
        Initialize Dual-Image ViT classifier

        Args:
            model_name: Pretrained model name from Hugging Face
            num_labels: Number of classification classes
            image_size: Input image size for each individual image
            pretrained: Whether to use pretrained weights
            concat_mode: How to concatenate images ("horizontal" or "vertical")
            freeze_backbone: Whether to freeze backbone parameters
        """
        super().__init__()

        self.num_labels = num_labels
        self.image_size = image_size
        self.concat_mode = concat_mode

        # Adjust image size based on concatenation mode
        if concat_mode == "horizontal":
            # Width doubles
            adjusted_size = (image_size, image_size * 2)
        elif concat_mode == "vertical":
            # Height doubles
            adjusted_size = (image_size * 2, image_size)
        else:
            raise ValueError(f"Invalid concat_mode: {concat_mode}")

        if pretrained:
            # Load pretrained model and adjust for concatenated input
            self.model = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
            # Note: May need to adjust patch embedding for different input size
        else:
            config = ViTConfig.from_pretrained(model_name)
            config.num_labels = num_labels
            config.image_size = adjusted_size
            self.model = ViTForImageClassification(config)

        if freeze_backbone:
            for param in self.model.vit.parameters():
                param.requires_grad = False

    def forward(self, pixel_values, labels=None):
        """
        Forward pass with dual images

        Args:
            pixel_values: Concatenated input images [batch_size, 3, height, width]
                         (already concatenated in dataset)
            labels: Ground truth labels [batch_size]

        Returns:
            Dictionary containing loss and logits
        """
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        return outputs


def create_vit_model(config: dict):
    """
    Factory function to create ViT model from config

    Args:
        config: Configuration dictionary

    Returns:
        ViT model instance
    """
    model_type = config.get("model_type", "single")

    common_params = {
        "model_name": config.get("model_name", "google/vit-base-patch16-224"),
        "num_labels": config.get("num_labels", 3),
        "image_size": config.get("image_size", 224),
        "pretrained": config.get("pretrained", True),
        "freeze_backbone": config.get("freeze_backbone", False)
    }

    if model_type == "dual":
        model = DualImageViTClassifier(
            **common_params,
            concat_mode=config.get("concat_mode", "horizontal")
        )
    else:
        model = ViTClassifier(**common_params)

    return model
