"""
Late Fusion Model: Eye Gaze Image + EEG Signal
Combines predictions from two pre-trained modality-specific models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from Models.backbones.vit import ViTClassifier
from Models.backbones.dual_eeg_transformer import DualEEGTransformer


class LateFusionModel(nn.Module):
    """
    Late Fusion for multimodal classification

    Strategy:
    1. Load pre-trained Image and EEG models
    2. Extract features or logits from each modality
    3. Fuse with weighted average or MLP

    Two fusion modes:
    - 'logits': Weighted average of output logits
    - 'features': Concatenate last hidden features → MLP
    """

    def __init__(
        self,
        # Image model config
        image_model_path: Optional[str] = None,
        image_size: int = 224,
        patch_size: int = 16,
        image_d_model: int = 768,
        image_num_layers: int = 12,
        image_num_heads: int = 12,

        # EEG model config
        eeg_model_path: Optional[str] = None,
        eeg_in_channels: int = 32,
        eeg_d_model: int = 256,
        eeg_num_layers: int = 6,
        eeg_num_heads: int = 8,
        eeg_d_ff: int = 1024,
        eeg_window_size: int = 1024,
        eeg_conv_kernel: int = 25,
        eeg_conv_stride: int = 4,
        eeg_conv_layers: int = 2,

        # Fusion config
        num_classes: int = 3,
        fusion_mode: str = 'features',  # 'logits' or 'features'
        fusion_dropout: float = 0.3,
        fusion_hidden_dim: int = 512,

        # Control
        freeze_image: bool = False,
        freeze_eeg: bool = False,
        image_weight: float = 0.5,
        eeg_weight: float = 0.5,
    ):
        super().__init__()

        self.fusion_mode = fusion_mode
        self.num_classes = num_classes
        self.image_weight = image_weight
        self.eeg_weight = eeg_weight

        # Initialize Image Model (ViT)
        self.image_model = ViTClassifier(
            model_name="google/vit-base-patch16-224",
            num_labels=num_classes,
            image_size=image_size,
            pretrained=True,
            freeze_backbone=False
        )

        # Load pre-trained image model if provided
        if image_model_path:
            self._load_pretrained_model(self.image_model, image_model_path, 'image')

        if freeze_image:
            for param in self.image_model.parameters():
                param.requires_grad = False

        # Initialize EEG Model (Dual EEG Transformer)
        self.eeg_model = DualEEGTransformer(
            in_channels=eeg_in_channels,
            num_classes=num_classes,
            d_model=eeg_d_model,
            num_layers=eeg_num_layers,
            num_heads=eeg_num_heads,
            d_ff=eeg_d_ff,
            dropout=0.1,
            max_len=eeg_window_size // (eeg_conv_stride ** eeg_conv_layers) + 2,
            conv_kernel_size=eeg_conv_kernel,
            conv_stride=eeg_conv_stride,
            conv_layers=eeg_conv_layers
        )

        # Load pre-trained EEG model if provided
        if eeg_model_path:
            self._load_pretrained_model(self.eeg_model, eeg_model_path, 'eeg')

        if freeze_eeg:
            for param in self.eeg_model.parameters():
                param.requires_grad = False

        # Fusion layers
        if fusion_mode == 'features':
            # Feature fusion: Concat last hidden states
            # ViT CLS (d_img) + EEG CLS averaged (d_eeg) + IBS token (d_eeg)
            fusion_input_dim = image_d_model + eeg_d_model * 2

            self.fusion_head = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_hidden_dim),
                nn.LayerNorm(fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
                nn.LayerNorm(fusion_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(fusion_hidden_dim // 2, num_classes)
            )
        elif fusion_mode == 'logits':
            # Logits fusion: Weighted average (no additional parameters)
            self.fusion_head = None
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

    def _load_pretrained_model(self, model: nn.Module, checkpoint_path: str, model_name: str):
        """Load pre-trained weights"""
        import os
        try:
            # If path is a directory, look for best_model.pt
            if os.path.isdir(checkpoint_path):
                checkpoint_file = os.path.join(checkpoint_path, 'best_model.pt')
                if not os.path.exists(checkpoint_file):
                    print(f"⚠ No checkpoint found in {checkpoint_path}, using random initialization")
                    return
            else:
                checkpoint_file = checkpoint_path

            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"✓ Loaded pre-trained {model_name} model from {checkpoint_file}")
        except Exception as e:
            print(f"✗ Failed to load {model_name} model: {e}")

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        eeg1: torch.Tensor,
        eeg2: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            img1, img2: (B, 3, H, W) - Eye gaze images
            eeg1, eeg2: (B, C, T) - EEG signals
            labels: (B,) - Ground truth labels

        Returns:
            dict with 'logits', 'loss' (if labels provided), 'logits_img', 'logits_eeg'
        """
        B = img1.shape[0]

        # ========== Image Branch ==========
        # Process player 1 and 2 images separately
        img_outputs_p1 = self.image_model(pixel_values=img1)
        img_outputs_p2 = self.image_model(pixel_values=img2)

        # Extract logits from HF outputs
        if hasattr(img_outputs_p1, 'logits'):
            logits_img_p1 = img_outputs_p1.logits
            logits_img_p2 = img_outputs_p2.logits
        elif isinstance(img_outputs_p1, dict):
            logits_img_p1 = img_outputs_p1['logits']
            logits_img_p2 = img_outputs_p2['logits']
        else:
            logits_img_p1 = img_outputs_p1
            logits_img_p2 = img_outputs_p2

        # Average logits from two players (symmetric)
        logits_img = (logits_img_p1 + logits_img_p2) / 2  # (B, num_classes)

        # Extract CLS features if needed
        if self.fusion_mode == 'features':
            # Get CLS token from ViT
            if hasattr(self.image_model, 'get_cls_features'):
                cls_img_p1 = self.image_model.get_cls_features(pixel_values=img1)
                cls_img_p2 = self.image_model.get_cls_features(pixel_values=img2)
                cls_img = (cls_img_p1 + cls_img_p2) / 2  # (B, d_img=768)
            else:
                # Fallback: use logits as features (not ideal but works)
                cls_img = logits_img  # (B, num_classes)

        # ========== EEG Branch ==========
        eeg_outputs = self.eeg_model(eeg1, eeg2, labels=None)

        logits_eeg = eeg_outputs['logits']  # (B, num_classes)

        # Extract EEG features if needed
        if self.fusion_mode == 'features':
            # Get intermediate features from EEG model
            # EEG model returns cls1, cls2, ibs_token
            cls1_eeg = eeg_outputs.get('cls1', None)  # (B, d_eeg)
            cls2_eeg = eeg_outputs.get('cls2', None)  # (B, d_eeg)
            ibs_token = eeg_outputs.get('ibs_token', None)  # (B, d_eeg)

            if cls1_eeg is not None:
                # Symmetric fusion of EEG CLS tokens
                cls_eeg = (cls1_eeg + cls2_eeg) / 2  # (B, d_eeg)
            else:
                # Fallback
                cls_eeg = logits_eeg  # (B, num_classes)

        # ========== Fusion ==========
        if self.fusion_mode == 'logits':
            # Weighted average of logits
            logits_fused = self.image_weight * logits_img + self.eeg_weight * logits_eeg

        elif self.fusion_mode == 'features':
            # Concatenate features and pass through MLP
            if cls1_eeg is not None and ibs_token is not None:
                # Full feature fusion
                features = torch.cat([cls_img, cls_eeg, ibs_token], dim=-1)  # (B, d_img + 2*d_eeg)
            else:
                features = torch.cat([cls_img, cls_eeg], dim=-1)

            logits_fused = self.fusion_head(features)  # (B, num_classes)

        # ========== Loss ==========
        output = {
            'logits': logits_fused,
            'logits_img': logits_img,
            'logits_eeg': logits_eeg
        }

        if labels is not None:
            loss_fused = F.cross_entropy(logits_fused, labels)
            loss_img = F.cross_entropy(logits_img, labels)
            loss_eeg = F.cross_entropy(logits_eeg, labels)

            # Total loss with auxiliary losses
            loss = loss_fused + 0.3 * loss_img + 0.3 * loss_eeg

            output['loss'] = loss
            output['loss_fused'] = loss_fused
            output['loss_img'] = loss_img
            output['loss_eeg'] = loss_eeg

        return output

    def get_num_params(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ========== Modified ViT to expose CLS features ==========
# We need to modify the ViT model to return CLS token
# For now, add this functionality