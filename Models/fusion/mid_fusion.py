"""
Mid Fusion Model: Four-Tower Architecture with Cross-Modal Interaction
Eye Gaze Image + EEG Signal fusion at intermediate feature level

Architecture:
1. Four-Tower Encoders: P1-Img, P2-Img, P1-EEG, P2-EEG
2. Intra-Modality Fusion: Symmetric operators for each modality
3. IBS Token: Inter-Brain Synchrony features from EEG
4. Cross-Modal Interaction: Bidirectional attention between image and EEG
5. Classification Head: Final fusion and prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from Models.backbones.vit import ViTClassifier
from Models.backbones.dual_eeg_transformer import (
    DualEEGTransformer,
    TemporalConvFrontend,
    IBSTokenGenerator
)
from Models.fusion.symmetric_fusion import SymmetricFusionOperators
from Models.fusion.cross_modal_attention import CrossModalAttention


class MidFusionModel(nn.Module):
    """
    Mid-level multimodal fusion with four-tower architecture

    Four independent encoders:
    - ViT Encoder 1 (for Player 1 image)
    - ViT Encoder 2 (for Player 2 image)
    - EEG Encoder 1 (for Player 1 EEG)
    - EEG Encoder 2 (for Player 2 EEG)

    Fusion strategy:
    1. Extract features from each tower
    2. Fuse within each modality (image-to-image, eeg-to-eeg) using symmetric operators
    3. Generate IBS token from raw EEG signals
    4. Cross-modal attention between image and EEG features
    5. Concatenate all features and classify
    """

    def __init__(
        self,
        # Image encoder config
        image_size: int = 224,
        patch_size: int = 16,
        image_d_model: int = 768,
        image_num_layers: int = 12,
        image_num_heads: int = 12,
        image_shared_weights: bool = True,  # Share weights between P1 and P2 image encoders

        # EEG encoder config
        eeg_in_channels: int = 32,
        eeg_d_model: int = 256,
        eeg_num_layers: int = 6,
        eeg_num_heads: int = 8,
        eeg_d_ff: int = 1024,
        eeg_window_size: int = 1024,
        eeg_conv_kernel: int = 25,
        eeg_conv_stride: int = 4,
        eeg_conv_layers: int = 2,
        eeg_shared_weights: bool = True,  # Share weights between P1 and P2 EEG encoders

        # Fusion config
        num_classes: int = 3,
        fusion_mode: str = 'basic',  # 'basic', 'all', 'simple'
        use_ibs_token: bool = True,
        use_cross_attention: bool = True,
        cross_attn_num_heads: int = 8,
        fusion_dropout: float = 0.3,
        fusion_hidden_dim: int = 512,

        # Pre-trained models
        image_model_path: Optional[str] = None,
        eeg_model_path: Optional[str] = None,
        freeze_image: bool = False,
        freeze_eeg: bool = False,
    ):
        super().__init__()

        self.image_d_model = image_d_model
        self.eeg_d_model = eeg_d_model
        self.image_shared_weights = image_shared_weights
        self.eeg_shared_weights = eeg_shared_weights
        self.use_ibs_token = use_ibs_token
        self.use_cross_attention = use_cross_attention
        self.num_classes = num_classes

        # ========== Image Encoders ==========
        if image_shared_weights:
            # Single shared ViT for both players
            self.image_encoder = ViTClassifier(
                model_name="google/vit-base-patch16-224",
                num_labels=num_classes,
                image_size=image_size,
                pretrained=True,
                freeze_backbone=freeze_image
            )

            if image_model_path:
                self._load_pretrained(self.image_encoder, image_model_path, 'image')

            self.image_encoder_p1 = self.image_encoder
            self.image_encoder_p2 = self.image_encoder
        else:
            # Separate encoders for each player
            self.image_encoder_p1 = ViTClassifier(
                model_name="google/vit-base-patch16-224",
                num_labels=num_classes,
                image_size=image_size,
                pretrained=True,
                freeze_backbone=freeze_image
            )
            self.image_encoder_p2 = ViTClassifier(
                model_name="google/vit-base-patch16-224",
                num_labels=num_classes,
                image_size=image_size,
                pretrained=True,
                freeze_backbone=freeze_image
            )

            if image_model_path:
                self._load_pretrained(self.image_encoder_p1, image_model_path, 'image_p1')
                self._load_pretrained(self.image_encoder_p2, image_model_path, 'image_p2')

        # ========== EEG Encoders ==========
        if eeg_shared_weights:
            # Single shared temporal conv + transformer for both players
            self.eeg_temporal_conv = TemporalConvFrontend(
                eeg_in_channels, eeg_d_model, eeg_conv_kernel, eeg_conv_stride, eeg_conv_layers
            )
            # For simplicity, use lightweight encoder instead of full DualEEGTransformer
            # Import transformer encoder from art.py
            from Models.backbones.art import TransformerEncoder

            self.eeg_transformer = TransformerEncoder(
                eeg_d_model, eeg_num_layers, eeg_num_heads, eeg_d_ff, 0.1, 0.1
            )

            if freeze_eeg:
                for param in self.eeg_temporal_conv.parameters():
                    param.requires_grad = False
                for param in self.eeg_transformer.parameters():
                    param.requires_grad = False

            self.eeg_encoder_p1 = (self.eeg_temporal_conv, self.eeg_transformer)
            self.eeg_encoder_p2 = (self.eeg_temporal_conv, self.eeg_transformer)
        else:
            # Separate encoders
            self.eeg_temporal_conv_p1 = TemporalConvFrontend(
                eeg_in_channels, eeg_d_model, eeg_conv_kernel, eeg_conv_stride, eeg_conv_layers
            )
            self.eeg_temporal_conv_p2 = TemporalConvFrontend(
                eeg_in_channels, eeg_d_model, eeg_conv_kernel, eeg_conv_stride, eeg_conv_layers
            )

            from Models.backbones.art import TransformerEncoder

            self.eeg_transformer_p1 = TransformerEncoder(
                eeg_d_model, eeg_num_layers, eeg_num_heads, eeg_d_ff, 0.1, 0.1
            )
            self.eeg_transformer_p2 = TransformerEncoder(
                eeg_d_model, eeg_num_layers, eeg_num_heads, eeg_d_ff, 0.1, 0.1
            )

            self.eeg_encoder_p1 = (self.eeg_temporal_conv_p1, self.eeg_transformer_p1)
            self.eeg_encoder_p2 = (self.eeg_temporal_conv_p2, self.eeg_transformer_p2)

        # ========== IBS Token Generator ==========
        if use_ibs_token:
            self.ibs_generator = IBSTokenGenerator(
                eeg_in_channels, eeg_d_model, num_freq_bands=4
            )

        # ========== Intra-Modality Fusion ==========
        # Symmetric fusion within each modality
        self.image_fusion = SymmetricFusionOperators(
            image_d_model, image_d_model, mode=fusion_mode
        )
        self.eeg_fusion = SymmetricFusionOperators(
            eeg_d_model, eeg_d_model, mode=fusion_mode
        )

        # ========== Cross-Modal Attention ==========
        if use_cross_attention:
            self.cross_modal_attn = CrossModalAttention(
                d_model_a=image_d_model,
                d_model_b=eeg_d_model,
                num_heads=cross_attn_num_heads,
                dropout=fusion_dropout,
                use_projection=True
            )

        # ========== Classification Head ==========
        # Input: [z_img_fused, z_eeg_fused, ibs_token (optional)]
        if use_ibs_token:
            classifier_input_dim = image_d_model + eeg_d_model * 2  # eeg_d_model*2 for eeg_fused + ibs
        else:
            classifier_input_dim = image_d_model + eeg_d_model

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.LayerNorm(fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim // 2, num_classes)
        )

    def _load_pretrained(self, model: nn.Module, checkpoint_path: str, model_name: str):
        """Load pre-trained weights"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"✓ Loaded pre-trained {model_name} from {checkpoint_path}")
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")

    def _extract_image_features(self, img: torch.Tensor, encoder) -> torch.Tensor:
        """Extract CLS features from ViT"""
        # Use the get_cls_features method from ViTClassifier
        cls_features = encoder.get_cls_features(pixel_values=img)  # (B, d_model)
        return cls_features

    def _extract_eeg_features(self, eeg: torch.Tensor, encoder_tuple: Tuple) -> torch.Tensor:
        """Extract features from EEG using temporal conv + transformer"""
        temporal_conv, transformer = encoder_tuple

        # Temporal convolution
        h = temporal_conv(eeg)  # (B, C, T) → (B, T̃, d)

        # Add CLS token
        B, T_tilde, d = h.shape
        cls = torch.zeros(B, 1, d, device=h.device)
        h_with_cls = torch.cat([cls, h], dim=1)  # (B, T̃+1, d)

        # Transformer
        z = transformer(h_with_cls)  # (B, T̃+1, d)

        # Extract CLS
        cls_features = z[:, 0]  # (B, d)

        return cls_features

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
            dict with 'logits', 'loss', etc.
        """
        B = img1.shape[0]

        # ========== Extract Features from Four Towers ==========
        # Image features
        z_img_p1 = self._extract_image_features(img1, self.image_encoder_p1)  # (B, d_img)
        z_img_p2 = self._extract_image_features(img2, self.image_encoder_p2)  # (B, d_img)

        # EEG features
        z_eeg_p1 = self._extract_eeg_features(eeg1, self.eeg_encoder_p1)  # (B, d_eeg)
        z_eeg_p2 = self._extract_eeg_features(eeg2, self.eeg_encoder_p2)  # (B, d_eeg)

        # ========== Intra-Modality Fusion ==========
        # Fuse image features (symmetric)
        z_img_fused = self.image_fusion(z_img_p1, z_img_p2)  # (B, d_img)

        # Fuse EEG features (symmetric)
        z_eeg_fused = self.eeg_fusion(z_eeg_p1, z_eeg_p2)  # (B, d_eeg)

        # ========== IBS Token ==========
        if self.use_ibs_token:
            ibs_token = self.ibs_generator(eeg1, eeg2)  # (B, d_eeg)
        else:
            ibs_token = None

        # ========== Cross-Modal Interaction ==========
        if self.use_cross_attention:
            # Bidirectional cross-attention
            z_img_cross, z_eeg_cross = self.cross_modal_attn(z_img_fused, z_eeg_fused)

            # Use cross-attended features
            z_img_final = z_img_cross
            z_eeg_final = z_eeg_cross
        else:
            # No cross-attention
            z_img_final = z_img_fused
            z_eeg_final = z_eeg_fused

        # ========== Concatenate All Features ==========
        if self.use_ibs_token:
            z_concat = torch.cat([z_img_final, z_eeg_final, ibs_token], dim=-1)  # (B, d_img + 2*d_eeg)
        else:
            z_concat = torch.cat([z_img_final, z_eeg_final], dim=-1)  # (B, d_img + d_eeg)

        # ========== Classification ==========
        logits = self.classifier(z_concat)  # (B, num_classes)

        # ========== Output ==========
        output = {
            'logits': logits,
            'z_img_fused': z_img_fused,
            'z_eeg_fused': z_eeg_fused,
            'ibs_token': ibs_token
        }

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            output['loss'] = loss

        return output

    def get_num_params(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ========== Simplified Mid Fusion (for ablation) ==========

class SimplifiedMidFusion(nn.Module):
    """
    Simplified version of Mid Fusion for ablation studies
    - No IBS token
    - No cross-attention
    - Only basic symmetric fusion
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        image_d_model: int = 768,
        image_num_layers: int = 12,
        image_num_heads: int = 12,
        eeg_in_channels: int = 32,
        eeg_d_model: int = 256,
        eeg_num_layers: int = 6,
        eeg_num_heads: int = 8,
        eeg_d_ff: int = 1024,
        eeg_window_size: int = 1024,
        eeg_conv_kernel: int = 25,
        eeg_conv_stride: int = 4,
        eeg_conv_layers: int = 2,
        num_classes: int = 3,
        fusion_dropout: float = 0.3,
    ):
        super().__init__()

        # Use full Mid Fusion but disable components
        self.model = MidFusionModel(
            image_size=image_size,
            patch_size=patch_size,
            image_d_model=image_d_model,
            image_num_layers=image_num_layers,
            image_num_heads=image_num_heads,
            eeg_in_channels=eeg_in_channels,
            eeg_d_model=eeg_d_model,
            eeg_num_layers=eeg_num_layers,
            eeg_num_heads=eeg_num_heads,
            eeg_d_ff=eeg_d_ff,
            eeg_window_size=eeg_window_size,
            eeg_conv_kernel=eeg_conv_kernel,
            eeg_conv_stride=eeg_conv_stride,
            eeg_conv_layers=eeg_conv_layers,
            num_classes=num_classes,
            fusion_mode='basic',
            use_ibs_token=False,  # Disable IBS token
            use_cross_attention=False,  # Disable cross-attention
            fusion_dropout=fusion_dropout
        )

    def forward(self, img1, img2, eeg1, eeg2, labels=None):
        return self.model(img1, img2, eeg1, eeg2, labels)

    def get_num_params(self):
        return self.model.get_num_params()
