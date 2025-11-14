"""
Early Fusion Model: Eye Gaze Image + EEG Signal
Converts EEG to time-frequency representation and stacks with images
Processes combined input through a single unified model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from Models.backbones.vit import ViTClassifier


class EEGToTimeFrequency(nn.Module):
    """
    Convert EEG signal to time-frequency representation (spectrogram-like)
    Makes EEG compatible with image-based models
    """

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 128,
        n_mels: int = 64,
        target_size: int = 224,
        num_channels: int = 32
    ):
        """
        Args:
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel frequency bins
            target_size: Target image size (H, W)
            num_channels: Number of EEG channels
        """
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.target_size = target_size
        self.num_channels = num_channels

        # Learnable projection to reduce EEG channels to image-like channels
        # We'll create a "pseudo-RGB" representation from EEG
        self.channel_proj = nn.Conv1d(num_channels, 3, kernel_size=1)

    def compute_spectrogram(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Compute spectrogram using Short-Time Fourier Transform

        Args:
            eeg: (B, C, T) - EEG signal

        Returns:
            spec: (B, C, F, T') - Spectrogram
        """
        B, C, T = eeg.shape

        # Compute STFT for each channel
        spectrograms = []

        for c in range(C):
            eeg_channel = eeg[:, c, :]  # (B, T)

            # STFT
            spec = torch.stft(
                eeg_channel,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=torch.hann_window(self.n_fft, device=eeg.device),
                return_complex=True
            )  # (B, F, T')

            # Magnitude
            spec_mag = torch.abs(spec)  # (B, F, T')
            spectrograms.append(spec_mag)

        # Stack all channels
        spec_all = torch.stack(spectrograms, dim=1)  # (B, C, F, T')

        return spec_all

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Convert EEG to pseudo-image

        Args:
            eeg: (B, C, T) - EEG signal

        Returns:
            eeg_image: (B, 3, H, W) - Pseudo-image from EEG
        """
        B = eeg.shape[0]

        # Compute spectrogram
        spec = self.compute_spectrogram(eeg)  # (B, C, F, T')

        # Average across channels to get single spectrogram
        spec_avg = spec.mean(dim=1)  # (B, F, T')

        # Resize to target size
        spec_resized = F.interpolate(
            spec_avg.unsqueeze(1),  # (B, 1, F, T')
            size=(self.target_size, self.target_size),
            mode='bilinear',
            align_corners=False
        )  # (B, 1, target_size, target_size)

        # Expand to 3 channels (pseudo-RGB)
        eeg_image = spec_resized.repeat(1, 3, 1, 1)  # (B, 3, H, W)

        # Normalize
        eeg_image = (eeg_image - eeg_image.mean()) / (eeg_image.std() + 1e-8)

        return eeg_image


class EarlyFusionModel(nn.Module):
    """
    Early Fusion: Stack EEG spectrogram with Eye Gaze images
    Feed concatenated input to a single ViT

    Two strategies:
    1. Simple: Average P1 and P2 images, average P1 and P2 EEG → single input
    2. Concatenate: Stack all 4 modalities as multi-channel input
    """

    def __init__(
        self,
        # ViT config
        image_size: int = 224,
        patch_size: int = 16,
        vit_d_model: int = 768,
        vit_num_layers: int = 12,
        vit_num_heads: int = 12,
        num_classes: int = 3,

        # EEG to image conversion
        eeg_n_fft: int = 256,
        eeg_hop_length: int = 128,
        eeg_n_mels: int = 64,
        eeg_num_channels: int = 32,

        # Fusion strategy
        fusion_strategy: str = 'average',  # 'average' or 'concatenate'

        # Pre-trained
        image_model_path: Optional[str] = None,
        freeze_pretrained: bool = False,
    ):
        """
        Args:
            fusion_strategy:
                - 'average': Average P1+P2 images and P1+P2 EEG → 6-channel input (RGB + RGB)
                - 'concatenate': Stack all → 12-channel input (RGB + RGB + RGB + RGB)
        """
        super().__init__()

        self.fusion_strategy = fusion_strategy
        self.image_size = image_size
        self.num_classes = num_classes

        # EEG to time-frequency converter
        self.eeg_to_tf = EEGToTimeFrequency(
            n_fft=eeg_n_fft,
            hop_length=eeg_hop_length,
            n_mels=eeg_n_mels,
            target_size=image_size,
            num_channels=eeg_num_channels
        )

        # Determine input channels for ViT
        if fusion_strategy == 'average':
            # Average images: (img1+img2)/2 → 3 channels
            # Average EEG: (eeg1+eeg2)/2 → 3 channels (pseudo-RGB)
            # Stack: 3+3 = 6 channels
            in_channels = 6
        elif fusion_strategy == 'concatenate':
            # Stack all: img1(3) + img2(3) + eeg1(3) + eeg2(3) = 12 channels
            in_channels = 12
        else:
            raise ValueError(f"Unknown fusion_strategy: {fusion_strategy}")

        # Modified ViT with custom input channels
        self.vit = self._create_modified_vit(
            in_channels=in_channels,
            image_size=image_size,
            patch_size=patch_size,
            d_model=vit_d_model,
            num_layers=vit_num_layers,
            num_heads=vit_num_heads,
            num_classes=num_classes
        )

        # Load pre-trained weights if provided
        if image_model_path:
            self._load_pretrained(image_model_path, freeze_pretrained)

    def _create_modified_vit(
        self,
        in_channels: int,
        image_size: int,
        patch_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        num_classes: int
    ) -> nn.Module:
        """Create ViT with modified input channels

        Uses a channel adapter to convert N-channel input to 3-channel,
        then applies standard ViTClassifier
        """

        # Channel adapter: N channels -> 3 channels
        if in_channels != 3:
            channel_adapter = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, kernel_size=3, padding=1),
                nn.Tanh()  # Output in [-1, 1] like normalized images
            )
        else:
            channel_adapter = nn.Identity()

        # Standard ViT (3 channels)
        vit_classifier = ViTClassifier(
            model_name="google/vit-base-patch16-224",
            num_labels=num_classes,
            image_size=image_size,
            pretrained=True,
            freeze_backbone=False
        )

        # Wrap both in a sequential module
        class ViTWithAdapter(nn.Module):
            def __init__(self, adapter, vit):
                super().__init__()
                self.adapter = adapter
                self.vit = vit

            def forward(self, x, labels=None):
                x_adapted = self.adapter(x)
                return self.vit(pixel_values=x_adapted, labels=labels)

        return ViTWithAdapter(channel_adapter, vit_classifier)

    def _load_pretrained(self, checkpoint_path: str, freeze: bool):
        """Load pre-trained weights"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                # Load what we can (will skip patch_embed due to channel mismatch)
                self.vit.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"✓ Loaded pre-trained ViT (patch_embed re-initialized for {self.fusion_strategy} fusion)")

            if freeze:
                for name, param in self.vit.named_parameters():
                    if 'patch_embed' not in name:  # Don't freeze patch embedding
                        param.requires_grad = False
                print(f"✓ Froze pre-trained layers (except patch_embed)")
        except Exception as e:
            print(f"✗ Failed to load pre-trained model: {e}")

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
            dict with 'logits', 'loss'
        """
        B = img1.shape[0]

        # Convert EEG to pseudo-images
        eeg1_image = self.eeg_to_tf(eeg1)  # (B, 3, H, W)
        eeg2_image = self.eeg_to_tf(eeg2)  # (B, 3, H, W)

        # Fuse according to strategy
        if self.fusion_strategy == 'average':
            # Average P1 and P2
            avg_image = (img1 + img2) / 2  # (B, 3, H, W)
            avg_eeg_image = (eeg1_image + eeg2_image) / 2  # (B, 3, H, W)

            # Concatenate
            fused_input = torch.cat([avg_image, avg_eeg_image], dim=1)  # (B, 6, H, W)

        elif self.fusion_strategy == 'concatenate':
            # Stack all four
            fused_input = torch.cat([img1, img2, eeg1_image, eeg2_image], dim=1)  # (B, 12, H, W)

        # ViT forward
        vit_output = self.vit(fused_input, labels=labels)

        # Extract logits (HF models return objects with .logits attribute)
        if hasattr(vit_output, 'logits'):
            logits = vit_output.logits
        elif isinstance(vit_output, dict) and 'logits' in vit_output:
            logits = vit_output['logits']
        else:
            logits = vit_output

        # Output
        output = {'logits': logits}

        if labels is not None:
            if hasattr(vit_output, 'loss') and vit_output.loss is not None:
                output['loss'] = vit_output.loss
            else:
                loss = F.cross_entropy(logits, labels)
                output['loss'] = loss

        return output

    def get_num_params(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ========== Alternative: Channel-wise Early Fusion ==========

class ChannelWiseEarlyFusion(nn.Module):
    """
    Alternative early fusion: Use multi-channel convolution to mix modalities
    More flexible than simple stacking
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        vit_d_model: int = 768,
        vit_num_layers: int = 12,
        vit_num_heads: int = 12,
        num_classes: int = 3,
        eeg_num_channels: int = 32,
    ):
        super().__init__()

        self.image_size = image_size

        # EEG converter
        self.eeg_to_tf = EEGToTimeFrequency(
            target_size=image_size,
            num_channels=eeg_num_channels
        )

        # Channel-wise mixing: 12 channels → 3 channels
        # Learn optimal combination of modalities
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # Standard ViT
        self.vit = ViTClassifier(
            model_name="google/vit-base-patch16-224",
            num_labels=num_classes,
            image_size=image_size,
            pretrained=True,
            freeze_backbone=False
        )

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        eeg1: torch.Tensor,
        eeg2: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""

        # Convert EEG to images
        eeg1_image = self.eeg_to_tf(eeg1)
        eeg2_image = self.eeg_to_tf(eeg2)

        # Stack all modalities
        multi_modal = torch.cat([img1, img2, eeg1_image, eeg2_image], dim=1)  # (B, 12, H, W)

        # Mix channels
        mixed = self.channel_mixer(multi_modal)  # (B, 3, H, W)

        # ViT
        vit_output = self.vit(pixel_values=mixed, labels=labels)

        # Extract logits (HF models return objects with .logits attribute)
        if hasattr(vit_output, 'logits'):
            logits = vit_output.logits
        elif isinstance(vit_output, dict) and 'logits' in vit_output:
            logits = vit_output['logits']
        else:
            logits = vit_output

        output = {'logits': logits}

        if labels is not None:
            if hasattr(vit_output, 'loss') and vit_output.loss is not None:
                output['loss'] = vit_output.loss
            else:
                loss = F.cross_entropy(logits, labels)
                output['loss'] = loss

        return output

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
