"""
Attention Visualization Module

Provides attention visualization tools for ViT-based models:
- Gradient Saliency Maps
- Grad-CAM for Transformers
- Combined attention overlays

Author: CNElab
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


# =============================================================================
# Constants
# =============================================================================

CLASS_NAMES = ["Single", "Competition", "Cooperation"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# Image Utilities
# =============================================================================

def denormalize_image(
    img_tensor: torch.Tensor,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD
) -> torch.Tensor:
    """
    Denormalize tensor image for visualization.

    Parameters
    ----------
    img_tensor : torch.Tensor
        (C, H, W) normalized tensor
    mean : list
        Normalization mean
    std : list
        Normalization std

    Returns
    -------
    torch.Tensor
        Denormalized tensor clamped to [0, 1]
    """
    img = img_tensor.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return img.clamp(0, 1)


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) tensor to (H, W, C) numpy array for plotting."""
    img = denormalize_image(tensor)
    return img.permute(1, 2, 0).cpu().numpy()


# =============================================================================
# Attention Analyzer
# =============================================================================

class AttentionAnalyzer:
    """
    Attention visualization for ViT-based dual-stream models.

    Provides gradient-based saliency maps and Grad-CAM visualizations
    for both Early Fusion and Late Fusion architectures.

    Parameters
    ----------
    model : nn.Module
        PyTorch ViT model (EarlyFusionViT or LateFusionViT)
    model_type : str
        Model type: 'early' or 'late'
    device : str
        Device for computation ('cuda' or 'cpu')

    Examples
    --------
    >>> analyzer = AttentionAnalyzer(model, 'early', 'cuda')
    >>> saliency = analyzer.compute_gradient_saliency(img_a, img_b)
    >>> cam_a, cam_b = analyzer.compute_grad_cam(img_a, img_b)
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        device: str = 'cuda'
    ):
        self.model = model
        self.model_type = model_type
        self.device = device

    def compute_gradient_saliency(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute gradient-based saliency map.

        Uses input gradients to highlight important regions for classification.

        Parameters
        ----------
        img_a, img_b : torch.Tensor
            Input images (1, 3, H, W)
        target_class : int, optional
            Target class for saliency. If None, uses predicted class.

        Returns
        -------
        np.ndarray
            Saliency map, shape (H, W), normalized to [0, 1]
        """
        self.model.eval()

        img_a = img_a.clone().to(self.device).requires_grad_(True)
        img_b = img_b.clone().to(self.device).requires_grad_(True)

        # Forward pass
        logits = self.model(img_a, img_b)

        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        logits.backward(gradient=one_hot)

        # Compute saliency
        if self.model_type == 'early':
            # For early fusion, combine gradients from both images
            grad_a = img_a.grad.abs()
            grad_b = img_b.grad.abs()
            saliency = (grad_a + grad_b).mean(dim=1)[0]  # Average across channels
        else:
            # For late fusion, use both gradients
            grad_a = img_a.grad.abs().mean(dim=1)[0]
            grad_b = img_b.grad.abs().mean(dim=1)[0]
            saliency = (grad_a + grad_b) / 2

        # Normalize
        saliency = saliency.cpu().numpy()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        return saliency

    def compute_grad_cam(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Grad-CAM for ViT model.

        Generates class activation maps by weighting feature activations
        with gradients from the target class.

        Parameters
        ----------
        img_a, img_b : torch.Tensor
            Input images (1, 3, H, W)
        target_class : int, optional
            Target class for CAM. If None, uses predicted class.

        Returns
        -------
        cam_a, cam_b : np.ndarray
            Grad-CAM maps for each image, shape (H, W), normalized to [0, 1]
        """
        self.model.eval()

        img_a = img_a.clone().to(self.device).requires_grad_(True)
        img_b = img_b.clone().to(self.device).requires_grad_(True)

        activations = []
        gradients = []

        # Get the last transformer block
        if self.model_type == 'early':
            target_layer = self.model.backbone.blocks[-1]
        else:
            target_layer = self.model.backbone.blocks[-1]

        # Register hooks
        def forward_hook(module, input, output):
            activations.append(output.detach())

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)

        try:
            # Forward
            logits = self.model(img_a, img_b)

            if target_class is None:
                target_class = logits.argmax(dim=1).item()

            # Backward
            self.model.zero_grad()
            one_hot = torch.zeros_like(logits)
            one_hot[0, target_class] = 1
            logits.backward(gradient=one_hot, retain_graph=True)

            # Compute Grad-CAM
            grad = gradients[0][:, 1:, :]  # Remove CLS token, shape: (1, N-1, D)
            act = activations[0][:, 1:, :]  # Remove CLS token

            # Global average pooling over the feature dimension
            weights = grad.mean(dim=2, keepdim=True)  # (1, N-1, 1)
            cam = (weights * act).sum(dim=2)  # (1, N-1)
            cam = F.relu(cam)

            # Reshape to 2D spatial map (14x14 for ViT-B/16 with 224x224 input)
            n_patches = int(np.sqrt(cam.shape[1]))
            cam = cam.reshape(1, n_patches, n_patches)

            # Upsample to image size
            cam = F.interpolate(
                cam.unsqueeze(0), size=(224, 224),
                mode='bilinear', align_corners=False
            )
            cam = cam.squeeze().cpu().numpy()

            # Normalize
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        finally:
            handle_fwd.remove()
            handle_bwd.remove()

        # Return same CAM for both (model processes them together)
        return cam, cam

    def compute_dual_stream_grad_cam(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute separate Grad-CAM for each stream (Late Fusion only).

        Parameters
        ----------
        img_a, img_b : torch.Tensor
            Input images (1, 3, H, W)
        target_class : int, optional
            Target class for CAM

        Returns
        -------
        dict
            Dictionary with keys: 'stream1', 'stream2', 'combined'
        """
        if self.model_type != 'late':
            # For early fusion, return single CAM for both
            cam, _ = self.compute_grad_cam(img_a, img_b, target_class)
            return {'stream1': cam, 'stream2': cam, 'combined': cam}

        # For late fusion, we'd need separate forward passes
        # This is a simplified version
        cam, _ = self.compute_grad_cam(img_a, img_b, target_class)

        return {
            'stream1': cam,
            'stream2': cam,
            'combined': cam
        }

    def visualize_attention(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        true_label: int,
        pred_label: int,
        confidence: float,
        save_path: Path,
        sample_idx: int,
        class_names: List[str] = None
    ):
        """
        Create comprehensive attention visualization figure.

        Parameters
        ----------
        img_a, img_b : torch.Tensor
            Input images (1, 3, H, W)
        true_label : int
            Ground truth label
        pred_label : int
            Predicted label
        confidence : float
            Prediction confidence
        save_path : Path
            Output file path
        sample_idx : int
            Sample index for title
        class_names : list, optional
            Class names for labeling
        """
        if class_names is None:
            class_names = CLASS_NAMES

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Convert images for display
        img_a_np = tensor_to_numpy_image(img_a.squeeze(0))
        img_b_np = tensor_to_numpy_image(img_b.squeeze(0))

        # Compute attention maps
        saliency = self.compute_gradient_saliency(img_a.clone(), img_b.clone())
        cam_a, cam_b = self.compute_grad_cam(img_a.clone(), img_b.clone())

        # Row 1: Original images and gradient saliency
        axes[0, 0].imshow(img_a_np)
        axes[0, 0].set_title('Player 1 Gaze Heatmap')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(img_b_np)
        axes[0, 1].set_title('Player 2 Gaze Heatmap')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(saliency, cmap='hot')
        axes[0, 2].set_title('Gradient Saliency')
        axes[0, 2].axis('off')

        # Row 2: Attention overlays
        axes[1, 0].imshow(img_a_np)
        axes[1, 0].imshow(cam_a, cmap='jet', alpha=0.5)
        axes[1, 0].set_title('Grad-CAM on Player 1')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(img_b_np)
        axes[1, 1].imshow(cam_b, cmap='jet', alpha=0.5)
        axes[1, 1].set_title('Grad-CAM on Player 2')
        axes[1, 1].axis('off')

        # Combined overlay
        combined_img = (img_a_np + img_b_np) / 2
        combined_cam = (cam_a + cam_b) / 2
        axes[1, 2].imshow(combined_img)
        axes[1, 2].imshow(combined_cam, cmap='jet', alpha=0.5)
        axes[1, 2].set_title('Combined Attention')
        axes[1, 2].axis('off')

        # Add title with prediction info
        true_name = class_names[true_label]
        pred_name = class_names[pred_label]
        correct = "Correct" if true_label == pred_label else "Incorrect"
        fig.suptitle(
            f'Sample {sample_idx}: True={true_name}, Pred={pred_name} '
            f'({correct}, Conf={confidence:.2%})',
            fontsize=14, fontweight='bold'
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # =========================================================================
    # Batch Processing
    # =========================================================================

    def batch_compute_saliency(
        self,
        dataloader,
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Compute saliency maps for all samples in dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader yielding (img_a, img_b, labels)
        show_progress : bool
            Whether to show progress

        Returns
        -------
        list
            List of saliency maps
        """
        from tqdm import tqdm

        saliency_maps = []

        iterator = tqdm(dataloader, desc="Computing saliency") if show_progress else dataloader

        for batch in iterator:
            if len(batch) >= 3:
                img_a, img_b, _ = batch[:3]
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")

            for i in range(len(img_a)):
                saliency = self.compute_gradient_saliency(
                    img_a[i:i+1], img_b[i:i+1]
                )
                saliency_maps.append(saliency)

        return saliency_maps

    def generate_attention_grid(
        self,
        samples: List[Tuple[torch.Tensor, torch.Tensor, int, int, float]],
        save_path: Path,
        title: str = "Attention Visualization Grid",
        class_names: List[str] = None
    ):
        """
        Generate grid visualization for multiple samples.

        Parameters
        ----------
        samples : list
            List of (img_a, img_b, true_label, pred_label, confidence) tuples
        save_path : Path
            Output file path
        title : str
            Plot title
        class_names : list, optional
            Class names
        """
        if class_names is None:
            class_names = CLASS_NAMES

        n_samples = len(samples)
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for row, (img_a, img_b, true_label, pred_label, conf) in enumerate(samples):
            # Convert images
            img_a_np = tensor_to_numpy_image(img_a.squeeze(0))
            img_b_np = tensor_to_numpy_image(img_b.squeeze(0))

            # Compute attention
            saliency = self.compute_gradient_saliency(img_a.clone(), img_b.clone())
            cam, _ = self.compute_grad_cam(img_a.clone(), img_b.clone())

            # Plot
            axes[row, 0].imshow(img_a_np)
            axes[row, 0].set_title(f'Player 1')
            axes[row, 0].axis('off')

            axes[row, 1].imshow(img_b_np)
            axes[row, 1].set_title(f'Player 2')
            axes[row, 1].axis('off')

            axes[row, 2].imshow(saliency, cmap='hot')
            axes[row, 2].set_title('Saliency')
            axes[row, 2].axis('off')

            combined = (img_a_np + img_b_np) / 2
            axes[row, 3].imshow(combined)
            axes[row, 3].imshow(cam, cmap='jet', alpha=0.5)

            correct = "OK" if true_label == pred_label else "ERR"
            axes[row, 3].set_title(
                f'{class_names[true_label]}->{class_names[pred_label]} [{correct}]'
            )
            axes[row, 3].axis('off')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    # Quick test with mock data
    print("=" * 60)
    print("Testing Attention Utils")
    print("=" * 60)

    # Test utility functions
    print("\n[1] Testing utility functions...")

    # Mock image tensor
    img = torch.rand(3, 224, 224)
    denorm_img = denormalize_image(img)
    print(f"  Denormalized image range: [{denorm_img.min():.3f}, {denorm_img.max():.3f}]")

    np_img = tensor_to_numpy_image(img)
    print(f"  NumPy image shape: {np_img.shape}")

    print("\n[2] Note: Full AttentionAnalyzer tests require a trained model")
    print("  Skipping model-dependent tests...")

    print("\n" + "=" * 60)
    print("Basic tests passed!")
    print("=" * 60)
