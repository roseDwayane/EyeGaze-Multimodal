"""
EEG Metrics Calculation Core

This module provides functions for computing metrics and extracting features
from trained EEG models for analysis.

Features:
- Confusion matrix and classification metrics
- IBS connectivity matrix extraction
- Frequency band sensitivity analysis
- Cross-attention weight extraction
- Feature embedding extraction (for t-SNE/UMAP)
- Grad-CAM computation

Author: Analysis Pipeline for EyeGaze-Multimodal Project
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from tqdm import tqdm

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)


# =============================================================================
# Core Classification Metrics
# =============================================================================

def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        class_names: List of class names (for label ordering)

    Returns:
        Confusion matrix (num_classes x num_classes)
    """
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        Dictionary with:
            - accuracy: Overall accuracy
            - macro_precision, macro_recall, macro_f1
            - per_class: Dict[class_name, Dict[metric, value]]
            - confusion_matrix: np.ndarray
    """
    accuracy = accuracy_score(y_true, y_pred)

    # Macro-averaged metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    # Per-class metrics
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            'precision': precision_per[i],
            'recall': recall_per[i],
            'f1': f1_per[i],
            'support': int(support_per[i])
        }

    cm = compute_confusion_matrix(y_true, y_pred, class_names)

    return {
        'accuracy': accuracy,
        'macro_precision': precision_macro,
        'macro_recall': recall_macro,
        'macro_f1': f1_macro,
        'per_class': per_class,
        'confusion_matrix': cm
    }


# =============================================================================
# Model Inference
# =============================================================================

@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, np.ndarray]:
    """
    Run inference on the entire dataset.

    Args:
        model: Trained model
        dataloader: DataLoader for test set
        device: Device to run inference on

    Returns:
        Dictionary with:
            - y_true: True labels (N,)
            - y_pred: Predicted labels (N,)
            - y_prob: Prediction probabilities (N, num_classes)
            - cls1: CLS token from brain 1 (N, d_model)
            - cls2: CLS token from brain 2 (N, d_model)
            - ibs_token: IBS token (N, d_model)
    """
    model.eval()

    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    all_cls1 = []
    all_cls2 = []
    all_ibs_token = []

    for batch in tqdm(dataloader, desc='Running inference'):
        eeg1, eeg2, labels = batch[0], batch[1], batch[2]
        eeg1 = eeg1.to(device)
        eeg2 = eeg2.to(device)

        outputs = model(eeg1, eeg2, labels=None)
        logits = outputs['logits']
        probs = F.softmax(logits, dim=-1)

        all_y_true.extend(labels.numpy())
        all_y_pred.extend(logits.argmax(dim=1).cpu().numpy())
        all_y_prob.append(probs.cpu().numpy())
        all_cls1.append(outputs['cls1'].cpu().numpy())
        all_cls2.append(outputs['cls2'].cpu().numpy())

        if outputs.get('ibs_token') is not None:
            all_ibs_token.append(outputs['ibs_token'].cpu().numpy())

    result = {
        'y_true': np.array(all_y_true),
        'y_pred': np.array(all_y_pred),
        'y_prob': np.vstack(all_y_prob),
        'cls1': np.vstack(all_cls1),
        'cls2': np.vstack(all_cls2),
    }

    if all_ibs_token:
        result['ibs_token'] = np.vstack(all_ibs_token)

    return result


# =============================================================================
# IBS Connectivity Matrix Extraction
# =============================================================================

class IBSMatrixExtractor:
    """
    Extract IBS connectivity matrices from the model.

    Uses forward hooks to capture intermediate outputs.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.ibs_matrices = []
        self.hook_handle = None

    def _hook_fn(self, module, input, output):
        """Hook function to capture IBS matrices."""
        # output shape: (B, 6, 7, C, C)
        self.ibs_matrices.append(output.detach().cpu().numpy())

    def register_hook(self):
        """Register forward hook on IBS matrix generator."""
        if hasattr(self.model, 'ibs_matrix_generator'):
            self.hook_handle = self.model.ibs_matrix_generator.register_forward_hook(
                self._hook_fn
            )
        else:
            print("[Warning] Model does not have ibs_matrix_generator")

    def remove_hook(self):
        """Remove the registered hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def clear(self):
        """Clear stored matrices."""
        self.ibs_matrices = []


@torch.no_grad()
def extract_ibs_matrices(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Extract IBS connectivity matrices for all samples.

    Args:
        model: Trained model with IBS matrix generator
        dataloader: DataLoader for test set
        device: Device to run on

    Returns:
        Dictionary with:
            - matrices: np.ndarray (N, 6, 7, C, C)
            - y_true: np.ndarray (N,)
            - y_pred: np.ndarray (N,)
    """
    model.eval()
    extractor = IBSMatrixExtractor(model)
    extractor.register_hook()

    all_y_true = []
    all_y_pred = []

    try:
        for batch in tqdm(dataloader, desc='Extracting IBS matrices'):
            eeg1, eeg2, labels = batch[0], batch[1], batch[2]
            eeg1 = eeg1.to(device)
            eeg2 = eeg2.to(device)

            outputs = model(eeg1, eeg2, labels=None)
            y_pred = outputs['logits'].argmax(dim=1).cpu().numpy()

            all_y_true.extend(labels.numpy())
            all_y_pred.extend(y_pred)
    finally:
        extractor.remove_hook()

    # Concatenate all matrices
    matrices = np.concatenate(extractor.ibs_matrices, axis=0)

    return {
        'matrices': matrices,
        'y_true': np.array(all_y_true),
        'y_pred': np.array(all_y_pred)
    }


def compute_mean_ibs_by_class(
    matrices: np.ndarray,
    labels: np.ndarray,
    class_names: List[str]
) -> Dict[str, np.ndarray]:
    """
    Compute mean IBS matrices for each class.

    Args:
        matrices: IBS matrices (N, 6, 7, C, C)
        labels: Class labels (N,)
        class_names: List of class names

    Returns:
        Dict mapping class_name to mean matrix (6, 7, C, C)
    """
    result = {}
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        if mask.sum() > 0:
            result[class_name] = matrices[mask].mean(axis=0)
        else:
            result[class_name] = np.zeros_like(matrices[0])
    return result


def compute_ibs_difference(
    coop_matrices: np.ndarray,
    comp_matrices: np.ndarray
) -> np.ndarray:
    """
    Compute difference matrix: Cooperation - Competition.

    Args:
        coop_matrices: Mean matrices for Cooperation (6, 7, C, C)
        comp_matrices: Mean matrices for Competition (6, 7, C, C)

    Returns:
        Difference matrix (6, 7, C, C)
    """
    return coop_matrices - comp_matrices


# =============================================================================
# Frequency Sensitivity Analysis
# =============================================================================

class FrequencyMaskHook:
    """
    Hook to mask specific frequency bands in IBS matrices.
    """

    def __init__(self, band_idx_to_mask: int):
        """
        Args:
            band_idx_to_mask: Index of band to mask (0-5)
                0: broadband, 1: delta, 2: theta, 3: alpha, 4: beta, 5: gamma
        """
        self.band_idx = band_idx_to_mask
        self.hook_handle = None

    def _hook_fn(self, module, input, output):
        """Mask the specified band in the output."""
        # output shape: (B, 6, 7, C, C)
        output[:, self.band_idx, :, :, :] = 0
        return output

    def register_hook(self, model: nn.Module):
        """Register hook on IBS matrix generator."""
        if hasattr(model, 'ibs_matrix_generator'):
            self.hook_handle = model.ibs_matrix_generator.register_forward_hook(
                self._hook_fn
            )

    def remove_hook(self):
        """Remove the hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


@torch.no_grad()
def compute_frequency_sensitivity(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    band_names: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute model sensitivity to each frequency band by masking.

    Args:
        model: Trained model
        dataloader: DataLoader for test set
        device: Device to run on
        band_names: List of band names (default: broadband, delta, theta, alpha, beta, gamma)

    Returns:
        Dict mapping band name to {'accuracy': ..., 'f1': ...}
    """
    if band_names is None:
        band_names = ['broadband', 'delta', 'theta', 'alpha', 'beta', 'gamma']

    model.eval()
    results = {}

    for band_idx, band_name in enumerate(band_names):
        print(f"  Testing with {band_name} masked...")

        mask_hook = FrequencyMaskHook(band_idx)
        mask_hook.register_hook(model)

        all_y_true = []
        all_y_pred = []

        try:
            for batch in dataloader:
                eeg1, eeg2, labels = batch[0], batch[1], batch[2]
                eeg1 = eeg1.to(device)
                eeg2 = eeg2.to(device)

                outputs = model(eeg1, eeg2, labels=None)
                y_pred = outputs['logits'].argmax(dim=1).cpu().numpy()

                all_y_true.extend(labels.numpy())
                all_y_pred.extend(y_pred)
        finally:
            mask_hook.remove_hook()

        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )

        results[band_name] = {
            'accuracy': accuracy,
            'f1': f1
        }

    return results


# =============================================================================
# Cross-Attention Weight Extraction
# =============================================================================

class AttentionWeightExtractor:
    """
    Extract cross-attention weights from CrossBrainAttention module.
    Hooks into the dropout layer of MultiHeadAttention to capture weights.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.weights = []
        self.hooks = []
        self._register_hook()

    def _register_hook(self):
        # Target: model.cross_attn.cross_attn.dropout
        # This is where 'attn' weights are passed through dropout in MultiHeadAttention
        try:
            # Check if cross_attn exists (it might be disabled in ablation)
            if hasattr(self.model, 'cross_attn'):
                target_layer = self.model.cross_attn.cross_attn.dropout
                self.hooks.append(target_layer.register_forward_hook(self._hook_fn))
            else:
                print("[Warning] Cross-attention module not found in model")
        except AttributeError:
             print("[Warning] Could not locate cross-attention dropout layer")

    def _hook_fn(self, module, input, output):
        # input is a tuple (attn_weights,), output is attn_weights (after dropout)
        # We prefer input[0] because it's the raw probabilities (before dropout mask)
        # But in eval mode, output is same as input.
        # Shape: (B, NumHeads, T, T)
        if isinstance(input, tuple):
            self.weights.append(input[0].detach().cpu())
        else:
            self.weights.append(input.detach().cpu())

    def reset(self):
        self.weights = []

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


@torch.no_grad()
def extract_attention_weights(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 200
) -> Dict[str, Any]:
    """
    Extract cross-attention weights and compute statistics.

    Args:
        model: Trained model
        dataloader: DataLoader for test set
        device: Device to run on
        max_samples: Maximum samples to extract

    Returns:
        Dictionary with attention statistics
    """
    print("[Metrics] Extracting attention weights...")
    model.eval()
    
    extractor = AttentionWeightExtractor(model)
    if not extractor.hooks:
        return {}

    # Store accumulated maps
    # We will average over heads and batch
    # But since T is large (~256+), storing (N, T, T) is heavy.
    # We'll compute running mean
    
    running_mean_map = None
    count = 0
    
    # Store diagonal stats per class
    # diagonals: List of (T,) arrays
    diagonals = defaultdict(list)
    
    try:
        for batch in tqdm(dataloader, desc='Extracting Attention'):
            if count >= max_samples:
                break
                
            eeg1, eeg2, labels = batch[0], batch[1], batch[2]
            eeg1 = eeg1.to(device)
            eeg2 = eeg2.to(device)
            labels = labels.cpu().numpy()
            
            extractor.reset()
            _ = model(eeg1, eeg2)
            
            # Extractor should have 2 tensors: [Attn_Z1_to_Z2, Attn_Z2_to_Z1]
            # Each (B, H, T, T)
            if len(extractor.weights) != 2:
                continue
                
            # Average the two directions and all heads
            # attn: (B, T, T)
            attn1 = extractor.weights[0].mean(dim=1) # Z1->Z2
            attn2 = extractor.weights[1].mean(dim=1) # Z2->Z1
            attn_avg = (attn1 + attn2) / 2.0
            
            # Update running mean map
            batch_sum = attn_avg.sum(dim=0) # (T, T)
            B = attn_avg.shape[0]
            
            if running_mean_map is None:
                running_mean_map = batch_sum
            else:
                if running_mean_map.shape == batch_sum.shape:
                    running_mean_map += batch_sum
                else:
                    # Shape mismatch (variable length?), skip
                    pass
            
            count += B
            
            # Extract diagonals
            for i in range(B):
                # diag: (T,)
                diag = torch.diagonal(attn_avg[i], offset=0)
                diagonals[labels[i]].append(diag.numpy())
                
    finally:
        extractor.remove_hooks()
        
    # Finalize stats
    if running_mean_map is not None:
        mean_map = (running_mean_map / count).numpy()
    else:
        mean_map = None
        
    # Aggregate diagonal stats
    diag_stats = {}
    class_names = ['Single', 'Competition', 'Cooperation'] # Assumption, should pass in
    
    all_diags = []
    
    for class_idx, diag_list in diagonals.items():
        # diag_list: List of (T,)
        # Stack might fail if lengths differ
        # We assume fixed length T for simplicity or truncate
        if not diag_list:
            continue
            
        # Check lengths
        min_len = min(len(d) for d in diag_list)
        diag_arr = np.array([d[:min_len] for d in diag_list])
        
        mean_diag = diag_arr.mean(axis=0) # (T,)
        mean_val = mean_diag.mean() # Scalar
        
        # Calculate "Diagonal Dominance": mean(diag) / mean(off_diag)
        # Since map sums to 1 per row, mean(row) = 1/T
        # Just reporting mean_val is comparable
        
        if class_idx < len(class_names):
            name = class_names[class_idx]
        else:
            name = str(class_idx)
            
        diag_stats[name] = {
            'mean_diagonal_vector': mean_diag,
            'mean_diagonal_value': mean_val,
            'count': len(diag_list)
        }

    return {
        'mean_map': mean_map,
        'diagonal_stats': diag_stats
    }


# =============================================================================
# Feature Embedding Extraction (for t-SNE/UMAP)
# =============================================================================

@torch.no_grad()
def extract_features_for_embedding(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, np.ndarray]:
    """
    Extract features suitable for t-SNE/UMAP visualization.

    Args:
        model: Trained model
        dataloader: DataLoader for test set
        device: Device to run on

    Returns:
        Dictionary with:
            - z_fuse: Fused features before classifier (N, 3*d_model)
            - ibs_token: IBS token representations (N, d_model)
            - cls1: CLS token from brain 1 (N, d_model)
            - cls2: CLS token from brain 2 (N, d_model)
            - y_true: True labels (N,)
            - y_pred: Predicted labels (N,)
    """
    model.eval()

    all_z_fuse = []
    all_ibs_token = []
    all_cls1 = []
    all_cls2 = []
    all_y_true = []
    all_y_pred = []

    for batch in tqdm(dataloader, desc='Extracting embeddings'):
        eeg1, eeg2, labels = batch[0], batch[1], batch[2]
        eeg1 = eeg1.to(device)
        eeg2 = eeg2.to(device)

        # Run forward pass
        outputs = model(eeg1, eeg2, labels=None)

        # Extract CLS tokens
        cls1 = outputs['cls1']  # (B, d_model)
        cls2 = outputs['cls2']  # (B, d_model)

        # Reconstruct z_fuse: [f_pair, mp1, mp2]
        # Since we don't have direct access, we approximate using cls tokens
        # For exact z_fuse, we would need to modify the model
        # Here we use concatenation of cls1 and cls2 as proxy
        z_fuse_proxy = torch.cat([cls1, cls2, torch.abs(cls1 - cls2)], dim=-1)

        all_z_fuse.append(z_fuse_proxy.cpu().numpy())
        all_cls1.append(cls1.cpu().numpy())
        all_cls2.append(cls2.cpu().numpy())

        if outputs.get('ibs_token') is not None:
            all_ibs_token.append(outputs['ibs_token'].cpu().numpy())

        y_pred = outputs['logits'].argmax(dim=1).cpu().numpy()
        all_y_true.extend(labels.numpy())
        all_y_pred.extend(y_pred)

    result = {
        'z_fuse': np.vstack(all_z_fuse),
        'cls1': np.vstack(all_cls1),
        'cls2': np.vstack(all_cls2),
        'y_true': np.array(all_y_true),
        'y_pred': np.array(all_y_pred)
    }

    if all_ibs_token:
        result['ibs_token'] = np.vstack(all_ibs_token)

    return result


def compute_tsne(
    features: np.ndarray,
    perplexity: int = 30,
    max_iter: int = 1000,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute t-SNE embedding.

    Args:
        features: Feature array (N, D)
        perplexity: t-SNE perplexity
        max_iter: Number of iterations
        random_state: Random seed

    Returns:
        2D embedding (N, 2)
    """
    from sklearn.manifold import TSNE

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
        init='pca'
    )
    return tsne.fit_transform(features)


def compute_umap(
    features: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute UMAP embedding.

    Args:
        features: Feature array (N, D)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random seed

    Returns:
        2D embedding (N, 2)
    """
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state
        )
        return reducer.fit_transform(features)
    except ImportError:
        print("[Warning] umap-learn not installed. Skipping UMAP.")
        return None


# =============================================================================
# Grad-CAM for Spectrogram
# =============================================================================

class GradCAM:
    """
    Grad-CAM implementation for EEG Spectrograms.
    Hooks into the specified layer to capture activations and gradients.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = []
        self.gradients = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations.append(output.detach())

        def backward_hook(module, grad_input, grad_output):
            # grad_output is a tuple, we want the first element
            self.gradients.append(grad_output[0].detach())

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def reset(self):
        self.activations = []
        self.gradients = []

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def generate_cam(self, activations: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
        """
        Generate CAM from activations and gradients.
        Args:
            activations: (N, C, H, W)
            gradients: (N, C, H, W)
        Returns:
            cam: (N, H, W)
        """
        # Global Average Pooling of gradients -> weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (N, C, 1, 1)
        
        # Weighted sum of activations
        cam = torch.sum(weights * activations, dim=1)  # (N, H, W)
        
        # ReLU
        cam = F.relu(cam)
        
        return cam.cpu().numpy()


@torch.no_grad()
def compute_gradcam_spectrogram(
    model: nn.Module,
    eeg1: torch.Tensor,
    eeg2: torch.Tensor,
    target_class: int,
    device: torch.device
) -> np.ndarray:
    """
    Compute Grad-CAM for a single sample (wrapper around batch version).
    """
    # This is a simplified wrapper, better to use batch processing
    pass


def compute_gradcam_batch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    target_class: int = None,
    max_samples: int = 100
) -> Dict[str, np.ndarray]:
    """
    Compute average Grad-CAM for each class.

    Args:
        model: Trained model
        dataloader: DataLoader
        device: Device
        target_class: Not used (we use predicted class or true class)
        max_samples: Max samples to process

    Returns:
        Dict mapping class_name to mean Grad-CAM heatmap (F, T)
    """
    print("[Metrics] initializing Grad-CAM...")
    
    # Identify target layer
    if not hasattr(model, 'spectrogram_generator'):
        print("[Error] Model has no spectrogram_generator")
        return {}
    
    # Target the second Conv2d layer in spec_conv (index 3)
    # Architecture: Conv2d(0) -> ReLU(1) -> MaxPool(2) -> Conv2d(3)
    try:
        target_layer = model.spectrogram_generator.spec_conv[3]
    except (AttributeError, IndexError):
        print("[Error] Could not find target layer in spectrogram_generator")
        return {}
        
    gradcam = GradCAM(model, target_layer)
    
    # Store accumulated CAMs per class
    # Key: class_idx, Value: list of CAMs
    class_cams = defaultdict(list)
    samples_count = 0
    
    model.eval()
    # We need gradients for input/activations, so we can't use standard no_grad
    # But we want to freeze model weights
    for param in model.parameters():
        param.requires_grad = False
        
    try:
        for batch in tqdm(dataloader, desc='Computing Grad-CAM'):
            if samples_count >= max_samples:
                break
                
            eeg1, eeg2, labels = batch[0], batch[1], batch[2]
            eeg1 = eeg1.to(device).requires_grad_(True)
            eeg2 = eeg2.to(device).requires_grad_(True)
            labels = labels.to(device)
            
            gradcam.reset()
            
            # Forward pass
            model.zero_grad()
            outputs = model(eeg1, eeg2)
            logits = outputs['logits']
            
            # We compute Grad-CAM for the *Predicted* class
            # (Alternatively, could use True class)
            preds = logits.argmax(dim=1)
            
            # Backward pass for each sample in batch
            # To do this efficiently for a batch, we can backward on sum of scores
            # But that mixes gradients. Correct way is per-sample or using specific masking.
            # Simplified approach: Backward on the sum of scores for the predicted class
            
            one_hot = F.one_hot(preds, num_classes=logits.shape[1]).float()
            score = (logits * one_hot).sum()
            score.backward()
            
            # Retrieve data
            # Activations: [act_brain1, act_brain2] (Forward order)
            # Gradients: [grad_brain2, grad_brain1] (Backward order - usually reversed)
            
            if len(gradcam.activations) != 2 or len(gradcam.gradients) != 2:
                # Might happen if spectrogram not used or structure differs
                continue
                
            act1 = gradcam.activations[0] # (B*C, 64, F', T')
            act2 = gradcam.activations[1]
            
            # Gradients are usually reversed in list due to backward hook execution order
            grad2 = gradcam.gradients[0]
            grad1 = gradcam.gradients[1]
            
            # Compute CAMs
            # These are (B*C, H, W)
            cam1 = gradcam.generate_cam(act1, grad1)
            cam2 = gradcam.generate_cam(act2, grad2)
            
            # Reshape to (B, C, H, W)
            B = eeg1.shape[0]
            C = eeg1.shape[1]
            
            cam1 = cam1.reshape(B, C, cam1.shape[1], cam1.shape[2])
            cam2 = cam2.reshape(B, C, cam2.shape[1], cam2.shape[2])
            
            # Average over channels to get (B, H, W)
            cam1_mean = cam1.mean(axis=1)
            cam2_mean = cam2.mean(axis=1)
            
            # Average Brain 1 and Brain 2
            cam_avg = (cam1_mean + cam2_mean) / 2.0
            
            # Upsample to common size for visualization
            # Target size: (64, 64) or similar (Freq bins, Time)
            # We'll rely on the analysis script to handle detailed upsampling or just resize here
            # For saving storage, we resize to (64, 64) here using cv2 or scipy
            # Using torch interpolate is easier
            
            cam_tensor = torch.from_numpy(cam_avg).unsqueeze(1) # (B, 1, H, W)
            cam_resized = F.interpolate(cam_tensor, size=(64, 64), mode='bilinear', align_corners=False)
            cam_resized = cam_resized.squeeze(1).numpy() # (B, 64, 64)
            
            # Store by true class
            labels_np = labels.cpu().numpy()
            for i in range(B):
                class_idx = labels_np[i]
                class_cams[class_idx].append(cam_resized[i])
                
            samples_count += B
            
    finally:
        gradcam.remove_hooks()
        # Re-enable grads if needed (though we are done)
        for param in model.parameters():
            param.requires_grad = True

    # Average per class
    results = {}
    for class_idx, cams in class_cams.items():
        if cams:
            results[class_idx] = np.mean(cams, axis=0)
            
    return results


# =============================================================================
# Utility Functions
# =============================================================================

def get_band_names() -> List[str]:
    """Return standard EEG frequency band names."""
    return ['broadband', 'delta', 'theta', 'alpha', 'beta', 'gamma']


def get_feature_names() -> List[str]:
    """Return IBS feature names."""
    return ['PLV', 'PLI', 'wPLI', 'Coherence', 'Power_Corr', 'Phase_Diff', 'Time_Corr']


def get_class_names() -> List[str]:
    """Return default class names."""
    return ['Single', 'Competition', 'Cooperation']


def get_channel_names(num_channels: int = 32) -> List[str]:
    """
    Return standard EEG channel names.

    For 32-channel setup, returns generic names.
    """
    # Standard 10-20 system channels (approximate for 32 channels)
    standard_channels = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'CP5', 'CP1', 'CP2', 'CP6',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'PO9', 'O1', 'Oz', 'O2', 'PO10'
    ]

    if num_channels <= len(standard_channels):
        return standard_channels[:num_channels]
    else:
        # Generate generic names for extra channels
        return [f'Ch{i}' for i in range(num_channels)]
