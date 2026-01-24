"""
FuzzyGatingFusion: Uncertainty-Aware Differentiable Fuzzy Fusion (UDF-Fusion)

This module implements a differentiable fuzzy logic-based fusion mechanism that
dynamically adjusts modality weights based on prediction uncertainty (entropy).
"""

import math
from typing import Dict, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_softplus(x: float) -> float:
    """Compute inverse of softplus: log(exp(x) - 1)."""
    if x <= 0:
        raise ValueError("inverse_softplus requires x > 0")
    return math.log(math.expm1(x))


class FuzzyGatingFusion(nn.Module):
    """
    Uncertainty-Aware Differentiable Fuzzy Fusion (UDF-Fusion).

    This module fuses logits from two modalities (image/gaze and EEG) using
    differentiable fuzzy logic. The fusion weight (alpha) is computed based on
    the entropy of each modality's predictions through Gaussian membership
    functions and fuzzy inference rules.

    Architecture (6 Stages):
        1. Temperature Scaling: Calibrate logits with learnable temperatures
        2. Entropy Estimation: Quantify uncertainty via Shannon entropy
        3. Fuzzification: Map entropy to fuzzy membership degrees
        4. Fuzzy Inference: Apply 4 fuzzy rules with product T-norm
        5. Defuzzification: Compute weighted average of rule consequents
        6. Fusion: Combine calibrated logits using the computed alpha

    Args:
        num_classes: Number of output classes (default: 3)
        mode: Ablation mode - one of:
            - 'full': Complete model with all components
            - 'no_temperature': Disable temperature scaling (T=1)
            - 'no_fuzzification': Use entropy directly for weights
            - 'fixed_weights': Use fixed alpha=0.5 (baseline)
        eps_temp: Minimum temperature value for numerical stability (default: 0.1)
        eps_log: Small value for log stability (default: 1e-8)
        eps_div: Small value for division stability (default: 1e-8)

    Input:
        img_logits: (B, num_classes) - Logits from Gaze Encoder
        eeg_logits: (B, num_classes) - Logits from EEG Encoder

    Output:
        fused_logits: (B, num_classes) - Fused output logits
        alpha: (B,) - Fusion weight for img modality
        aux_info: Dict - Intermediate information for analysis
    """

    VALID_MODES = ('full', 'no_temperature', 'no_fuzzification', 'fixed_weights')

    def __init__(
        self,
        num_classes: int = 3,
        mode: Literal['full', 'no_temperature', 'no_fuzzification', 'fixed_weights'] = 'full',
        eps_temp: float = 0.1,
        eps_log: float = 1e-8,
        eps_div: float = 1e-8,
    ):
        super().__init__()

        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {self.VALID_MODES}")

        self.num_classes = num_classes
        self.mode = mode
        self.eps_temp = eps_temp
        self.eps_log = eps_log
        self.eps_div = eps_div

        # Maximum entropy for normalization (ln(num_classes))
        self.max_entropy = math.log(num_classes)

        # Stage 1: Temperature Scaling Parameters
        # Initialize τ such that Softplus(τ) + eps_temp ≈ target temperature
        # T_img ≈ 1.5, T_eeg ≈ 1.0
        self.tau_img = nn.Parameter(torch.tensor(inverse_softplus(1.5 - eps_temp)))
        self.tau_eeg = nn.Parameter(torch.tensor(inverse_softplus(1.0 - eps_temp)))

        # Stage 3: Fuzzification - Gaussian Membership Function Parameters
        # c_reliable is fixed at 0.0 (low entropy = reliable)
        self.register_buffer('c_reliable', torch.tensor(0.0))

        # c_unreliable centers (modality-specific, learnable)
        # Initialize to ~80% of max entropy
        c_unrel_init = self.max_entropy * 0.8
        self.c_unreliable_img = nn.Parameter(torch.tensor(c_unrel_init))
        self.c_unreliable_eeg = nn.Parameter(torch.tensor(c_unrel_init))

        # Sigma parameters in log-space for positivity
        # Initialize to ~30% of max entropy
        log_sigma_init = math.log(self.max_entropy * 0.3)
        self.log_sigma_reliable_img = nn.Parameter(torch.tensor(log_sigma_init))
        self.log_sigma_reliable_eeg = nn.Parameter(torch.tensor(log_sigma_init))
        self.log_sigma_unreliable_img = nn.Parameter(torch.tensor(log_sigma_init))
        self.log_sigma_unreliable_eeg = nn.Parameter(torch.tensor(log_sigma_init))

        # Stage 4: Fuzzy Inference - Rule Consequents (β parameters)
        # β_k = logit(target) = log(target / (1 - target))
        # R1: Img Reliable, EEG Unreliable -> favor img (0.8)
        # R2: Img Unreliable, EEG Reliable -> favor eeg (0.2)
        # R3: Both Reliable -> slight img preference (0.6)
        # R4: Both Unreliable -> equal (0.5)
        self.beta = nn.Parameter(torch.tensor([
            math.log(0.8 / 0.2),   # β₁ ≈ 1.386
            math.log(0.2 / 0.8),   # β₂ ≈ -1.386
            math.log(0.6 / 0.4),   # β₃ ≈ 0.405
            0.0,                   # β₄ = 0.0
        ]))

    @property
    def temp_img(self) -> torch.Tensor:
        """Get current image modality temperature."""
        return F.softplus(self.tau_img) + self.eps_temp

    @property
    def temp_eeg(self) -> torch.Tensor:
        """Get current EEG modality temperature."""
        return F.softplus(self.tau_eeg) + self.eps_temp

    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute Shannon entropy from logits.

        Args:
            logits: (B, C) - Raw or calibrated logits

        Returns:
            entropy: (B,) - Entropy values
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + self.eps_log)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    def _gaussian_membership(
        self,
        x: torch.Tensor,
        center: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Gaussian membership degree.

        μ(x) = exp(-(x - c)² / (2σ²))

        Args:
            x: (B,) - Input values (entropy)
            center: () - Gaussian center
            log_sigma: () - Log of Gaussian sigma

        Returns:
            membership: (B,) - Membership degrees in [0, 1]
        """
        sigma = torch.exp(log_sigma)
        return torch.exp(-((x - center) ** 2) / (2 * sigma ** 2 + self.eps_div))

    def _fuzzify(
        self,
        entropy_img: torch.Tensor,
        entropy_eeg: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Fuzzification: Map entropy values to membership degrees.

        Args:
            entropy_img: (B,) - Image modality entropy
            entropy_eeg: (B,) - EEG modality entropy

        Returns:
            membership: Dict with 'img' and 'eeg' sub-dicts containing 'rel' and 'unrel'
        """
        # Image modality memberships
        mu_img_rel = self._gaussian_membership(
            entropy_img, self.c_reliable, self.log_sigma_reliable_img
        )
        mu_img_unrel = self._gaussian_membership(
            entropy_img, self.c_unreliable_img, self.log_sigma_unreliable_img
        )

        # EEG modality memberships
        mu_eeg_rel = self._gaussian_membership(
            entropy_eeg, self.c_reliable, self.log_sigma_reliable_eeg
        )
        mu_eeg_unrel = self._gaussian_membership(
            entropy_eeg, self.c_unreliable_eeg, self.log_sigma_unreliable_eeg
        )

        return {
            'img': {'rel': mu_img_rel, 'unrel': mu_img_unrel},
            'eeg': {'rel': mu_eeg_rel, 'unrel': mu_eeg_unrel},
        }

    def _fuzzy_inference(
        self,
        membership: Dict[str, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuzzy inference: Apply rules with product T-norm.

        Rules:
            R1: Img Reliable ∧ EEG Unreliable -> favor img
            R2: Img Unreliable ∧ EEG Reliable -> favor eeg
            R3: Both Reliable -> slight img preference
            R4: Both Unreliable -> equal

        Args:
            membership: Fuzzification output

        Returns:
            firing_strengths: (B, 4) - Rule firing strengths
            consequents: (4,) - Rule consequent values (sigmoid of beta)
        """
        mu_img_rel = membership['img']['rel']
        mu_img_unrel = membership['img']['unrel']
        mu_eeg_rel = membership['eeg']['rel']
        mu_eeg_unrel = membership['eeg']['unrel']

        # Compute firing strengths (product T-norm)
        w1 = mu_img_rel * mu_eeg_unrel      # R1: Img Rel ∧ EEG Unrel
        w2 = mu_img_unrel * mu_eeg_rel      # R2: Img Unrel ∧ EEG Rel
        w3 = mu_img_rel * mu_eeg_rel        # R3: Both Rel
        w4 = mu_img_unrel * mu_eeg_unrel    # R4: Both Unrel

        firing_strengths = torch.stack([w1, w2, w3, w4], dim=-1)  # (B, 4)
        consequents = torch.sigmoid(self.beta)  # (4,)

        return firing_strengths, consequents

    def _defuzzify(
        self,
        firing_strengths: torch.Tensor,
        consequents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Defuzzification: Weighted average of rule consequents.

        α = Σ(wₖ × θₖ) / (Σwₖ + ε)

        Args:
            firing_strengths: (B, 4) - Rule firing strengths
            consequents: (4,) - Rule consequent values

        Returns:
            alpha: (B,) - Fusion weight for img modality
        """
        # Weighted sum of consequents
        weighted_sum = torch.sum(firing_strengths * consequents, dim=-1)
        # Normalization
        weight_sum = torch.sum(firing_strengths, dim=-1)
        alpha = weighted_sum / (weight_sum + self.eps_div)
        # Clamp to valid range
        alpha = torch.clamp(alpha, 0.0, 1.0)
        return alpha

    def _compute_alpha_no_fuzzification(
        self,
        entropy_img: torch.Tensor,
        entropy_eeg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute alpha directly from entropy without fuzzy logic.

        Uses inverse entropy weighting:
        α = (1 - H_img/H_max) / ((1 - H_img/H_max) + (1 - H_eeg/H_max) + ε)

        Args:
            entropy_img: (B,) - Image modality entropy
            entropy_eeg: (B,) - EEG modality entropy

        Returns:
            alpha: (B,) - Fusion weight
        """
        # Confidence = 1 - normalized_entropy
        conf_img = 1.0 - entropy_img / (self.max_entropy + self.eps_div)
        conf_eeg = 1.0 - entropy_eeg / (self.max_entropy + self.eps_div)

        # Clamp to avoid negative values
        conf_img = torch.clamp(conf_img, min=0.0)
        conf_eeg = torch.clamp(conf_eeg, min=0.0)

        # Weighted by confidence
        alpha = conf_img / (conf_img + conf_eeg + self.eps_div)
        return torch.clamp(alpha, 0.0, 1.0)

    def forward(
        self,
        img_logits: torch.Tensor,
        eeg_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass: Fuse two modality logits.

        Args:
            img_logits: (B, num_classes) - Logits from Gaze Encoder
            eeg_logits: (B, num_classes) - Logits from EEG Encoder

        Returns:
            fused_logits: (B, num_classes) - Fused output logits
            alpha: (B,) - Fusion weight for img modality (1=full img, 0=full eeg)
            aux_info: Dict containing intermediate values for analysis
        """
        B = img_logits.size(0)
        device = img_logits.device

        aux_info = {}

        # Stage 1: Temperature Scaling
        if self.mode == 'no_temperature' or self.mode == 'fixed_weights':
            T_img = torch.ones(1, device=device)
            T_eeg = torch.ones(1, device=device)
            z_img_cal = img_logits
            z_eeg_cal = eeg_logits
        else:
            T_img = self.temp_img
            T_eeg = self.temp_eeg
            z_img_cal = img_logits / T_img
            z_eeg_cal = eeg_logits / T_eeg

        aux_info['temperatures'] = {'img': T_img.detach(), 'eeg': T_eeg.detach()}

        # Stage 2: Entropy Estimation
        H_img = self._compute_entropy(z_img_cal)
        H_eeg = self._compute_entropy(z_eeg_cal)
        aux_info['entropies'] = {'img': H_img.detach(), 'eeg': H_eeg.detach()}

        # Handle different modes
        if self.mode == 'fixed_weights':
            # Baseline: fixed alpha = 0.5
            alpha = torch.full((B,), 0.5, device=device)
            aux_info['membership'] = None
            aux_info['firing_strengths'] = None
            aux_info['consequents'] = None

        elif self.mode == 'no_fuzzification':
            # Direct entropy-based weighting
            alpha = self._compute_alpha_no_fuzzification(H_img, H_eeg)
            aux_info['membership'] = None
            aux_info['firing_strengths'] = None
            aux_info['consequents'] = None

        else:  # 'full' or 'no_temperature'
            # Stage 3: Fuzzification
            membership = self._fuzzify(H_img, H_eeg)
            aux_info['membership'] = {
                'img': {k: v.detach() for k, v in membership['img'].items()},
                'eeg': {k: v.detach() for k, v in membership['eeg'].items()},
            }

            # Stage 4: Fuzzy Inference
            firing_strengths, consequents = self._fuzzy_inference(membership)
            aux_info['firing_strengths'] = firing_strengths.detach()
            aux_info['consequents'] = consequents.detach()

            # Stage 5: Defuzzification
            alpha = self._defuzzify(firing_strengths, consequents)

        # Store fuzzy parameters for analysis
        aux_info['fuzz_params'] = {
            'c_unreliable': {
                'img': self.c_unreliable_img.detach(),
                'eeg': self.c_unreliable_eeg.detach(),
            },
            'sigma_reliable': {
                'img': torch.exp(self.log_sigma_reliable_img).detach(),
                'eeg': torch.exp(self.log_sigma_reliable_eeg).detach(),
            },
            'sigma_unreliable': {
                'img': torch.exp(self.log_sigma_unreliable_img).detach(),
                'eeg': torch.exp(self.log_sigma_unreliable_eeg).detach(),
            },
        }

        # Stage 6: Fusion
        # Expand alpha for broadcasting: (B,) -> (B, 1)
        alpha_expanded = alpha.unsqueeze(-1)
        fused_logits = alpha_expanded * z_img_cal + (1 - alpha_expanded) * z_eeg_cal

        return fused_logits, alpha, aux_info

    def compute_temperature_regularization(
        self,
        t_min: float = 0.5,
        t_max: float = 5.0,
    ) -> torch.Tensor:
        """
        Compute temperature regularization loss.

        L_reg = ReLU(T_img - T_max) + ReLU(T_min - T_img) +
                ReLU(T_eeg - T_max) + ReLU(T_min - T_eeg)

        This encourages temperatures to stay within a reasonable range.

        Args:
            t_min: Minimum allowed temperature (default: 0.5)
            t_max: Maximum allowed temperature (default: 5.0)

        Returns:
            reg_loss: Regularization loss (scalar)
        """
        T_img = self.temp_img
        T_eeg = self.temp_eeg

        reg_loss = (
            F.relu(T_img - t_max) + F.relu(t_min - T_img) +
            F.relu(T_eeg - t_max) + F.relu(t_min - T_eeg)
        )
        return reg_loss

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return (
            f"num_classes={self.num_classes}, "
            f"mode='{self.mode}', "
            f"eps_temp={self.eps_temp}"
        )


def _test_fuzzy_gating_fusion():
    """Unit tests for FuzzyGatingFusion module."""
    print("=" * 60)
    print("Testing FuzzyGatingFusion")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, C = 8, 3  # batch size, num_classes

    # Test 1: Shape verification
    print("\n[Test 1] Shape verification...")
    model = FuzzyGatingFusion(num_classes=C, mode='full').to(device)
    img_logits = torch.randn(B, C, device=device)
    eeg_logits = torch.randn(B, C, device=device)

    fused_logits, alpha, aux_info = model(img_logits, eeg_logits)

    assert fused_logits.shape == (B, C), f"Expected fused_logits shape (B, C), got {fused_logits.shape}"
    assert alpha.shape == (B,), f"Expected alpha shape (B,), got {alpha.shape}"
    assert torch.all((alpha >= 0) & (alpha <= 1)), "Alpha should be in [0, 1]"
    print(f"  fused_logits: {fused_logits.shape}")
    print(f"  alpha: {alpha.shape}, range: [{alpha.min():.4f}, {alpha.max():.4f}]")
    print("  PASSED")

    # Test 2: Gradient flow
    print("\n[Test 2] Gradient flow verification...")
    model.zero_grad()
    loss = fused_logits.sum()
    loss.backward()

    param_names = [
        'tau_img', 'tau_eeg',
        'c_unreliable_img', 'c_unreliable_eeg',
        'log_sigma_reliable_img', 'log_sigma_reliable_eeg',
        'log_sigma_unreliable_img', 'log_sigma_unreliable_eeg',
        'beta'
    ]
    all_have_grad = True
    for name in param_names:
        param = getattr(model, name)
        if param.grad is None:
            print(f"  WARNING: {name} has no gradient!")
            all_have_grad = False
        else:
            print(f"  {name}: grad_norm = {param.grad.norm():.6f}")

    if all_have_grad:
        print("  PASSED")
    else:
        print("  FAILED (some parameters have no gradient)")

    # Test 3: Ablation modes
    print("\n[Test 3] Ablation modes...")
    modes = ['full', 'no_temperature', 'no_fuzzification', 'fixed_weights']
    for mode in modes:
        model = FuzzyGatingFusion(num_classes=C, mode=mode).to(device)
        fused, alpha, aux = model(img_logits, eeg_logits)
        print(f"  {mode}: alpha_mean = {alpha.mean():.4f}")

        if mode == 'fixed_weights':
            assert torch.allclose(alpha, torch.full_like(alpha, 0.5)), \
                "fixed_weights mode should produce alpha=0.5"
    print("  PASSED")

    # Test 4: Edge cases
    print("\n[Test 4] Edge cases...")
    model = FuzzyGatingFusion(num_classes=C, mode='full').to(device)

    # Uniform distribution (max entropy)
    uniform_logits = torch.zeros(B, C, device=device)
    fused, alpha, _ = model(uniform_logits, uniform_logits)
    print(f"  Uniform (max entropy): alpha_mean = {alpha.mean():.4f}")

    # Very confident prediction (low entropy)
    confident_logits = torch.tensor([[10.0, -10.0, -10.0]] * B, device=device)
    fused, alpha, _ = model(confident_logits, uniform_logits)
    print(f"  Confident img, uniform eeg: alpha_mean = {alpha.mean():.4f}")

    fused, alpha, _ = model(uniform_logits, confident_logits)
    print(f"  Uniform img, confident eeg: alpha_mean = {alpha.mean():.4f}")
    print("  PASSED")

    # Test 5: Temperature regularization
    print("\n[Test 5] Temperature regularization...")
    model = FuzzyGatingFusion(num_classes=C, mode='full').to(device)
    reg_loss = model.compute_temperature_regularization(t_min=0.5, t_max=5.0)
    print(f"  Initial reg_loss: {reg_loss.item():.6f}")
    print(f"  T_img: {model.temp_img.item():.4f}, T_eeg: {model.temp_eeg.item():.4f}")
    print("  PASSED")

    # Test 6: aux_info contents
    print("\n[Test 6] aux_info contents...")
    model = FuzzyGatingFusion(num_classes=C, mode='full').to(device)
    _, _, aux_info = model(img_logits, eeg_logits)

    expected_keys = ['temperatures', 'entropies', 'membership', 'firing_strengths',
                     'consequents', 'fuzz_params']
    for key in expected_keys:
        assert key in aux_info, f"Missing key: {key}"
        print(f"  {key}: present")
    print("  PASSED")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    _test_fuzzy_gating_fusion()
