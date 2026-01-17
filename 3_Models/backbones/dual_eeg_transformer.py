"""
Dual EEG Transformer for Inter-Brain Synchrony Classification
Fuses two players' EEG signals with IBS (Inter-Brain Synchrony) token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

# Try relative import first, then absolute import, fall back to sys.modules
try:
    from .art import (
        TransformerEncoder,
        TransformerEncoderBlock,
        MultiHeadAttention,
        PositionalEmbedding
    )
except ImportError:
    try:
        from art import (
            TransformerEncoder,
            TransformerEncoderBlock,
            MultiHeadAttention,
            PositionalEmbedding
        )
    except ImportError:
        import sys
        if 'art' in sys.modules:
            art = sys.modules['art']
            TransformerEncoder = art.TransformerEncoder
            TransformerEncoderBlock = art.TransformerEncoderBlock
            MultiHeadAttention = art.MultiHeadAttention
            PositionalEmbedding = art.PositionalEmbedding
        else:
            raise ImportError("Could not import from art module")


class SpectrogramTokenGenerator(nn.Module):
    """
    Generates per-channel Spectrogram tokens using STFT
    为每个 EEG channel 生成一个 Spectrogram token

    使用短时距傅立叶变换 (STFT) 将原始 EEG 转换为 Time-Frequency 表示
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        n_fft: int = 128,           # FFT window size (~0.5s at 256Hz)
        hop_length: int = 64,        # Stride between windows (50% overlap)
        sampling_rate: int = 256,    # EEG sampling rate
        freq_bins: int = 64,         # Number of frequency bins to use
        use_log_magnitude: bool = True,  # Use log scale for magnitude
    ):
        super().__init__()
        self.d_model = d_model
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.freq_bins = freq_bins
        self.use_log_magnitude = use_log_magnitude

        # Register Hann window as buffer (not a parameter)
        self.register_buffer('window', torch.hann_window(n_fft))

        # 2D Conv to process spectrogram (F, T_spec) → fixed dimension
        # Input: (B*C, 1, F, T_spec) where F = freq_bins
        self.spec_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Fixed output size: 4x4
        )

        # Project flattened spectrogram to d_model
        # After AdaptiveAvgPool: (B*C, 64, 4, 4) → flatten to (B*C, 64*4*4=1024)
        self.proj = nn.Sequential(
            nn.Linear(64 * 4 * 4, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) - Raw EEG signal
        Returns:
            spec_tokens: (B, C, d_model) - Per-channel spectrogram tokens
        """
        B, C, T = x.shape

        # Reshape to (B*C, T) for STFT
        x_flat = x.reshape(B * C, T)  # (B*C, T)

        # Compute STFT
        # stft: (B*C, F, T_spec) complex tensor
        stft = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=True
        )

        # Take magnitude and optionally apply log scale
        magnitude = torch.abs(stft)  # (B*C, F, T_spec)

        # Only use first freq_bins frequency bins (e.g., 1-45 Hz range)
        magnitude = magnitude[:, :self.freq_bins, :]  # (B*C, freq_bins, T_spec)

        if self.use_log_magnitude:
            magnitude = torch.log(magnitude + 1e-8)  # Log scale for better dynamic range

        # Add channel dimension for Conv2D: (B*C, 1, F, T_spec)
        magnitude = magnitude.unsqueeze(1)

        # Apply 2D convolution
        spec_features = self.spec_conv(magnitude)  # (B*C, 64, 4, 4)

        # Flatten
        spec_features = spec_features.flatten(start_dim=1)  # (B*C, 1024)

        # Project to d_model
        spec_tokens = self.proj(spec_features)  # (B*C, d_model)

        # Reshape back to (B, C, d_model)
        spec_tokens = spec_tokens.reshape(B, C, self.d_model)

        return spec_tokens


class TemporalConvFrontend(nn.Module):
    """
    Temporal convolution frontend to downsample and embed EEG channels
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        kernel_size: int = 25,
        stride: int = 4,
        num_layers: int = 2
    ):
        super().__init__()
        self.convs = nn.ModuleList()

        # First conv layer
        self.convs.append(nn.Conv1d(in_channels, d_model, kernel_size, stride, padding=kernel_size//2))

        # Additional conv layers
        for _ in range(num_layers - 1):
            self.convs.append(nn.Conv1d(d_model, d_model, kernel_size, stride, padding=kernel_size//2))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) - EEG signal
        Returns:
            (B, T̃, d) - Downsampled and embedded
        """
        for conv in self.convs:
            x = self.dropout(self.relu(conv(x)))

        # Permute to (B, T, d)
        x = x.permute(0, 2, 1)
        return x


class IBSTokenGenerator(nn.Module):
    """
    Generates Inter-Brain Synchrony (IBS) token from dual EEG signals
    Computes cross-brain connectivity features (PLV, PLI, wPLI, Coherence, etc.)
    真正分頻段計算：theta (4-8Hz), alpha (8-13Hz), beta (13-30Hz), gamma (30-45Hz)

    Enhanced version with:
    - 7 features per band: PLV, PLI, wPLI, Coherence, Power Correlation, Phase Difference, Time-domain Correlation
    - Total: 4 bands * 7 features = 28 features
    - Optional LayerNorm (disabled by default to preserve feature relationships)
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        sampling_rate: int = 256,  # EEG 採樣率
        use_layernorm: bool = False,  # 預設關閉 LayerNorm
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.use_layernorm = use_layernorm

        # 定義頻段範圍 (Hz)
        self.freq_bands = {
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

        # Feature dimension: 4 bands * 7 features = 28
        # Features: PLV, PLI, wPLI, Coherence, Power Correlation, Phase Difference, Time-domain Correlation
        feature_dim = len(self.freq_bands) * 7

        # 使用更深的 projection network
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )

        # Optional LayerNorm
        if self.use_layernorm:
            self.norm = nn.LayerNorm(d_model)

    def bandpass_filter_fft(
        self,
        signal: torch.Tensor,
        low_freq: float,
        high_freq: float
    ) -> torch.Tensor:
        """
        使用 FFT 的 bandpass filter
        Args:
            signal: (B, C, T) - EEG signal
            low_freq: 低頻截止 (Hz)
            high_freq: 高頻截止 (Hz)
        Returns:
            filtered: (B, C, T) - 濾波後的信號
        """
        B, C, T = signal.shape

        # FFT
        fft_signal = torch.fft.rfft(signal, dim=2)  # (B, C, F)

        # 頻率軸
        freqs = torch.fft.rfftfreq(T, d=1.0/self.sampling_rate).to(signal.device)

        # 創建 bandpass mask
        mask = ((freqs >= low_freq) & (freqs <= high_freq)).float()
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, F)

        # 應用 mask
        fft_filtered = fft_signal * mask

        # IFFT 回時域
        filtered = torch.fft.irfft(fft_filtered, n=T, dim=2)

        return filtered

    def compute_plv(self, phase1: torch.Tensor, phase2: torch.Tensor) -> torch.Tensor:
        """
        Compute Phase Locking Value between two signals
        Args:
            phase1, phase2: (B, C, T) - phase of signals
        Returns:
            PLV: (B,) - scalar PLV value per batch
        """
        phase_diff = phase1 - phase2
        # 使用 complex exponential，然後取平均的絕對值
        complex_diff = torch.exp(1j * phase_diff)
        plv = torch.abs(torch.mean(complex_diff, dim=(1, 2)))
        return plv

    def compute_power_correlation(self, power1: torch.Tensor, power2: torch.Tensor) -> torch.Tensor:
        """
        Compute power correlation between two signals
        Args:
            power1, power2: (B, C, T)
        Returns:
            corr: (B,) - correlation coefficient per batch
        """
        p1_flat = power1.flatten(1)  # (B, C*T)
        p2_flat = power2.flatten(1)

        # Normalize
        p1_norm = (p1_flat - p1_flat.mean(dim=1, keepdim=True)) / (p1_flat.std(dim=1, keepdim=True) + 1e-8)
        p2_norm = (p2_flat - p2_flat.mean(dim=1, keepdim=True)) / (p2_flat.std(dim=1, keepdim=True) + 1e-8)

        # Compute correlation
        corr = (p1_norm * p2_norm).mean(dim=1)
        return corr

    def compute_phase(self, signal: torch.Tensor) -> torch.Tensor:
        """
        使用 Hilbert transform 計算瞬時相位
        使用 FFT-based Hilbert transform
        Args:
            signal: (B, C, T)
        Returns:
            phase: (B, C, T)
        """
        # FFT
        fft_signal = torch.fft.fft(signal, dim=2)

        # Hilbert transform in frequency domain
        T = signal.shape[2]
        h = torch.zeros(T, device=signal.device)
        if T % 2 == 0:
            h[0] = h[T // 2] = 1
            h[1:T // 2] = 2
        else:
            h[0] = 1
            h[1:(T + 1) // 2] = 2

        h = h.unsqueeze(0).unsqueeze(0)  # (1, 1, T)

        # Apply Hilbert transform
        analytic_signal = torch.fft.ifft(fft_signal * h, dim=2)

        # Extract phase
        phase = torch.angle(analytic_signal)

        return phase

    def compute_phase_lag_index(self, phase1: torch.Tensor, phase2: torch.Tensor) -> torch.Tensor:
        """
        Compute Phase Lag Index (PLI) - more robust than PLV
        PLI = |E[sign(Δφ)]|

        Args:
            phase1, phase2: (B, C, T) - phase of signals
        Returns:
            PLI: (B,) - scalar PLI value per batch
        """
        phase_diff = phase1 - phase2  # (B, C, T)
        pli = torch.abs(torch.sign(phase_diff).mean(dim=(1, 2)))  # (B,)
        return pli

    def compute_weighted_pli(
        self,
        phase1: torch.Tensor,
        phase2: torch.Tensor,
        power1: torch.Tensor,
        power2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted PLI (wPLI) - power-weighted version
        More robust to volume conduction

        Args:
            phase1, phase2: (B, C, T) - phase of signals
            power1, power2: (B, C, T) - power of signals
        Returns:
            wPLI: (B,) - scalar wPLI value per batch
        """
        phase_diff = phase1 - phase2  # (B, C, T)

        # Weights: average power (normalized)
        weights = (power1 + power2) / 2  # (B, C, T)
        weights = weights / (weights.sum(dim=(1, 2), keepdim=True) + 1e-8)

        # Weighted sign
        weighted_sign = (torch.sign(phase_diff) * weights).sum(dim=(1, 2))  # (B,)
        wpli = torch.abs(weighted_sign)

        return wpli

    def compute_coherence(self, eeg1_band: torch.Tensor, eeg2_band: torch.Tensor) -> torch.Tensor:
        """
        Compute magnitude-squared coherence
        Coh(f) = |Pxy(f)|^2 / (Pxx(f) * Pyy(f))

        Args:
            eeg1_band, eeg2_band: (B, C, T) - bandpass-filtered signals
        Returns:
            coherence: (B,) - average coherence across frequencies
        """
        # FFT
        fft1 = torch.fft.rfft(eeg1_band, dim=2)  # (B, C, F)
        fft2 = torch.fft.rfft(eeg2_band, dim=2)

        # Cross-spectral density (average over channels)
        pxy = (fft1 * fft2.conj()).mean(dim=1)  # (B, F)

        # Power spectral densities
        pxx = (fft1 * fft1.conj()).mean(dim=1).real  # (B, F)
        pyy = (fft2 * fft2.conj()).mean(dim=1).real

        # Coherence
        coherence = (pxy.abs() ** 2) / (pxx * pyy + 1e-8)  # (B, F)

        # Average over frequencies
        coherence = coherence.mean(dim=1)  # (B,)

        return coherence

    def compute_time_correlation(self, eeg1_band: torch.Tensor, eeg2_band: torch.Tensor) -> torch.Tensor:
        """
        Time-domain Pearson correlation

        Args:
            eeg1_band, eeg2_band: (B, C, T) - bandpass-filtered signals
        Returns:
            correlation: (B,) - Pearson correlation coefficient
        """
        # Average over channels
        eeg1_avg = eeg1_band.mean(dim=1)  # (B, T)
        eeg2_avg = eeg2_band.mean(dim=1)

        # Normalize (zero mean, unit variance)
        eeg1_norm = (eeg1_avg - eeg1_avg.mean(dim=1, keepdim=True)) / (eeg1_avg.std(dim=1, keepdim=True) + 1e-8)
        eeg2_norm = (eeg2_avg - eeg2_avg.mean(dim=1, keepdim=True)) / (eeg2_avg.std(dim=1, keepdim=True) + 1e-8)

        # Correlation
        corr = (eeg1_norm * eeg2_norm).mean(dim=1)  # (B,)

        return corr

    def forward(
        self,
        eeg1: torch.Tensor,
        eeg2: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate IBS token from dual EEG signals
        真正分頻段計算 7 種特徵：PLV, PLI, wPLI, Coherence, Power Correlation, Phase Difference, Time-domain Correlation

        Args:
            eeg1, eeg2: (B, C, T) - EEG signals
        Returns:
            ibs_token: (B, d_model) - IBS token
        """
        B = eeg1.shape[0]
        features = []

        # 對每個頻段分別計算 7 種特徵
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            # 1. Bandpass filter
            eeg1_band = self.bandpass_filter_fft(eeg1, low_freq, high_freq)
            eeg2_band = self.bandpass_filter_fft(eeg2, low_freq, high_freq)

            # 2. Compute power
            power1 = eeg1_band ** 2
            power2 = eeg2_band ** 2

            # 3. Compute phase using Hilbert transform
            phase1 = self.compute_phase(eeg1_band)
            phase2 = self.compute_phase(eeg2_band)

            # 4. Compute all 7 features
            plv = self.compute_plv(phase1, phase2)  # (B,)
            pli = self.compute_phase_lag_index(phase1, phase2)  # (B,)
            wpli = self.compute_weighted_pli(phase1, phase2, power1, power2)  # (B,)
            coherence = self.compute_coherence(eeg1_band, eeg2_band)  # (B,)
            power_corr = self.compute_power_correlation(power1, power2)  # (B,)
            phase_diff = torch.abs(torch.mean(phase1 - phase2, dim=(1, 2)))  # (B,)
            time_corr = self.compute_time_correlation(eeg1_band, eeg2_band)  # (B,)

            features.extend([plv, pli, wpli, coherence, power_corr, phase_diff, time_corr])

        # Stack features: (B, 28)
        features = torch.stack(features, dim=1)

        # Project to d_model
        ibs_token = self.proj(features)  # (B, d_model)

        # Optional LayerNorm
        if self.use_layernorm:
            ibs_token = self.norm(ibs_token)

        return ibs_token


class IBSConnectivityMatrixGenerator(nn.Module):
    """
    生成完整的 IBS Connectivity Matrices (channel-to-channel)

    相比原始 IBSTokenGenerator 的改进：
    - 不再全局平均，保留完整的 (C1, C2) 空间结构
    - 支持 6 个频段（包含 broadband 和 delta）
    - 计算 7 种特征的完整 connectivity matrices
    - 支持消融實驗的特徵選擇 (feature_type)

    输出形状: (B, 6, num_features, C, C) 其中 C=in_channels
    - feature_type="all": num_features=7
    - feature_type="phase": num_features=4 (PLV, PLI, wPLI, Phase_Diff)
    - feature_type="amplitude": num_features=3 (Coherence, Power_Corr, Time_Corr)
    """
    def __init__(
        self,
        in_channels: int,
        sampling_rate: int = 256,
        feature_type: str = "all",  # NEW: "all" | "phase" | "amplitude"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.sampling_rate = sampling_rate
        self.feature_type = feature_type

        # 定义 6 个频段范围 (Hz) - 新增 broadband 和 delta
        self.freq_bands = {
            'broadband': (0.5, 45),   # 全频段
            'delta': (0.5, 4),         # Delta 波
            'theta': (4, 8),           # Theta 波
            'alpha': (8, 13),          # Alpha 波
            'beta': (13, 30),          # Beta 波
            'gamma': (30, 45)          # Gamma 波
        }

        self.band_names = ['broadband', 'delta', 'theta', 'alpha', 'beta', 'gamma']
        self.feature_names = ['PLV', 'PLI', 'wPLI', 'Coherence', 'Power_Corr', 'Phase_Diff', 'Time_Corr']

        # Feature indices for ablation studies
        # [PLV, PLI, wPLI, Coherence, Power_Corr, Phase_Diff, Time_Corr]
        # [  0,   1,    2,         3,          4,          5,         6]
        if feature_type == "phase":
            # Phase-based features: PLV, PLI, wPLI, Phase_Diff
            self.feature_indices = [0, 1, 2, 5]
            self.num_features = 4
        elif feature_type == "amplitude":
            # Amplitude-based features: Coherence, Power_Corr, Time_Corr
            self.feature_indices = [3, 4, 6]
            self.num_features = 3
        else:  # "all"
            self.feature_indices = list(range(7))
            self.num_features = 7

    def bandpass_filter_fft(
        self,
        signal: torch.Tensor,
        low_freq: float,
        high_freq: float
    ) -> torch.Tensor:
        """
        使用 FFT 的 bandpass filter
        Args:
            signal: (B, C, T) - EEG signal
            low_freq: 低频截止 (Hz)
            high_freq: 高频截止 (Hz)
        Returns:
            filtered: (B, C, T) - 滤波后的信号
        """
        B, C, T = signal.shape

        # FFT
        fft_signal = torch.fft.rfft(signal, dim=2)  # (B, C, F)

        # 频率轴
        freqs = torch.fft.rfftfreq(T, d=1.0/self.sampling_rate).to(signal.device)

        # 创建 bandpass mask
        mask = ((freqs >= low_freq) & (freqs <= high_freq)).float()
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, F)

        # 应用 mask
        fft_filtered = fft_signal * mask

        # IFFT 回时域
        filtered = torch.fft.irfft(fft_filtered, n=T, dim=2)

        return filtered

    def compute_phase(self, signal: torch.Tensor) -> torch.Tensor:
        """
        使用 Hilbert transform 计算瞬时相位
        Args:
            signal: (B, C, T)
        Returns:
            phase: (B, C, T)
        """
        # FFT
        fft_signal = torch.fft.fft(signal, dim=2)

        # Hilbert transform in frequency domain
        T = signal.shape[2]
        h = torch.zeros(T, device=signal.device)
        if T % 2 == 0:
            h[0] = h[T // 2] = 1
            h[1:T // 2] = 2
        else:
            h[0] = 1
            h[1:(T + 1) // 2] = 2

        h = h.unsqueeze(0).unsqueeze(0)  # (1, 1, T)

        # Apply Hilbert transform
        analytic_signal = torch.fft.ifft(fft_signal * h, dim=2)

        # Extract phase
        phase = torch.angle(analytic_signal)

        return phase

    def compute_plv_matrix(self, phase1: torch.Tensor, phase2: torch.Tensor) -> torch.Tensor:
        """
        计算 Phase Locking Value 矩阵 (channel-to-channel)
        Args:
            phase1, phase2: (B, C, T) - phase of signals
        Returns:
            PLV matrix: (B, C, C) - channel-to-channel PLV
        """
        B, C, T = phase1.shape
        plv_matrix = torch.zeros(B, C, C, device=phase1.device)

        for i in range(C):
            for j in range(C):
                phase_diff = phase1[:, i, :] - phase2[:, j, :]  # (B, T)
                complex_diff = torch.exp(1j * phase_diff)
                plv = torch.abs(torch.mean(complex_diff, dim=1))  # (B,)
                plv_matrix[:, i, j] = plv

        return plv_matrix

    def compute_pli_matrix(self, phase1: torch.Tensor, phase2: torch.Tensor) -> torch.Tensor:
        """
        计算 Phase Lag Index 矩阵
        Args:
            phase1, phase2: (B, C, T)
        Returns:
            PLI matrix: (B, C, C)
        """
        B, C, T = phase1.shape
        pli_matrix = torch.zeros(B, C, C, device=phase1.device)

        for i in range(C):
            for j in range(C):
                phase_diff = phase1[:, i, :] - phase2[:, j, :]  # (B, T)
                pli = torch.abs(torch.sign(phase_diff).mean(dim=1))  # (B,)
                pli_matrix[:, i, j] = pli

        return pli_matrix

    def compute_wpli_matrix(
        self,
        phase1: torch.Tensor,
        phase2: torch.Tensor,
        power1: torch.Tensor,
        power2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 weighted PLI 矩阵
        Args:
            phase1, phase2: (B, C, T)
            power1, power2: (B, C, T)
        Returns:
            wPLI matrix: (B, C, C)
        """
        B, C, T = phase1.shape
        wpli_matrix = torch.zeros(B, C, C, device=phase1.device)

        for i in range(C):
            for j in range(C):
                phase_diff = phase1[:, i, :] - phase2[:, j, :]  # (B, T)
                weights = (power1[:, i, :] + power2[:, j, :]) / 2  # (B, T)
                weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
                weighted_sign = (torch.sign(phase_diff) * weights).sum(dim=1)  # (B,)
                wpli_matrix[:, i, j] = torch.abs(weighted_sign)

        return wpli_matrix

    def compute_coherence_matrix(self, eeg1: torch.Tensor, eeg2: torch.Tensor) -> torch.Tensor:
        """
        计算 Coherence 矩阵
        Args:
            eeg1, eeg2: (B, C, T)
        Returns:
            Coherence matrix: (B, C, C)
        """
        B, C, T = eeg1.shape
        coh_matrix = torch.zeros(B, C, C, device=eeg1.device)

        # FFT
        fft1 = torch.fft.rfft(eeg1, dim=2)  # (B, C, F)
        fft2 = torch.fft.rfft(eeg2, dim=2)

        for i in range(C):
            for j in range(C):
                # Cross-spectral density
                pxy = fft1[:, i, :] * fft2[:, j, :].conj()  # (B, F)

                # Power spectral densities
                pxx = (fft1[:, i, :] * fft1[:, i, :].conj()).real  # (B, F)
                pyy = (fft2[:, j, :] * fft2[:, j, :].conj()).real

                # Coherence
                coherence = (pxy.abs() ** 2) / (pxx * pyy + 1e-8)  # (B, F)
                coh_matrix[:, i, j] = coherence.mean(dim=1)  # Average over frequencies

        return coh_matrix

    def compute_power_corr_matrix(self, power1: torch.Tensor, power2: torch.Tensor) -> torch.Tensor:
        """
        计算 Power Correlation 矩阵
        Args:
            power1, power2: (B, C, T)
        Returns:
            Power Corr matrix: (B, C, C)
        """
        B, C, T = power1.shape
        corr_matrix = torch.zeros(B, C, C, device=power1.device)

        for i in range(C):
            for j in range(C):
                p1 = power1[:, i, :]  # (B, T)
                p2 = power2[:, j, :]

                # Normalize
                p1_norm = (p1 - p1.mean(dim=1, keepdim=True)) / (p1.std(dim=1, keepdim=True) + 1e-8)
                p2_norm = (p2 - p2.mean(dim=1, keepdim=True)) / (p2.std(dim=1, keepdim=True) + 1e-8)

                # Correlation
                corr = (p1_norm * p2_norm).mean(dim=1)  # (B,)
                corr_matrix[:, i, j] = corr

        return corr_matrix

    def compute_phase_diff_matrix(self, phase1: torch.Tensor, phase2: torch.Tensor) -> torch.Tensor:
        """
        计算 Phase Difference 矩阵
        Args:
            phase1, phase2: (B, C, T)
        Returns:
            Phase Diff matrix: (B, C, C)
        """
        B, C, T = phase1.shape
        diff_matrix = torch.zeros(B, C, C, device=phase1.device)

        for i in range(C):
            for j in range(C):
                phase_diff = torch.abs(phase1[:, i, :] - phase2[:, j, :])  # (B, T)
                diff_matrix[:, i, j] = phase_diff.mean(dim=1)  # (B,)

        return diff_matrix

    def compute_time_corr_matrix(self, eeg1: torch.Tensor, eeg2: torch.Tensor) -> torch.Tensor:
        """
        计算 Time-domain Correlation 矩阵
        Args:
            eeg1, eeg2: (B, C, T)
        Returns:
            Time Corr matrix: (B, C, C)
        """
        B, C, T = eeg1.shape
        corr_matrix = torch.zeros(B, C, C, device=eeg1.device)

        for i in range(C):
            for j in range(C):
                sig1 = eeg1[:, i, :]  # (B, T)
                sig2 = eeg2[:, j, :]

                # Normalize
                sig1_norm = (sig1 - sig1.mean(dim=1, keepdim=True)) / (sig1.std(dim=1, keepdim=True) + 1e-8)
                sig2_norm = (sig2 - sig2.mean(dim=1, keepdim=True)) / (sig2.std(dim=1, keepdim=True) + 1e-8)

                # Correlation
                corr = (sig1_norm * sig2_norm).mean(dim=1)  # (B,)
                corr_matrix[:, i, j] = corr

        return corr_matrix

    def forward(self, eeg1: torch.Tensor, eeg2: torch.Tensor) -> torch.Tensor:
        """
        生成完整的 IBS Connectivity Matrices

        Args:
            eeg1, eeg2: (B, C, T) - EEG signals
        Returns:
            connectivity_matrices: (B, 6, num_features, C, C) - Filtered connectivity matrices
            - feature_type="all": (B, 6, 7, C, C)
            - feature_type="phase": (B, 6, 4, C, C)
            - feature_type="amplitude": (B, 6, 3, C, C)
        """
        B, C, T = eeg1.shape

        # Initialize output tensor: (B, 6 bands, 7 features, C, C)
        all_connectivity_matrices = torch.zeros(
            B, len(self.band_names), len(self.feature_names), C, C,
            device=eeg1.device
        )

        # 对每个频段分别计算 7 种特征的矩阵
        for band_idx, band_name in enumerate(self.band_names):
            low_freq, high_freq = self.freq_bands[band_name]

            # 1. Bandpass filter
            eeg1_band = self.bandpass_filter_fft(eeg1, low_freq, high_freq)
            eeg2_band = self.bandpass_filter_fft(eeg2, low_freq, high_freq)

            # 2. Compute power
            power1 = eeg1_band ** 2
            power2 = eeg2_band ** 2

            # 3. Compute phase
            phase1 = self.compute_phase(eeg1_band)
            phase2 = self.compute_phase(eeg2_band)

            # 4. Compute all 7 features as matrices
            plv_mat = self.compute_plv_matrix(phase1, phase2)  # (B, C, C)
            pli_mat = self.compute_pli_matrix(phase1, phase2)
            wpli_mat = self.compute_wpli_matrix(phase1, phase2, power1, power2)
            coh_mat = self.compute_coherence_matrix(eeg1_band, eeg2_band)
            power_corr_mat = self.compute_power_corr_matrix(power1, power2)
            phase_diff_mat = self.compute_phase_diff_matrix(phase1, phase2)
            time_corr_mat = self.compute_time_corr_matrix(eeg1_band, eeg2_band)

            # 5. Store in output tensor
            all_connectivity_matrices[:, band_idx, 0, :, :] = plv_mat
            all_connectivity_matrices[:, band_idx, 1, :, :] = pli_mat
            all_connectivity_matrices[:, band_idx, 2, :, :] = wpli_mat
            all_connectivity_matrices[:, band_idx, 3, :, :] = coh_mat
            all_connectivity_matrices[:, band_idx, 4, :, :] = power_corr_mat
            all_connectivity_matrices[:, band_idx, 5, :, :] = phase_diff_mat
            all_connectivity_matrices[:, band_idx, 6, :, :] = time_corr_mat

        # 6. Filter features based on feature_type (for ablation studies)
        # self.feature_indices is set in __init__ based on feature_type
        connectivity_matrices = all_connectivity_matrices[:, :, self.feature_indices, :, :]
        # Output shape: (B, 6, num_features, C, C)

        return connectivity_matrices


class RobustIBSTokenizer(nn.Module):
    """
    将 IBS Connectivity Matrices 转换为序列化的 Token Embeddings

    架构：
    1. Flatten & Reshape: (B, 6, num_features, C, C) -> (B, num_tokens, C*C)
    2. Instance Normalization: 放大微弱信号 (可選，用於消融實驗)
    3. Bottleneck MLP: C*C -> 64 -> d_model (去噪)
    4. Type Embedding: 标识不同的频段和特征

    输出: (B, num_tokens, d_model) 可以直接输入 Transformer
    - feature_type="all": num_tokens = 6 bands × 7 features = 42
    - feature_type="phase": num_tokens = 6 bands × 4 features = 24
    - feature_type="amplitude": num_tokens = 6 bands × 3 features = 18
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        use_instance_norm: bool = True,  # NEW: Ablation - disable instance norm
        num_features: int = 7,            # NEW: Dynamic feature count (7, 4, or 3)
        num_bands: int = 6,               # Fixed: 6 frequency bands
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.use_instance_norm = use_instance_norm
        self.num_features = num_features
        self.num_bands = num_bands
        self.num_tokens = num_bands * num_features  # Dynamic: 42, 24, or 18

        matrix_dim = in_channels * in_channels  # C * C = 1024 for C=32

        # Step 2: Instance Normalization (对每个 connectivity matrix 独立归一化)
        # 关键：强制将数值分布拉开，解决数值过小问题
        # NEW: 可通過 use_instance_norm=False 關閉（消融實驗）
        if use_instance_norm:
            self.instance_norm = nn.InstanceNorm1d(matrix_dim, affine=True)

        # Step 3: Bottleneck MLP (去噪 + 降维)
        # 通过 64 维瓶颈层过滤 Grid Pattern 伪影
        self.bottleneck = nn.Sequential(
            nn.Linear(matrix_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, d_model)
        )

        # Step 4: Type Embedding (可学习的位置/类型编码)
        # 让 Transformer 知道每个 token 代表哪个频段和特征
        # NEW: 動態大小，根據 feature_type 調整
        # Shape: (1, num_tokens, d_model) where num_tokens = num_bands * num_features
        self.type_embedding = nn.Parameter(torch.randn(1, self.num_tokens, d_model))

        # 初始化
        nn.init.normal_(self.type_embedding, std=0.02)

    def forward(self, connectivity_matrices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            connectivity_matrices: (B, num_bands, num_features, C, C) - IBS connectivity matrices
        Returns:
            ibs_tokens: (B, num_tokens, d_model) - Tokenized IBS features
        """
        B, num_bands, num_features, C1, C2 = connectivity_matrices.shape
        assert C1 == C2 == self.in_channels, "Channel dimension mismatch"
        assert num_bands == self.num_bands, f"Expected {self.num_bands} bands, got {num_bands}"
        assert num_features == self.num_features, f"Expected {self.num_features} features, got {num_features}"

        # Step 1: Flatten & Reshape
        # (B, num_bands, num_features, C, C) -> (B, num_tokens, C*C)
        x = connectivity_matrices.reshape(B, num_bands * num_features, C1 * C2)

        # Step 2: Instance Normalization (conditional for ablation)
        # 对每个样本的每个 matrix (C*C 维) 独立归一化
        if self.use_instance_norm:
            # Input: (B, num_tokens, 1024) -> permute to (B, 1024, num_tokens) for InstanceNorm1d
            x = x.permute(0, 2, 1)  # (B, 1024, num_tokens)
            x = self.instance_norm(x)  # (B, 1024, num_tokens)
            x = x.permute(0, 2, 1)  # (B, num_tokens, 1024)

        # Step 3: Bottleneck Projection
        # (B, num_tokens, 1024) -> (B, num_tokens, 64) -> (B, num_tokens, d_model)
        x = self.bottleneck(x)  # (B, num_tokens, d_model)

        # Step 4: Add Type Embedding
        # 将频段和特征的身份信息加入
        x = x + self.type_embedding  # (B, num_tokens, d_model)

        return x


class SymmetricFusion(nn.Module):
    """
    Symmetric operator for fusing two representations
    Ensures permutation invariance: f(z1, z2) = f(z2, z1)
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Operations: element-wise sum, product, abs difference
        # Total: d_model * 3
        self.proj = nn.Linear(d_model * 3, d_model)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: (B, d_model)
        Returns:
            fused: (B, d_model)
        """
        # Symmetric operations
        add = z1 + z2
        mul = z1 * z2
        abs_diff = torch.abs(z1 - z2)

        # Concat all symmetric operations and project
        combined = torch.cat([add, mul, abs_diff], dim=-1)
        fused = self.proj(combined)

        return fused


class CrossBrainAttention(nn.Module):
    """
    Cross-attention between two brain sequences
    Z1' = CrossAttn(Z1, Z2), Z2' = CrossAttn(Z2, Z1)
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z1, z2: (B, T, d_model) - encoded sequences
        Returns:
            z1', z2': (B, T, d_model) - cross-attended sequences
        """
        # Z1 attends to Z2
        z1_cross = self.cross_attn(q=z1, k=z2, v=z2)
        z1_out = self.norm(z1 + self.dropout(z1_cross))

        # Z2 attends to Z1
        z2_cross = self.cross_attn(q=z2, k=z1, v=z1)
        z2_out = self.norm(z2 + self.dropout(z2_cross))

        return z1_out, z2_out


class DualEEGTransformer(nn.Module):
    """
    Dual-stream EEG Transformer with Inter-Brain Synchrony token

    Architecture:
    1. Temporal convolution frontend for each player
    2. IBS token generation from cross-brain features (optional)
    3. Token sequence: [CLS, IBS, H1(1), ..., H1(T̃)] or [CLS, H1(1), ..., H1(T̃)]
    4. Shared Transformer Encoder (Siamese)
    5. Cross-brain attention (optional)
    6. Symmetric fusion and classification

    Ablation Study Parameters:
    - use_ibs: Enable/disable IBS tokens
    - use_cross_attention: Enable/disable CrossBrainAttention
    - ibs_instance_norm: Enable/disable instance norm in RobustIBSTokenizer
    - ibs_feature_type: "all" | "phase" | "amplitude"
    """
    def __init__(
        self,
        in_channels: int = 62,  # Number of EEG channels
        num_classes: int = 3,    # Single, Competition, Cooperation
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 2048,
        conv_kernel_size: int = 25,
        conv_stride: int = 4,
        conv_layers: int = 2,
        sampling_rate: int = 256,  # EEG sampling rate
        # Spectrogram parameters
        use_spectrogram: bool = True,  # Enable/disable Spectrogram tokens
        spec_n_fft: int = 128,
        spec_hop_length: int = 64,
        spec_freq_bins: int = 64,
        # IBS tokenization mode
        use_robust_ibs: bool = True,  # Use new RobustIBSTokenizer (recommended)
        # ===== Ablation Study Parameters (NEW) =====
        use_ibs: bool = True,               # A: Enable/disable IBS tokens completely
        use_cross_attention: bool = True,   # C: Enable/disable CrossBrainAttention
        ibs_instance_norm: bool = True,     # B: Instance norm in RobustIBSTokenizer
        ibs_feature_type: str = "all",      # B: Feature type: "all" | "phase" | "amplitude"
    ):
        super().__init__()

        self.d_model = d_model
        self.in_channels = in_channels
        self.use_spectrogram = use_spectrogram
        self.use_robust_ibs = use_robust_ibs

        # ===== Ablation flags =====
        self.use_ibs = use_ibs
        self.use_cross_attention = use_cross_attention
        self.ibs_feature_type = ibs_feature_type

        # Calculate dynamic IBS token count based on feature_type
        feature_counts = {"all": 7, "phase": 4, "amplitude": 3}
        self.num_ibs_features = feature_counts.get(ibs_feature_type, 7)
        if use_ibs:
            if use_robust_ibs:
                self.num_ibs_tokens = 6 * self.num_ibs_features  # 42, 24, or 18
            else:
                self.num_ibs_tokens = 1  # Scalar IBS
        else:
            self.num_ibs_tokens = 0  # No IBS tokens

        # Temporal convolution frontend
        self.temporal_conv = TemporalConvFrontend(
            in_channels, d_model, conv_kernel_size, conv_stride, conv_layers
        )

        # Spectrogram token generator
        if use_spectrogram:
            self.spectrogram_generator = SpectrogramTokenGenerator(
                in_channels, d_model, spec_n_fft, spec_hop_length,
                sampling_rate, spec_freq_bins
            )

        # IBS token generation (conditional for ablation)
        if use_ibs:
            if use_robust_ibs:
                # NEW: Generate full connectivity matrices + robust tokenization
                self.ibs_matrix_generator = IBSConnectivityMatrixGenerator(
                    in_channels, sampling_rate, feature_type=ibs_feature_type
                )
                self.ibs_tokenizer = RobustIBSTokenizer(
                    in_channels, d_model,
                    use_instance_norm=ibs_instance_norm,
                    num_features=self.num_ibs_features
                )
            else:
                # OLD: Original IBS token generator (scalar features, global average)
                self.ibs_generator = IBSTokenGenerator(in_channels, d_model, sampling_rate, use_layernorm=False)

            # IBS token 專用分類頭（直接監督）- only when IBS is enabled
            self.ibs_classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(d_model // 2, num_classes)
            )

        # CLS tokens (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embedding
        self.pos_embed = PositionalEmbedding(max_len, d_model, mode='learned')

        # Shared Transformer Encoder (Siamese)
        self.encoder = TransformerEncoder(
            d_model, num_layers, num_heads, d_ff, dropout, dropout
        )

        # Cross-brain attention (conditional for ablation)
        if use_cross_attention:
            self.cross_attn = CrossBrainAttention(d_model, num_heads, dropout)

        # Symmetric fusion
        self.symmetric_fusion = SymmetricFusion(d_model)

        # Main classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # [f_pair, mp1', mp2']
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes

    def forward(
        self,
        eeg1: torch.Tensor,
        eeg2: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            eeg1, eeg2: (B, C, T) - EEG signals for player 1 and 2
            labels: (B,) - class labels (optional, for computing loss)

        Returns:
            dict with 'logits', 'loss' (if labels provided)
        """
        B = eeg1.shape[0]

        # 1. Temporal convolution frontend
        h1 = self.temporal_conv(eeg1)  # (B, T̃, d)
        h2 = self.temporal_conv(eeg2)  # (B, T̃, d)

        # 2. Generate IBS tokens (conditional for ablation)
        ibs_tokens = None
        if self.use_ibs:
            if self.use_robust_ibs:
                # NEW: Full connectivity matrices + robust tokenization
                # Step 1: Generate connectivity matrices (B, 6, num_features, C, C)
                connectivity_matrices = self.ibs_matrix_generator(eeg1, eeg2)
                # Step 2: Tokenize into sequence (B, num_ibs_tokens, d_model)
                ibs_tokens = self.ibs_tokenizer(connectivity_matrices)
            else:
                # OLD: Scalar IBS features
                ibs_token_single = self.ibs_generator(eeg1, eeg2)  # (B, d)
                ibs_tokens = ibs_token_single.unsqueeze(1)  # (B, 1, d)

        # 2.5 Generate Spectrogram tokens
        spec1 = None
        spec2 = None
        if self.use_spectrogram:
            spec1 = self.spectrogram_generator(eeg1)  # (B, C, d)
            spec2 = self.spectrogram_generator(eeg2)  # (B, C, d)

        # 3. Build sequences dynamically based on ablation settings
        # Possible formats:
        # - Full: [CLS, IBS(1..N), Spec(1..C), H(1..T̃)]
        # - No IBS: [CLS, Spec(1..C), H(1..T̃)]
        # - No Spec: [CLS, IBS(1..N), H(1..T̃)]
        # - Baseline: [CLS, H(1..T̃)]
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d)

        # Build sequence by concatenating available components
        seq1_components = [cls]
        seq2_components = [cls]

        if ibs_tokens is not None:
            seq1_components.append(ibs_tokens)
            seq2_components.append(ibs_tokens)

        if self.use_spectrogram and spec1 is not None:
            seq1_components.append(spec1)
            seq2_components.append(spec2)

        seq1_components.append(h1)
        seq2_components.append(h2)

        seq1 = torch.cat(seq1_components, dim=1)
        seq2 = torch.cat(seq2_components, dim=1)

        # Add positional embedding
        seq1 = self.pos_embed(seq1)
        seq2 = self.pos_embed(seq2)

        # 4. Shared Transformer Encoder (Siamese)
        z1 = self.encoder(seq1)
        z2 = self.encoder(seq2)

        # 5. Cross-brain attention (conditional for ablation)
        if self.use_cross_attention:
            z1_cross, z2_cross = self.cross_attn(z1, z2)
        else:
            # Skip cross-brain attention (ablation: No Cross-Attention)
            z1_cross, z2_cross = z1, z2

        # 6. Extract CLS tokens AFTER cross-attention
        cls1 = z1_cross[:, 0, :]  # (B, d)
        cls2 = z2_cross[:, 0, :]  # (B, d)

        # Calculate dynamic offset for mean pooling based on ablation settings
        # offset = CLS(1) + IBS(num_ibs_tokens) + Spec(in_channels)
        offset = 1  # CLS token
        if self.use_ibs:
            offset += self.num_ibs_tokens
        if self.use_spectrogram:
            offset += self.in_channels

        # Mean pooling (only use temporal conv tokens)
        mp1 = z1_cross[:, offset:, :].mean(dim=1)  # (B, d)
        mp2 = z2_cross[:, offset:, :].mean(dim=1)  # (B, d)

        # 7. Symmetric fusion
        f_pair = self.symmetric_fusion(cls1, cls2)  # (B, d)

        # 8. Concatenate and classify
        z_fuse = torch.cat([f_pair, mp1, mp2], dim=-1)  # (B, 3*d)
        logits = self.classifier(z_fuse)  # (B, num_classes)

        # 9. IBS token classification (only if IBS is enabled)
        ibs_logits = None
        ibs_token_output = None
        if self.use_ibs:
            if self.use_robust_ibs:
                # Pool all IBS tokens for classification
                # Extract IBS tokens from sequence: indices [1:1+num_ibs_tokens] (after CLS)
                ibs_tokens_from_seq = z1_cross[:, 1:1+self.num_ibs_tokens, :]
                ibs_pooled = ibs_tokens_from_seq.mean(dim=1)  # (B, d)
                ibs_logits = self.ibs_classifier(ibs_pooled)
                ibs_token_output = ibs_pooled
            else:
                # Single IBS token
                ibs_token_from_seq = z1_cross[:, 1, :]  # (B, d)
                ibs_logits = self.ibs_classifier(ibs_token_from_seq)
                ibs_token_output = ibs_token_from_seq

        output = {
            'logits': logits,
            'cls1': cls1,
            'cls2': cls2,
        }

        # Only include IBS-related outputs if IBS is enabled
        if self.use_ibs:
            output['ibs_logits'] = ibs_logits
            output['ibs_token'] = ibs_token_output

        if labels is not None:
            loss_ce = F.cross_entropy(logits, labels)
            output['loss'] = loss_ce
            output['loss_ce'] = loss_ce

            # IBS classification loss (only if IBS is enabled)
            if self.use_ibs and ibs_logits is not None:
                loss_ibs_cls = F.cross_entropy(ibs_logits, labels)
                output['loss_ibs_cls'] = loss_ibs_cls

        return output

    def compute_symmetry_loss(self, cls1: torch.Tensor, cls2: torch.Tensor) -> torch.Tensor:
        """
        Symmetry loss: encourages similar CLS representations
        Only use when task requires symmetry (e.g., cooperation)
        """
        return F.mse_loss(cls1, cls2)

    def compute_ibs_alignment_loss(
        self,
        ibs_token: torch.Tensor,
        cls1: torch.Tensor,
        cls2: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        IBS alignment loss using InfoNCE
        Encourages IBS token to align with CLS tokens from the same window

        Args:
            ibs_token: (B, d) - IBS tokens
            cls1, cls2: (B, d) - CLS tokens
            temperature: softmax temperature

        Returns:
            loss: scalar
        """
        B = ibs_token.shape[0]

        # Normalize
        ibs_norm = F.normalize(ibs_token, dim=-1)
        cls1_norm = F.normalize(cls1, dim=-1)
        cls2_norm = F.normalize(cls2, dim=-1)

        # Positive pairs: IBS with corresponding CLS1 and CLS2
        pos_sim1 = (ibs_norm * cls1_norm).sum(dim=-1) / temperature  # (B,)
        pos_sim2 = (ibs_norm * cls2_norm).sum(dim=-1) / temperature  # (B,)

        # Negative pairs: IBS with all other CLS tokens in the batch
        all_cls = torch.cat([cls1_norm, cls2_norm], dim=0)  # (2B, d)
        neg_sim = torch.matmul(ibs_norm, all_cls.T) / temperature  # (B, 2B)

        # InfoNCE loss
        # For each IBS, positive examples are cls1 and cls2 from same sample
        # Negative examples are all other cls tokens in batch

        # Simplified: treat cls1 as positive, all others as negative
        labels = torch.arange(B, device=ibs_token.device)
        loss = F.cross_entropy(neg_sim, labels)

        return loss

    def compute_ibs_contrastive_loss(
        self,
        ibs_tokens: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Contrastive learning loss for IBS tokens
        強迫同類別的 IBS tokens 接近，不同類別的 IBS tokens 遠離

        This directly optimizes angular separation which is our evaluation metric!

        Args:
            ibs_tokens: (B, d) - IBS token embeddings
            labels: (B,) - Class labels
            temperature: Temperature parameter for InfoNCE

        Returns:
            loss: Contrastive loss (scalar)
        """
        B = ibs_tokens.shape[0]

        # Normalize to unit sphere (angular distance is what matters)
        ibs_tokens = F.normalize(ibs_tokens, p=2, dim=1)  # (B, d)

        # Compute similarity matrix (cosine similarity)
        sim_matrix = torch.matmul(ibs_tokens, ibs_tokens.t()) / temperature  # (B, B)

        # Create label matrix: 1 if same class, 0 otherwise
        labels_expanded = labels.unsqueeze(1)  # (B, 1)
        label_matrix = (labels_expanded == labels_expanded.t()).float()  # (B, B)

        # Mask out diagonal (self-similarity)
        mask = torch.eye(B, device=ibs_tokens.device).bool()
        label_matrix = label_matrix.masked_fill(mask, 0)

        # Compute positive and negative masks
        pos_mask = label_matrix  # Same class
        neg_mask = 1 - label_matrix  # Different class

        # Check if any sample has positive pairs
        has_pos = (pos_mask.sum(dim=1) > 0)

        if has_pos.sum() == 0:
            # No positive pairs in this batch (all different classes)
            return torch.tensor(0.0, device=ibs_tokens.device)

        # Numerator: sum of exp(sim) for positive pairs
        exp_sim = torch.exp(sim_matrix)
        pos_sim = (exp_sim * pos_mask).sum(dim=1)  # (B,)

        # Denominator: sum of exp(sim) for all pairs (excluding self)
        exp_sim_masked = exp_sim.masked_fill(mask, 0)
        all_sim = exp_sim_masked.sum(dim=1)  # (B,)

        # Loss: -log(pos / all)
        # 對於同類樣本，希望 pos_sim 大；對於不同類，希望在 all_sim 中占比小
        loss = -torch.log(pos_sim / (all_sim + 1e-8) + 1e-8)

        # Only compute loss for samples that have at least one positive pair
        if has_pos.sum() > 0:
            loss = loss[has_pos].mean()
        else:
            loss = torch.tensor(0.0, device=ibs_tokens.device)

        return loss
