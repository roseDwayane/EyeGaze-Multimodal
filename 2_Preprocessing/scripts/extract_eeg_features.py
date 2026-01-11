"""
EEG Feature Extraction Script for Hyperscanning Data

Extracts comprehensive features from dual-person EEG trials:
- Time domain: Raw signals
- Frequency domain: PSD via Welch
- Band energy: Power in 5 frequency bands (Delta, Theta, Alpha, Beta, Gamma)
- Intra-brain connectivity: 7 metrics x 5 bands x 32x32 (per player)
- Inter-brain connectivity: 7 metrics x 5 bands x 32x32 (between players)

Connectivity Metrics:
    0: Pearson Correlation (time-domain)
    1: Power Correlation (amplitude envelope)
    2: PLV (Phase Locking Value)
    3: PLI (Phase Lag Index)
    4: wPLI (Weighted Phase Lag Index)
    5: Spectral Coherence
    6: Mean Phase Difference

Frequency Bands:
    0: Delta (0.5-4 Hz)
    1: Theta (4-7 Hz)
    2: Alpha (8-12 Hz)
    3: Beta (12-28 Hz)
    4: Gamma (28-50 Hz)

Output Structure (per trial .npy):
    {
        'time_domain': (2, 32, T),           # Players x Channels x Time
        'freq_domain': (2, 32, num_freqs),   # PSD
        'freq_bins': (num_freqs,),           # Frequency axis
        'bands_energy': (2, 32, 5),          # Band power
        'intra_con': (2, 7, 5, 32, 32),      # Intra-brain connectivity
        'inter_con': (7, 5, 32, 32),         # Inter-brain connectivity
        'metadata': dict                      # Trial info
    }

Usage:
    python 2_Preprocessing/scripts/extract_eeg_features.py
    python 2_Preprocessing/scripts/extract_eeg_features.py --workers 8 --resume
    python 2_Preprocessing/scripts/extract_eeg_features.py --input-dir path/to/eeg --output-dir path/to/features
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# scipy for signal processing
try:
    from scipy import signal
    from scipy.signal import butter, filtfilt, hilbert, welch, csd
    from scipy.fft import rfft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[Error] scipy is required for this script. Please install: pip install scipy")
    sys.exit(1)

# joblib for parallel processing
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("[Warning] joblib not installed. Parallel processing disabled.")

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Constants
# =============================================================================

FREQUENCY_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 7),
    'alpha': (8, 12),
    'beta': (12, 28),
    'gamma': (28, 50),
}

BAND_NAMES = list(FREQUENCY_BANDS.keys())
BAND_INDICES = {name: i for i, name in enumerate(BAND_NAMES)}

METRIC_NAMES = [
    'pearson',      # 0: Pearson Correlation
    'power_corr',   # 1: Power Correlation
    'plv',          # 2: Phase Locking Value
    'pli',          # 3: Phase Lag Index
    'wpli',         # 4: Weighted PLI
    'coherence',    # 5: Spectral Coherence
    'phase_diff',   # 6: Mean Phase Difference
]

METRIC_INDICES = {name: i for i, name in enumerate(METRIC_NAMES)}

CLASS_MAPPING = {
    'Single': 0,
    'Competition': 1,
    'Comp': 1,
    'Cooperation': 2,
    'Coop': 2,
}


# =============================================================================
# Preprocessing Functions
# =============================================================================

def load_eeg_csv(path: Path, num_channels: int = 32) -> np.ndarray:
    """
    Load EEG from CSV file.

    Args:
        path: Path to CSV file
        num_channels: Expected number of channels

    Returns:
        eeg: (C, T) numpy array, float32
    """
    df = pd.read_csv(path, header=None)
    eeg = df.values.astype(np.float32)

    # Ensure shape is (C, T)
    if eeg.ndim == 1:
        eeg = eeg.reshape(1, -1)
    elif eeg.shape[0] > eeg.shape[1]:
        eeg = eeg.T

    # Adjust channels
    if eeg.shape[0] > num_channels:
        eeg = eeg[:num_channels, :]
    elif eeg.shape[0] < num_channels:
        pad = np.zeros((num_channels - eeg.shape[0], eeg.shape[1]), dtype=np.float32)
        eeg = np.vstack([eeg, pad])

    return eeg


def bandpass_filter(
    eeg: np.ndarray,
    low_freq: float,
    high_freq: float,
    sampling_rate: int = 250,
    order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to EEG signal.

    Args:
        eeg: (C, T) EEG signal
        low_freq: Low cutoff frequency (Hz)
        high_freq: High cutoff frequency (Hz)
        sampling_rate: Sampling rate (Hz)
        order: Filter order

    Returns:
        Filtered EEG signal (C, T)
    """
    nyquist = sampling_rate / 2
    low = max(low_freq / nyquist, 0.001)
    high = min(high_freq / nyquist, 0.99)

    try:
        b, a = butter(order, [low, high], btype='band')
        eeg_filtered = filtfilt(b, a, eeg, axis=1)
        return eeg_filtered.astype(np.float32)
    except Exception as e:
        logging.warning(f"Bandpass filter failed: {e}")
        return eeg


def common_average_reference(eeg: np.ndarray) -> np.ndarray:
    """Apply Common Average Reference (CAR)."""
    car = eeg.mean(axis=0, keepdims=True)
    return (eeg - car).astype(np.float32)


def preprocess_eeg(
    eeg: np.ndarray,
    sampling_rate: int = 250,
    low_freq: float = 0.5,
    high_freq: float = 50.0
) -> np.ndarray:
    """
    Basic preprocessing: Bandpass filter + CAR.

    Args:
        eeg: (C, T) raw EEG signal
        sampling_rate: Sampling rate in Hz
        low_freq: Bandpass low cutoff
        high_freq: Bandpass high cutoff

    Returns:
        Preprocessed EEG signal (C, T)
    """
    # Handle NaN/Inf values
    if np.isnan(eeg).any():
        eeg = np.nan_to_num(eeg, nan=0.0)
    if np.isinf(eeg).any():
        eeg = np.clip(eeg, -1e6, 1e6)

    # Bandpass filter
    eeg = bandpass_filter(eeg, low_freq, high_freq, sampling_rate)

    # Common Average Reference
    eeg = common_average_reference(eeg)

    return eeg


# =============================================================================
# Frequency Domain Features
# =============================================================================

def compute_psd(
    eeg: np.ndarray,
    sampling_rate: int = 250,
    nperseg: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method.

    Args:
        eeg: (C, T) EEG signal
        sampling_rate: Sampling rate
        nperseg: Segment length for Welch

    Returns:
        freqs: (num_freqs,) frequency bins
        psd: (C, num_freqs) power spectral density
    """
    # Welch handles all channels at once
    freqs, psd = welch(eeg, fs=sampling_rate, nperseg=nperseg, axis=1)
    return freqs.astype(np.float32), psd.astype(np.float32)


def compute_band_energy(
    psd: np.ndarray,
    freqs: np.ndarray,
    bands: Dict[str, Tuple[float, float]]
) -> np.ndarray:
    """
    Compute average power in each frequency band.

    Args:
        psd: (C, num_freqs) power spectral density
        freqs: (num_freqs,) frequency bins
        bands: Dict of band_name -> (low, high)

    Returns:
        (C, num_bands) band energy for each channel
    """
    n_channels = psd.shape[0]
    n_bands = len(bands)
    band_energy = np.zeros((n_channels, n_bands), dtype=np.float32)

    for i, (band_name, (low, high)) in enumerate(bands.items()):
        mask = (freqs >= low) & (freqs <= high)
        if mask.sum() > 0:
            band_energy[:, i] = psd[:, mask].mean(axis=1)

    return band_energy


# =============================================================================
# Hilbert Transform
# =============================================================================

def compute_analytic_signal(band_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute analytic signal using Hilbert transform.

    Args:
        band_signal: (C, T) band-filtered signal

    Returns:
        amplitude: (C, T) instantaneous amplitude
        phase: (C, T) instantaneous phase
    """
    # Hilbert handles all channels at once (axis=1)
    analytic = hilbert(band_signal, axis=1)
    amplitude = np.abs(analytic).astype(np.float32)
    phase = np.angle(analytic).astype(np.float32)
    return amplitude, phase


# =============================================================================
# Connectivity Metrics (Vectorized)
# =============================================================================

def compute_pearson_correlation(x: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation matrix.

    Args:
        x: (C, T) time series

    Returns:
        (C, C) correlation matrix
    """
    # Z-score normalize
    x_mean = x.mean(axis=1, keepdims=True)
    x_std = x.std(axis=1, keepdims=True) + 1e-8
    x_norm = (x - x_mean) / x_std

    # Matrix multiply: (C, T) @ (T, C) -> (C, C)
    n_timepoints = x.shape[1]
    corr = (x_norm @ x_norm.T) / n_timepoints

    return corr.astype(np.float32)


def compute_power_correlation(amplitude: np.ndarray) -> np.ndarray:
    """
    Compute correlation of instantaneous amplitude (power envelope).

    Args:
        amplitude: (C, T) from Hilbert transform

    Returns:
        (C, C) power correlation matrix
    """
    return compute_pearson_correlation(amplitude)


def compute_plv(phase: np.ndarray) -> np.ndarray:
    """
    Compute Phase Locking Value.
    PLV = |mean(exp(i * (phi_i - phi_j)))|

    Args:
        phase: (C, T) instantaneous phase from Hilbert

    Returns:
        (C, C) PLV matrix
    """
    # Phase difference: (C, 1, T) - (1, C, T) -> (C, C, T)
    phase_diff = phase[:, np.newaxis, :] - phase[np.newaxis, :, :]

    # Complex exponential and mean over time
    complex_phase = np.exp(1j * phase_diff)
    plv = np.abs(complex_phase.mean(axis=2))

    return plv.astype(np.float32)


def compute_pli(phase: np.ndarray) -> np.ndarray:
    """
    Compute Phase Lag Index.
    PLI = |mean(sign(sin(phi_i - phi_j)))|

    Args:
        phase: (C, T) instantaneous phase

    Returns:
        (C, C) PLI matrix
    """
    # Phase difference: (C, C, T)
    phase_diff = phase[:, np.newaxis, :] - phase[np.newaxis, :, :]

    # Sign of sine (detects phase lead/lag)
    sign_sin = np.sign(np.sin(phase_diff))

    # Mean over time, then absolute value
    pli = np.abs(sign_sin.mean(axis=2))

    return pli.astype(np.float32)


def compute_wpli(phase: np.ndarray) -> np.ndarray:
    """
    Compute Weighted Phase Lag Index.
    wPLI = |mean(|sin(dphi)| * sign(sin(dphi)))| / mean(|sin(dphi)|)

    Args:
        phase: (C, T) instantaneous phase

    Returns:
        (C, C) wPLI matrix
    """
    # Phase difference: (C, C, T)
    phase_diff = phase[:, np.newaxis, :] - phase[np.newaxis, :, :]

    sin_dphi = np.sin(phase_diff)
    abs_sin = np.abs(sin_dphi)

    # Numerator: weighted by |sin|
    numerator = np.abs((abs_sin * np.sign(sin_dphi)).mean(axis=2))

    # Denominator: mean of |sin|
    denominator = abs_sin.mean(axis=2) + 1e-8

    wpli = numerator / denominator

    return wpli.astype(np.float32)


def compute_coherence_fast(
    x: np.ndarray,
    sampling_rate: int = 250,
    nperseg: int = 256
) -> np.ndarray:
    """
    Compute spectral coherence using FFT-based approach.
    Mean coherence across all frequencies.

    Args:
        x: (C, T) time series (band-filtered)
        sampling_rate: Sampling rate
        nperseg: Segment length

    Returns:
        (C, C) mean coherence matrix
    """
    n_channels, n_timepoints = x.shape

    # Ensure we have enough data
    if n_timepoints < nperseg:
        nperseg = n_timepoints // 2
        if nperseg < 4:
            return np.eye(n_channels, dtype=np.float32)

    n_segments = n_timepoints // nperseg
    if n_segments < 1:
        return np.eye(n_channels, dtype=np.float32)

    # Reshape for segment-wise FFT: (C, n_segments, nperseg)
    segments = x[:, :n_segments * nperseg].reshape(n_channels, n_segments, nperseg)

    # Apply Hanning window
    window = np.hanning(nperseg).astype(np.float32)
    segments = segments * window

    # FFT of all segments: (C, n_segments, nperseg//2+1)
    X = rfft(segments, axis=2)

    # Auto-spectra: mean(|X|^2) over segments -> (C, freqs)
    Pxx = (np.abs(X) ** 2).mean(axis=1)

    # Cross-spectra: (C, C, freqs) using einsum
    # X[i,s,f] * conj(X[j,s,f]) averaged over segments
    Pxy = np.einsum('isf,jsf->ijf', X, np.conj(X)) / n_segments

    # Coherence = |Pxy|^2 / (Pxx_i * Pxx_j)
    Pxx_outer = Pxx[:, np.newaxis, :] * Pxx[np.newaxis, :, :]  # (C, C, freqs)
    coherence = np.abs(Pxy) ** 2 / (Pxx_outer + 1e-8)

    # Mean coherence across frequencies
    mean_coh = coherence.mean(axis=2).real

    return mean_coh.astype(np.float32)


def compute_mean_phase_diff(phase: np.ndarray) -> np.ndarray:
    """
    Compute circular mean of phase difference.

    Args:
        phase: (C, T) instantaneous phase

    Returns:
        (C, C) mean phase difference (in radians)
    """
    # Phase difference: (C, C, T)
    phase_diff = phase[:, np.newaxis, :] - phase[np.newaxis, :, :]

    # Circular mean using complex exponentials
    mean_complex = np.exp(1j * phase_diff).mean(axis=2)
    mean_phase = np.angle(mean_complex)

    return mean_phase.astype(np.float32)


# =============================================================================
# Inter-Brain Connectivity (Between Two Players)
# =============================================================================

def compute_inter_pearson(x_a: np.ndarray, x_b: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation between Player A and B channels.

    Args:
        x_a: (C, T) Player A time series
        x_b: (C, T) Player B time series

    Returns:
        (C, C) correlation matrix (A channels x B channels)
    """
    # Z-score normalize
    a_mean = x_a.mean(axis=1, keepdims=True)
    a_std = x_a.std(axis=1, keepdims=True) + 1e-8
    a_norm = (x_a - a_mean) / a_std

    b_mean = x_b.mean(axis=1, keepdims=True)
    b_std = x_b.std(axis=1, keepdims=True) + 1e-8
    b_norm = (x_b - b_mean) / b_std

    # Cross-correlation: (C_A, T) @ (T, C_B) -> (C_A, C_B)
    n_timepoints = x_a.shape[1]
    corr = (a_norm @ b_norm.T) / n_timepoints

    return corr.astype(np.float32)


def compute_inter_power_correlation(amp_a: np.ndarray, amp_b: np.ndarray) -> np.ndarray:
    """
    Compute power correlation between Player A and B.

    Args:
        amp_a: (C, T) amplitude envelope of Player A
        amp_b: (C, T) amplitude envelope of Player B

    Returns:
        (C, C) power correlation matrix
    """
    return compute_inter_pearson(amp_a, amp_b)


def compute_inter_plv(phase_a: np.ndarray, phase_b: np.ndarray) -> np.ndarray:
    """
    Compute PLV between Player A and B.

    Args:
        phase_a: (C, T) phase of Player A
        phase_b: (C, T) phase of Player B

    Returns:
        (C, C) inter-brain PLV matrix
    """
    # Phase difference: A channels x B channels
    # (C_A, 1, T) - (1, C_B, T) -> (C_A, C_B, T)
    phase_diff = phase_a[:, np.newaxis, :] - phase_b[np.newaxis, :, :]

    # PLV
    plv = np.abs(np.exp(1j * phase_diff).mean(axis=2))

    return plv.astype(np.float32)


def compute_inter_pli(phase_a: np.ndarray, phase_b: np.ndarray) -> np.ndarray:
    """
    Compute PLI between Player A and B.

    Args:
        phase_a: (C, T) phase of Player A
        phase_b: (C, T) phase of Player B

    Returns:
        (C, C) inter-brain PLI matrix
    """
    phase_diff = phase_a[:, np.newaxis, :] - phase_b[np.newaxis, :, :]
    pli = np.abs(np.sign(np.sin(phase_diff)).mean(axis=2))
    return pli.astype(np.float32)


def compute_inter_wpli(phase_a: np.ndarray, phase_b: np.ndarray) -> np.ndarray:
    """
    Compute wPLI between Player A and B.

    Args:
        phase_a: (C, T) phase of Player A
        phase_b: (C, T) phase of Player B

    Returns:
        (C, C) inter-brain wPLI matrix
    """
    phase_diff = phase_a[:, np.newaxis, :] - phase_b[np.newaxis, :, :]

    sin_dphi = np.sin(phase_diff)
    abs_sin = np.abs(sin_dphi)

    numerator = np.abs((abs_sin * np.sign(sin_dphi)).mean(axis=2))
    denominator = abs_sin.mean(axis=2) + 1e-8

    wpli = numerator / denominator
    return wpli.astype(np.float32)


def compute_inter_coherence(
    x_a: np.ndarray,
    x_b: np.ndarray,
    sampling_rate: int = 250,
    nperseg: int = 256
) -> np.ndarray:
    """
    Compute spectral coherence between Player A and B.

    Args:
        x_a: (C, T) Player A time series
        x_b: (C, T) Player B time series
        sampling_rate: Sampling rate
        nperseg: Segment length

    Returns:
        (C, C) inter-brain coherence matrix
    """
    n_channels = x_a.shape[0]
    n_timepoints = x_a.shape[1]

    # Ensure we have enough data
    if n_timepoints < nperseg:
        nperseg = n_timepoints // 2
        if nperseg < 4:
            return np.zeros((n_channels, n_channels), dtype=np.float32)

    n_segments = n_timepoints // nperseg
    if n_segments < 1:
        return np.zeros((n_channels, n_channels), dtype=np.float32)

    # Reshape for segment-wise FFT
    seg_a = x_a[:, :n_segments * nperseg].reshape(n_channels, n_segments, nperseg)
    seg_b = x_b[:, :n_segments * nperseg].reshape(n_channels, n_segments, nperseg)

    # Apply Hanning window
    window = np.hanning(nperseg).astype(np.float32)
    seg_a = seg_a * window
    seg_b = seg_b * window

    # FFT
    X_a = rfft(seg_a, axis=2)
    X_b = rfft(seg_b, axis=2)

    # Auto-spectra
    Paa = (np.abs(X_a) ** 2).mean(axis=1)  # (C, freqs)
    Pbb = (np.abs(X_b) ** 2).mean(axis=1)  # (C, freqs)

    # Cross-spectra: (C_A, C_B, freqs)
    Pab = np.einsum('isf,jsf->ijf', X_a, np.conj(X_b)) / n_segments

    # Coherence = |Pab|^2 / (Paa * Pbb)
    Pxx_outer = Paa[:, np.newaxis, :] * Pbb[np.newaxis, :, :]
    coherence = np.abs(Pab) ** 2 / (Pxx_outer + 1e-8)

    # Mean coherence across frequencies
    mean_coh = coherence.mean(axis=2).real

    return mean_coh.astype(np.float32)


def compute_inter_phase_diff(phase_a: np.ndarray, phase_b: np.ndarray) -> np.ndarray:
    """
    Compute mean phase difference between Player A and B.

    Args:
        phase_a: (C, T) phase of Player A
        phase_b: (C, T) phase of Player B

    Returns:
        (C, C) mean phase difference matrix
    """
    phase_diff = phase_a[:, np.newaxis, :] - phase_b[np.newaxis, :, :]
    mean_complex = np.exp(1j * phase_diff).mean(axis=2)
    mean_phase = np.angle(mean_complex)
    return mean_phase.astype(np.float32)


# =============================================================================
# Full Connectivity Computation
# =============================================================================

def compute_intra_connectivity(
    eeg: np.ndarray,
    bands: Dict[str, Tuple[float, float]],
    sampling_rate: int = 250
) -> np.ndarray:
    """
    Compute all intra-brain connectivity metrics for one player.

    Args:
        eeg: (C, T) preprocessed EEG
        bands: Frequency band definitions
        sampling_rate: Sampling rate

    Returns:
        (7, 5, C, C) connectivity matrices (Metrics x Bands x Ch x Ch)
    """
    n_channels = eeg.shape[0]
    n_metrics = len(METRIC_NAMES)
    n_bands = len(bands)

    intra_con = np.zeros((n_metrics, n_bands, n_channels, n_channels), dtype=np.float32)

    for band_idx, (band_name, (low, high)) in enumerate(bands.items()):
        # Band-pass filter
        band_signal = bandpass_filter(eeg, low, high, sampling_rate)

        # Hilbert transform for phase and amplitude
        amplitude, phase = compute_analytic_signal(band_signal)

        # Compute all 7 metrics
        intra_con[0, band_idx] = compute_pearson_correlation(band_signal)
        intra_con[1, band_idx] = compute_power_correlation(amplitude)
        intra_con[2, band_idx] = compute_plv(phase)
        intra_con[3, band_idx] = compute_pli(phase)
        intra_con[4, band_idx] = compute_wpli(phase)
        intra_con[5, band_idx] = compute_coherence_fast(band_signal, sampling_rate)
        intra_con[6, band_idx] = compute_mean_phase_diff(phase)

    return intra_con


def compute_inter_connectivity(
    eeg1: np.ndarray,
    eeg2: np.ndarray,
    bands: Dict[str, Tuple[float, float]],
    sampling_rate: int = 250
) -> np.ndarray:
    """
    Compute all inter-brain connectivity metrics between two players.

    Args:
        eeg1: (C, T) preprocessed EEG of Player 1
        eeg2: (C, T) preprocessed EEG of Player 2
        bands: Frequency band definitions
        sampling_rate: Sampling rate

    Returns:
        (7, 5, C, C) connectivity matrices (Metrics x Bands x Ch1 x Ch2)
    """
    n_channels = eeg1.shape[0]
    n_metrics = len(METRIC_NAMES)
    n_bands = len(bands)

    inter_con = np.zeros((n_metrics, n_bands, n_channels, n_channels), dtype=np.float32)

    for band_idx, (band_name, (low, high)) in enumerate(bands.items()):
        # Band-pass filter for both players
        band1 = bandpass_filter(eeg1, low, high, sampling_rate)
        band2 = bandpass_filter(eeg2, low, high, sampling_rate)

        # Hilbert transform
        amp1, phase1 = compute_analytic_signal(band1)
        amp2, phase2 = compute_analytic_signal(band2)

        # Compute all 7 metrics
        inter_con[0, band_idx] = compute_inter_pearson(band1, band2)
        inter_con[1, band_idx] = compute_inter_power_correlation(amp1, amp2)
        inter_con[2, band_idx] = compute_inter_plv(phase1, phase2)
        inter_con[3, band_idx] = compute_inter_pli(phase1, phase2)
        inter_con[4, band_idx] = compute_inter_wpli(phase1, phase2)
        inter_con[5, band_idx] = compute_inter_coherence(band1, band2, sampling_rate)
        inter_con[6, band_idx] = compute_inter_phase_diff(phase1, phase2)

    return inter_con


# =============================================================================
# Trial Processing
# =============================================================================

def process_trial(
    sample: Dict[str, Any],
    config: Dict[str, Any],
    bands: Dict[str, Tuple[float, float]]
) -> Dict[str, Any]:
    """
    Process a single trial and extract all features.

    Args:
        sample: Metadata dict with 'player1', 'player2', 'class', 'pair'
        config: Configuration dict
        bands: Frequency band definitions

    Returns:
        Result dict with status and output path
    """
    trial_id = f"{sample['player1']}_{sample['player2']}"

    try:
        # 1. Construct paths
        player1_path = Path(config['input_dir']) / f"{sample['player1']}.csv"
        player2_path = Path(config['input_dir']) / f"{sample['player2']}.csv"

        if not player1_path.exists():
            return {'status': 'error', 'trial_id': trial_id, 'error': f"File not found: {player1_path}"}
        if not player2_path.exists():
            return {'status': 'error', 'trial_id': trial_id, 'error': f"File not found: {player2_path}"}

        # 2. Load EEG data
        eeg1 = load_eeg_csv(player1_path, config['num_channels'])
        eeg2 = load_eeg_csv(player2_path, config['num_channels'])

        # 3. Align lengths
        min_len = min(eeg1.shape[1], eeg2.shape[1])

        # Check minimum length (at least 2 seconds)
        min_samples = config['sampling_rate'] * 2
        if min_len < min_samples:
            return {'status': 'error', 'trial_id': trial_id,
                    'error': f"Trial too short: {min_len} samples (min: {min_samples})"}

        eeg1 = eeg1[:, :min_len]
        eeg2 = eeg2[:, :min_len]

        # 4. Basic preprocessing (bandpass + CAR)
        eeg1_proc = preprocess_eeg(eeg1, config['sampling_rate'])
        eeg2_proc = preprocess_eeg(eeg2, config['sampling_rate'])

        # 5. Extract features
        features = {}

        # 5.1 Time domain (raw preprocessed signals)
        features['time_domain'] = np.stack([eeg1_proc, eeg2_proc], axis=0)  # (2, 32, T)

        # 5.2 Frequency domain (PSD)
        freqs, psd1 = compute_psd(eeg1_proc, config['sampling_rate'])
        _, psd2 = compute_psd(eeg2_proc, config['sampling_rate'])
        features['freq_domain'] = np.stack([psd1, psd2], axis=0)  # (2, 32, num_freqs)
        features['freq_bins'] = freqs  # (num_freqs,)

        # 5.3 Band energy
        band_energy1 = compute_band_energy(psd1, freqs, bands)
        band_energy2 = compute_band_energy(psd2, freqs, bands)
        features['bands_energy'] = np.stack([band_energy1, band_energy2], axis=0)  # (2, 32, 5)

        # 5.4 Intra-brain connectivity
        intra1 = compute_intra_connectivity(eeg1_proc, bands, config['sampling_rate'])
        intra2 = compute_intra_connectivity(eeg2_proc, bands, config['sampling_rate'])
        features['intra_con'] = np.stack([intra1, intra2], axis=0)  # (2, 7, 5, 32, 32)

        # 5.5 Inter-brain connectivity
        features['inter_con'] = compute_inter_connectivity(
            eeg1_proc, eeg2_proc, bands, config['sampling_rate']
        )  # (7, 5, 32, 32)

        # 6. Add metadata
        features['metadata'] = {
            'player1': sample['player1'],
            'player2': sample['player2'],
            'class': sample.get('class', 'Unknown'),
            'class_idx': CLASS_MAPPING.get(sample.get('class', ''), -1),
            'pair': sample.get('pair', -1),
            'timepoints': min_len,
            'sampling_rate': config['sampling_rate'],
            'bands': list(bands.keys()),
            'metrics': METRIC_NAMES,
        }

        # 7. Save to file
        output_path = Path(config['output_dir']) / f"{trial_id}.npy"
        np.save(output_path, features, allow_pickle=True)

        return {
            'status': 'success',
            'trial_id': trial_id,
            'output_path': str(output_path),
            'timepoints': min_len
        }

    except Exception as e:
        return {
            'status': 'error',
            'trial_id': trial_id,
            'error': str(e)
        }


# =============================================================================
# Parallel Execution
# =============================================================================

def get_completed_trials(output_dir: Path) -> set:
    """Get set of already completed trial IDs."""
    completed = set()
    for npy_file in output_dir.glob("*.npy"):
        trial_id = npy_file.stem
        completed.add(trial_id)
    return completed


def run_extraction(
    metadata: List[Dict],
    config: Dict[str, Any],
    bands: Dict[str, Tuple[float, float]],
    n_workers: int = 8,
    resume: bool = False
) -> List[Dict]:
    """
    Run feature extraction on all trials.

    Args:
        metadata: List of sample metadata dicts
        config: Configuration dict
        bands: Frequency band definitions
        n_workers: Number of parallel workers
        resume: If True, skip already processed trials

    Returns:
        List of result dicts
    """
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out completed trials if resuming
    trials_to_process = metadata
    if resume:
        completed = get_completed_trials(output_dir)
        trials_to_process = []
        for sample in metadata:
            trial_id = f"{sample['player1']}_{sample['player2']}"
            if trial_id not in completed:
                trials_to_process.append(sample)
        logging.info(f"Resuming: {len(completed)} already completed, {len(trials_to_process)} remaining")

    if len(trials_to_process) == 0:
        logging.info("All trials already processed!")
        return []

    logging.info(f"Processing {len(trials_to_process)} trials with {n_workers} workers...")

    # Run in parallel or sequentially
    if JOBLIB_AVAILABLE and n_workers > 1:
        results = Parallel(n_jobs=n_workers, backend='loky', verbose=0)(
            delayed(process_trial)(sample, config, bands)
            for sample in tqdm(trials_to_process, desc="Extracting features")
        )
    else:
        results = []
        for sample in tqdm(trials_to_process, desc="Extracting features"):
            results.append(process_trial(sample, config, bands))

    # Summarize results
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')

    logging.info(f"Completed: {success_count} success, {error_count} errors")

    # Log errors
    errors = [r for r in results if r['status'] == 'error']
    if errors:
        logging.warning(f"Errors ({len(errors)}):")
        for e in errors[:10]:  # Log first 10 errors
            logging.warning(f"  {e['trial_id']}: {e.get('error', 'Unknown error')}")

    return results


def save_summary(
    results: List[Dict],
    config: Dict[str, Any],
    output_dir: Path
):
    """Save processing summary to JSON."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
        'total_trials': len(results),
        'success': sum(1 for r in results if r['status'] == 'success'),
        'errors': sum(1 for r in results if r['status'] == 'error'),
        'error_details': [r for r in results if r['status'] == 'error'][:20],
        'bands': BAND_NAMES,
        'metrics': METRIC_NAMES,
    }

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info(f"Summary saved to: {summary_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract EEG features for Hyperscanning data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default='1_Data/datasets/EEGseg',
        help='Directory containing raw EEG CSV files'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='1_Data/datasets/EEGseg_features',
        help='Directory to save extracted features'
    )

    parser.add_argument(
        '--metadata',
        type=str,
        default='1_Data/metadata/complete_metadata.json',
        help='Path to metadata JSON file'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel workers'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip already processed trials'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--sampling-rate',
        type=int,
        default=250,
        help='EEG sampling rate in Hz'
    )

    parser.add_argument(
        '--num-channels',
        type=int,
        default=32,
        help='Number of EEG channels'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Resolve paths relative to project root
    input_dir = PROJECT_ROOT / args.input_dir
    output_dir = PROJECT_ROOT / args.output_dir
    metadata_path = PROJECT_ROOT / args.metadata

    # Add file handler for logging
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / 'extraction.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    logging.info("=" * 60)
    logging.info("EEG Feature Extraction Script")
    logging.info("=" * 60)
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Metadata path: {metadata_path}")
    logging.info(f"Workers: {args.workers}")
    logging.info(f"Resume: {args.resume}")
    logging.info(f"Sampling rate: {args.sampling_rate} Hz")
    logging.info(f"Frequency bands: {BAND_NAMES}")
    logging.info(f"Connectivity metrics: {METRIC_NAMES}")

    # Load metadata
    if not metadata_path.exists():
        logging.error(f"Metadata file not found: {metadata_path}")
        sys.exit(1)

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    logging.info(f"Loaded {len(metadata)} trials from metadata")

    # Check input directory
    if not input_dir.exists():
        logging.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Configuration
    config = {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'sampling_rate': args.sampling_rate,
        'num_channels': args.num_channels,
    }

    # Run extraction
    results = run_extraction(
        metadata=metadata,
        config=config,
        bands=FREQUENCY_BANDS,
        n_workers=args.workers,
        resume=args.resume
    )

    # Save summary
    if results:
        save_summary(results, config, output_dir)

    logging.info("Feature extraction complete!")


if __name__ == '__main__':
    main()
