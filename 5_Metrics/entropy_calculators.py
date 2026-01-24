"""
Entropy Calculators for Multimodal Physiological Signals

This module provides entropy calculation methods for:
1. Spatial Entropy: For Gaze heatmap images
2. Spectral Entropy: For EEG time series data

Author: CNElab
Date: 2024
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
from scipy.stats import entropy as scipy_entropy
from scipy.signal import butter, filtfilt, welch


class BaseEntropyCalculator(ABC):
    """
    Abstract base class for entropy calculators.

    All entropy calculators should inherit from this class and implement
    the `compute` method.
    """

    @abstractmethod
    def compute(self, data: np.ndarray) -> np.ndarray:
        """
        Compute entropy from input data.

        Parameters
        ----------
        data : np.ndarray
            Input data array (format depends on subclass)

        Returns
        -------
        np.ndarray
            Computed entropy value(s)
        """
        pass

    @staticmethod
    def _normalize_to_probability(data: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """
        Normalize data to a valid probability distribution.

        Parameters
        ----------
        data : np.ndarray
            Input data (must be non-negative)
        eps : float
            Small constant to avoid log(0)

        Returns
        -------
        np.ndarray
            Normalized probability distribution (sums to 1)
        """
        # Ensure non-negative
        data = np.abs(data)

        # Add small epsilon to avoid zeros
        data = data + eps

        # Normalize to sum to 1
        return data / np.sum(data)


class SpatialEntropyCalculator(BaseEntropyCalculator):
    """
    Spatial Entropy Calculator for Gaze Heatmap Images.

    Computes Shannon entropy of a 2D heatmap treated as a spatial
    probability distribution. Higher entropy indicates more dispersed
    gaze patterns, while lower entropy indicates focused attention.

    Parameters
    ----------
    normalize_input : bool, default=True
        Whether to normalize input image to [0, 1] range

    Attributes
    ----------
    normalize_input : bool
        Whether to normalize input image

    Examples
    --------
    >>> calculator = SpatialEntropyCalculator()
    >>> heatmap = np.random.rand(224, 224)  # Grayscale heatmap
    >>> entropy = calculator.compute(heatmap)
    >>> print(f"Spatial Entropy: {entropy:.4f} bits")

    Notes
    -----
    - Input can be grayscale (H, W) or RGB (H, W, 3) or (3, H, W)
    - RGB images are converted to grayscale using luminosity method
    - Output is in bits (log base 2)
    """

    def __init__(self, normalize_input: bool = True):
        self.normalize_input = normalize_input

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to grayscale.

        Uses luminosity method: 0.299*R + 0.587*G + 0.114*B

        Parameters
        ----------
        image : np.ndarray
            Input image, can be:
            - (H, W): Already grayscale
            - (H, W, 3): RGB format
            - (3, H, W): Channel-first RGB format

        Returns
        -------
        np.ndarray
            Grayscale image of shape (H, W)
        """
        if image.ndim == 2:
            # Already grayscale
            return image

        elif image.ndim == 3:
            if image.shape[0] == 3:
                # Channel-first format (3, H, W) -> (H, W, 3)
                image = np.transpose(image, (1, 2, 0))

            if image.shape[2] == 3:
                # RGB to grayscale using luminosity method
                return (0.299 * image[:, :, 0] +
                        0.587 * image[:, :, 1] +
                        0.114 * image[:, :, 2])
            else:
                raise ValueError(f"Expected 3 channels, got {image.shape[2]}")

        else:
            raise ValueError(f"Expected 2D or 3D array, got {image.ndim}D")

    def compute(self, data: np.ndarray) -> float:
        """
        Compute spatial entropy of a heatmap image.

        Parameters
        ----------
        data : np.ndarray
            Input heatmap image of shape (H, W), (H, W, 3), or (3, H, W)

        Returns
        -------
        float
            Spatial entropy in bits

        Notes
        -----
        The entropy is computed as:
        H = -sum(p * log2(p))

        where p is the normalized pixel intensity distribution.
        """
        # Convert to grayscale if necessary
        grayscale = self._to_grayscale(data)

        # Normalize to [0, 1] if requested
        if self.normalize_input:
            grayscale = (grayscale - grayscale.min()) / (grayscale.max() - grayscale.min() + 1e-10)

        # Flatten to 1D for probability distribution
        flat = grayscale.flatten()

        # Normalize to probability distribution
        prob = self._normalize_to_probability(flat)

        # Compute Shannon entropy (base 2 for bits)
        return scipy_entropy(prob, base=2)


class SpectralEntropyCalculator(BaseEntropyCalculator):
    """
    Spectral Entropy Calculator for EEG Time Series.

    Computes Shannon entropy of the power spectral density (PSD) of
    EEG signals. Higher spectral entropy indicates a flatter, more
    uniform frequency distribution (similar to white noise), while
    lower entropy indicates energy concentrated in specific frequency bands.

    Parameters
    ----------
    sampling_rate : float, default=250.0
        Sampling rate of the EEG signal in Hz
    filter_low : float, default=0.5
        Low cutoff frequency for bandpass filter in Hz
    filter_high : float, default=50.0
        High cutoff frequency for bandpass filter in Hz
    filter_order : int, default=4
        Order of the Butterworth filter
    nperseg : int, default=256
        Length of each segment for Welch's method
    apply_filter : bool, default=True
        Whether to apply bandpass filter before computing PSD

    Attributes
    ----------
    sampling_rate : float
        Sampling rate in Hz
    filter_low : float
        Low cutoff frequency
    filter_high : float
        High cutoff frequency
    filter_order : int
        Butterworth filter order
    nperseg : int
        Segment length for Welch's method
    apply_filter : bool
        Whether to apply bandpass filter

    Examples
    --------
    >>> calculator = SpectralEntropyCalculator(sampling_rate=250)
    >>> eeg_data = np.random.randn(32, 1024)  # 32 channels, 1024 timepoints
    >>> entropy = calculator.compute(eeg_data)
    >>> print(f"Mean Spectral Entropy: {entropy.mean():.4f} bits")
    >>> print(f"Per-channel shape: {entropy.shape}")  # (32,)

    Notes
    -----
    - Input shape: (C, T) where C is channels and T is timepoints
    - Output shape: (C,) - one entropy value per channel
    - Uses Welch's method for robust PSD estimation
    - Filter is applied to remove DC offset and high-frequency noise
    """

    def __init__(
        self,
        sampling_rate: float = 250.0,
        filter_low: float = 0.5,
        filter_high: float = 50.0,
        filter_order: int = 4,
        nperseg: int = 256,
        apply_filter: bool = True
    ):
        self.sampling_rate = sampling_rate
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_order = filter_order
        self.nperseg = nperseg
        self.apply_filter = apply_filter

        # Pre-compute filter coefficients
        if self.apply_filter:
            self._b, self._a = self._design_bandpass_filter()

    def _design_bandpass_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design Butterworth bandpass filter.

        Returns
        -------
        b, a : np.ndarray
            Numerator and denominator coefficients of the filter
        """
        nyquist = self.sampling_rate / 2.0
        low = self.filter_low / nyquist
        high = self.filter_high / nyquist

        # Ensure frequencies are within valid range
        low = max(low, 0.001)
        high = min(high, 0.999)

        b, a = butter(self.filter_order, [low, high], btype='band')
        return b, a

    def _apply_bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to signal.

        Parameters
        ----------
        signal : np.ndarray
            Input signal of shape (T,) or (C, T)

        Returns
        -------
        np.ndarray
            Filtered signal of same shape
        """
        # Use filtfilt for zero-phase filtering
        if signal.ndim == 1:
            return filtfilt(self._b, self._a, signal)
        else:
            # Apply filter to each channel
            return np.array([filtfilt(self._b, self._a, ch) for ch in signal])

    def _compute_psd(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density using Welch's method.

        Parameters
        ----------
        signal : np.ndarray
            Input signal of shape (T,)

        Returns
        -------
        freqs : np.ndarray
            Frequency bins
        psd : np.ndarray
            Power spectral density values
        """
        freqs, psd = welch(
            signal,
            fs=self.sampling_rate,
            nperseg=min(self.nperseg, len(signal)),
            noverlap=self.nperseg // 2
        )
        return freqs, psd

    def _compute_single_channel_entropy(self, signal: np.ndarray) -> float:
        """
        Compute spectral entropy for a single channel.

        Parameters
        ----------
        signal : np.ndarray
            1D signal of shape (T,)

        Returns
        -------
        float
            Spectral entropy in bits
        """
        # Compute PSD
        freqs, psd = self._compute_psd(signal)

        # Normalize PSD to probability distribution
        prob = self._normalize_to_probability(psd)

        # Compute Shannon entropy (base 2 for bits)
        return scipy_entropy(prob, base=2)

    def compute(self, data: np.ndarray) -> np.ndarray:
        """
        Compute spectral entropy for multi-channel EEG data.

        Parameters
        ----------
        data : np.ndarray
            Input EEG data of shape (C, T) where:
            - C: number of channels (e.g., 32)
            - T: number of timepoints

        Returns
        -------
        np.ndarray
            Spectral entropy for each channel, shape (C,)

        Raises
        ------
        ValueError
            If input is not 2D array
        """
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array (C, T), got {data.ndim}D")

        n_channels = data.shape[0]

        # Apply bandpass filter if enabled
        if self.apply_filter:
            data = self._apply_bandpass_filter(data)

        # Compute entropy for each channel
        entropies = np.zeros(n_channels)
        for ch in range(n_channels):
            entropies[ch] = self._compute_single_channel_entropy(data[ch])

        return entropies


# =============================================================================
# Utility Functions
# =============================================================================

def compute_batch_spatial_entropy(
    images: np.ndarray,
    calculator: Optional[SpatialEntropyCalculator] = None
) -> np.ndarray:
    """
    Compute spatial entropy for a batch of images.

    Parameters
    ----------
    images : np.ndarray
        Batch of images, shape (N, H, W) or (N, H, W, 3) or (N, 3, H, W)
    calculator : SpatialEntropyCalculator, optional
        Pre-configured calculator. If None, creates default one.

    Returns
    -------
    np.ndarray
        Entropy values, shape (N,)
    """
    if calculator is None:
        calculator = SpatialEntropyCalculator()

    n_samples = images.shape[0]
    entropies = np.zeros(n_samples)

    for i in range(n_samples):
        entropies[i] = calculator.compute(images[i])

    return entropies


def compute_batch_spectral_entropy(
    eeg_data: np.ndarray,
    calculator: Optional[SpectralEntropyCalculator] = None,
    return_mean: bool = False
) -> np.ndarray:
    """
    Compute spectral entropy for a batch of EEG trials.

    Parameters
    ----------
    eeg_data : np.ndarray
        Batch of EEG data, shape (N, C, T) where:
        - N: number of trials
        - C: number of channels
        - T: number of timepoints
    calculator : SpectralEntropyCalculator, optional
        Pre-configured calculator. If None, creates default one.
    return_mean : bool, default=False
        If True, return mean entropy across channels for each trial.
        If False, return per-channel entropy.

    Returns
    -------
    np.ndarray
        If return_mean=False: shape (N, C)
        If return_mean=True: shape (N,)
    """
    if calculator is None:
        calculator = SpectralEntropyCalculator()

    n_samples = eeg_data.shape[0]
    n_channels = eeg_data.shape[1]

    entropies = np.zeros((n_samples, n_channels))

    for i in range(n_samples):
        entropies[i] = calculator.compute(eeg_data[i])

    if return_mean:
        return entropies.mean(axis=1)

    return entropies


# =============================================================================
# Standard 10-20 System Channel Information
# =============================================================================

# Standard 32-channel 10-20 system electrode names and positions
STANDARD_32_CHANNELS = [
    'Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3',
    'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1',
    'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz',
    'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2'
]

# 2D positions for topomap visualization (normalized coordinates)
# X: left(-1) to right(+1), Y: back(-1) to front(+1)
CHANNEL_POSITIONS_2D = {
    'Fp1': (-0.3, 0.9),   'Fp2': (0.3, 0.9),
    'F7': (-0.7, 0.5),    'F3': (-0.35, 0.5),   'Fz': (0.0, 0.5),    'F4': (0.35, 0.5),   'F8': (0.7, 0.5),
    'FT9': (-0.9, 0.3),   'FC5': (-0.55, 0.3),  'FC1': (-0.2, 0.3),  'FC2': (0.2, 0.3),   'FC6': (0.55, 0.3),  'FT10': (0.9, 0.3),
    'T7': (-0.9, 0.0),    'C3': (-0.45, 0.0),   'Cz': (0.0, 0.0),    'C4': (0.45, 0.0),   'T8': (0.9, 0.0),
    'TP9': (-0.9, -0.3),  'CP5': (-0.55, -0.3), 'CP1': (-0.2, -0.3), 'CP2': (0.2, -0.3),  'CP6': (0.55, -0.3), 'TP10': (0.9, -0.3),
    'P7': (-0.7, -0.5),   'P3': (-0.35, -0.5),  'Pz': (0.0, -0.5),   'P4': (0.35, -0.5),  'P8': (0.7, -0.5),
    'O1': (-0.3, -0.8),   'Oz': (0.0, -0.8),    'O2': (0.3, -0.8)
}


if __name__ == '__main__':
    # Quick test with synthetic data
    print("=" * 60)
    print("Testing Entropy Calculators")
    print("=" * 60)

    # Test Spatial Entropy
    print("\n[1] Spatial Entropy Calculator")
    print("-" * 40)

    spatial_calc = SpatialEntropyCalculator()

    # Uniform distribution (high entropy)
    uniform_heatmap = np.ones((224, 224))
    uniform_entropy = spatial_calc.compute(uniform_heatmap)
    print(f"Uniform heatmap entropy: {uniform_entropy:.4f} bits")

    # Focused distribution (low entropy)
    focused_heatmap = np.zeros((224, 224))
    focused_heatmap[100:124, 100:124] = 1.0  # Small bright region
    focused_entropy = spatial_calc.compute(focused_heatmap)
    print(f"Focused heatmap entropy: {focused_entropy:.4f} bits")

    # RGB image test
    rgb_image = np.random.rand(224, 224, 3)
    rgb_entropy = spatial_calc.compute(rgb_image)
    print(f"Random RGB image entropy: {rgb_entropy:.4f} bits")

    # Test Spectral Entropy
    print("\n[2] Spectral Entropy Calculator")
    print("-" * 40)

    spectral_calc = SpectralEntropyCalculator(sampling_rate=250)

    # White noise (high entropy)
    white_noise = np.random.randn(32, 1024)
    noise_entropy = spectral_calc.compute(white_noise)
    print(f"White noise - Mean entropy: {noise_entropy.mean():.4f} bits")
    print(f"             Std: {noise_entropy.std():.4f}")

    # Sinusoidal signal (low entropy - concentrated frequency)
    t = np.linspace(0, 4.096, 1024)  # 4.096 seconds at 250 Hz
    sine_wave = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
    sine_signal = np.tile(sine_wave, (32, 1))  # Repeat for all channels
    sine_entropy = spectral_calc.compute(sine_signal)
    print(f"10 Hz sine - Mean entropy: {sine_entropy.mean():.4f} bits")
    print(f"             Std: {sine_entropy.std():.4f}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
