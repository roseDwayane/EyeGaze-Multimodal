"""
Dual EEG Dataset for Inter-Brain Synchrony Classification
Loads paired EEG signals and applies preprocessing
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class DualEEGDataset(Dataset):
    """
    Dataset for dual-EEG classification (player1 + player2)
    Loads EEG CSV files and applies preprocessing
    """

    def __init__(
        self,
        dataset,
        eeg_base_path: str,
        label2id: Dict[str, int],
        window_size: int = 1000,  # Number of time points per window
        stride: int = 500,         # Stride for sliding window
        sampling_rate: int = 250,  # EEG sampling rate (Hz)
        filter_low: float = 1.0,   # Low-pass filter (Hz)
        filter_high: float = 45.0, # High-pass filter (Hz)
        enable_preprocessing: bool = True,  # Enable EEG preprocessing (CAR, bandpass, z-score)
    ):
        """
        Initialize dataset

        Args:
            dataset: HuggingFace dataset object with metadata
            eeg_base_path: Base path for EEG CSV files
            label2id: Label to ID mapping
            window_size: Window size in samples
            stride: Stride for sliding window
            sampling_rate: EEG sampling rate
            filter_low: Low-pass filter frequency
            filter_high: High-pass filter frequency
        """
        self.dataset = dataset
        self.eeg_base_path = Path(eeg_base_path)
        self.label2id = label2id
        self.window_size = window_size
        self.stride = stride
        self.sampling_rate = sampling_rate
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.enable_preprocessing = enable_preprocessing

        # Precompute valid windows
        self.valid_windows = []
        self._prepare_windows()

    def _prepare_windows(self):
        """Precompute valid sliding windows from all samples"""
        logger.info(f"Preparing windows from {len(self.dataset)} samples...")

        for idx in range(len(self.dataset)):
            if idx % 100 == 0:
                logger.info(f"Processing sample {idx}/{len(self.dataset)}...")

            item = self.dataset[idx]

            player1_path = self.eeg_base_path / f"{item['player1']}.csv"
            player2_path = self.eeg_base_path / f"{item['player2']}.csv"

            if not player1_path.exists() or not player2_path.exists():
                if idx < 10:  # Only log first 10 missing files
                    logger.warning(f"Missing EEG files for idx {idx}: {item['player1']}, {item['player2']}")
                continue

            try:
                # Quick check: just get shape without loading full data
                # Assume EEG files have consistent format
                import pandas as pd

                # Read only first row to get number of columns (timepoints)
                df1 = pd.read_csv(player1_path, header=None, nrows=1)
                df2 = pd.read_csv(player2_path, header=None, nrows=1)

                len1 = df1.shape[1]  # Number of timepoints
                len2 = df2.shape[1]

                min_len = min(len1, len2)

                # Skip if too short
                if min_len < self.window_size:
                    continue

                # Create sliding windows
                num_windows = (min_len - self.window_size) // self.stride + 1

                for win_idx in range(num_windows):
                    start = win_idx * self.stride
                    end = start + self.window_size

                    if end <= min_len:
                        self.valid_windows.append({
                            'dataset_idx': idx,
                            'start': start,
                            'end': end,
                            'player1': item['player1'],
                            'player2': item['player2'],
                            'class': item['class']
                        })

            except Exception as e:
                if idx < 10:  # Only log first 10 errors
                    logger.error(f"Error processing idx {idx}: {e}")
                continue

        logger.info(f"Created {len(self.valid_windows)} valid windows from {len(self.dataset)} samples")

    def _load_eeg(self, path: Path) -> np.ndarray:
        """
        Load EEG from CSV file

        Args:
            path: Path to CSV file

        Returns:
            EEG array of shape (C, T) where C is channels, T is time
        """
        # Load CSV (assuming each row is a channel)
        df = pd.read_csv(path, header=None)
        eeg = df.values  # (C, T)

        # If transposed (each column is a channel), fix it
        if eeg.shape[0] > eeg.shape[1]:
            eeg = eeg.T

        return eeg.astype(np.float32)

    def _preprocess_eeg(self, eeg: np.ndarray) -> np.ndarray:
        """
        Preprocess EEG signal
        1. Bandpass filter (1-45 Hz)
        2. Re-reference (CAR - Common Average Reference)
        3. Z-score normalization

        Args:
            eeg: (C, T) raw EEG

        Returns:
            preprocessed: (C, T) preprocessed EEG
        """
        # 1. Bandpass filter (simplified - in practice use scipy.signal)
        # For now, just do basic filtering approximation
        # TODO: Implement proper butterworth bandpass filter

        # 2. Common Average Reference (CAR)
        car = eeg.mean(axis=0, keepdims=True)  # (1, T)
        eeg = eeg - car  # (C, T)

        # 3. Z-score normalization per channel
        mean = eeg.mean(axis=1, keepdims=True)  # (C, 1)
        std = eeg.std(axis=1, keepdims=True) + 1e-8  # (C, 1)
        eeg = (eeg - mean) / std

        return eeg

    def __len__(self):
        return len(self.valid_windows)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Get a windowed sample"""
        window_info = self.valid_windows[idx]

        # Load EEG files
        player1_path = self.eeg_base_path / f"{window_info['player1']}.csv"
        player2_path = self.eeg_base_path / f"{window_info['player2']}.csv"

        try:
            eeg1 = self._load_eeg(player1_path)
            eeg2 = self._load_eeg(player2_path)

            # Extract window
            start, end = window_info['start'], window_info['end']
            eeg1_win = eeg1[:, start:end]  # (C, T)
            eeg2_win = eeg2[:, start:end]  # (C, T)

            # Ensure same number of channels
            min_channels = min(eeg1_win.shape[0], eeg2_win.shape[0])
            eeg1_win = eeg1_win[:min_channels, :]
            eeg2_win = eeg2_win[:min_channels, :]

            # Preprocess (if enabled)
            if self.enable_preprocessing:
                eeg1_win = self._preprocess_eeg(eeg1_win)
                eeg2_win = self._preprocess_eeg(eeg2_win)
            else:
                # Simple normalization only
                eeg1_win = (eeg1_win - eeg1_win.mean()) / (eeg1_win.std() + 1e-8)
                eeg2_win = (eeg2_win - eeg2_win.mean()) / (eeg2_win.std() + 1e-8)

            # Convert to tensors
            eeg1_tensor = torch.from_numpy(eeg1_win).float()  # (C, T)
            eeg2_tensor = torch.from_numpy(eeg2_win).float()  # (C, T)

            # Get label
            label = self.label2id[window_info['class']]
            label_tensor = torch.tensor(label, dtype=torch.long)

            return {
                'eeg1': eeg1_tensor,
                'eeg2': eeg2_tensor,
                'labels': label_tensor,
                'dataset_idx': window_info['dataset_idx'],
                'player1': window_info['player1'],
                'player2': window_info['player2'],
                'class': window_info['class']
            }

        except Exception as e:
            logger.error(f"Error loading window {idx}: {e}")
            # Return a dummy sample
            dummy_eeg = torch.zeros((62, self.window_size), dtype=torch.float32)
            return {
                'eeg1': dummy_eeg,
                'eeg2': dummy_eeg,
                'labels': torch.tensor(0, dtype=torch.long),
                'player1': '',
                'player2': '',
                'class': 'Single'
            }


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    eeg1 = torch.stack([item['eeg1'] for item in batch])  # (B, C, T)
    eeg2 = torch.stack([item['eeg2'] for item in batch])  # (B, C, T)
    labels = torch.stack([item['labels'] for item in batch])  # (B,)
    dataset_idx = [item['dataset_idx'] for item in batch]  # List of integers

    return {
        'eeg1': eeg1,
        'eeg2': eeg2,
        'labels': labels,
        'dataset_idx': dataset_idx
    }


class SimpleEEGPreprocessor:
    """
    Simple EEG preprocessor for filtering
    Uses basic filtering techniques
    """
    def __init__(self, sampling_rate: int = 250, low_freq: float = 1.0, high_freq: float = 45.0):
        self.sampling_rate = sampling_rate
        self.low_freq = low_freq
        self.high_freq = high_freq

    def bandpass_filter(self, eeg: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter (simplified)
        In practice, use scipy.signal.butter + filtfilt

        Args:
            eeg: (C, T) raw signal

        Returns:
            filtered: (C, T) filtered signal
        """
        # Placeholder: In real implementation, use proper filter
        # For now, return as is
        # TODO: Implement butterworth bandpass filter

        from scipy import signal

        try:
            # Design Butterworth bandpass filter
            nyquist = self.sampling_rate / 2
            low = self.low_freq / nyquist
            high = self.high_freq / nyquist

            b, a = signal.butter(4, [low, high], btype='band')

            # Apply filter to each channel
            filtered = signal.filtfilt(b, a, eeg, axis=1)

            return filtered.astype(np.float32)

        except ImportError:
            logger.warning("scipy not available, skipping bandpass filter")
            return eeg
        except Exception as e:
            logger.error(f"Error in bandpass filter: {e}")
            return eeg
