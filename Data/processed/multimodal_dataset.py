"""
Multimodal Dataset: Eye Gaze Images + EEG Signals
Loads paired data for cross-modal fusion
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    """
    Dataset for multimodal fusion: Eye Gaze Images + EEG Signals

    For each sample, loads:
    - Player 1 Eye Gaze image
    - Player 2 Eye Gaze image
    - Player 1 EEG signal
    - Player 2 EEG signal
    - Interaction label (Single/Competition/Cooperation)
    """

    def __init__(
        self,
        dataset,
        image_base_path: str,
        eeg_base_path: str,
        label2id: Dict[str, int],
        image_size: int = 224,
        eeg_window_size: int = 1024,
        eeg_stride: int = 512,
        enable_eeg_preprocessing: bool = False,
        mode: str = 'train'
    ):
        """
        Initialize multimodal dataset

        Args:
            dataset: HuggingFace dataset with metadata
            image_base_path: Base path for eye gaze images
            eeg_base_path: Base path for EEG CSV files
            label2id: Label to ID mapping
            image_size: Image resize dimension
            eeg_window_size: EEG window size in samples
            eeg_stride: EEG sliding window stride
            enable_eeg_preprocessing: Enable EEG preprocessing (CAR, bandpass, etc.)
            mode: 'train' or 'test' (affects augmentation)
        """
        self.dataset = dataset
        self.image_base_path = Path(image_base_path)
        self.eeg_base_path = Path(eeg_base_path)
        self.label2id = label2id
        self.image_size = image_size
        self.eeg_window_size = eeg_window_size
        self.eeg_stride = eeg_stride
        self.enable_eeg_preprocessing = enable_eeg_preprocessing
        self.mode = mode

        # Image transforms
        if mode == 'train':
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

        # Precompute valid samples
        self.valid_samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        """Precompute valid samples (both image and EEG exist)"""
        logger.info(f"Preparing multimodal samples from {len(self.dataset)} items...")

        for idx in range(len(self.dataset)):
            if idx % 100 == 0:
                logger.info(f"Processing sample {idx}/{len(self.dataset)}...")

            item = self.dataset[idx]

            # Image paths
            img1_path = self.image_base_path / f"{item['player1']}.jpg"
            img2_path = self.image_base_path / f"{item['player2']}.jpg"

            # EEG paths
            eeg1_path = self.eeg_base_path / f"{item['player1']}.csv"
            eeg2_path = self.eeg_base_path / f"{item['player2']}.csv"

            # Check if all files exist
            if not (img1_path.exists() and img2_path.exists() and
                    eeg1_path.exists() and eeg2_path.exists()):
                if idx < 10:
                    logger.warning(f"Missing files for idx {idx}: {item['player1']}, {item['player2']}")
                continue

            try:
                # Quick check EEG length
                df1 = pd.read_csv(eeg1_path, header=None, nrows=1)
                df2 = pd.read_csv(eeg2_path, header=None, nrows=1)

                eeg_len1 = df1.shape[1]
                eeg_len2 = df2.shape[1]
                min_len = min(eeg_len1, eeg_len2)

                # Skip if EEG too short
                if min_len < self.eeg_window_size:
                    continue

                # Create sliding windows for EEG
                num_windows = (min_len - self.eeg_window_size) // self.eeg_stride + 1

                for win_idx in range(num_windows):
                    start = win_idx * self.eeg_stride
                    end = start + self.eeg_window_size

                    if end <= min_len:
                        self.valid_samples.append({
                            'dataset_idx': idx,
                            'start': start,
                            'end': end,
                            'player1': item['player1'],
                            'player2': item['player2'],
                            'class': item['class']
                        })

            except Exception as e:
                if idx < 10:
                    logger.error(f"Error processing idx {idx}: {e}")
                continue

        logger.info(f"Created {len(self.valid_samples)} valid multimodal samples from {len(self.dataset)} items")

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load and transform image"""
        img = Image.open(path).convert('RGB')
        return self.image_transform(img)

    def _load_eeg(self, path: Path, start: int, end: int) -> np.ndarray:
        """Load EEG window from CSV"""
        df = pd.read_csv(path, header=None)
        eeg = df.values  # (C, T)

        # Transpose if needed
        if eeg.shape[0] > eeg.shape[1]:
            eeg = eeg.T

        # Extract window
        eeg_win = eeg[:, start:end]  # (C, window_size)

        return eeg_win.astype(np.float32)

    def _preprocess_eeg(self, eeg: np.ndarray) -> np.ndarray:
        """Preprocess EEG signal (CAR, z-score)"""
        # Common Average Reference
        car = eeg.mean(axis=0, keepdims=True)
        eeg = eeg - car

        # Z-score normalization per channel
        mean = eeg.mean(axis=1, keepdims=True)
        std = eeg.std(axis=1, keepdims=True) + 1e-8
        eeg = (eeg - mean) / std

        return eeg

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Get a multimodal sample"""
        sample_info = self.valid_samples[idx]

        # Image paths
        img1_path = self.image_base_path / f"{sample_info['player1']}.jpg"
        img2_path = self.image_base_path / f"{sample_info['player2']}.jpg"

        # EEG paths
        eeg1_path = self.eeg_base_path / f"{sample_info['player1']}.csv"
        eeg2_path = self.eeg_base_path / f"{sample_info['player2']}.csv"

        try:
            # Load images
            img1 = self._load_image(img1_path)  # (3, H, W)
            img2 = self._load_image(img2_path)  # (3, H, W)

            # Load EEG
            start, end = sample_info['start'], sample_info['end']
            eeg1 = self._load_eeg(eeg1_path, start, end)  # (C, T)
            eeg2 = self._load_eeg(eeg2_path, start, end)  # (C, T)

            # Ensure same number of channels
            min_channels = min(eeg1.shape[0], eeg2.shape[0])
            eeg1 = eeg1[:min_channels, :]
            eeg2 = eeg2[:min_channels, :]

            # Preprocess EEG
            if self.enable_eeg_preprocessing:
                eeg1 = self._preprocess_eeg(eeg1)
                eeg2 = self._preprocess_eeg(eeg2)
            else:
                # Simple normalization
                eeg1 = (eeg1 - eeg1.mean()) / (eeg1.std() + 1e-8)
                eeg2 = (eeg2 - eeg2.mean()) / (eeg2.std() + 1e-8)

            # Convert to tensors
            img1_tensor = img1  # Already tensor from transform
            img2_tensor = img2
            eeg1_tensor = torch.from_numpy(eeg1).float()
            eeg2_tensor = torch.from_numpy(eeg2).float()

            # Get label
            label = self.label2id[sample_info['class']]
            label_tensor = torch.tensor(label, dtype=torch.long)

            return {
                'img1': img1_tensor,      # (3, H, W)
                'img2': img2_tensor,      # (3, H, W)
                'eeg1': eeg1_tensor,      # (C, T)
                'eeg2': eeg2_tensor,      # (C, T)
                'labels': label_tensor,   # scalar
                'player1': sample_info['player1'],
                'player2': sample_info['player2'],
                'class': sample_info['class']
            }

        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return dummy sample
            dummy_img = torch.zeros((3, self.image_size, self.image_size))
            dummy_eeg = torch.zeros((32, self.eeg_window_size))

            return {
                'img1': dummy_img,
                'img2': dummy_img,
                'eeg1': dummy_eeg,
                'eeg2': dummy_eeg,
                'labels': torch.tensor(0, dtype=torch.long),
                'player1': '',
                'player2': '',
                'class': 'Single'
            }


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    img1 = torch.stack([item['img1'] for item in batch])
    img2 = torch.stack([item['img2'] for item in batch])
    eeg1 = torch.stack([item['eeg1'] for item in batch])
    eeg2 = torch.stack([item['eeg2'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'img1': img1,
        'img2': img2,
        'eeg1': eeg1,
        'eeg2': eeg2,
        'labels': labels
    }
