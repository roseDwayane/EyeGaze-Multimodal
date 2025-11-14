# Early Fusion實現指南

## 概述

Early Fusion是最簡單的跨模態融合策略，將EEG信號轉換為圖像表示（時頻譜圖），然後與Eye Gaze圖像堆疊，送入單一ViT模型處理。

**優點**:
- 架構簡單，易於實現
- 單一統一模型，參數共享
- 可利用預訓練ViT的圖像處理能力

**缺點**:
- 模態差異大時融合效果可能不佳
- EEG時頻轉換會損失部分時域信息
- 作為消融對照實驗

---

## 架構設計

### 1. EEG到時頻表示轉換

```
EEG Signal (B, C, T)
    │
    ▼
STFT (Short-Time Fourier Transform)
    │
    ▼
Spectrogram (B, C, F, T')
    │
    ▼
Average across channels → (B, F, T')
    │
    ▼
Resize to (224, 224)
    │
    ▼
Expand to 3 channels (pseudo-RGB)
    │
    ▼
Pseudo-Image (B, 3, 224, 224)
```

**關鍵參數**:
- `n_fft`: 256 (FFT窗口大小)
- `hop_length`: 128 (跳躍長度)
- `n_mels`: 64 (Mel濾波器數量)

### 2. 融合策略

#### 策略A: Average Fusion (6通道)
```
Image_P1 (3 channels)  ┐
Image_P2 (3 channels)  ├─ Average → Avg_Image (3 channels)
                       ┘

EEG_P1 → Spectrogram (3 channels)  ┐
EEG_P2 → Spectrogram (3 channels)  ├─ Average → Avg_EEG (3 channels)
                                   ┘

Stack: [Avg_Image, Avg_EEG] → (B, 6, 224, 224)
    │
    ▼
Modified ViT (6-channel input)
```

#### 策略B: Concatenate Fusion (12通道)
```
Image_P1 (3 channels)         ┐
Image_P2 (3 channels)         │
EEG_P1 → Spectrogram (3)     │── Concatenate → (B, 12, 224, 224)
EEG_P2 → Spectrogram (3)     │
                              ┘
    │
    ▼
Modified ViT (12-channel input)
```

### 3. 修改ViT輸入層

```python
# 原始ViT patch embedding
Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

# 修改後
Conv2d(6 or 12, d_model, kernel_size=patch_size, stride=patch_size)

# 權重初始化策略：
# 如果有預訓練3通道模型，將權重重複並歸一化
new_weight = old_weight.repeat(1, n_channels//3, 1, 1)
new_weight = new_weight / (n_channels / 3)
```

---

## 實現細節

### EEGToTimeFrequency模塊

```python
class EEGToTimeFrequency(nn.Module):
    def __init__(self, n_fft=256, hop_length=128, n_mels=64,
                 target_size=224, num_channels=32):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # ...

    def compute_spectrogram(self, eeg):
        # STFT for each channel
        spec = torch.stft(eeg, n_fft=self.n_fft,
                         hop_length=self.hop_length,
                         return_complex=True)
        spec_mag = torch.abs(spec)
        return spec_mag

    def forward(self, eeg):
        # (B, C, T) → (B, 3, H, W)
        spec = self.compute_spectrogram(eeg)
        spec_avg = spec.mean(dim=1)
        spec_resized = F.interpolate(spec_avg, size=(224, 224))
        eeg_image = spec_resized.repeat(1, 3, 1, 1)
        return eeg_image
```

### EarlyFusionModel

```python
class EarlyFusionModel(nn.Module):
    def __init__(self, fusion_strategy='average', ...):
        super().__init__()

        # EEG轉換器
        self.eeg_to_tf = EEGToTimeFrequency(...)

        # 修改ViT
        if fusion_strategy == 'average':
            in_channels = 6
        else:  # concatenate
            in_channels = 12

        self.vit = self._create_modified_vit(in_channels, ...)

    def forward(self, img1, img2, eeg1, eeg2, labels):
        # 轉換EEG
        eeg1_image = self.eeg_to_tf(eeg1)
        eeg2_image = self.eeg_to_tf(eeg2)

        # 融合
        if self.fusion_strategy == 'average':
            avg_img = (img1 + img2) / 2
            avg_eeg = (eeg1_image + eeg2_image) / 2
            fused = torch.cat([avg_img, avg_eeg], dim=1)
        else:
            fused = torch.cat([img1, img2, eeg1_image, eeg2_image], dim=1)

        # ViT
        logits = self.vit(fused)
        return {'logits': logits, 'loss': ...}
```

---

## 使用方法

### 訓練

```bash
# 使用average策略（6通道）
python Experiments/scripts/train_early_fusion.py \
    --config Experiments/configs/early_fusion.yaml

# 使用concatenate策略（12通道）
# 修改配置文件：fusion_strategy: "concatenate"
```

### 配置選項

```yaml
model:
  fusion_strategy: "average"  # 或 "concatenate"

  # EEG時頻轉換參數
  eeg_n_fft: 256
  eeg_hop_length: 128
  eeg_n_mels: 64

  # ViT配置
  vit_d_model: 768
  vit_num_layers: 12
  vit_num_heads: 12
```

---

## 變體：Channel-Wise Early Fusion

更靈活的方式，使用卷積層混合通道：

```
12-channel input
    │
    ▼
Conv2d(12, 64, k=3) + BN + ReLU
    │
    ▼
Conv2d(64, 32, k=3) + BN + ReLU
    │
    ▼
Conv2d(32, 3, k=3) + Tanh
    │
    ▼
3-channel mixed representation
    │
    ▼
Standard ViT (3-channel)
```

**優勢**:
- 學習最優的模態混合方式
- 可使用標準預訓練ViT
- 更靈活的特徵融合

**使用方法**:
```python
from Models.fusion.early_fusion import ChannelWiseEarlyFusion

model = ChannelWiseEarlyFusion(
    image_size=224,
    vit_d_model=768,
    vit_num_layers=12,
    ...
)
```

---

## 預期性能

| 配置 | 準確率 | F1 Score | 參數量 | 備註 |
|------|-------|----------|--------|------|
| Average (6 ch) | ~72% | ~0.65 | ~86M | 簡單穩定 |
| Concatenate (12 ch) | ~73% | ~0.66 | ~86M | 略優 |
| Channel-Wise | ~74% | ~0.67 | ~87M | 最靈活 |

**對比其他方法**:
- Late Fusion: ~75% (F1: 0.68)
- **Mid Fusion: ~80% (F1: 0.75)** ← 最佳
- Early Fusion: ~72% (F1: 0.65)

---

## 優化建議

### 1. EEG時頻轉換優化

**當前**: 簡單STFT + 平均通道
**改進**:
- 使用Mel頻譜圖（更適合低頻EEG）
- 保留多通道信息（不平均）
- 使用小波變換（更好的時頻分辨率）

```python
# Mel spectrogram
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=256,
    n_fft=256,
    n_mels=64
)(eeg)
```

### 2. 通道混合策略

**當前**: 直接堆疊或平均
**改進**:
- 學習通道注意力權重
- 使用1×1卷積動態混合
- 分層融合（不同層融合不同模態）

```python
# Learnable channel attention
class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        gap = x.mean(dim=(2, 3))  # (B, C)
        weights = self.fc(gap).unsqueeze(-1).unsqueeze(-1)
        return x * weights
```

### 3. 多尺度融合

在不同patch size處理不同模態：

```python
# Image: 16×16 patches (細粒度)
# EEG: 32×32 patches (粗粒度，EEG分辨率低)
```

---

## 消融實驗建議

### 實驗1: 融合策略對比
- Average (6 ch) vs Concatenate (12 ch) vs Channel-Wise

### 實驗2: EEG表示方式
- STFT vs Mel Spectrogram vs Wavelet Transform

### 實驗3: 通道數影響
- 單通道（灰度） vs 3通道（pseudo-RGB） vs 原始多通道

### 實驗4: 時頻參數
- 不同n_fft: 128, 256, 512
- 不同hop_length: 64, 128, 256

---

## 可視化分析

### 1. EEG時頻譜圖

```python
import matplotlib.pyplot as plt

# 可視化轉換後的EEG
eeg_image = model.eeg_to_tf(eeg)
plt.imshow(eeg_image[0, 0].cpu().numpy())
plt.title("EEG Spectrogram (Pseudo-Image)")
plt.colorbar()
```

### 2. 輸入通道可視化

```python
# 可視化融合後的12通道輸入
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for i in range(12):
    ax = axes[i//4, i%4]
    ax.imshow(fused_input[0, i].cpu().numpy(), cmap='viridis')
    ax.set_title(f"Channel {i}")
```

### 3. 注意力圖

```python
# 可視化ViT的注意力權重
# 看哪些patch對預測最重要
attention_weights = model.vit.get_attention_weights(fused_input)
```

---

## 常見問題

### Q1: 為什麼性能不如Mid Fusion?

**A**: Early Fusion的主要限制：
1. EEG轉時頻會損失時域信息
2. 模態差異大，直接堆疊可能不是最優
3. ViT對低質量"圖像"（EEG頻譜圖）處理能力有限

### Q2: 如何改進Early Fusion?

**A**: 建議：
1. 使用更好的EEG表示（小波、連續小波變換）
2. 添加通道注意力機制
3. 使用預訓練的頻譜圖處理模型
4. 嘗試其他架構（CNN而非ViT）

### Q3: 何時使用Early Fusion?

**A**: 適用場景：
1. 快速baseline實驗
2. 計算資源有限（單一模型）
3. 消融實驗對照組
4. 模態相似度高的情況

---

## 總結

Early Fusion作為最簡單的融合策略，適合：
- ✅ 快速驗證跨模態融合的可行性
- ✅ 計算資源受限的場景
- ✅ 作為消融實驗的對照

但對於最佳性能，推薦使用：
- **Mid Fusion**: 四塔架構 + IBS token + 跨模態注意力（最佳）
- **Late Fusion**: 後期融合（穩定基線）

---

**相關文檔**:
- `MULTIMODAL_FUSION_PLAN.md` - 完整融合策略規劃
- `MULTIMODAL_FUSION_SUMMARY.md` - 實現總結
- `Models/fusion/early_fusion.py` - 源代碼

**訓練命令**:
```bash
python Experiments/scripts/train_early_fusion.py \
    --config Experiments/configs/early_fusion.yaml
```
