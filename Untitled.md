# EEG Analysis Implementation Plan

  

## 模型架構摘要 (來自 Codebase 探索)

  

- **模型**: `DualEEGTransformer`

- **IBS 矩陣形狀**: `(B, 6, 7, 32, 32)` — 6 bands × 7 features × 32² channels

- **6 Bands**: broadband(0.5-45Hz), delta, theta, alpha, beta, gamma

- **7 Features**: PLV, PLI, wPLI, Coherence, PowerCorr, PhaseDiff, TimeCorr

- **z_fuse 形狀**: `(B, 768)` — 分類前的融合特徵

- **Checkpoint 位置**: `4_Experiments/runs/dualEEG/*/best_model.pt`

  

---

  

## 一、CSV 輸出結構表 (CSV Output Schema)

  

### 1. 核心指標 (Core Metrics)

  

| 檔案名稱                        | 說明           | 欄位                                                                                       |
| --------------------------- | ------------ | ---------------------------------------------------------------------------------------- |
| `confusion_matrix.csv`      | 3×3 混淆矩陣     | `Predicted_Single`, `Predicted_Competition`, `Predicted_Cooperation` (rows: True labels) |
| `classification_report.csv` | Per-class 指標 | `Class`, `Precision`, `Recall`, `F1`, `Support`                                          |
| `overall_metrics.csv`       | 整體指標         | `Metric`, `Value` (Accuracy, Macro-F1, etc.)                                             |

  

### 2. 頻段敏感度分析 (Frequency Sensitivity)

  

| 檔案名稱                        | 說明             | 欄位                                                                 |
| --------------------------- | -------------- | ------------------------------------------------------------------ |
| `frequency_sensitivity.csv` | Mask 各頻段後的性能變化 | `Band`, `Masked_Accuracy`, `Masked_F1`, `Accuracy_Drop`, `F1_Drop` |

  

### 3. IBS Connectivity 矩陣 (高維數據)

  

**策略**: 採用 **Flattened Long Format** + **獨立矩陣檔案** 雙軌制

  

| 檔案名稱                              | 說明                      | 欄位/格式                                                                                          |
| --------------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------- |
| `ibs_connectivity_long.csv`       | 所有 IBS 數據的 Long Format  | `Subject_ID`, `True_Label`, `Pred_Label`, `Band`, `Feature`, `Channel_1`, `Channel_2`, `Value` |
| `ibs_mean_by_class/`              | 按類別平均的矩陣目錄              |                                                                                                |
| ├─ `{Class}_{Band}_{Feature}.csv` | 32×32 矩陣 (無 header，純數值) | MATLAB `readmatrix()` 直接讀取                                                                     |
| └─ `channel_names.csv`            | 通道名稱對照                  | `Index`, `Channel_Name`                                                                        |
| `ibs_difference_coop_vs_comp/`    | Coop - Comp 差異矩陣        | 同上格式                                                                                           |

  

### 4. Cross-Attention Weights

  

| 檔案名稱                                | 說明                     | 欄位/格式                                                    |
| ----------------------------------- | ---------------------- | -------------------------------------------------------- |
| `attention_weights/`                | Attention 矩陣目錄         |                                                          |
| ├─ `sample_{idx}_class_{label}.csv` | 單樣本 attention (T×T 矩陣) | 無 header，MATLAB 直接讀取                                     |
| └─ `attention_summary.csv`          | 統計摘要                   | `Class`, `Mean_Diagonal`, `Std_Diagonal`, `Mean_OffDiag` |
| `attention_mean_by_class/`          | 按類別平均的 attention       |                                                          |
| └─ `{Class}_mean_attention.csv`     | 平均矩陣                   |                                                          |

  

### 5. t-SNE / UMAP 特徵分佈

  

| 檔案名稱                 | 說明                   | 欄位                                                          |
| -------------------- | -------------------- | ----------------------------------------------------------- |
| `tsne_z_fuse.csv`    | z_fuse 的 t-SNE 結果    | `Sample_ID`, `True_Label`, `Pred_Label`, `TSNE_1`, `TSNE_2` |
| `tsne_ibs_token.csv` | ibs_token 的 t-SNE 結果 | 同上                                                          |
| `umap_z_fuse.csv`    | z_fuse 的 UMAP 結果     | `Sample_ID`, `True_Label`, `Pred_Label`, `UMAP_1`, `UMAP_2` |
| `umap_ibs_token.csv` | ibs_token 的 UMAP 結果  | 同上                                                          |

  

### 6. Grad-CAM 分析

  

| 檔案名稱                                | 說明                         | 欄位/格式                                              |
| ----------------------------------- | -------------------------- | -------------------------------------------------- |
| `gradcam_spectrogram/`              | Spectrogram Grad-CAM 目錄    |                                                    |
| ├─ `sample_{idx}_class_{label}.csv` | Time-Freq heatmap (F×T 矩陣) | 無 header                                           |
| └─ `gradcam_mean_by_class/`         | 按類別平均的 Grad-CAM            |                                                    |
|     └─ `{Class}_mean_gradcam.csv`   | 平均 heatmap                 |                                                    |
| `gradcam_metadata.csv`              | 時頻軸資訊                      | `Axis`, `Index`, `Value` (Time: seconds, Freq: Hz) |

  

---

  

## 二、程式碼模組規劃

  

### 檔案結構

```

EyeGaze-Multimodal_new/

├── 5_Metrics/

│   └── eeg_metrics.py          # 計算核心

├── 6_Utils/

│   └── io_utils.py             # CSV 導出工具

└── 7_Analysis/

    ├── python_scripts/

    │   └── analyze_eeg.py      # 主執行檔

    └── outputs/                # CSV 輸出目錄 (自動建立)

        ├── core_metrics/

        ├── frequency_sensitivity/

        ├── ibs_connectivity/

        ├── attention_weights/

        ├── feature_embeddings/

        └── gradcam/

```

  

### 模組功能分配

  

#### `5_Metrics/eeg_metrics.py`

```python

# 核心計算函數

def compute_confusion_matrix(y_true, y_pred, class_names) -> pd.DataFrame

def compute_classification_report(y_true, y_pred, class_names) -> pd.DataFrame

def compute_frequency_sensitivity(model, dataloader, device, bands) -> pd.DataFrame

def extract_ibs_matrices(model, dataloader, device) -> dict  # 返回按類別分組的矩陣

def compute_ibs_difference(coop_matrices, comp_matrices) -> dict

def extract_attention_weights(model, dataloader, device) -> dict

def extract_features_for_embedding(model, dataloader, device) -> dict  # z_fuse, ibs_token

def compute_gradcam(model, dataloader, device, target_class) -> dict

```

  

#### `6_Utils/io_utils.py`

```python

# CSV 導出函數

def save_confusion_matrix(cm, path, class_names)

def save_classification_report(report_df, path)

def save_ibs_long_format(ibs_data, path)

def save_ibs_matrices(matrices, output_dir, class_name, band, feature)

def save_attention_weights(attn_data, output_dir)

def save_embedding_results(embedding_df, path)

def save_gradcam_results(gradcam_data, output_dir, metadata)

def ensure_output_dirs(base_path) -> dict  # 建立所有子目錄

```

  

#### `7_Analysis/python_scripts/analyze_eeg.py`

```python

# 主程式流程

def main():

    # 1. 載入配置與模型

    # 2. 載入測試數據

    # 3. 執行各項分析

    # 4. 生成預覽圖

    # 5. 導出 CSV

  

# 分析函數

def run_core_metrics_analysis(model, dataloader, output_dir)

def run_frequency_sensitivity_analysis(model, dataloader, output_dir)

def run_ibs_connectivity_analysis(model, dataloader, output_dir)

def run_attention_analysis(model, dataloader, output_dir)

def run_embedding_analysis(model, dataloader, output_dir)

def run_gradcam_analysis(model, dataloader, output_dir)

```

  

---

  

## 三、分析任務實作細節

  

### Task A: 核心指標

  

**步驟**:

1. 載入 `best_model.pt`，設為 eval 模式

2. 對測試集進行推論，收集 `y_true`, `y_pred`

3. 計算 confusion matrix, precision/recall/F1

4. 生成 heatmap 預覽圖

5. 導出 CSV

  

**預覽圖**: `confusion_matrix_heatmap.png`

  

---

  

### Task B: 頻段敏感度分析

  

**步驟**:

1. 定義 6 個頻段: broadband, delta, theta, alpha, beta, gamma

2. 對每個頻段:

   - 在 `IBSConnectivityMatrixGenerator` 的輸出中，將該頻段的 slice 設為 0

   - 重新計算準確率和 F1

3. 計算性能下降幅度

4. 生成 bar chart 預覽圖

5. 導出 CSV

  

**技術實作**: 使用 forward hook 在 `ibs_tokenizer` 之前攔截並修改 `connectivity_matrices`

  

**預覽圖**: `frequency_sensitivity_barplot.png`

  

---

  

### Task C: IBS Connectivity 可視化

  

**步驟**:

1. 收集所有測試樣本的 IBS 矩陣 `(N, 6, 7, 32, 32)`

2. 按 true label 分組並計算平均

3. 計算 Cooperation vs Competition 差異矩陣

4. 生成預覽圖 (使用 `mne.viz.plot_connectivity_circle` 或 `seaborn.heatmap`)

5. 導出:

   - Long format CSV (完整數據)

   - 獨立矩陣 CSV (MATLAB 讀取用)

  

**預覽圖**:

- `ibs_mean_{class}_{band}_{feature}.png` (選擇性，主要頻段)

- `ibs_diff_coop_vs_comp_theta_plv.png` (關鍵差異圖)

  

**Hook 設置**:

```python

def hook_ibs_matrices(module, input, output):

    # output shape: (B, 6, 7, 32, 32)

    self.ibs_matrices.append(output.cpu().numpy())

```

  

---

  

### Task D: Cross-Attention Weights 可視化

  

**步驟**:

1. 修改 `CrossBrainAttention.forward()` 返回 attention weights (或使用 hook)

2. 收集測試集的 attention matrices

3. 按類別計算平均 attention pattern

4. 分析對角線 vs 非對角線 attention 強度

5. 生成 heatmap 預覽圖

6. 導出 CSV

  

**預覽圖**:

- `attention_heatmap_{class}.png`

- `attention_diagonal_comparison.png`

  

**技術細節**: PyTorch `MultiheadAttention` 可設置 `need_weights=True` 返回 attention weights

  

---

  

### Task E: t-SNE / UMAP 特徵分佈

  

**步驟**:

1. 收集所有測試樣本的 `z_fuse` (768 dims) 和 `ibs_token` (256 dims)

2. 使用 sklearn 的 TSNE 和 umap-learn 的 UMAP

3. 設置參數: perplexity=30, n_iter=1000 (t-SNE); n_neighbors=15 (UMAP)

4. 生成 scatter plot，按類別著色

5. 導出 CSV

  

**預覽圖**:

- `tsne_z_fuse.png`

- `tsne_ibs_token.png`

- `umap_z_fuse.png`

- `umap_ibs_token.png`

  

---

  

### Task F: Spectrogram Grad-CAM

  

**步驟**:

1. 對 `SpectrogramTokenGenerator` 的輸入進行 Grad-CAM

2. 計算目標類別 (如 Cooperation) 對 spectrogram 的梯度

3. 生成 Time-Frequency activation map

4. 按類別平均

5. 導出 CSV

  

**預覽圖**:

- `gradcam_spectrogram_{class}.png`

  

**技術實作**:

```python

# Grad-CAM 核心邏輯

spectrogram = model.spectrogram_generator.get_spectrogram(eeg)  # (B, C, F, T)

spectrogram.requires_grad_(True)

output = model(eeg1, eeg2)

target_score = output['logits'][:, target_class].sum()

target_score.backward()

gradients = spectrogram.grad  # (B, C, F, T)

weights = gradients.mean(dim=(2, 3), keepdim=True)  # Global Average Pooling

cam = (weights * spectrogram).sum(dim=1)  # (B, F, T)

cam = F.relu(cam)  # 只保留正向影響

```

  

---

  

## 四、執行順序與依賴

  

```

1. analyze_eeg.py 載入模型與數據

   │

   ├─ 2. run_core_metrics_analysis()      [無依賴]

   │

   ├─ 3. run_frequency_sensitivity_analysis()  [無依賴]

   │

   ├─ 4. run_ibs_connectivity_analysis()  [無依賴]

   │

   ├─ 5. run_attention_analysis()         [無依賴]

   │

   ├─ 6. run_embedding_analysis()         [無依賴]

   │

   └─ 7. run_gradcam_analysis()           [無依賴]

  

所有分析任務互相獨立，可並行執行或按需選擇執行。

```

  

---

  

## 五、命令列介面設計

  

```bash

python 7_Analysis/python_scripts/analyze_eeg.py \

    --checkpoint 4_Experiments/runs/dualEEG/old_eeg/best_model.pt \

    --config 4_Experiments/configs/dual_eeg_transformer.yaml \

    --output_dir 7_Analysis/outputs \

    --analyses all  # 或 metrics,ibs,attention,embedding,gradcam,frequency

    --device cuda:0 \

    --batch_size 32

```

  

---

  

## 六、預期輸出檔案清單

  

```

7_Analysis/outputs/

├── core_metrics/

│   ├── confusion_matrix.csv

│   ├── confusion_matrix_heatmap.png

│   ├── classification_report.csv

│   └── overall_metrics.csv

│

├── frequency_sensitivity/

│   ├── frequency_sensitivity.csv

│   └── frequency_sensitivity_barplot.png

│

├── ibs_connectivity/

│   ├── ibs_connectivity_long.csv

│   ├── channel_names.csv

│   ├── ibs_mean_by_class/

│   │   ├── Single_theta_PLV.csv

│   │   ├── Competition_theta_PLV.csv

│   │   ├── Cooperation_theta_PLV.csv

│   │   └── ... (6 bands × 7 features × 3 classes = 126 files)

│   ├── ibs_difference_coop_vs_comp/

│   │   ├── diff_theta_PLV.csv

│   │   └── ... (42 files)

│   ├── ibs_mean_cooperation_theta_plv.png

│   └── ibs_diff_coop_vs_comp_theta_plv.png

│

├── attention_weights/

│   ├── attention_summary.csv

│   ├── attention_mean_by_class/

│   │   ├── Single_mean_attention.csv

│   │   ├── Competition_mean_attention.csv

│   │   └── Cooperation_mean_attention.csv

│   ├── attention_heatmap_Single.png

│   ├── attention_heatmap_Competition.png

│   └── attention_heatmap_Cooperation.png

│

├── feature_embeddings/

│   ├── tsne_z_fuse.csv

│   ├── tsne_ibs_token.csv

│   ├── umap_z_fuse.csv

│   ├── umap_ibs_token.csv

│   ├── tsne_z_fuse.png

│   ├── tsne_ibs_token.png

│   ├── umap_z_fuse.png

│   └── umap_ibs_token.png

│

└── gradcam/

    ├── gradcam_metadata.csv

    ├── gradcam_mean_by_class/

    │   ├── Single_mean_gradcam.csv

    │   ├── Competition_mean_gradcam.csv

    │   └── Cooperation_mean_gradcam.csv

    ├── gradcam_spectrogram_Single.png

    ├── gradcam_spectrogram_Competition.png

    └── gradcam_spectrogram_Cooperation.png

```

  

---

  

## 七、所需套件

  

```python

# 核心

import torch

import numpy as np

import pandas as pd

  

# 可視化

import matplotlib.pyplot as plt

import seaborn as sns

  

# 降維

from sklearn.manifold import TSNE

import umap  # pip install umap-learn

  

# 配置

import yaml

import argparse

  

# 進度條

from tqdm import tqdm

```

  

---

  

## 八、實作優先順序建議

  

1. **Phase 1 (核心)**:

   - `io_utils.py` - CSV 導出基礎設施

   - `eeg_metrics.py` - confusion matrix, classification report

   - `analyze_eeg.py` - 模型載入、基本框架

  

2. **Phase 2 (IBS 分析)**:

   - IBS 矩陣提取與導出

   - 差異矩陣計算

  

3. **Phase 3 (可解釋性)**:

   - t-SNE/UMAP embedding

   - Cross-Attention weights

  

4. **Phase 4 (進階)**:

   - 頻段敏感度分析

   - Grad-CAM

  

---

  

## 九、注意事項

  

1. **記憶體管理**: IBS 矩陣 `(N, 6, 7, 32, 32)` 可能很大，建議分批處理後取平均

2. **Hook 清理**: 使用 forward hook 後記得 `hook.remove()` 避免記憶體洩漏

3. **隨機種子**: t-SNE/UMAP 設置固定種子確保可重現性

4. **Grad-CAM**: 需確保 spectrogram 在計算圖中，可能需要修改模型 forward 邏輯