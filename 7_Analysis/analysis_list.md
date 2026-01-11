# 分析腳本使用指南

本文件說明專案中兩套分析工具的使用方法：
- **analyze_gaze.py** - Gaze 模型分類效能分析
- **analyze_entropy.py** - 多模態訊息熵分析

---

## 目錄

1. [概述與架構](#概述與架構)
2. [環境需求](#環境需求)
3. [Part 1: Gaze 模型分析](#part-1-gaze-模型分析)
4. [Part 2: 熵分析](#part-2-熵分析)
5. [模組化組件說明](#模組化組件說明)
6. [常見問題](#常見問題)

---

## 概述與架構

### 兩套分析工具

| 腳本 | 用途 | 輸入 | 輸出 |
|------|------|------|------|
| `analyze_gaze.py` | 分類模型效能評估 | 訓練好的 ViT 模型 checkpoint | 混淆矩陣、ROC 曲線、t-SNE、注意力圖 |
| `analyze_entropy.py` | 訊息熵與資料複雜度分析 | Gaze 影像 + EEG CSV 檔 | 空間熵、頻譜熵、統計摘要、Topomap |

### 模組化架構

兩套腳本共享重構後的模組化組件，遵循相同的設計模式：

```
EyeGaze-Multimodal_new/
├── 5_Metrics/                    # 指標計算模組
│   ├── classification_metrics.py # 分類指標（Accuracy, F1, ROC）
│   ├── feature_extractors.py     # 特徵提取（CLS token, t-SNE）
│   └── entropy_calculators.py    # 熵計算（空間熵、頻譜熵）
│
├── 6_Utils/                      # 工具與視覺化模組
│   ├── visualizers.py            # 繪圖函數（共 14 種圖表）
│   ├── attention_utils.py        # 注意力分析（Grad-CAM, Saliency）
│   ├── error_analysis.py         # 錯誤分析（Pair-wise, 機制驗證）
│   ├── learning_curves.py        # 學習曲線（Wandb, Checkpoint）
│   └── model_comparison.py       # 多模型比較
│
└── 7_Analysis/
    └── python_scripts/
        ├── analyze_gaze.py       # 主腳本 1（~530 行）
        └── analyze_entropy.py    # 主腳本 2（~900 行）
```

---

## 環境需求

```bash
# 必要套件
pip install torch torchvision timm
pip install numpy pandas scipy scikit-learn
pip install matplotlib seaborn
pip install pyyaml tqdm pillow

# 可選套件
pip install wandb  # 用於學習曲線
```

---

## Part 1: Gaze 模型分析

### 1.1 用途

分析訓練好的 Early Fusion / Late Fusion ViT 模型的分類效能：
- **定量分析**：Accuracy, Precision, Recall, F1, 混淆矩陣, ROC 曲線
- **定性分析**：t-SNE 特徵視覺化、注意力熱圖（Grad-CAM）
- **錯誤分析**：Per-pair 準確率、困難樣本識別
- **機制分析**：空間敏感度（Early）、特徵相關性（Late）

### 1.2 單一模型分析

#### 基本指令格式

```bash
python 7_Analysis/python_scripts/analyze_gaze.py \
    --config <配置檔路徑> \
    --checkpoint <模型權重路徑> \
    --model_type <early|late> \
    --exp_name <實驗名稱>
```

#### 參數說明

| 參數 | 必填 | 說明 |
|------|------|------|
| `--config` | ✅ | YAML 配置檔路徑 |
| `--checkpoint` | ✅ | 模型 checkpoint 路徑 (.pt 檔案) |
| `--model_type` | ✅ | 模型類型：`early` (Early Fusion) 或 `late` (Late Fusion) |
| `--fusion_mode` | ❌ | 覆蓋配置檔中的 fusion_mode（可選） |
| `--device` | ❌ | 指定裝置：`cuda` 或 `cpu`（預設自動偵測） |
| `--exp_name` | ❌ | 實驗名稱，用於輸出資料夾命名 |

#### 範例 1：分析 Early Fusion (Concat) 模型

```bash
python 7_Analysis/python_scripts/analyze_gaze.py \
    --config 4_Experiments/configs/gaze_earlyfusion.yaml \
    --checkpoint 4_Experiments/runs/gaze_earlyfusion/concat/best_model.pt \
    --model_type early \
    --exp_name earlyfusion_concat
```

#### 範例 2：分析 Late Fusion (Full) 模型

```bash
python 7_Analysis/python_scripts/analyze_gaze.py \
    --config 4_Experiments/configs/gaze_latefusion.yaml \
    --checkpoint 4_Experiments/runs/gaze_latefusion/full/best_model.pt \
    --model_type late \
    --exp_name latefusion_full
```

#### 範例 3：使用 CPU 執行

```bash
python 7_Analysis/python_scripts/analyze_gaze.py \
    --config 4_Experiments/configs/gaze_earlyfusion.yaml \
    --checkpoint 4_Experiments/runs/gaze_earlyfusion/concat/best_model.pt \
    --model_type early \
    --device cpu \
    --exp_name earlyfusion_concat_cpu
```

### 1.3 多模型比較

#### 基本指令格式

```bash
python 7_Analysis/python_scripts/analyze_gaze.py \
    --compare \
    --checkpoints <模型1.pt> <模型2.pt> ... \
    --configs <配置1.yaml> <配置2.yaml> ... \
    --model_types <類型1> <類型2> ... \
    --labels <標籤1> <標籤2> ...
```

#### 範例 4：比較 Early Fusion vs Late Fusion

```bash
python 7_Analysis/python_scripts/analyze_gaze.py \
    --compare \
    --checkpoints \
        4_Experiments/runs/gaze_earlyfusion/concat/best_model.pt \
        4_Experiments/runs/gaze_latefusion/full/best_model.pt \
    --configs \
        4_Experiments/configs/gaze_earlyfusion.yaml \
        4_Experiments/configs/gaze_latefusion.yaml \
    --model_types early late \
    --labels "Early-Concat" "Late-Full"
```

#### 範例 5：比較多種 Late Fusion 模式

```bash
python 7_Analysis/python_scripts/analyze_gaze.py \
    --compare \
    --checkpoints \
        4_Experiments/runs/gaze_latefusion/concat/best_model.pt \
        4_Experiments/runs/gaze_latefusion/add/best_model.pt \
        4_Experiments/runs/gaze_latefusion/multiply/best_model.pt \
        4_Experiments/runs/gaze_latefusion/full/best_model.pt \
    --configs \
        4_Experiments/configs/gaze_latefusion.yaml \
        4_Experiments/configs/gaze_latefusion.yaml \
        4_Experiments/configs/gaze_latefusion.yaml \
        4_Experiments/configs/gaze_latefusion.yaml \
    --model_types late late late late \
    --labels "Concat" "Add" "Multiply" "Full"
```

### 1.4 輸出檔案說明

#### 原始資料 (Raw Data CSV)

位置：`7_Analysis/raw_result/{exp_name}/`

| 檔案 | 說明 | 格式 | 用途 |
|------|------|------|------|
| `predictions.csv` | 所有樣本的預測結果與機率 | (N × 8) | 錯誤分析 |
| `metrics.csv` | 分類指標 | 單列多欄 | 論文 Table |
| `conf_mat.csv` | 混淆矩陣 | (3 × 3) | MATLAB 後處理 |
| `roc_data.csv` | ROC 曲線資料（FPR, TPR, AUC） | 多列 | MATLAB 繪圖 |
| `tsne_coords.csv` | t-SNE 降維座標 | (N × 4) | 視覺化 |
| `pair_stats.csv` | 各 Pair 的準確率統計 | (8 × 4) | 受試者分析 |

#### 論文圖表 (Publication Figures)

位置：`7_Analysis/figures/{exp_name}/`

| 檔案 | 說明 | 論文章節 |
|------|------|----------|
| `fig_conf_mat.pdf` | 混淆矩陣熱圖（含百分比） | Results |
| `fig_roc_curves.pdf` | 多類別 ROC 曲線（含 Micro/Macro） | Results |
| `fig_tsne.pdf` | 特徵空間 t-SNE 視覺化（正確/錯誤標記） | Results |
| `fig_pair_accuracy.png` | 各 Pair 準確率長條圖 | Discussion |

#### 比較模式輸出

位置：`7_Analysis/figures/comparison/`

| 檔案 | 說明 |
|------|------|
| `fig_compare_metrics.pdf` | 指標比較長條圖（含數值標註） |
| `fig_compare_conf_mat.pdf` | 並排混淆矩陣 |
| `fig_compare_roc.pdf` | 重疊 ROC 曲線（含 AUC） |

位置：`7_Analysis/tables/`

| 檔案 | 說明 |
|------|------|
| `table_comparison.csv` | 多模型比較表（所有指標） |

### 1.5 分析流程說明

執行腳本後，會依序進行以下步驟：

```
[Step 1] Running inference...
    → 載入模型與資料，執行驗證集推論
    → 使用 ClassificationMetrics 計算預測
    → 輸出: predictions.csv

[Step 2] Quantitative Analysis...
    → 計算分類指標、混淆矩陣、ROC 曲線
    → 使用 visualizers.plot_confusion_matrix()
    → 使用 visualizers.plot_roc_curves()
    → 輸出: metrics.csv, conf_mat.csv, roc_data.csv
    → 圖表: fig_conf_mat.pdf, fig_roc_curves.pdf

[Step 3] Qualitative Analysis...
    → 使用 FeatureExtractor 提取 CLS token 特徵
    → 執行 t-SNE 降維
    → 使用 visualizers.plot_tsne()
    → 輸出: tsne_coords.csv
    → 圖表: fig_tsne.pdf

[Step 4] Error & Mechanism Analysis...
    → 使用 ErrorAnalyzer 計算 per-pair 準確率
    → 使用 MechanismAnalyzer 分析空間/相關性
    → 使用 visualizers.plot_pair_accuracy()
    → 輸出: pair_stats.csv
    → 圖表: fig_pair_accuracy.png
```

---

## Part 2: 熵分析

### 2.1 用途

分析 Gaze 影像與 EEG 訊號的訊息熵，評估資料複雜度與變異性：
- **Gaze 空間熵**：量化注視分布的集中程度（Shannon Entropy on 2D heatmap）
- **EEG 頻譜熵**：量化腦波功率譜的複雜度（Shannon Entropy on PSD）
- **跨受試者分析**：Per-trial 計算 + Per-subject 統計
- **跨模態關聯**：Gaze vs EEG 熵的相關性分析

### 2.2 基本指令格式

```bash
python 7_Analysis/python_scripts/analyze_entropy.py \
    --modality <gaze|eeg|both> \
    [--use_mock] \
    [--eeg_path <路徑>] \
    [--gaze_path <路徑>] \
    [--output_dir <路徑>]
```

### 2.3 參數說明

| 參數 | 必填 | 說明 |
|------|------|------|
| `--modality` | ✅ | 分析模態：`gaze`（Gaze 影像）、`eeg`（EEG 訊號）、`both`（兩者） |
| `--use_mock` | ❌ | 使用模擬資料測試（無需真實資料） |
| `--eeg_path` | ❌ | 覆蓋預設 EEG 資料路徑 |
| `--gaze_path` | ❌ | 覆蓋預設 Gaze 資料路徑 |
| `--output_dir` | ❌ | 覆蓋預設輸出目錄 |

### 2.4 使用範例

#### 範例 1：分析 Gaze 空間熵

```bash
python 7_Analysis/python_scripts/analyze_entropy.py --modality gaze
```

#### 範例 2：分析 EEG 頻譜熵

```bash
python 7_Analysis/python_scripts/analyze_entropy.py --modality eeg
```

#### 範例 3：同時分析兩種模態

```bash
python 7_Analysis/python_scripts/analyze_entropy.py --modality both
```

#### 範例 4：使用模擬資料測試

```bash
python 7_Analysis/python_scripts/analyze_entropy.py --modality both --use_mock
```

#### 範例 5：指定自訂路徑

```bash
python 7_Analysis/python_scripts/analyze_entropy.py \
    --modality both \
    --eeg_path "D:/MyData/EEGseg" \
    --gaze_path "E:/Gaze/Heatmaps" \
    --output_dir "C:/Results/entropy"
```

### 2.5 輸出檔案說明

#### 原始資料 (Raw Data CSV)

位置：`7_Analysis/raw_result/entropy_analysis/`

| 檔案 | 說明 | 格式 | 內容 |
|------|------|------|------|
| `gaze_entropy_raw.csv` | Gaze 空間熵（每個 trial） | (N × 6) | Subject, Condition, Trial, Player, Entropy, Image_Path |
| `eeg_entropy_raw.csv` | EEG 頻譜熵（每個 trial） | (N × 38) | Subject, Condition, Trial, Player, ch1~ch32 熵, mean_entropy, std_entropy |
| `gaze_entropy_summary.csv` | Gaze 受試者統計摘要 | (40 × 5) | Subject, Condition, Mean, Std, N_Trials |
| `eeg_entropy_summary.csv` | EEG 受試者統計摘要 | (40 × 5) | Subject, Condition, Mean, Std, N_Trials |
| `cross_modality_entropy.csv` | 跨模態合併資料 | (N × 5) | Subject, Condition, Trial, gaze_entropy, eeg_entropy |

**註**：
- N = 所有 trials 總數（~160 trials/subject × 10 subjects）
- EEG 熵為 32 通道的平均值（`mean_entropy` 欄位）

#### 視覺化圖表 (Figures)

位置：`7_Analysis/figures/entropy_analysis/`

| 檔案 | 說明 | 圖表類型 | 用途 |
|------|------|----------|------|
| `gaze_entropy_boxplot.pdf` | Gaze 盒鬚圖（按 Condition） | Per-subject 分布 | 論文 Results |
| `gaze_entropy_kde.pdf` | Gaze 密度估計（3 條 KDE 曲線） | Group-level 分布 | 論文 Results |
| `eeg_entropy_boxplot.pdf` | EEG 盒鬚圖（按 Condition） | Per-subject 分布 | 論文 Results |
| `eeg_entropy_kde.pdf` | EEG 密度估計（3 條 KDE 曲線） | Group-level 分布 | 論文 Results |
| `eeg_entropy_topomap.pdf` | EEG Topomap（按 Condition） | Per-channel 空間分布 | 論文 Results |
| `cross_modality_correlation.pdf` | Gaze vs EEG 散點圖（含 R²） | 跨模態相關性 | 論文 Discussion |
| `gaze_entropy_violin.pdf` | Gaze 小提琴圖（可選） | 分布細節 | 補充材料 |
| `eeg_entropy_heatmap.pdf` | EEG 熱圖（Subject × Condition） | 受試者差異 | 補充材料 |

### 2.6 分析流程說明

```
[Phase 1] Data Loading
    → 讀取 Gaze 影像（PNG）或 EEG CSV 檔
    → 解析檔名取得 Subject, Condition, Trial, Player

[Phase 2] Entropy Calculation
    → Gaze: 使用 SpatialEntropyCalculator
        - 將 RGB 影像轉為灰階
        - 正規化為機率分布
        - 計算 Shannon Entropy
    → EEG: 使用 SpectralEntropyCalculator
        - 執行 Bandpass Filter (0.5-50 Hz)
        - 計算 Welch PSD
        - 對 32 通道分別計算頻譜熵
    → 輸出: *_entropy_raw.csv

[Phase 3] Statistical Summary
    → Per-subject 計算 Mean, Std
    → 輸出: *_entropy_summary.csv

[Phase 4] Visualization
    → 使用 visualizers.plot_entropy_boxplot()
    → 使用 visualizers.plot_entropy_kde()
    → 使用 visualizers.plot_entropy_topomap()
    → 使用 visualizers.plot_entropy_correlation()
    → 輸出: 8 張圖表（PDF）

[Phase 5] Cross-Modality Analysis (if both)
    → 合併 Gaze + EEG 資料
    → 計算相關係數
    → 輸出: cross_modality_entropy.csv
```

### 2.7 資料格式說明

#### Gaze 影像檔案命名規則

```
格式：Pair-{id}-{condition}_trial{num}_{player}.png
範例：Pair-1-Single_trial001_player1.png
      Pair-2-Competition_trial042_player2.png
      Pair-5-Cooperation_trial120_player1.png
```

#### EEG CSV 檔案命名規則與格式

```
命名格式：Pair-{id}-{condition}_trial{num}_{player}.csv
範例：Pair-1-Single_trial001_player1.csv

CSV 格式：
- 行數：32 行（對應 32 個 EEG 通道）
- 列數：T 列（時間點，fs=250 Hz）
- 通道順序：遵循 10-20 系統
- 數值：μV（浮點數）

10-20 系統 32 通道：
['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7',
 'FC3', 'FCz', 'FC4', 'FT8', 'T7', 'C3', 'Cz', 'C4',
 'T8', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P7', 'P3',
 'Pz', 'P4', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8']
```

---

## 模組化組件說明

### 5_Metrics/ - 指標計算模組

#### `classification_metrics.py`

**類別**：`ClassificationMetrics`

**功能**：
- 計算多類別分類指標（Accuracy, Precision, Recall, F1）
- 計算混淆矩陣
- 計算 ROC 曲線與 AUC（含 Micro/Macro 平均）
- 生成分類報告

**主要方法**：
- `compute_metrics(y_true, y_pred)` → Dict[str, float]
- `compute_confusion_matrix(y_true, y_pred)` → np.ndarray
- `compute_roc_data(y_true, y_prob)` → Dict
- `save_metrics_csv()`, `save_confusion_matrix_csv()`, `save_roc_data_csv()`

**使用者**：`analyze_gaze.py`

---

#### `feature_extractors.py`

**類別**：`FeatureExtractor`

**功能**：
- 從 ViT 模型提取 CLS token 特徵
- 執行 t-SNE / PCA 降維
- 計算特徵統計（centroids, intra-class variance）
- 計算 cosine similarity, Euclidean distance

**主要方法**：
- `extract_features(dataloader)` → (features, labels, metadata)
- `compute_tsne(features)` → np.ndarray
- `compute_pca(features)` → (pca_features, pca_obj)
- `save_features_csv()`, `save_tsne_csv()`

**使用者**：`analyze_gaze.py`

---

#### `entropy_calculators.py`

**類別**：
- `SpatialEntropyCalculator` - Gaze 空間熵
- `SpectralEntropyCalculator` - EEG 頻譜熵

**功能**：
- **Spatial Entropy**：計算 2D 灰階影像的 Shannon Entropy
- **Spectral Entropy**：計算 EEG 訊號的功率譜熵（Welch PSD）

**主要方法**：
- `SpatialEntropyCalculator.compute(image_path)` → float
- `SpectralEntropyCalculator.compute(eeg_data, fs)` → np.ndarray (32,)

**使用者**：`analyze_entropy.py`

---

### 6_Utils/ - 工具與視覺化模組

#### `visualizers.py`

**功能**：提供 14 種出版級圖表函數

**分類相關**：
- `plot_confusion_matrix()` - 混淆矩陣熱圖（含百分比）
- `plot_roc_curves()` - 多類別 ROC 曲線
- `plot_tsne()` - t-SNE 散點圖（雙子圖：真實標籤 + 錯誤標記）
- `plot_learning_curves()` - 訓練曲線（Loss + Accuracy + F1）
- `plot_metrics_comparison()` - 模型比較長條圖
- `plot_pair_accuracy()` - Per-pair 準確率長條圖
- `plot_mechanism_analysis()` - 機制分析圖（空間敏感度/特徵相關性）

**熵分析相關**：
- `plot_entropy_boxplot()` - 盒鬚圖（按 Condition）
- `plot_entropy_kde()` - KDE 密度估計（3 條曲線）
- `plot_entropy_topomap()` - EEG Topomap（10-20 系統）
- `plot_entropy_correlation()` - Gaze vs EEG 散點圖（含迴歸線）
- `plot_entropy_violin()` - 小提琴圖
- `plot_entropy_heatmap()` - Subject × Condition 熱圖

**樣式設定**：
- `setup_academic_style()` - 配置出版級圖表樣式

**使用者**：`analyze_gaze.py`, `analyze_entropy.py`

---

#### `attention_utils.py`

**類別**：`AttentionAnalyzer`

**功能**：
- 計算 Gradient Saliency Map
- 計算 Grad-CAM（針對 ViT 的 Class Activation Mapping）
- 生成注意力視覺化圖（原始影像 + 熱圖疊加）
- 批次處理與網格視覺化

**主要方法**：
- `compute_gradient_saliency(img_a, img_b)` → np.ndarray
- `compute_grad_cam(img_a, img_b)` → (cam_a, cam_b)
- `visualize_attention()` - 生成 2×3 子圖（含預測資訊）

**使用者**：`analyze_gaze.py`

---

#### `error_analysis.py`

**類別**：
- `ErrorAnalyzer` - 錯誤分析
- `MechanismAnalyzer` - 機制驗證

**功能**：
- **ErrorAnalyzer**：
  - Per-pair 準確率統計
  - 識別困難 pairs（bottom 20%）
  - 錯誤分布分析（confusion patterns）
- **MechanismAnalyzer**：
  - 計算 Gaze 中心距離（Center of Mass）
  - 計算 Gaze 重疊度（IoU）
  - 計算 Late Fusion 特徵相關性（Cosine Similarity）
  - 統計檢定（ANOVA, t-test with Bonferroni correction, Cohen's d）

**主要方法**：
- `ErrorAnalyzer.analyze_pair_performance(df)` → pd.DataFrame
- `MechanismAnalyzer.compute_gaze_distance(img_a, img_b)` → float
- `MechanismAnalyzer.compute_feature_correlation(model, dataloader)` → (similarities, labels)
- `MechanismAnalyzer.statistical_tests_by_class(values, labels)` → pd.DataFrame

**使用者**：`analyze_gaze.py`

---

#### `learning_curves.py`

**類別**：`LearningCurveAnalyzer`

**功能**：
- 從 Wandb 抓取訓練歷史
- 從 Checkpoint 檔案提取訓練歷史
- 繪製學習曲線（Loss, Accuracy, F1, Learning Rate）
- 計算訓練統計（best epoch, convergence info）

**主要方法**：
- `fetch_wandb_history(project, run_name)` → pd.DataFrame
- `extract_from_checkpoints(checkpoint_dir)` → pd.DataFrame
- `plot_learning_curves()` - 繪製 Loss + Acc + F1 曲線
- `get_best_epoch(metric, mode)` → Dict

**使用者**：`analyze_gaze.py`

---

#### `model_comparison.py`

**類別**：
- `ModelResults` - 模型結果容器
- `MultiModelComparator` - 多模型比較器

**功能**：
- 比較多個模型的指標
- 生成比較表（CSV + LaTeX）
- 統計顯著性檢定（McNemar's test）
- 繪製比較圖（並排混淆矩陣、重疊 ROC、指標長條圖、Radar chart）

**主要方法**：
- `MultiModelComparator.compare_metrics()` → pd.DataFrame
- `MultiModelComparator.compute_statistical_significance()` → pd.DataFrame (p-values)
- `MultiModelComparator.plot_metrics_comparison()`
- `MultiModelComparator.plot_confusion_matrices()`
- `MultiModelComparator.plot_roc_comparison()`
- `MultiModelComparator.generate_latex_table()` → str

**使用者**：`analyze_gaze.py`

---

## 常見問題

### Q1: 出現 CUDA out of memory 錯誤

**Gaze 模型分析**：使用 CPU 執行
```bash
python 7_Analysis/python_scripts/analyze_gaze.py \
    --config ... --checkpoint ... --model_type early \
    --device cpu
```

**熵分析**：降低批次處理量（修改腳本內的 batch size）

---

### Q2: 找不到模型檔案

確認 checkpoint 路徑正確，可用以下指令列出可用的模型：

```bash
# Windows
dir /s /b 4_Experiments\runs\*.pt

# Linux/Mac
find 4_Experiments/runs -name "*.pt"
```

---

### Q3: 找不到 EEG 或 Gaze 資料

**檢查資料路徑**：
```bash
# EEG
dir C:\Users\user\pythonproject\EyeGaze-Multimodal_new\1_Data\datasets\EEGseg\*.csv

# Gaze
dir "G:\共用雲端硬碟\CNElab_林佳誼_Gaze\B.GazeImage\01.data\bgOn_heatmapOn_trajOn\*.png"
```

**使用自訂路徑**：
```bash
python 7_Analysis/python_scripts/analyze_entropy.py \
    --modality both \
    --eeg_path "D:/MyData/EEG" \
    --gaze_path "E:/Gaze"
```

---

### Q4: 圖表字體顯示問題

如果中文或特殊字元顯示為方塊，修改字體設定：

```python
# 在腳本開頭加入（或修改 visualizers.py）
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft JhengHei'  # Windows 繁體中文
# plt.rcParams['font.family'] = 'Arial Unicode MS'  # macOS
```

---

### Q5: ImportError: No module named 'sklearn'

安裝缺少的套件：
```bash
pip install scikit-learn
```

---

### Q6: 熵分析中 EEG Topomap 顯示異常

確認 EEG 檔案格式正確：
- 32 行（通道）× T 列（時間點）
- CSV 格式，無標題行
- 通道順序符合 10-20 系統

---

### Q7: 多模型比較時長度不匹配

確保所有參數列表長度相同：
```bash
# 錯誤：checkpoints 有 3 個，但 configs 只有 2 個
--checkpoints model1.pt model2.pt model3.pt \
--configs config1.yaml config2.yaml  # ❌

# 正確：所有列表長度一致
--checkpoints model1.pt model2.pt model3.pt \
--configs config1.yaml config2.yaml config3.yaml \
--model_types early late early \
--labels "Model1" "Model2" "Model3"  # ✅
```

---

## 快速開始

### Gaze 模型分析

```bash
# 1. 確認環境
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 2. 執行分析（以 Early Fusion Concat 為例）
python 7_Analysis/python_scripts/analyze_gaze.py \
    --config 4_Experiments/configs/gaze_earlyfusion.yaml \
    --checkpoint 4_Experiments/runs/gaze_earlyfusion/concat/best_model.pt \
    --model_type early \
    --exp_name earlyfusion_concat

# 3. 查看結果
dir 7_Analysis\raw_result\earlyfusion_concat\
dir 7_Analysis\figures\earlyfusion_concat\
```

### 熵分析

```bash
# 1. 測試（使用模擬資料）
python 7_Analysis/python_scripts/analyze_entropy.py --modality both --use_mock

# 2. 實際分析
python 7_Analysis/python_scripts/analyze_entropy.py --modality both

# 3. 查看結果
dir 7_Analysis\raw_result\entropy_analysis\
dir 7_Analysis\figures\entropy_analysis\
```

---

## 進階使用

### 批次處理多個模型

建立批次腳本 `batch_analyze.bat`（Windows）：

```batch
@echo off
setlocal enabledelayedexpansion

set MODELS=concat add subtract multiply full
set CONFIG=4_Experiments/configs/gaze_latefusion.yaml

for %%m in (%MODELS%) do (
    echo Analyzing Late Fusion - %%m
    python 7_Analysis/python_scripts/analyze_gaze.py ^
        --config %CONFIG% ^
        --checkpoint 4_Experiments/runs/gaze_latefusion/%%m/best_model.pt ^
        --model_type late ^
        --exp_name latefusion_%%m
)

echo All models analyzed!
```

或 Bash 腳本 `batch_analyze.sh`（Linux/Mac）：

```bash
#!/bin/bash

MODELS=(concat add subtract multiply full)
CONFIG="4_Experiments/configs/gaze_latefusion.yaml"

for model in "${MODELS[@]}"; do
    echo "Analyzing Late Fusion - $model"
    python 7_Analysis/python_scripts/analyze_gaze.py \
        --config "$CONFIG" \
        --checkpoint "4_Experiments/runs/gaze_latefusion/$model/best_model.pt" \
        --model_type late \
        --exp_name "latefusion_$model"
done

echo "All models analyzed!"
```

---

## 聯絡資訊

如有問題，請參考：
- 專案說明：`CLAUDE.md`
- 程式碼文件：各模組的 docstrings
- 專案負責人：Kong-Yi Chang

---

**最後更新**：2026-01-11
**版本**：v2.0（模組化重構版）
