# Gaze 模型分析腳本使用指南

本文件說明如何使用 `analyze_gaze.py` 進行模型分析。

---

## 目錄

1. [環境需求](#環境需求)
2. [單一模型分析](#單一模型分析)
3. [多模型比較](#多模型比較)
4. [輸出檔案說明](#輸出檔案說明)
5. [常見問題](#常見問題)

---

## 環境需求

```bash
# 必要套件
pip install torch torchvision timm
pip install numpy pandas scipy scikit-learn
pip install matplotlib seaborn
pip install pyyaml tqdm

# 可選（用於學習曲線）
pip install wandb
```

---

## 單一模型分析

### 基本指令格式

```bash
python 7_Analysis/python_scripts/analyze_gaze.py \
    --config <配置檔路徑> \
    --checkpoint <模型權重路徑> \
    --model_type <early|late> \
    --exp_name <實驗名稱>
```

### 參數說明

| 參數 | 必填 | 說明 |
|------|------|------|
| `--config` | ✅ | YAML 配置檔路徑 |
| `--checkpoint` | ✅ | 模型 checkpoint 路徑 (.pt 檔案) |
| `--model_type` | ✅ | 模型類型：`early` (Early Fusion) 或 `late` (Late Fusion) |
| `--fusion_mode` | ❌ | 覆蓋配置檔中的 fusion_mode（可選） |
| `--device` | ❌ | 指定裝置：`cuda` 或 `cpu`（預設自動偵測） |
| `--exp_name` | ❌ | 實驗名稱，用於輸出資料夾命名 |
| `--wandb_project` | ❌ | Wandb 專案名稱（用於學習曲線） |
| `--wandb_run` | ❌ | Wandb 執行名稱 |

---

### 範例 1：分析 Early Fusion (Concat) 模型

```bash
python 7_Analysis/python_scripts/analyze_gaze.py --config 4_Experiments/configs/gaze_earlyfusion.yaml --checkpoint 4_Experiments/runs/gaze_earlyfusion/concate/best_model.pt --model_type early --exp_name earlyfusion_concat
```

### 範例 2：分析 Late Fusion (Full) 模型

```bash
python 7_Analysis/python_scripts/analyze_gaze.py \
    --config 4_Experiments/configs/gaze_latefusion.yaml \
    --checkpoint 4_Experiments/runs/gaze_latefusion/full/best_model.pt \
    --model_type late \
    --exp_name latefusion_full
```

### 範例 3：分析並載入 Wandb 學習曲線

```bash
python 7_Analysis/python_scripts/analyze_gaze.py \
    --config 4_Experiments/configs/gaze_earlyfusion.yaml \
    --checkpoint 4_Experiments/runs/gaze_earlyfusion/concat/best_model.pt \
    --model_type early \
    --exp_name earlyfusion_concat \
    --wandb_project Multimodal_Gaze \
    --wandb_run early_fusion_vit
```

### 範例 4：使用 CPU 執行

```bash
python 7_Analysis/python_scripts/analyze_gaze.py \
    --config 4_Experiments/configs/gaze_earlyfusion.yaml \
    --checkpoint 4_Experiments/runs/gaze_earlyfusion/concat/best_model.pt \
    --model_type early \
    --device cpu \
    --exp_name earlyfusion_concat_cpu
```

---

## 多模型比較

### 基本指令格式

```bash
python 7_Analysis/python_scripts/analyze_gaze.py \
    --compare \
    --checkpoints <模型1.pt> <模型2.pt> ... \
    --configs <配置1.yaml> <配置2.yaml> ... \
    --model_types <類型1> <類型2> ... \
    --labels <標籤1> <標籤2> ...
```

### 範例 5：比較 Early Fusion vs Late Fusion

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

### 範例 6：比較多種 Late Fusion 模式

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

---

## 輸出檔案說明

### 原始資料 (Raw Data)

位置：`7_Analysis/raw_result/{exp_name}/`

| 檔案 | 說明 | 用途 |
|------|------|------|
| `metrics.csv` | 分類指標（Accuracy, F1, Precision, Recall） | 論文 Table |
| `conf_mat.csv` | 混淆矩陣 | MATLAB 後處理 |
| `roc_data.csv` | ROC 曲線資料（FPR, TPR, AUC） | MATLAB 繪圖 |
| `features.csv` | CLS token 特徵向量 (N × 768) | 特徵分析 |
| `tsne_coords.csv` | t-SNE 降維座標 (N × 2) | 視覺化 |
| `predictions.csv` | 所有樣本的預測結果與機率 | 錯誤分析 |
| `pair_stats.csv` | 各 Pair 的準確率統計 | 受試者分析 |
| `statistical_tests.csv` | 統計檢定結果（t-test, ANOVA） | 論文 Discussion |
| `training_history.csv` | 訓練歷史（Loss, Accuracy） | 學習曲線 |

### 論文圖表 (Publication Assets)

位置：`manuscript/figures/{exp_name}/`

| 檔案 | 說明 | 論文章節 |
|------|------|----------|
| `fig_conf_mat.pdf` | 混淆矩陣熱圖 | Results |
| `fig_roc_curves.pdf` | 多類別 ROC 曲線 | Results |
| `fig_tsne.pdf` | 特徵空間 t-SNE 視覺化 | Results |
| `fig_attention_sample_*.png` | 注意力視覺化 | Results |
| `fig_pair_accuracy.png` | 各 Pair 準確率長條圖 | Discussion |
| `fig_mechanism_analysis.pdf` | 機制分析圖（空間敏感度/特徵相關性） | Discussion |
| `fig_learning_curves.pdf` | 學習曲線 | Results |

位置：`manuscript/tables/`

| 檔案 | 說明 |
|------|------|
| `table_performance_{exp}.csv` | 單一模型效能表 |
| `table_comparison.csv` | 多模型比較表 |

### 比較模式輸出

位置：`manuscript/figures/comparison/`

| 檔案 | 說明 |
|------|------|
| `fig_compare_conf_mat.pdf` | 並排混淆矩陣 |
| `fig_compare_roc.pdf` | 重疊 ROC 曲線 |
| `fig_compare_metrics.pdf` | 指標比較長條圖 |

---

## 分析流程說明

執行腳本後，會依序進行以下步驟：

```
[Step 1] Running inference...
    → 載入模型與資料，執行驗證集推論
    → 輸出: predictions.csv

[Step 2] Quantitative Analysis...
    → 計算分類指標、混淆矩陣、ROC 曲線
    → 輸出: metrics.csv, conf_mat.csv, roc_data.csv
    → 圖表: fig_conf_mat.pdf, fig_roc_curves.pdf

[Step 3] Qualitative Analysis...
    → 提取特徵向量、執行 t-SNE 降維
    → 生成注意力視覺化
    → 輸出: features.csv, tsne_coords.csv
    → 圖表: fig_tsne.pdf, fig_attention_*.png

[Step 4] Error & Mechanism Analysis...
    → 計算各 Pair 準確率、識別困難樣本
    → Early Fusion: 分析空間敏感度
    → Late Fusion: 分析特徵相關性
    → 輸出: pair_stats.csv, statistical_tests.csv
    → 圖表: fig_pair_accuracy.png, fig_mechanism_analysis.pdf

[Step 5] Learning Curves...
    → 從 Wandb 或 checkpoint 載入訓練歷史
    → 輸出: training_history.csv
    → 圖表: fig_learning_curves.pdf
```

---

## 常見問題

### Q1: 出現 CUDA out of memory 錯誤

使用 CPU 執行或減少 batch size：

```bash
python 7_Analysis/python_scripts/analyze_gaze.py \
    --config ... --checkpoint ... --model_type early \
    --device cpu
```

### Q2: 找不到模型檔案

確認 checkpoint 路徑正確，可用以下指令列出可用的模型：

```bash
dir /s /b 4_Experiments\runs\*.pt
```

### Q3: 無法載入學習曲線

如果沒有使用 Wandb，腳本會嘗試從 checkpoint 目錄載入歷史。
確保 checkpoint 目錄中有 `checkpoint_epoch_*.pt` 檔案。

### Q4: 圖表字體顯示問題

如果中文或特殊字元顯示為方塊，安裝字體：

```python
# 在腳本開頭加入
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft JhengHei'  # Windows 繁體中文
```

---

## 快速開始

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
dir manuscript\figures\earlyfusion_concat\
```

---

## 聯絡資訊

如有問題，請參考 `CLAUDE.md` 或聯繫專案負責人。
