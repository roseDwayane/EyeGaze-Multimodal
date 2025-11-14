# Eye Gaze × EEG Multimodal Fusion - Implementation Complete

## 完成狀態 (Completion Status)

✅ **所有三種融合策略已實現並除錯完成**

### 1. Late Fusion (Strategy A) - 決策層融合
- **檔案**: `Models/fusion/late_fusion.py`
- **配置**: `Experiments/configs/late_fusion.yaml`
- **訓練腳本**: `Experiments/scripts/train_late_fusion.py`
- **狀態**: ✅ 已除錯完成，可以開始訓練
- **模式**:
  - `logits`: 加權平均輸出logits
  - `features`: 特徵拼接 + MLP分類器

**修復問題**:
- ✅ 添加 `ViTClassifier.get_cls_features()` 方法
- ✅ 修正融合維度計算 (768 + 256×2 = 1280)
- ✅ 修正 Hugging Face API 參數名稱
- ✅ 測試通過 (`test_late_fusion_fix.py`)

### 2. Mid Fusion (Strategy B) - 中間層融合 【主要貢獻】
- **檔案**: `Models/fusion/mid_fusion.py`
- **配置**: `Experiments/configs/mid_fusion.yaml`
- **訓練腳本**: `Experiments/scripts/train_mid_fusion.py`
- **狀態**: ✅ 已除錯完成，訓練中
- **核心組件**:
  - Four-tower 架構 (P1-Img, P2-Img, P1-EEG, P2-EEG)
  - Symmetric Fusion Operators (sum, product, abs diff)
  - IBS Token Generator (PLV, power correlation)
  - Cross-Modal Bidirectional Attention

**修復問題**:
- ✅ 修正 `_extract_image_features()` 使用 `get_cls_features()` API
- ✅ 測試通過 (`test_mid_fusion_fix.py`)
- ✅ 訓練已啟動 (WandB run: mid-fusion-full)

### 3. Early Fusion (Strategy C) - 輸入層融合
- **檔案**: `Models/fusion/early_fusion.py`
- **配置**: `Experiments/configs/early_fusion.yaml`
- **訓練腳本**: `Experiments/scripts/train_early_fusion.py`
- **狀態**: ✅ 實現完成
- **策略**:
  - `average`: EEG轉頻譜圖後平均 (6通道)
  - `concatenate`: 直接拼接所有通道 (12通道)

---

## 快速開始訓練 (Quick Start Training)

### Late Fusion 訓練
```bash
python Experiments/scripts/train_late_fusion.py --config Experiments/configs/late_fusion.yaml
```

**參數調整** (`Experiments/configs/late_fusion.yaml`):
```yaml
model:
  fusion_mode: "features"  # 或 "logits"
  freeze_image: false      # 是否凍結影像模型
  freeze_eeg: false        # 是否凍結EEG模型
  image_weight: 0.5        # logits模式下的影像權重
  eeg_weight: 0.5          # logits模式下的EEG權重

training:
  learning_rate: 1.0e-4
  num_train_epochs: 50
  per_device_train_batch_size: 16
```

### Mid Fusion 訓練
```bash
python Experiments/scripts/train_mid_fusion.py --config Experiments/configs/mid_fusion.yaml
```

**參數調整** (`Experiments/configs/mid_fusion.yaml`):
```yaml
model:
  use_ibs_token: true         # 是否使用IBS token
  use_cross_attention: true   # 是否使用跨模態注意力
  image_shared_weights: true  # 影像編碼器是否共享權重
  eeg_shared_weights: true    # EEG編碼器是否共享權重
  symmetric_fusion_mode: "full"  # full, basic, learnable
```

### Early Fusion 訓練
```bash
python Experiments/scripts/train_early_fusion.py --config Experiments/configs/early_fusion.yaml
```

**參數調整** (`Experiments/configs/early_fusion.yaml`):
```yaml
model:
  fusion_strategy: "average"  # 或 "concatenate"
  n_fft: 256                  # STFT 參數
  hop_length: 64
  freq_bins: 64
```

---

## 資料配置 (Data Configuration)

所有配置檔案中的資料路徑 (確認這些路徑正確):
```yaml
data:
  metadata_path: "Data/metadata/complete_metadata.json"
  image_base_path: "G:/共用雲端硬碟/CNElab_林佳誼_Gaze/B.GazeImage/01.data/bgOn_heatmapOn_trajOn"
  eeg_base_path: "G:/共用雲端硬碟/CNElab_林佳誼_Gaze/B.GazeImage/01.data/EEGseg"

  enable_eeg_preprocessing: false  # 已關閉預處理
```

預訓練模型路徑:
```yaml
model:
  image_model_path: "C:/Users/user/pythonproject/EyeGaze-Multimodal/Experiments/outputs/vit_class_multiply"
  eeg_model_path: "C:/Users/user/pythonproject/EyeGaze-Multimodal/Experiments/outputs/dual_eeg_transformer"
```

---

## 模型參數量統計 (Model Parameters)

| 模型 | 總參數量 | 可訓練參數 |
|------|----------|------------|
| Late Fusion (features) | ~93.9M | 取決於freeze設定 |
| Late Fusion (logits) | ~93.4M | 取決於freeze設定 |
| Mid Fusion (full) | ~110M | 取決於shared weights |
| Early Fusion (avg) | ~86M | ~86M |
| Early Fusion (concat) | ~87M | ~87M |

---

## WandB 監控 (Monitoring)

所有訓練會自動上傳到 Weights & Biases:

**專案名稱**: `eyegaze-eeg-classification`

**Run 名稱**:
- `late-fusion-baseline` (Late Fusion)
- `mid-fusion-main` (Mid Fusion)
- `early-fusion-baseline` (Early Fusion)

**監控指標**:
```python
# Late Fusion
- train/loss (總損失)
- train/loss_fused (融合損失)
- train/loss_img (影像分支損失)
- train/loss_eeg (EEG分支損失)
- eval/accuracy, eval/f1, eval/precision, eval/recall

# Mid Fusion
- train/loss (總損失)
- train/loss_cls (分類損失)
- train/loss_ibs (IBS token損失)
- eval/accuracy, eval/f1

# Early Fusion
- train/loss
- eval/accuracy, eval/f1
```

---

## 消融實驗建議 (Ablation Study Recommendations)

### Mid Fusion 消融實驗 (最重要)

**A1: IBS Token 影響**
```bash
# 有 IBS token (baseline)
python Experiments/scripts/train_mid_fusion.py --config Experiments/configs/mid_fusion.yaml

# 無 IBS token
# 修改 mid_fusion.yaml: use_ibs_token: false
python Experiments/scripts/train_mid_fusion.py --config Experiments/configs/mid_fusion.yaml
```

**A2: Cross-Modal Attention 影響**
```bash
# 有 cross-attention (baseline)
# 無 cross-attention
# 修改 mid_fusion.yaml: use_cross_attention: false
```

**A3: Symmetric Fusion Mode**
```bash
# Full mode (sum + mul + diff)
# 修改 mid_fusion.yaml: symmetric_fusion_mode: "full"

# Basic mode (僅 sum + mul)
# symmetric_fusion_mode: "basic"

# Learnable mode (可學習權重)
# symmetric_fusion_mode: "learnable"
```

**A4: Weight Sharing Strategy**
```bash
# 影像共享權重 vs 獨立編碼器
# image_shared_weights: true/false

# EEG共享權重 vs 獨立編碼器
# eeg_shared_weights: true/false
```

### Late Fusion 消融實驗

**B1: Fusion Mode**
```bash
# Features mode (baseline)
# fusion_mode: "features"

# Logits mode
# fusion_mode: "logits"
```

**B2: Freeze Strategy**
```bash
# 全部微調 (baseline)
# freeze_image: false, freeze_eeg: false

# 僅微調融合層
# freeze_image: true, freeze_eeg: true

# 僅微調影像模型
# freeze_image: false, freeze_eeg: true
```

### Early Fusion 消融實驗

**C1: Fusion Strategy**
```bash
# Average mode
# fusion_strategy: "average"

# Concatenate mode
# fusion_strategy: "concatenate"
```

---

## 技術文件 (Documentation)

### 已完成文件
1. **`TECHNICAL_WHITEPAPER_MULTIMODAL_FUSION.md`** (52+ 頁)
   - 完整學術白皮書
   - 架構圖 (ASCII)
   - 數學推導
   - 實現細節
   - 消融實驗設計
   - 適合作為論文基礎

2. **`MULTIMODAL_FUSION_PLAN.md`**
   - 初始規劃文件
   - 詳細實現步驟

3. **`COMPLETE_FUSION_SUMMARY.md`**
   - 實現總結
   - 三種策略對比

4. **`EARLY_FUSION_GUIDE.md`**
   - Early Fusion 詳細指南
   - STFT 參數優化建議

5. **`TECHNICAL_WHITEPAPER_DUAL_EEG.md`**
   - ART (Dual EEG Transformer) 技術白皮書

---

## 預期實驗結果 (Expected Results)

### 性能預測 (基於文獻與架構複雜度)

| 策略 | 預期準確率 | 預期 F1 | 訓練時間 | 推論時間 |
|------|-----------|---------|----------|----------|
| **Early Fusion** | 65-72% | 0.62-0.70 | 快 (~2h) | 快 |
| **Late Fusion (logits)** | 70-75% | 0.68-0.73 | 中 (~3h) | 中 |
| **Late Fusion (features)** | 72-78% | 0.70-0.76 | 中 (~3h) | 中 |
| **Mid Fusion** | **75-82%** | **0.73-0.80** | 慢 (~5h) | 慢 |

**預期發現**:
- Early Fusion: 基準線，但損失EEG時序信息
- Late Fusion (logits): 簡單有效，但缺乏跨模態交互
- Late Fusion (features): 比logits好，特徵融合更靈活
- **Mid Fusion**: **最佳性能**，IBS token與cross-attention帶來顯著提升

### 消融實驗預期影響

**Mid Fusion 組件貢獻**:
- IBS Token: +2-4% 準確率提升
- Cross-Modal Attention: +3-5% 準確率提升
- Symmetric Fusion (full vs basic): +1-2% 準確率提升
- Shared Weights vs Independent: 性能相近，但參數量減半

---

## 下一步行動 (Next Steps)

### 立即執行
1. **開始訓練三個基準模型**:
   ```bash
   # Terminal 1
   python Experiments/scripts/train_late_fusion.py --config Experiments/configs/late_fusion.yaml

   # Terminal 2
   python Experiments/scripts/train_mid_fusion.py --config Experiments/configs/mid_fusion.yaml

   # Terminal 3
   python Experiments/scripts/train_early_fusion.py --config Experiments/configs/early_fusion.yaml
   ```

2. **監控訓練**:
   - 登入 WandB: https://wandb.ai/
   - 查看專案: `eyegaze-eeg-classification`
   - 比較三個 runs

### 中期任務
3. **執行消融實驗** (訓練完成後):
   - Mid Fusion 消融 (4個實驗)
   - Late Fusion 消融 (2個實驗)
   - Early Fusion 消融 (1個實驗)

4. **結果分析**:
   - 生成混淆矩陣
   - t-SNE 特徵可視化
   - 注意力權重可視化

### 論文撰寫
5. **基於技術白皮書撰寫正式論文**:
   - 使用 `TECHNICAL_WHITEPAPER_MULTIMODAL_FUSION.md` 作為基礎
   - 添加實驗結果與分析
   - 準備投稿至 NeurIPS/ICCV/CVPR/IEEE TPAMI

---

## 故障排除 (Troubleshooting)

### 常見問題

**Q1: CUDA Out of Memory**
```bash
# 解決方案：減少 batch size
# 修改配置文件:
per_device_train_batch_size: 8  # 原本 16
per_device_eval_batch_size: 16  # 原本 32
```

**Q2: 資料讀取錯誤**
```bash
# 確認 G: 磁碟已掛載
# 或修改為絕對路徑
image_base_path: "G:/共用雲端硬碟/..."
```

**Q3: 預訓練模型找不到**
```bash
# 確認路徑存在
ls "C:/Users/user/pythonproject/EyeGaze-Multimodal/Experiments/outputs/vit_class_multiply/best_model.pt"
ls "C:/Users/user/pythonproject/EyeGaze-Multimodal/Experiments/outputs/dual_eeg_transformer/best_model.pt"

# 或設置為 null 從頭訓練
image_model_path: null
eeg_model_path: null
```

**Q4: WandB 登入問題**
```bash
# 首次使用需要登入
wandb login
# 輸入 API key (從 https://wandb.ai/authorize 獲取)
```

---

## 聯絡資訊 (Contact)

如有問題，請檢查:
1. 技術白皮書: `TECHNICAL_WHITEPAPER_MULTIMODAL_FUSION.md`
2. 程式碼註解: 所有模型檔案都有詳細註解
3. 配置檔案: `Experiments/configs/*.yaml`

---

**祝訓練順利！Good luck with training! 🚀**

生成時間: 2025-11-13
版本: v1.0 - Implementation Complete
