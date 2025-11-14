# 跨模態融合系統實現總結

## 完成狀態：✅ 全部完成

本文檔總結已實現的 Eye Gaze Image × EEG Signal 跨模態融合系統。

---

## 📁 已實現的文件結構

```
EyeGaze-Multimodal/
│
├── Models/fusion/
│   ├── __init__.py                   ✅ 模組導出
│   ├── late_fusion.py                ✅ Late Fusion（基線）
│   ├── mid_fusion.py                 ✅ Mid Fusion（主模型）
│   ├── symmetric_fusion.py           ✅ 對稱融合算子
│   └── cross_modal_attention.py      ✅ 跨模態注意力
│
├── Data/processed/
│   └── multimodal_dataset.py         ✅ 跨模態數據加載器
│
├── Experiments/
│   ├── scripts/
│   │   ├── train_late_fusion.py     ✅ Late Fusion 訓練腳本
│   │   └── train_mid_fusion.py      ✅ Mid Fusion 訓練腳本
│   │
│   └── configs/
│       ├── late_fusion.yaml         ✅ Late Fusion 配置
│       └── mid_fusion.yaml          ✅ Mid Fusion 配置
│
└── Documentation/
    ├── MULTIMODAL_FUSION_PLAN.md    ✅ 詳細規劃文檔
    └── MULTIMODAL_FUSION_SUMMARY.md ✅ 本文檔
```

---

## 🎯 三層融合策略實現

### A. Late Fusion（已實現）✅

**描述**: 後期融合，兩個預訓練模態在最後階段融合

**架構**:
```
[ViT-Img1, ViT-Img2] → logits_img
[EEG-P1, EEG-P2] → logits_eeg
─────────────────────────────
Fusion: Weighted Average or MLP
─────────────────────────────
Final Logits
```

**特點**:
- ✅ 簡單穩定
- ✅ 可獨立訓練單模態模型
- ✅ 支持logits fusion和features fusion兩種模式
- ✅ 包含輔助損失（L_img, L_eeg）

**訓練命令**:
```bash
python Experiments/scripts/train_late_fusion.py \
    --config Experiments/configs/late_fusion.yaml
```

---

### B. Mid Fusion（已實現）✅ **【主要貢獻】**

**描述**: 中層融合，四塔架構，跨模態交互，IBS token

**架構**:
```
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ ViT-P1  │ │ ViT-P2  │ │ EEG-P1  │ │ EEG-P2  │
│ Encoder │ │ Encoder │ │ Encoder │ │ Encoder │
└────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
     │           │            │           │
   z_img_p1   z_img_p2    z_eeg_p1   z_eeg_p2
     │           │            │           │
     └─────┬─────┘            └─────┬─────┘
           │                        │
  ┌────────▼────────┐      ┌────────▼────────┐
  │ Symmetric Fusion│      │ Symmetric Fusion│
  │ (Sum,Mul,Diff)  │      │ (Sum,Mul,Diff)  │
  └────────┬────────┘      └────────┬────────┘
           │                        │
        z_img_fused              z_eeg_fused
           │                        │
           │        ┌──────────┐    │
           │        │IBS Token │    │
           │        │ (PLV+...)│    │
           │        └────┬─────┘    │
           │             │          │
           └──────┬──────┴──────┬───┘
                  │             │
         ┌────────▼─────────────▼────────┐
         │  Cross-Modal Attention        │
         │  (Bidirectional)              │
         └────────┬──────────────────────┘
                  │
         [z_img', z_eeg', ibs_token]
                  │
         ┌────────▼────────┐
         │ Classification  │
         │ Head (MLP)      │
         └─────────────────┘
```

**關鍵組件**:

1. **對稱融合算子** (`symmetric_fusion.py`)
   - ✅ Sum: z1 + z2
   - ✅ Hadamard Product: z1 * z2
   - ✅ Absolute Difference: |z1 - z2|
   - ✅ 保證排列不變性

2. **跨模態注意力** (`cross_modal_attention.py`)
   - ✅ 雙向交叉注意力
   - ✅ Co-Attention機制
   - ✅ 門控融合
   - ✅ 多模態Transformer Block

3. **IBS Token**
   - ✅ Phase Locking Value (PLV)
   - ✅ 功率相關性
   - ✅ 相位差
   - ✅ 多頻段特徵（θ, α, β, γ）

4. **四塔編碼器**
   - ✅ 支持共享權重（Siamese）或獨立權重
   - ✅ 圖像：ViT編碼器
   - ✅ EEG：Temporal Conv + Transformer編碼器

**訓練命令**:
```bash
python Experiments/scripts/train_mid_fusion.py \
    --config Experiments/configs/mid_fusion.yaml
```

**配置選項**:
```yaml
model:
  # 控制組件啟用
  use_ibs_token: true          # 是否使用IBS token
  use_cross_attention: true    # 是否使用跨模態注意力
  fusion_mode: "basic"         # 對稱融合模式：basic/all/simple

  # 權重共享
  image_shared_weights: true   # ViT是否共享權重
  eeg_shared_weights: true     # EEG encoder是否共享權重
```

---

### C. Early Fusion（未實現）

**描述**: 早期融合，將EEG轉換為頻譜圖，與圖像堆疊

**狀態**: ⚠️ 待實現（可作為消融實驗對照）

**簡單實現建議**:
```python
# Convert EEG to spectrogram
eeg_spectrogram = stft(eeg)  # (B, C, F, T)

# Concatenate with image
combined = torch.cat([img, eeg_spectrogram], dim=1)  # (B, 3+C, H, W)

# Single ViT with modified input channels
vit = VisionTransformer(in_channels=3+C, ...)
```

---

## 📊 數據流程

### MultimodalDataset (`multimodal_dataset.py`)

**功能**:
- ✅ 同時加載 Eye Gaze images 和 EEG signals
- ✅ 滑動窗口分割EEG（window_size=1024, stride=512）
- ✅ 圖像數據增強（訓練時）
- ✅ EEG預處理選項（CAR, bandpass, z-score）

**輸出格式**:
```python
{
    'img1': (3, 224, 224),      # Player 1 圖像
    'img2': (3, 224, 224),      # Player 2 圖像
    'eeg1': (32, 1024),         # Player 1 EEG
    'eeg2': (32, 1024),         # Player 2 EEG
    'labels': scalar            # 類別標籤 (0/1/2)
}
```

---

## 🔧 核心技術創新

### 1. 對稱融合算子

**數學表達**:
```python
z_sum = z1 + z2                    # 加法（捕捉共同模式）
z_mul = z1 * z2                    # 乘法（建模交互）
z_diff = |z1 - z2|                 # 差異（量化不同）
z_fused = Proj([z_sum, z_mul, z_diff])
```

**優勢**:
- ✅ 排列不變性：f(z1, z2) = f(z2, z1)
- ✅ 適用於對稱任務（兩個玩家地位平等）
- ✅ 可解釋性強

### 2. IBS Token

**計算方式**:
```python
# 對於每個頻段 (θ, α, β, γ)
for freq_band in [theta, alpha, beta, gamma]:
    # 計算同步度量
    plv = compute_plv(phase1, phase2)
    pow_corr = compute_power_correlation(power1, power2)
    phase_diff = mean(phase1 - phase2)

    features.append([plv, pow_corr, phase_diff])

# 投影到模型維度
ibs_token = MLP(features)  # (B, d_model)
```

**意義**:
- ✅ 顯式建模跨腦同步
- ✅ 結合領域知識（神經科學）
- ✅ 可解釋性強（可視化PLV等指標）

### 3. 跨模態交互

**雙向Cross-Attention**:
```python
# Image attends to EEG
z_img' = z_img + CrossAttn(Q=z_img, K=z_eeg, V=z_eeg)

# EEG attends to Image
z_eeg' = z_eeg + CrossAttn(Q=z_eeg, K=z_img, V=z_img)
```

**優勢**:
- ✅ 明確的跨模態信息流
- ✅ 捕捉不同模態間的依賴關係
- ✅ 端到端可微分學習

---

## 📈 預期性能

| 方法 | 預期準確率 | 預期F1 | 參數量 | 特點 |
|------|-----------|--------|--------|------|
| Image-only (ViT) | ~65% | ~0.55 | ~86M | 基線 |
| EEG-only (Dual Transformer) | ~70% | ~0.60 | ~7M | 基線 |
| **Late Fusion** | **~75%** | **~0.68** | **~93M** | 簡單穩定 |
| **Mid Fusion (Full)** | **~80%** | **~75%** | **~95M** | 🎯 主模型 |
| Mid Fusion (No IBS) | ~78% | ~0.72 | ~94M | 消融 |
| Mid Fusion (No Cross-Attn) | ~77% | ~0.70 | ~93M | 消融 |
| Early Fusion | ~72% | ~0.65 | ~90M | 對照 |

---

## 🚀 使用指南

### 快速開始

**1. 訓練Late Fusion（基線）**
```bash
# 使用預訓練模型
python Experiments/scripts/train_late_fusion.py \
    --config Experiments/configs/late_fusion.yaml

# 配置文件設置預訓練路徑
# late_fusion.yaml:
#   image_model_path: "path/to/vit_best_model.pt"
#   eeg_model_path: "path/to/eeg_best_model.pt"
```

**2. 訓練Mid Fusion（主模型）**
```bash
python Experiments/scripts/train_mid_fusion.py \
    --config Experiments/configs/mid_fusion.yaml
```

**3. 消融實驗**

禁用IBS token:
```yaml
# mid_fusion.yaml
model:
  use_ibs_token: false
```

禁用Cross-Attention:
```yaml
model:
  use_cross_attention: false
```

使用簡化對稱融合:
```yaml
model:
  fusion_mode: "simple"  # 只用sum和mul
```

---

## 📝 實驗建議

### 消融實驗設計

| 實驗 | IBS Token | Cross-Attn | 融合模式 | 目的 |
|------|-----------|-----------|---------|------|
| Full Model | ✅ | ✅ | basic | 完整模型 |
| No IBS | ❌ | ✅ | basic | IBS token作用 |
| No Cross-Attn | ✅ | ❌ | basic | 跨模態注意力作用 |
| Simple Fusion | ✅ | ✅ | simple | 對稱算子數量 |
| All Fusion | ✅ | ✅ | all | 增加concat算子 |

### 對比實驗

1. **單模態 vs 多模態**
   - Image-only
   - EEG-only
   - Late Fusion
   - Mid Fusion

2. **融合層次**
   - Early Fusion（待實現）
   - Mid Fusion
   - Late Fusion

3. **權重共享策略**
   - Siamese（共享權重）
   - Independent（獨立權重）

---

## 🔬 可視化分析

### 建議的可視化

1. **注意力圖**
   - 跨模態注意力權重
   - 哪些圖像區域關注哪些EEG時段

2. **IBS Token分析**
   - PLV值分布（不同類別）
   - 頻段特異性同步模式

3. **特徵空間**
   - t-SNE可視化融合後的特徵
   - 不同模態特徵的分離度

4. **混淆矩陣**
   - 各個模型的類別預測性能
   - 哪些類別最難區分

---

## 📚 相關文獻

### 跨模態融合
- Baltrusaitis et al., "Multimodal Machine Learning: A Survey", PAMI 2019
- Ngiam et al., "Multimodal Deep Learning", ICML 2011

### 對稱架構
- Vaswani et al., "Attention Is All You Need", NeurIPS 2017
- Zhang et al., "Dual Attention Networks", CVPR 2019

### EEG與腦同步
- Hasson et al., "Brain-to-brain coupling", TiCS 2012
- Jiang et al., "Leader emergence through interpersonal neural synchronization", PNAS 2015

---

## 🎓 論文寫作建議

### 核心貢獻點

1. **四塔架構 + IBS Token**
   - 首次將Inter-Brain Synchrony顯式建模為learnable token
   - 結合神經科學先驗（PLV等）與深度學習

2. **對稱融合算子**
   - 確保排列不變性
   - 適用於對稱社交交互任務

3. **跨模態交互機制**
   - 雙向Cross-Attention
   - 允許圖像和EEG相互影響

### 實驗部分

1. **主實驗**: 三種融合策略對比
2. **消融實驗**: 各個組件的貢獻
3. **可視化分析**: 注意力圖、IBS token分析
4. **錯誤分析**: 哪些樣本被錯誤分類

---

## ✅ 實現檢查清單

### 核心組件
- [x] MultimodalDataset（跨模態數據加載）
- [x] SymmetricFusionOperators（對稱融合）
- [x] CrossModalAttention（跨模態注意力）
- [x] IBSTokenGenerator（IBS token生成）
- [x] LateFusionModel（後期融合基線）
- [x] MidFusionModel（中層融合主模型）

### 訓練與配置
- [x] Late Fusion訓練腳本
- [x] Mid Fusion訓練腳本
- [x] Late Fusion配置文件
- [x] Mid Fusion配置文件

### 文檔
- [x] 詳細規劃文檔（MULTIMODAL_FUSION_PLAN.md）
- [x] 實現總結文檔（本文檔）
- [x] 技術白皮書（TECHNICAL_WHITEPAPER_DUAL_EEG.md）

### 待完成（可選）
- [ ] Early Fusion實現
- [ ] 可視化工具（注意力圖、t-SNE）
- [ ] 推理腳本（inference.py）
- [ ] Demo程序

---

## 💡 關鍵takeaways

1. **Late Fusion**: 最簡單穩定，適合快速驗證跨模態有效性
2. **Mid Fusion**: 論文核心貢獻，四塔架構 + IBS token + 跨模態注意力
3. **IBS Token**: 結合領域知識的創新，可解釋性強
4. **對稱融合**: 確保排列不變性，適用於對稱任務

---

## 📞 聯繫與支持

如有問題或需要進一步實現，請參考：
- 詳細規劃：`MULTIMODAL_FUSION_PLAN.md`
- 技術白皮書：`TECHNICAL_WHITEPAPER_DUAL_EEG.md`
- 代碼文檔：各模塊的docstrings

---

**Last Updated**: 2025-11-13
**Status**: ✅ 完整實現完成
**Next Steps**: 開始訓練並進行實驗！
