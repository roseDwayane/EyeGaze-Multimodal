## 1. 專案概述區塊 (Project Overview)
### 1.1 研究背景與目標

本計畫旨在開發一套多模態深度學習框架，利用眼動影像 (Eye-Gaze Images) 與 腦電圖 (EEG) 訊號，精確辨識與分類雙人互動情境。具體研究目標如下：

| 類別          | 說明           |
| ----------- | ------------ |
| Single      | 單人模式 - 無社交互動 |
| Competition | 競爭模式 - 對抗性互動 |
| Cooperation | 合作模式 - 協作性互動 |

**核心價值**：結合視覺注意力（Eye Gaze 顯式行為）與神經活動（EEG 隱式腦狀態），實現對社交互動的多維度理解。

### 1.2 資料型態

```text
輸入資料 (每個樣本包含):
├── Eye Gaze 影像
│   ├── Player 1: img1 (3, 224, 224) - RGB 熱力圖
│   └── Player 2: img2 (3, 224, 224) - RGB 熱力圖
│
└── EEG 訊號
    ├── Player 1: eeg1 (C, T) - C 通道, T 時間點
    └── Player 2: eeg2 (C, T) - C 通道, T 時間點
```


輸出:
└── 類別標籤: y ∈ {0: Single, 1: Competition, 2: Cooperation}

### 1.3 技術框架：五大核心模組

#### 模組 1｜Vision Transformer 視覺注視融合
針對雙人眼動影像，探討不同融合策略對分類效能的影響。

| 組件       | 說明                                |
| -------- | --------------------------------- |
| **骨幹網路** | Vision Transformer (ViT) 處理雙人注視熱圖 |
| **融合機制** | Dual-Image Fusion，支援多種模式          |

**融合模式比較**：
- **空間拼接**：Horizontal / Vertical Concatenation
- **像素運算**：Add (共同特徵)、Multiply (重疊強化)、Subtract (差異凸顯)

##### 已實作：Early Fusion ViT (Channel Concatenation)

**技術實作**：
- **框架**: PyTorch + timm (PyTorch Image Models)
- **模型**: `vit_base_patch16_224` (預訓練 ImageNet-21K)
- **融合策略**: Channel Concatenation (6-channel input)

**架構流程**：
```
img_a (B, 3, 224, 224) ──┐
                        ├── torch.cat(dim=1) ──► (B, 6, 224, 224) ──► ViT ──► (B, 3)
img_b (B, 3, 224, 224) ──┘
```

**關鍵修改 - 6 Channel Patch Embedding**：
- 原始 ViT 的 `patch_embed.proj` 為 Conv2d(3, 768, 16, 16)
- 修改為 Conv2d(6, 768, 16, 16)
- 權重初始化策略：複製原始 3-channel 權重到前後兩組 channels，保留預訓練特徵提取能力

**程式碼位置**: `3_Models/backbones/early_fusion_vit.py`

#### 模組 2｜Dual EEG Transformer 跨腦同步建模

基於 Artifact Removal Transformer 架構改良，專為捕捉雙人神經同步特徵設計。

|                         | Dual EEG Transformer                                |
| ----------------------- | --------------------------------------------------- |
| Cross-Brain Comm.跨腦通訊機制 | Siamese Encoder + Cross-Brain Attn 雙向資訊流動學習         |
| IBS Token 腦際同步標記        | PLV + Power Correlation Theta/Alpha/Beta/Gamma 頻帶量化 |
| Symmetric Fusion 對稱性融合  | f(P1,P2) = f(P2,P1) 排列不變性 + Symmetry Loss           |

#### 模組 3｜Early Fusion — 輸入層融合

在進入編碼器前整合雙人視覺資訊，保留原始空間關聯：
- **融合時機**：編碼前（Input Level）
- **目標**：保留雙方視線重疊與差異的原始空間特徵
- **策略**：Dual-Image Fusion（拼接 / 像素運算）

#### 模組 4｜Mid Fusion — 特徵層融合 

在編碼過程中實現跨模態深度交互，捕捉神經耦合動態：
- **融合時機**：特徵提取中（Feature Level）
- **核心機制**：Cross-Brain Attention 雙向資訊流動
- **捕捉目標**：
- 合作情境 → 同步增強特徵
- 競爭情境 → 對抗差異特徵

#### 模組 5｜Late Fusion — 決策層融合

在最終決策階段整合視覺與腦波模態，採用不確定性感知策略。
- **融合時機**：分類前（Decision Level）
- **核心機制**：Uncertainty-Aware Fuzzy Gating
- **運作方式**：根據各模態可信度動態調整權重

Gaze Encoder ──┐
			 ├──▶ Fuzzy Gating ──▶ 加權融合 ──▶ 最終分類
EEG Encoder  ──┘      (不確定性感知)

#### 融合層級對比

| 層級        | 融合時機 | 核心技術                  | 交互深度   |
| --------- | ---- | --------------------- | ------ |
| **Early** | 編碼前  | Dual-Image Fusion     | 淺層（空間） |
| **Mid**   | 編碼中  | Cross-Brain Attention | 深層（特徵） |
| **Late**  | 分類前  | Fuzzy Gating          | 決策級    |

### 1.4 專案結構總覽

```text
EyeGaze-Multimodal/
│
├─ data/                       # [原有] 數據資料
│  ├─ raw/
│  │  ├─ eeg/
│  │  └─ gaze/
│  ├─ processed/
│  └─ metadata/
│
├─ preprocessing/              # [原有] 預處理代碼
│  ├─ eeg/
│  ├─ gaze/
│  └─ sync/                    # 同步 gaze-EEG、切段、label 對齊
│
├─ models/                     # [原有] 模型架構
│  ├─ backbones/               # CNN, Transformer, etc.
│  ├─ fusion/                  # EEG + gaze 多模態融合
│  └─ heads/                   # classifier / regressor / decoder
│
├─ experiments/                # [原有] 實驗執行
│  ├─ configs/                 # yaml/json 實驗設定
│  ├─ runs/                    # 機器生成的 Log, ckpt
│  └─ scripts/                 # train.py / eval.py / crossval.py
│
├─ metrics/                    # [原有] 評估指標
│  ├─ classification.py
│  ├─ regression.py
│  └─ stats/                   # 統計檢定, p-value, effect size
│
├─ utils/                      # [原有] 通用工具
│  ├─ io.py
│  ├─ viz.py
│  └─ constants.py
│
├─ analysis/                   # ★ 新增：實驗後的分析與解釋 (Human-readable)
│  ├─ notebooks/               # Jupyter Notebooks (EDA, 錯誤分析, 可視化草稿)
│  └─ reports/                 # 實驗總結報告 (Markdown/PDF)
│       ├─ exp_001_baseline.md # 針對特定實驗的詳細說明文檔
│       └─ exp_002_fusion.md   
│
└─ manuscript/                 # ★ 新增：論文寫作區
   ├─ src/                     # LaTeX 源碼或 Markdown 草稿
   │  ├─ 01_introduction.tex
   │  ├─ 02_methods.tex
   │  ├─ 03_results.tex
   │  ├─ 04_discussion.tex
   │  └─ 05_conclusion.tex
   │
   ├─ figures/                 # 論文用的最終圖檔 (由 utils.viz 或 notebooks 生成)
   │  ├─ fig1_system_arch.pdf
   │  ├─ fig2_eeg_preprocessing.png
   │  └─ ...
   │
   ├─ tables/                  # 論文用的表格 (Excel/CSV 或 LaTeX code)
   │  ├─ table1_demographics.csv
   │  └─ table2_performance_comparison.tex
   │
   └─ main.tex                 # 論文主文件 (匯總 src 內的章節)
```

### 1.5 技術棧

| 類別         | 技術                                |
| ---------- | --------------------------------- |
| **深度學習框架** | PyTorch, HuggingFace Transformers |
| **實驗追蹤**   | Weights & Biases (wandb)          |
| **資料處理**   | NumPy, SciPy (訊號處理)               |
| **可視化**    | matplotlib, seaborn               |
| **配置管理**   | YAML                              |
| **環境**     | Windows 或 Linux, CUDA             |

### 1.6 研究目標與預期成果

  **短期目標**：
  - 建立三種融合策略的基準模型
  - 完成消融實驗驗證各組件貢獻

  **中期目標**：
  - Mid Fusion 達到 ~80% F1-score
  - 完成論文撰寫

  **長期目標**：
  - 投稿頂級會議/期刊 (NeurIPS, ICCV, IEEE TPAMI)
  - 開源專案程式碼