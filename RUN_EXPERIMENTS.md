# 實驗執行指令快速參考 (Quick Experiment Commands)

## 🚀 基準模型訓練 (Baseline Training)

### 1. Late Fusion - 決策層融合
```bash
# Features mode (推薦)
python Experiments/scripts/train_late_fusion.py --config Experiments/configs/late_fusion.yaml

# 如果要測試 logits mode，修改 late_fusion.yaml:
# fusion_mode: "logits"
```

**預期輸出**:
- 訓練集準確率: ~70-80%
- 驗證集準確率: ~65-75%
- F1 分數: ~0.70-0.76
- 訓練時間: ~3小時 (50 epochs)

---

### 2. Mid Fusion - 中間層融合 【主要貢獻】
```bash
# Full configuration (所有組件開啟)
python Experiments/scripts/train_mid_fusion.py --config Experiments/configs/mid_fusion.yaml
```

**預期輸出**:
- 訓練集準確率: ~75-85%
- 驗證集準確率: ~70-82%
- F1 分數: ~0.73-0.80
- 訓練時間: ~5小時 (50 epochs)

---

### 3. Early Fusion - 輸入層融合
```bash
# Average mode (推薦)
python Experiments/scripts/train_early_fusion.py --config Experiments/configs/early_fusion.yaml

# 如果要測試 concatenate mode，修改 early_fusion.yaml:
# fusion_strategy: "concatenate"
```

**預期輸出**:
- 訓練集準確率: ~65-75%
- 驗證集準確率: ~60-72%
- F1 分數: ~0.62-0.70
- 訓練時間: ~2小時 (50 epochs)

---

## 🔬 消融實驗 (Ablation Studies)

### Mid Fusion 消融實驗 (最重要)

#### A1: 移除 IBS Token
```bash
# Step 1: 修改配置
# 編輯 Experiments/configs/mid_fusion.yaml
# use_ibs_token: false

# Step 2: 修改 run_name
# run_name: "mid-fusion-no-ibs"

# Step 3: 運行訓練
python Experiments/scripts/train_mid_fusion.py --config Experiments/configs/mid_fusion.yaml
```

**預期影響**: 準確率下降 2-4%

---

#### A2: 移除 Cross-Modal Attention
```bash
# Step 1: 修改配置
# use_cross_attention: false
# run_name: "mid-fusion-no-cross-attn"

# Step 2: 運行訓練
python Experiments/scripts/train_mid_fusion.py --config Experiments/configs/mid_fusion.yaml
```

**預期影響**: 準確率下降 3-5%

---

#### A3: 移除兩者 (IBS + Cross-Attention)
```bash
# Step 1: 修改配置
# use_ibs_token: false
# use_cross_attention: false
# run_name: "mid-fusion-minimal"

# Step 2: 運行訓練
python Experiments/scripts/train_mid_fusion.py --config Experiments/configs/mid_fusion.yaml
```

**預期影響**: 準確率下降 5-8%，接近 Late Fusion 性能

---

#### A4: Symmetric Fusion Mode 對比
```bash
# Basic mode (sum + mul 僅)
# symmetric_fusion_mode: "basic"
# run_name: "mid-fusion-basic-sym"

# Learnable mode (可學習權重)
# symmetric_fusion_mode: "learnable"
# run_name: "mid-fusion-learnable-sym"

# Full mode (baseline)
# symmetric_fusion_mode: "full"
# run_name: "mid-fusion-full-sym"
```

---

#### A5: Weight Sharing 策略
```bash
# 獨立影像編碼器
# image_shared_weights: false
# run_name: "mid-fusion-independent-img"

# 獨立 EEG 編碼器
# eeg_shared_weights: false
# run_name: "mid-fusion-independent-eeg"

# 全部獨立
# image_shared_weights: false
# eeg_shared_weights: false
# run_name: "mid-fusion-all-independent"
```

**預期影響**: 性能相近，但參數量增加 ~2x

---

### Late Fusion 消融實驗

#### B1: Fusion Mode 對比
```bash
# Logits mode
# fusion_mode: "logits"
# run_name: "late-fusion-logits"

# Features mode (baseline)
# fusion_mode: "features"
# run_name: "late-fusion-features"
```

**預期影響**: Features mode 比 logits mode 高 2-3%

---

#### B2: Freeze Strategy 對比
```bash
# 凍結所有預訓練模型
# freeze_image: true
# freeze_eeg: true
# run_name: "late-fusion-frozen"

# 僅凍結 EEG
# freeze_image: false
# freeze_eeg: true
# run_name: "late-fusion-frozen-eeg"

# 全部微調 (baseline)
# freeze_image: false
# freeze_eeg: false
# run_name: "late-fusion-finetune-all"
```

---

### Early Fusion 消融實驗

#### C1: Fusion Strategy 對比
```bash
# Concatenate mode
# fusion_strategy: "concatenate"
# run_name: "early-fusion-concat"

# Average mode (baseline)
# fusion_strategy: "average"
# run_name: "early-fusion-avg"
```

---

## 📊 完整實驗矩陣 (Complete Experimental Matrix)

### 建議執行順序

**階段 1: 基準模型** (同時運行)
```bash
# Terminal 1
python Experiments/scripts/train_early_fusion.py --config Experiments/configs/early_fusion.yaml

# Terminal 2
python Experiments/scripts/train_late_fusion.py --config Experiments/configs/late_fusion.yaml

# Terminal 3
python Experiments/scripts/train_mid_fusion.py --config Experiments/configs/mid_fusion.yaml
```

**階段 2: Mid Fusion 消融** (順序運行，每個 ~5小時)
1. 無 IBS token
2. 無 Cross-Attention
3. 無兩者
4. Symmetric mode (basic, learnable)
5. Independent encoders

**階段 3: Late Fusion 消融** (順序運行，每個 ~3小時)
1. Logits mode
2. Frozen models
3. Frozen EEG only

**階段 4: Early Fusion 消融** (順序運行，每個 ~2小時)
1. Concatenate mode

**總預估時間**: ~60-70 小時 (如果順序運行)

---

## 🎯 關鍵結果指標 (Key Metrics to Track)

### WandB 監控
登入: https://wandb.ai/
專案: `eyegaze-eeg-classification`

### 主要指標
- **Accuracy**: 整體準確率
- **F1-Score**: 加權 F1 分數 (3類別)
- **Per-Class F1**: Single, Competition, Cooperation 各別 F1
- **Loss Curves**: 訓練/驗證損失曲線

### Mid Fusion 額外指標
- **loss_cls**: 分類損失
- **loss_ibs**: IBS token 重建損失
- **Cross-Attention Weights**: 跨模態注意力權重分布

### Late Fusion 額外指標
- **loss_fused**: 融合分支損失
- **loss_img**: 影像分支損失
- **loss_eeg**: EEG 分支損失

---

## 🛠 除錯指令 (Debugging Commands)

### 檢查資料載入
```bash
python -c "
import sys
sys.path.append('.')
from Data.processed.multimodal_dataset import MultimodalDataset
import json

with open('Data/metadata/complete_metadata.json', 'r') as f:
    metadata = json.load(f)

dataset = MultimodalDataset(
    metadata=metadata,
    image_base_path='G:/共用雲端硬碟/CNElab_林佳誼_Gaze/B.GazeImage/01.data/bgOn_heatmapOn_trajOn',
    eeg_base_path='G:/共用雲端硬碟/CNElab_林佳誼_Gaze/B.GazeImage/01.data/EEGseg',
    mode='train',
    train_test_split=0.2,
    random_seed=42
)

print(f'Dataset size: {len(dataset)}')
print(f'First sample keys: {dataset[0].keys()}')
print(f'Image shape: {dataset[0][\"img1\"].shape}')
print(f'EEG shape: {dataset[0][\"eeg1\"].shape}')
"
```

### 測試 Late Fusion 前向傳播
```bash
python test_late_fusion_fix.py
```

### 快速測試訓練 (1 epoch)
```bash
# 修改配置: num_train_epochs: 1
python Experiments/scripts/train_late_fusion.py --config Experiments/configs/late_fusion.yaml
```

### 檢查 GPU 使用
```bash
nvidia-smi
```

### 檢查模型參數量
```bash
python -c "
import sys
sys.path.append('.')
from Models.fusion.late_fusion import LateFusionModel

model = LateFusionModel(num_classes=3, fusion_mode='features')
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Total params: {total_params/1e6:.1f}M')
print(f'Trainable params: {trainable_params/1e6:.1f}M')
"
```

---

## 📈 結果分析腳本 (Result Analysis)

### 比較三種融合策略
```python
# analyze_results.py (需要創建)
import wandb

api = wandb.Api()
runs = api.runs("your-entity/eyegaze-eeg-classification")

results = []
for run in runs:
    if run.state == "finished":
        results.append({
            'name': run.name,
            'accuracy': run.summary.get('eval/accuracy', 0),
            'f1': run.summary.get('eval/f1', 0),
            'best_epoch': run.summary.get('best_epoch', 0)
        })

# 排序並打印
results.sort(key=lambda x: x['f1'], reverse=True)
for r in results:
    print(f"{r['name']:30s} | Acc: {r['accuracy']:.3f} | F1: {r['f1']:.3f}")
```

### 生成混淆矩陣
```python
# 在評估後添加
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=['Single', 'Competition', 'Cooperation'])
disp.plot()
plt.savefig('confusion_matrix.png')
```

---

## 💡 優化建議 (Optimization Tips)

### 如果顯存不足 (OOM)
1. 減少 batch size: `per_device_train_batch_size: 8`
2. 使用 gradient accumulation (需修改訓練腳本)
3. 使用混合精度訓練 (需修改訓練腳本)

### 如果訓練太慢
1. 增加 num_workers: `num_workers: 8`
2. 使用更小的模型 (修改 d_model)
3. 減少 epochs: `num_train_epochs: 30`

### 如果過擬合
1. 增加 dropout: `fusion_dropout: 0.5`
2. 增加 weight_decay: `weight_decay: 0.05`
3. 使用 data augmentation (需修改 dataset)

### 如果欠擬合
1. 增加模型容量: `fusion_hidden_dim: 1024`
2. 減少 weight_decay: `weight_decay: 0.001`
3. 訓練更多 epochs: `num_train_epochs: 100`

---

## ✅ 檢查清單 (Checklist)

開始訓練前:
- [ ] 確認 G: 磁碟已掛載
- [ ] 確認預訓練模型存在 (或設為 null)
- [ ] 確認 WandB 已登入 (`wandb login`)
- [ ] 確認 GPU 可用 (`nvidia-smi`)
- [ ] 確認配置檔案正確

訓練過程中:
- [ ] 監控 WandB 訓練曲線
- [ ] 檢查 loss 是否下降
- [ ] 檢查準確率是否上升
- [ ] 監控 GPU 記憶體使用

訓練完成後:
- [ ] 保存最佳模型 checkpoint
- [ ] 生成評估報告
- [ ] 可視化結果
- [ ] 更新實驗記錄

---

**準備開始實驗！Ready to start experiments! 🚀**
