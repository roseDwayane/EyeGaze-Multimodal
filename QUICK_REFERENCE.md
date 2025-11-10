# 快速參考卡 (Quick Reference)

## 🚀 訓練命令

### 基本訓練
```bash
# 正常訓練（從頭開始）
python Experiments/scripts/train_vit.py

# 自動恢復（推薦）⭐
python Experiments/scripts/train_vit.py --resume

# 指定 checkpoint 恢復
python Experiments/scripts/train_vit.py --checkpoint path/to/checkpoint-500
```

---

## 🎯 融合模式選擇

編輯配置文件 `Experiments/configs/vit_single_vs_competition.yaml`:

```yaml
model:
  concat_mode: "horizontal"  # 改為以下任一種：
```

| 模式 | 效果 | 尺寸 | 適用 |
|-----|------|------|------|
| `horizontal` | 左右拼接 | 6000×1583 | Baseline |
| `vertical` | 上下拼接 | 3000×3166 | 完整信息 |
| `add` | 相加平均 | 3000×1583 | 共同特徵 |
| `multiply` | 相乘 | 3000×1583 | 重疊區域 |
| `subtract` | 相減 | 3000×1583 | 差異分析 |

---

## 📊 監控訓練

### Wandb 可視化
```bash
# 訓練開始後，點擊終端顯示的 URL
wandb: 🚀 View run at https://wandb.ai/...
```

### 本地日誌
```bash
# 查看訓練日誌
tail -f Experiments/outputs/vit_class_subtract/logs/events.out.tfevents.*
```

---

## 🔧 常用修改

### 調整訓練參數

```yaml
training:
  num_train_epochs: 20              # 訓練輪數
  per_device_train_batch_size: 16   # Batch size
  learning_rate: 5.0e-5             # 學習率
```

### 更換模型

```yaml
model:
  model_name: "google/vit-large-patch16-224"  # 更大的模型
```

---

## ⚡ 快速測試融合模式

```bash
# 測試不同融合模式的視覺效果
python Data/processed/test_fusion_simple.py --concat-mode add --num-samples 5
python Data/processed/test_fusion_simple.py --concat-mode multiply --num-samples 5
python Data/processed/test_fusion_simple.py --concat-mode subtract --num-samples 5

# 查看輸出
ls Data/processed/test_outputs/
```

---

## 🛠️ 故障排除

### Out of Memory
```yaml
# 減小 batch size
training:
  per_device_train_batch_size: 4
```

### 訓練太慢
```yaml
# 減少 workers
training:
  dataloader_num_workers: 2
```

### Resume 不工作
```bash
# 手動指定 checkpoint
python train_vit.py --checkpoint Experiments/outputs/vit_class_subtract/checkpoint-500
```

---

## 📁 重要路徑

| 內容 | 路徑 |
|-----|------|
| 訓練腳本 | `Experiments/scripts/train_vit.py` |
| 配置文件 | `Experiments/configs/vit_single_vs_competition.yaml` |
| 融合模組 | `Data/processed/two_image_fusion.py` |
| Checkpoints | `Experiments/outputs/vit_class_subtract/checkpoint-*` |
| 測試輸出 | `Data/processed/test_outputs/` |

---

## 📖 完整文檔

- **Resume 指南**: `RESUME_TRAINING_GUIDE.md`
- **融合模式**: `FUSION_MODES.md`
- **Wandb 設置**: `WANDB_SETUP.md`
- **快速開始**: `QUICKSTART.md`

---

## 🎯 推薦工作流程

```bash
# 1. 測試融合模式
python Data/processed/test_fusion_simple.py --concat-mode subtract --num-samples 10

# 2. 修改配置文件
vim Experiments/configs/vit_single_vs_competition.yaml
# 設置 concat_mode: "subtract"

# 3. 開始訓練
python Experiments/scripts/train_vit.py

# 4. 如果中斷，恢復訓練
python Experiments/scripts/train_vit.py --resume

# 5. 在 wandb 查看結果
# 訪問終端顯示的 URL
```

---

**需要幫助？查看詳細文檔或運行 `python train_vit.py --help`**
