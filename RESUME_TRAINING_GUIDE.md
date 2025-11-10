# 訓練中斷恢復指南 (Resume Training Guide)

## ✅ 已實現的功能

訓練腳本現在完全支持中斷後恢復訓練！

---

## 🎯 功能說明

### 1. 自動檢測最新 Checkpoint ✅
- 訓練腳本會自動在輸出目錄中尋找最新的 checkpoint
- 使用 `--resume` 參數即可自動恢復

### 2. 保持 Wandb Run 連續 ✅
- Resume 時會嘗試繼續同一個 wandb run
- 訓練曲線不會中斷，保持連續性

### 3. 命令行控制 ✅
- `--resume`: 自動檢測並從最新 checkpoint 恢復
- `--checkpoint`: 指定特定的 checkpoint 路徑

### 4. 取消 Early Stopping ✅
- 已移除 early stopping 機制
- 訓練會完整運行設定的 epoch 數

---

## 🚀 使用方式

### 方法 1: 自動恢復（推薦）

```bash
# 正常訓練
python Experiments/scripts/train_vit.py

# 訓練中斷（Ctrl+C 或意外中斷）
# ... 中斷 ...

# 自動從最新 checkpoint 恢復
python Experiments/scripts/train_vit.py --resume
```

**輸出範例**:
```
INFO - Auto-detected checkpoint: Experiments/outputs/vit_class_subtract/checkpoint-500
INFO - Resuming wandb run: abc123xyz
INFO - Training will resume from checkpoint: checkpoint-500
INFO - Resuming training from: Experiments/outputs/vit_class_subtract/checkpoint-500
```

### 方法 2: 指定 Checkpoint

```bash
# 從特定 checkpoint 恢復
python Experiments/scripts/train_vit.py \
  --checkpoint Experiments/outputs/vit_class_subtract/checkpoint-500
```

### 方法 3: 從頭開始（預設）

```bash
# 不使用任何參數，從頭開始訓練
python Experiments/scripts/train_vit.py
```

---

## 📊 Checkpoint 結構

訓練時會自動保存 checkpoint：

```
Experiments/outputs/vit_class_subtract/
├── checkpoint-100/              # Epoch 1 結束
│   ├── config.json
│   ├── model.safetensors
│   ├── optimizer.pt             # 優化器狀態
│   ├── scheduler.pt             # 學習率調度器
│   ├── trainer_state.json       # 訓練狀態
│   └── training_args.bin
├── checkpoint-200/              # Epoch 2 結束
├── checkpoint-300/              # Epoch 3 結束（最新）
└── wandb/                       # Wandb 運行數據
    └── run-20231029_123456-abc123/
```

**配置說明**:
```yaml
training:
  save_strategy: "epoch"      # 每個 epoch 保存
  save_total_limit: 3         # 只保留最近 3 個
```

---

## 🔍 Resume 恢復的內容

使用 `--resume` 時，以下內容會完整恢復：

### ✅ 模型狀態
- 模型權重（所有參數）
- 分類頭權重

### ✅ 優化器狀態
- Adam 優化器的動量
- 參數的一階和二階矩估計

### ✅ 學習率調度器
- 當前學習率
- Warmup 進度
- Cosine decay 狀態

### ✅ 訓練進度
- 當前 epoch
- Global step (已訓練的 batch 數)
- 最佳指標值

### ✅ 隨機數狀態
- PyTorch 隨機數生成器
- NumPy 隨機數生成器

### ✅ Wandb 運行
- 繼續同一個 run_id
- 保持訓練曲線連續

---

## 📋 常見場景

### 場景 1: 訓練中手動中斷（Ctrl+C）

```bash
# 訓練中...
python Experiments/scripts/train_vit.py
# 訓練到 Epoch 5/10，按 Ctrl+C 中斷

# 稍後繼續
python Experiments/scripts/train_vit.py --resume
# ✅ 從 Epoch 5 繼續訓練到 Epoch 10
```

### 場景 2: GPU 記憶體溢出

```bash
# 訓練崩潰
python Experiments/scripts/train_vit.py
# Out of memory 錯誤

# 調整 batch size 後繼續
# 修改 config: per_device_train_batch_size: 4
python Experiments/scripts/train_vit.py --resume
# ⚠️ 注意：改變 batch size 可能影響訓練動態
```

### 場景 3: 服務器重啟/斷電

```bash
# 訓練中斷
# ... 服務器重啟 ...

# 重啟後繼續
python Experiments/scripts/train_vit.py --resume
# ✅ 自動從最新 checkpoint 恢復
```

### 場景 4: 想從特定 checkpoint 重新開始

```bash
# 發現 Epoch 5 效果最好，想從那裡繼續調整
python Experiments/scripts/train_vit.py \
  --checkpoint Experiments/outputs/vit_class_subtract/checkpoint-500
```

### 場景 5: 完全重新訓練

```bash
# 刪除舊的 checkpoints
rm -rf Experiments/outputs/vit_class_subtract/

# 從頭開始
python Experiments/scripts/train_vit.py
```

---

## ⚙️ 配置說明

### Checkpoint 保存策略

```yaml
training:
  # 保存策略
  save_strategy: "epoch"           # 每個 epoch 保存
  # save_strategy: "steps"         # 或每 N steps 保存
  # save_steps: 500                # steps 模式下的間隔

  # 保存限制
  save_total_limit: 3              # 最多保留 3 個 checkpoint

  # 最佳模型
  load_best_model_at_end: true     # 訓練結束載入最佳模型
  metric_for_best_model: "f1"      # 用於判斷最佳的指標
  greater_is_better: true          # F1 越大越好
```

### Early Stopping（已禁用）

```yaml
# Early stopping (DISABLED)
# early_stopping_patience: 3
# early_stopping_threshold: 0.001
```

現在訓練會運行完整的 `num_train_epochs`，不會提前停止。

---

## 🔧 技術細節

### 自動檢測邏輯

```python
def get_last_checkpoint(output_dir):
    """尋找最新的 checkpoint"""
    checkpoints = [d for d in os.listdir(output_dir)
                   if d.startswith('checkpoint-')]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    return checkpoints[-1]  # 返回編號最大的
```

### Wandb Run ID 恢復

```python
# 從 checkpoint 目錄中提取 wandb run_id
wandb_run_path = os.path.join(output_dir, 'wandb')
if os.path.exists(wandb_run_path):
    run_dirs = [d for d in os.listdir(wandb_run_path)
                if d.startswith('run-')]
    latest_run = sorted(run_dirs)[-1]
    wandb_id = latest_run.split('-')[-1]
    wandb.init(id=wandb_id, resume="must")
```

---

## 📝 日誌輸出

### Resume 成功

```
2025-10-29 16:30:00 - INFO - Loading configuration from config.yaml
2025-10-29 16:30:01 - INFO - Auto-detected checkpoint: checkpoint-500
2025-10-29 16:30:02 - INFO - Resuming wandb run: abc123xyz
2025-10-29 16:30:03 - INFO - Wandb run initialized: vit-subtract-run-1
2025-10-29 16:30:04 - INFO - Training will resume from checkpoint: checkpoint-500
2025-10-29 16:30:10 - INFO - Resuming training from: checkpoint-500
```

### 無 Checkpoint（從頭開始）

```
2025-10-29 16:30:00 - INFO - Loading configuration from config.yaml
2025-10-29 16:30:01 - INFO - No checkpoint found, starting from scratch
2025-10-29 16:30:02 - INFO - Wandb run initialized: vit-subtract-run-2
2025-10-29 16:30:10 - INFO - Starting training from scratch
```

---

## ⚠️ 注意事項

### 1. 配置文件變更

Resume 時應避免改變：
- ❌ `model_name` (模型架構)
- ❌ `num_labels` (分類類別數)
- ❌ `concat_mode` (融合模式)
- ⚠️ `learning_rate` (可以改，但會影響訓練)
- ⚠️ `batch_size` (可以改，但不推薦)

可以安全改變：
- ✅ `num_train_epochs` (延長訓練)
- ✅ `logging_steps` (日誌頻率)
- ✅ `save_total_limit` (保存數量)

### 2. 數據集變更

- ❌ 不要改變訓練數據
- ❌ 不要改變 `train_test_split` 比例
- ❌ 不要改變 `random_seed`

### 3. Checkpoint 管理

```yaml
save_total_limit: 3  # 只保留最近 3 個
```

舊的 checkpoint 會自動刪除，注意：
- 如果想保留某個 checkpoint，複製到別處
- 刪除 checkpoint 後無法恢復

### 4. 磁碟空間

每個 checkpoint 約佔用 ~350MB（ViT-base）：
- 3 個 checkpoint ≈ 1GB
- 確保有足夠的磁碟空間

---

## 🛠️ 故障排除

### 問題 1: "No checkpoint found" 但確實有 checkpoint

**原因**: Checkpoint 目錄名稱不符合格式

**解決**:
```bash
# 檢查目錄名稱
ls Experiments/outputs/vit_class_subtract/
# 應該看到 checkpoint-100, checkpoint-200 等

# 如果格式不對，手動指定
python train_vit.py --checkpoint path/to/checkpoint
```

### 問題 2: Resume 後 wandb 創建了新的 run

**原因**: 無法找到原 wandb run_id

**解決**:
- 正常情況會自動處理
- 如需手動指定，修改代碼中的 wandb.init()

### 問題 3: Resume 後訓練從 epoch 0 開始

**原因**: Checkpoint 可能損壞

**檢查**:
```bash
# 查看 trainer_state.json
cat checkpoint-500/trainer_state.json | grep epoch
```

**解決**: 使用更早的 checkpoint

### 問題 4: 改了配置後 resume 出錯

**原因**: 配置與 checkpoint 不兼容

**解決**:
- 恢復原配置
- 或刪除 checkpoint 從頭訓練

---

## 📚 參考資料

- HuggingFace Trainer: https://huggingface.co/docs/transformers/main_classes/trainer
- Resume Training: https://huggingface.co/docs/transformers/main_classes/trainer#resuming-training
- Wandb Resume: https://docs.wandb.ai/guides/runs/resuming

---

## ✅ 檢查清單

訓練前確認：
- [ ] 有足夠的磁碟空間（> 5GB）
- [ ] 配置文件正確
- [ ] 知道如何使用 `--resume`

Resume 前確認：
- [ ] Checkpoint 存在
- [ ] 配置文件未改變關鍵參數
- [ ] 數據集未變更

---

**🎉 現在可以放心訓練，不用擔心中斷了！**

## 快速命令

```bash
# 正常訓練
python Experiments/scripts/train_vit.py

# 恢復訓練（推薦）
python Experiments/scripts/train_vit.py --resume

# 指定 checkpoint
python Experiments/scripts/train_vit.py --checkpoint path/to/checkpoint

# 查看幫助
python Experiments/scripts/train_vit.py --help
```
