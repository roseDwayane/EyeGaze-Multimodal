# 變更日誌：TensorBoard → Wandb

## 修改摘要

已將訓練可視化從 TensorBoard 改為 Weights & Biases (wandb)

---

## 修改的文件

### 1. `Experiments/scripts/train_vit.py`

#### 添加的功能

**Import wandb** (第 35 行):
```python
import wandb
```

**Wandb 初始化** (第 114-127 行):
```python
# Initialize wandb
wandb_config = config.get('wandb', {})
wandb.init(
    project=wandb_config.get('project', 'eyegaze-vit-classification'),
    name=wandb_config.get('run_name', None),
    config={
        'model': config['model'],
        'training': config['training'],
        'data': {k: v for k, v in config['data'].items() if k != 'image_base_path'},
    },
    tags=wandb_config.get('tags', ['vit', 'dual-image', 'eyegaze']),
    notes=wandb_config.get('notes', 'ViT training for dual-image eye-gaze classification'),
)
logger.info(f"Wandb run initialized: {wandb.run.name}")
```

**記錄測試結果** (第 250 行):
```python
# Log final test results to wandb
wandb.log({f"test/{k}": v for k, v in test_results.items()})
```

**結束 wandb run** (第 263-264 行):
```python
# Finish wandb run
wandb.finish()
logger.info("Wandb run finished")
```

---

### 2. `Experiments/configs/vit_single_vs_competition.yaml`

**修改 report_to** (第 64 行):
```yaml
report_to: ["wandb"]  # Changed from tensorboard to wandb
```

**添加 wandb 配置** (第 110-120 行):
```yaml
# Weights & Biases Configuration
wandb:
  project: "eyegaze-vit-classification"
  run_name: null  # Auto-generate if null
  tags:
    - "vit"
    - "dual-image"
    - "eyegaze"
    - "single-vs-competition"
  notes: "ViT training for dual-image eye-gaze classification (Single/Competition/Cooperation)"
  entity: null  # Set your wandb username/team here if needed
```

---

### 3. `requirements.txt`

**替換 tensorboard 為 wandb** (第 16 行):
```txt
wandb>=0.15.0
```

---

### 4. 新增文件

- **`WANDB_SETUP.md`**: Wandb 完整使用指南
- **`CHANGELOG_WANDB.md`**: 本文件

---

## 使用方式

### 安裝 wandb

```bash
pip install wandb
```

### 首次使用：登入 wandb

```bash
wandb login
```

這會打開瀏覽器讓你登入並獲取 API key。

### 運行訓練

```bash
python Experiments/scripts/train_vit.py
```

### 查看結果

訓練開始後，終端會顯示 wandb dashboard 的 URL：

```
wandb: 🚀 View run at https://wandb.ai/username/eyegaze-vit-classification/runs/run-id
```

點擊該 URL 即可即時查看訓練過程！

---

## Wandb 追蹤的內容

### 自動追蹤（由 HuggingFace Trainer 提供）

- ✅ 訓練 loss
- ✅ 驗證 loss
- ✅ 學習率變化
- ✅ 每個 epoch 的評估指標 (accuracy, precision, recall, F1)
- ✅ GPU/CPU 使用率
- ✅ 訓練速度和進度

### 額外記錄

- ✅ 最終測試集結果
- ✅ 模型配置
- ✅ 訓練配置
- ✅ 資料配置

---

## 配置選項

你可以在 `Experiments/configs/vit_single_vs_competition.yaml` 中自定義：

```yaml
wandb:
  project: "your-project-name"      # 專案名稱
  run_name: "custom-run-name"       # 自定義 run 名稱（null 為自動生成）
  tags:                              # 添加標籤
    - "your-tag"
    - "experiment-1"
  notes: "Your experiment notes"     # 實驗說明
  entity: "your-username"            # 你的 wandb username
```

---

## 優勢

### Wandb vs TensorBoard

| 功能 | Wandb | TensorBoard |
|-----|-------|-------------|
| 即時雲端同步 | ✅ | ❌ |
| 多實驗比較 | ✅ 簡單直觀 | ⚠️ 需要手動設置 |
| 團隊協作 | ✅ | ❌ |
| 系統監控 | ✅ GPU/CPU/Memory | ⚠️ 有限 |
| 超參數搜索 | ✅ Sweeps | ❌ |
| 模型版本管理 | ✅ Artifacts | ❌ |
| 手機 App | ✅ | ❌ |

---

## 進階功能

### 1. 比較多次實驗

在 wandb dashboard 中選擇多個 runs，點擊 "Compare" 即可並排比較。

### 2. 超參數搜索

創建 `sweep.yaml`:
```yaml
program: Experiments/scripts/train_vit.py
method: bayes
metric:
  name: eval/f1
  goal: maximize
parameters:
  learning_rate:
    min: 1e-5
    max: 5e-5
  per_device_train_batch_size:
    values: [8, 16, 32]
```

運行：
```bash
wandb sweep sweep.yaml
wandb agent your-sweep-id
```

### 3. 保存最佳模型

Wandb 會自動保存檢查點，你可以在 Files 標籤中下載。

---

## 離線模式

如果沒有網路：

```bash
export WANDB_MODE=offline
python Experiments/scripts/train_vit.py
```

稍後同步：
```bash
wandb sync wandb/offline-run-*
```

---

## 故障排除

### 問題：找不到 wandb 模組

```bash
pip install wandb
```

### 問題：API key 錯誤

```bash
wandb login --relogin
```

### 問題：不想使用 wandb

修改配置文件：
```yaml
training:
  report_to: []  # 空列表
```

---

## 資源

- 📚 Wandb 文檔: https://docs.wandb.ai/
- 🎓 快速開始: https://docs.wandb.ai/quickstart
- 🤗 HuggingFace 整合: https://docs.wandb.ai/guides/integrations/huggingface
- 📖 完整指南: 參考 `WANDB_SETUP.md`

---

## 回滾到 TensorBoard

如果想回到 TensorBoard，只需修改：

1. `requirements.txt`: `wandb` → `tensorboard`
2. `vit_single_vs_competition.yaml`: `report_to: ["tensorboard"]`
3. `train_vit.py`: 移除 wandb import 和相關代碼

---

**🎉 享受更好的訓練可視化體驗！**
