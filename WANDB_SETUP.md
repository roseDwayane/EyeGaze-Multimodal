# Wandb Setup Guide

本專案使用 **Weights & Biases (wandb)** 來追蹤和可視化訓練過程。

## 安裝 Wandb

```bash
pip install wandb
```

## 初始化 Wandb

首次使用時需要登入：

```bash
wandb login
```

這會打開瀏覽器讓你登入 wandb 帳號並獲取 API key。

如果你還沒有 wandb 帳號：
1. 前往 https://wandb.ai/
2. 免費註冊一個帳號
3. 複製你的 API key

## 配置

在 `Experiments/configs/vit_single_vs_competition.yaml` 中已經配置好 wandb：

```yaml
wandb:
  project: "eyegaze-vit-classification"  # 專案名稱
  run_name: null                          # 自動生成或自定義
  tags:                                   # 標籤
    - "vit"
    - "dual-image"
    - "eyegaze"
    - "single-vs-competition"
  notes: "ViT training for dual-image eye-gaze classification"
  entity: null                            # 你的 wandb username/team
```

### 自定義配置

你可以修改：

1. **專案名稱** (`project`): 在 wandb 中組織你的實驗
2. **Run 名稱** (`run_name`): 每次訓練的名稱，null 為自動生成
3. **標籤** (`tags`): 方便過濾和搜尋實驗
4. **Entity** (`entity`): 你的 wandb username 或 team name

## 訓練時使用 Wandb

直接運行訓練腳本即可：

```bash
python Experiments/scripts/train_vit.py
```

訓練過程會自動記錄：
- 訓練和驗證的 loss
- 評估指標 (accuracy, precision, recall, F1)
- 學習率變化
- 模型配置和超參數
- 最終測試結果

## 查看訓練結果

訓練開始後，終端會顯示 wandb run 的 URL，例如：

```
wandb: 🚀 View run at https://wandb.ai/your-username/eyegaze-vit-classification/runs/abc123
```

點擊該 URL 或前往 https://wandb.ai 查看：

### Wandb Dashboard 功能

1. **Charts**: 即時查看訓練曲線
   - Loss curves (train/eval)
   - Metrics (accuracy, F1, etc.)
   - Learning rate schedule

2. **System Metrics**:
   - GPU 使用率
   - CPU/Memory 使用
   - 訓練速度

3. **Config**:
   - 所有超參數
   - 模型配置
   - 資料配置

4. **Files**:
   - 保存的模型檔案
   - 訓練日誌

5. **Artifacts**:
   - 最佳模型版本
   - 評估結果

## Wandb 追蹤的指標

訓練腳本會自動記錄：

### 訓練過程中
- `train/loss`: 訓練 loss
- `eval/loss`: 驗證 loss
- `eval/accuracy`: 驗證準確率
- `eval/precision`: 驗證精確率（macro）
- `eval/recall`: 驗證召回率（macro）
- `eval/f1`: 驗證 F1 分數（macro）
- `train/learning_rate`: 當前學習率
- `train/epoch`: 當前 epoch

### 訓練結束後
- `test/eval_loss`: 測試集 loss
- `test/eval_accuracy`: 測試集準確率
- `test/eval_precision`: 測試集精確率
- `test/eval_recall`: 測試集召回率
- `test/eval_f1`: 測試集 F1 分數

## 離線模式

如果沒有網路連接，可以使用離線模式：

```bash
export WANDB_MODE=offline
python Experiments/scripts/train_vit.py
```

訓練完成後同步：

```bash
wandb sync wandb/offline-run-*
```

## 比較多次實驗

Wandb 最強大的功能之一是比較不同實驗：

1. 進入你的專案頁面
2. 選擇多個 runs
3. 點擊 "Compare"
4. 查看並排的圖表和指標對比

## 進階功能

### 自定義 Run 名稱

在配置文件中設置：

```yaml
wandb:
  run_name: "vit-base-lr2e5-bs8"
```

### 添加自定義標籤

```yaml
wandb:
  tags:
    - "vit"
    - "experiment-1"
    - "high-lr"
```

### 記錄額外資訊

在 `train_vit.py` 中可以添加：

```python
wandb.log({"custom_metric": value})
```

## 常見問題

### Q: 如何停止記錄到 wandb？

修改配置文件：

```yaml
training:
  report_to: []  # 空列表表示不報告
```

### Q: 如何刪除失敗的 runs？

在 wandb dashboard 中選擇 run → Settings → Delete run

### Q: 可以在訓練期間修改 notes 嗎？

可以！在 wandb dashboard 的 run 頁面直接編輯

### Q: 如何下載訓練好的模型？

在 wandb dashboard → Files → 下載 pytorch_model.bin

## Wandb vs TensorBoard

| 特性 | Wandb | TensorBoard |
|-----|-------|-------------|
| 雲端同步 | ✅ 自動 | ❌ 需要手動 |
| 多實驗比較 | ✅ 易用 | ⚠️ 複雜 |
| 系統監控 | ✅ GPU/CPU/Memory | ⚠️ 有限 |
| 團隊協作 | ✅ 內建 | ❌ 需要額外設置 |
| 模型版本管理 | ✅ Artifacts | ❌ 無 |
| 超參數搜索 | ✅ Sweeps | ❌ 需要其他工具 |
| 離線使用 | ✅ 支持 | ✅ 原生支持 |

## 資源連結

- Wandb 官網: https://wandb.ai/
- 文檔: https://docs.wandb.ai/
- 快速開始: https://docs.wandb.ai/quickstart
- Hugging Face 整合: https://docs.wandb.ai/guides/integrations/huggingface

## 小技巧

1. **使用 Sweep 進行超參數搜索**
   ```bash
   wandb sweep sweep.yaml
   wandb agent your-sweep-id
   ```

2. **Group 相關實驗**
   ```python
   wandb.init(group="experiment-group-1")
   ```

3. **保存最佳模型為 Artifact**
   ```python
   artifact = wandb.Artifact('model', type='model')
   artifact.add_file('model.pth')
   wandb.log_artifact(artifact)
   ```

---

現在你可以開始訓練並在 wandb 上追蹤你的實驗了！🚀
