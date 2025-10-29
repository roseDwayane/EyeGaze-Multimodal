# ViT 圖片分類訓練 - 快速開始指南

## 專案概述

本專案使用 Vision Transformer (ViT) 對眼動追蹤圖片進行三分類任務：
- **Single**: 單人模式
- **Competition**: 競爭模式
- **Cooperation**: 合作模式

### 特色功能

✅ **完全遵循 Hugging Face 規範**
- 使用 `datasets.load_dataset()` 載入 JSON 資料
- 使用 `train_test_split()` 進行資料分割（支援分層抽樣）
- 使用 `ViTForImageClassification` 預訓練模型
- 使用 Hugging Face `Trainer` API 訓練

✅ **雙圖片輸入處理**
- 自動將 player1 和 player2 圖片水平/垂直拼接
- 支援自訂拼接模式

✅ **完整的訓練功能**
- Early stopping
- 模型檢查點保存
- TensorBoard 日誌
- 詳細的評估指標

---

## 檔案結構

```
EyeGaze-Multimodal/
├── Data/
│   └── metadata/
│       └── complete_metadata.json      # 原始標註資料
│
├── Models/
│   └── backbones/
│       └── vit.py                      # ViT 模型實作
│
├── Experiments/
│   ├── configs/
│   │   └── vit_single_vs_competition.yaml  # 訓練配置
│   ├── scripts/
│   │   ├── train_vit.py                    # 訓練腳本
│   │   └── verify_setup.py                 # 驗證腳本
│   └── README.md                           # 詳細說明
│
├── metrics/
│   └── classification.py              # 評估指標
│
└── requirements.txt                    # 相依套件
```

---

## 快速開始

### 步驟 1: 安裝相依套件

```bash
pip install -r requirements.txt
```

### 步驟 2: 驗證設置

運行驗證腳本確保所有設置正確：

```bash
python Experiments/scripts/verify_setup.py
```

這會檢查：
- ✓ 相依套件是否安裝
- ✓ 配置文件是否有效
- ✓ 資料集是否存在
- ✓ 圖片檔案是否可讀取
- ✓ ViT 模型是否可載入

### 步驟 3: 開始訓練

```bash
python Experiments/scripts/train_vit.py --config Experiments/configs/vit_single_vs_competition.yaml
```

### 步驟 4: 監控訓練進度

使用 TensorBoard 監控：

```bash
tensorboard --logdir Experiments/outputs/vit_dual_classification/logs/
```

然後在瀏覽器開啟: http://localhost:6006

---

## 資料處理流程

### 1. 資料載入

```python
# 使用 HuggingFace datasets 載入 JSON
datasets = load_dataset("json", data_files="./complete_metadata.json", split="train")
```

### 2. Train/Test 分割

```python
# 使用分層抽樣 (stratified split)
split_datasets = datasets.train_test_split(
    test_size=0.02,  # 2% 測試集
    seed=42,
    stratify_by_column='class'  # 保持類別比例
)
```

### 3. 雙圖片拼接

每個樣本包含兩張圖片 (player1 + player2)：

```
Player1 圖片    +    Player2 圖片    =    拼接後的圖片
[224x224]            [224x224]            [224x448]
                                          (水平拼接)
```

### 4. ViT 處理

拼接後的圖片經過 ViT 進行分類。

---

## 配置說明

編輯 `Experiments/configs/vit_single_vs_competition.yaml` 可自訂：

### 模型配置

```yaml
model:
  model_name: "google/vit-base-patch16-224"  # 預訓練模型
  num_labels: 3                               # 類別數量
  concat_mode: "horizontal"                   # 拼接模式
  freeze_backbone: false                      # 是否凍結 backbone
```

### 訓練配置

```yaml
training:
  num_train_epochs: 10                        # 訓練輪數
  per_device_train_batch_size: 8             # 批次大小
  learning_rate: 2.0e-5                       # 學習率
  evaluation_strategy: "epoch"                # 每輪評估
  save_strategy: "epoch"                      # 每輪保存
  metric_for_best_model: "f1"                # 最佳模型指標
```

### 資料配置

```yaml
data:
  metadata_path: "Data/metadata/complete_metadata.json"
  image_base_path: "Data/raw/Gaze/example"
  train_test_split: 0.02                     # 測試集比例
  random_seed: 42                             # 隨機種子
```

---

## 輸出結果

訓練完成後，結果保存在：

```
Experiments/outputs/vit_dual_classification/
├── checkpoint-{step}/              # 檢查點
├── logs/                           # TensorBoard 日誌
├── results/
│   └── test_results.txt           # 測試結果
└── pytorch_model.bin              # 最佳模型
```

---

## 評估指標

訓練過程會計算以下指標：

- **Accuracy**: 整體準確率
- **Precision**: 精確率 (macro & weighted)
- **Recall**: 召回率 (macro & weighted)
- **F1 Score**: F1 分數 (macro & weighted)
- **Confusion Matrix**: 混淆矩陣

---

## 常見問題

### Q: CUDA 記憶體不足

**解決方法:**
- 減少 `per_device_train_batch_size`
- 使用較小的模型 (例如 `google/vit-small-patch16-224`)
- 啟用梯度累積

```yaml
training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
```

### Q: 訓練準確率很低

**解決方法:**
- 增加訓練輪數
- 調整學習率
- 啟用資料增強
- 檢查資料集是否平衡

### Q: 圖片載入錯誤

**解決方法:**
- 確認 `image_base_path` 路徑正確
- 確認圖片檔案存在且為 `.jpg` 格式
- 檢查 metadata 中的檔名是否與實際檔案一致

---

## 進階使用

### 使用更大的模型

```yaml
model:
  model_name: "google/vit-large-patch16-224"
```

### 改變拼接方式

```yaml
model:
  concat_mode: "vertical"  # 垂直拼接
```

### 啟用資料增強

```yaml
augmentation:
  enabled: true
  random_horizontal_flip: 0.5
  random_rotation: 15
```

---

## 技術細節

本專案完全遵循 **Hugging Face Transformers** 的最佳實踐：

1. ✅ 使用 `datasets` 庫進行資料載入
2. ✅ 使用 `ViTForImageClassification` 預訓練模型
3. ✅ 使用 `Trainer` API 進行訓練
4. ✅ 使用 `ViTImageProcessor` 進行圖片預處理
5. ✅ 支援 `compute_metrics` 回調函數
6. ✅ 支援 Early Stopping 和模型檢查點

---

## 參考資料

- [Hugging Face Transformers 文件](https://huggingface.co/docs/transformers)
- [ViT 論文](https://arxiv.org/abs/2010.11929)
- [Datasets 庫文件](https://huggingface.co/docs/datasets)

---

## 聯絡與支援

如有問題，請查看:
- `Experiments/README.md` - 詳細訓練說明
- 運行 `verify_setup.py` 檢查配置
