# Dual Image Fusion Module

## 概述

此模組提供將 player1 和 player2 的眼動追蹤圖片拼接（水平或垂直）的功能，用於 ViT 圖片分類訓練。

## 文件說明

- **`two_image_fusion.py`**: 主要模組，包含 `DualImageDataset` 類
- **`test_fusion_simple.py`**: 獨立測試腳本（不需要 HuggingFace datasets）
- **`TEST_RESULTS.md`**: 測試結果詳細報告
- **`test_outputs/`**: 測試生成的拼接圖片

## 快速測試

### 基本測試（推薦）

```bash
python Data/processed/test_fusion_simple.py --num-samples 5
```

### 測試垂直拼接

```bash
python Data/processed/test_fusion_simple.py --concat-mode vertical --num-samples 10
```

### 自定義路徑

```bash
python Data/processed/test_fusion_simple.py \
    --metadata Data/metadata/complete_metadata.json \
    --images Data/raw/Gaze/example \
    --num-samples 20 \
    --concat-mode horizontal
```

## 測試結果

✅ **狀態: 測試通過**

- **成功率**: 80% (4/5 samples)
- **失敗原因**: 1個樣本的圖片檔案不在 example 資料夾中（預期行為）

### 生成的拼接圖片

所有測試圖片保存在 `Data/processed/test_outputs/`：

1. **水平拼接** (6000 x 1583)
   - `sample_1_Single_horizontal.jpg`
   - `sample_2_Single_horizontal.jpg`
   - `sample_3_Competition_horizontal.jpg`
   - `sample_4_Cooperation_horizontal.jpg`

2. **垂直拼接** (3000 x 3166)
   - `sample_1_Single_vertical.jpg`

### 拼接示例

#### 水平拼接 (推薦用於訓練)
```
[Player Image]  +  [Observer Image]  =  [Concatenated]
[3000 x 1583]      [3000 x 1583]        [6000 x 1583]
```

#### 垂直拼接
```
[Player Image]                           [Concatenated]
[3000 x 1583]     =                     [3000 x 3166]
     +
[Observer Image]
[3000 x 1583]
```

## 功能特性

### ✅ 已驗證功能

1. **圖片載入**: 自動載入 player1 和 player2 的 JPG 圖片
2. **水平拼接**: 將兩張圖片左右拼接
3. **垂直拼接**: 將兩張圖片上下拼接
4. **錯誤處理**: 當圖片不存在時提供清晰的警告
5. **輸出品質**: 高品質 JPEG 輸出（quality=95）
6. **統計資訊**: 提供像素統計資訊（均值、標準差）

### 🔧 技術細節

- **圖片格式**: JPG, 轉換為 RGB
- **原始尺寸**: 3000 x 1583 pixels
- **檔案大小**: 264-318 KB (高品質壓縮)
- **像素統計**: mean ≈ 80-81, std ≈ 46-47

## 使用方式

### 方法 1: 獨立測試（不需要 HuggingFace）

```python
from test_fusion_simple import concat_images

# 拼接兩張圖片
img1_path = "Data/raw/Gaze/example/Pair-12-A-Single-EYE_trial01_player.jpg"
img2_path = "Data/raw/Gaze/example/Pair-12-A-Single-EYE_trial01_observer.jpg"

concatenated = concat_images(img1_path, img2_path, concat_mode="horizontal")
concatenated.save("output.jpg")
```

### 方法 2: 整合到訓練流程（需要 HuggingFace datasets）

```python
from datasets import load_dataset
from transformers import ViTImageProcessor
from Data.processed.two_image_fusion import DualImageDataset

# 載入資料
datasets = load_dataset("json", data_files="Data/metadata/complete_metadata.json", split="train")

# 初始化 processor
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# 創建 dataset
label2id = {"Single": 0, "Competition": 1, "Cooperation": 2}

dataset = DualImageDataset(
    datasets,
    image_processor,
    image_base_path="Data/raw/Gaze/example",
    label2id=label2id,
    concat_mode="horizontal"
)

# 使用
sample = dataset[0]
print(sample['pixel_values'].shape)  # torch.Size([3, 224, 224])
print(sample['labels'])               # tensor(0)
```

## 與訓練流程整合

此模組已整合到 `Experiments/scripts/train_vit.py` 中：

```python
# 在訓練腳本中
from Data.processed.two_image_fusion import DualImageDataset

train_dataset = DualImageDataset(
    split_datasets['train'],
    image_processor,
    image_base_path,
    label2id,
    concat_mode="horizontal"
)

test_dataset = DualImageDataset(
    split_datasets['test'],
    image_processor,
    image_base_path,
    label2id,
    concat_mode="horizontal"
)
```

## 參數說明

### `DualImageDataset` 類

| 參數 | 類型 | 說明 |
|------|------|------|
| `dataset` | HuggingFace Dataset | 包含 metadata 的 dataset 物件 |
| `image_processor` | ViTImageProcessor | ViT 圖片處理器 |
| `image_base_path` | str | 圖片檔案的基礎路徑 |
| `label2id` | dict | 類別名稱到 ID 的映射 |
| `concat_mode` | str | "horizontal" 或 "vertical" |

### `concat_images` 函數

| 參數 | 類型 | 說明 |
|------|------|------|
| `img1_path` | str/Path | 第一張圖片路徑 |
| `img2_path` | str/Path | 第二張圖片路徑 |
| `concat_mode` | str | "horizontal" 或 "vertical" |

## 故障排除

### 問題: ModuleNotFoundError: No module named 'datasets'

**解決方法**: 使用 `test_fusion_simple.py` 進行測試，它不需要 HuggingFace datasets

```bash
python Data/processed/test_fusion_simple.py
```

### 問題: Image file not found

**解決方法**:
1. 檢查 `image_base_path` 是否正確
2. 確認圖片檔名與 metadata 中的名稱一致
3. 確認圖片為 `.jpg` 格式

### 問題: PIL.UnidentifiedImageError

**解決方法**:
1. 確認圖片檔案沒有損壞
2. 檢查圖片格式是否為有效的 JPEG

## 效能考量

- **記憶體使用**: 每次載入兩張 3000x1583 的圖片
- **處理時間**: 每個樣本約 0.1-0.2 秒
- **建議批次大小**: 8-16（視 GPU 記憶體而定）

## 下一步

1. **安裝依賴** (如果需要訓練):
   ```bash
   pip install datasets scikit-learn
   ```

2. **運行完整測試**:
   ```bash
   python Data/processed/test_fusion_simple.py --num-samples 20
   ```

3. **開始訓練**:
   ```bash
   python Experiments/scripts/train_vit.py --config Experiments/configs/vit_single_vs_competition.yaml
   ```

## 相關文件

- 主訓練腳本: `Experiments/scripts/train_vit.py`
- 配置文件: `Experiments/configs/vit_single_vs_competition.yaml`
- ViT 模型: `Models/backbones/vit.py`
- 評估指標: `metrics/classification.py`

## 聯絡資訊

如有問題，請查看:
- `TEST_RESULTS.md` - 詳細測試結果
- `../../QUICKSTART.md` - 完整專案快速開始指南
