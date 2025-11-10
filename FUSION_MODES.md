# Image Fusion Modes 圖片融合模式

本專案支援多種圖片融合策略，用於將 player1 和 player2 的眼動追蹤圖片融合成單一輸入。

---

## 支援的融合模式

### 1. 拼接模式 (Concatenation)

#### `horizontal` - 水平拼接
- **描述**: 將兩張圖片左右並排拼接
- **輸出尺寸**: `[width1 + width2, max(height1, height2)]`
- **範例**: 3000×1583 + 3000×1583 → 6000×1583
- **用途**: 保留兩張圖片的完整資訊

```yaml
model:
  concat_mode: "horizontal"
```

#### `vertical` - 垂直拼接
- **描述**: 將兩張圖片上下堆疊拼接
- **輸出尺寸**: `[max(width1, width2), height1 + height2]`
- **範例**: 3000×1583 + 3000×1583 → 3000×3166
- **用途**: 保留兩張圖片的完整資訊

```yaml
model:
  concat_mode: "vertical"
```

---

### 2. 像素級融合 (Pixel-level Fusion)

這些模式會先將兩張圖片調整為相同尺寸，然後進行像素級運算。

#### `add` - 相加融合
- **描述**: 像素級相加後取平均
- **公式**: `fused = (img1 + img2) / 2`
- **輸出尺寸**: 與 img1 相同
- **特性**:
  - 保留兩張圖片的共同特徵
  - 增強重疊區域的信號
  - 值域保持在 [0, 255]
- **適用場景**: 當兩張圖片互補時效果好

```yaml
model:
  concat_mode: "add"
```

**視覺效果**: 兩張圖片的疊加，亮度增加

#### `multiply` - 相乘融合
- **描述**: 像素級相乘（歸一化）
- **公式**: `fused = (img1/255) * (img2/255) * 255`
- **輸出尺寸**: 與 img1 相同
- **特性**:
  - 強調兩張圖片的共同活躍區域
  - 抑制非共同區域
  - 對比度增強
- **適用場景**: 尋找兩個玩家共同關注的區域

```yaml
model:
  concat_mode: "multiply"
```

**視覺效果**: 只保留兩張圖片都有高亮的區域，整體變暗

#### `subtract` - 相減融合
- **描述**: 像素級相減取絕對值
- **公式**: `fused = |img1 - img2|`
- **輸出尺寸**: 與 img1 相同
- **特性**:
  - 強調兩張圖片的差異
  - 突出不同的關注區域
  - 相同區域變黑
- **適用場景**: 分析兩個玩家關注點的差異

```yaml
model:
  concat_mode: "subtract"
```

**視覺效果**: 差異越大越亮，相同區域變黑

---

## 使用方式

### 1. 修改配置文件

編輯 `Experiments/configs/vit_single_vs_competition.yaml`:

```yaml
model:
  concat_mode: "add"  # 改為你想要的融合模式
```

### 2. 運行訓練

```bash
python Experiments/scripts/train_vit.py
```

訓練腳本會自動使用配置文件中指定的融合模式。

### 3. 測試不同融合模式

使用測試腳本預覽融合效果：

```bash
# 測試相加融合
python Data/processed/test_fusion_simple.py --concat-mode add --num-samples 5

# 測試相乘融合
python Data/processed/test_fusion_simple.py --concat-mode multiply --num-samples 5

# 測試相減融合
python Data/processed/test_fusion_simple.py --concat-mode subtract --num-samples 5
```

生成的圖片會保存在 `Data/processed/test_outputs/`

---

## 融合模式比較

| 模式 | 輸出尺寸 | 信息保留 | 計算成本 | 適用場景 |
|-----|---------|---------|---------|---------|
| `horizontal` | 加倍寬度 | 100% | 低 | 保留完整信息 |
| `vertical` | 加倍高度 | 100% | 低 | 保留完整信息 |
| `add` | 不變 | 高 | 中 | 尋找共同特徵 |
| `multiply` | 不變 | 中 | 中 | 強調共同關注區域 |
| `subtract` | 不變 | 中 | 中 | 分析差異 |

---

## 實驗建議

### 快速實驗流程

1. **測試視覺效果**:
   ```bash
   python Data/processed/test_fusion_simple.py --concat-mode add --num-samples 10
   python Data/processed/test_fusion_simple.py --concat-mode multiply --num-samples 10
   python Data/processed/test_fusion_simple.py --concat-mode subtract --num-samples 10
   ```

2. **查看生成的圖片**，選擇最合適的融合模式

3. **運行完整訓練**，比較不同融合模式的性能

### 系統化實驗

為每種融合模式創建單獨的配置文件：

```bash
# 複製配置文件
cp Experiments/configs/vit_single_vs_competition.yaml \
   Experiments/configs/vit_fusion_add.yaml

# 修改 concat_mode
# 運行訓練
python Experiments/scripts/train_vit.py --config Experiments/configs/vit_fusion_add.yaml
```

在 wandb 中使用標籤來組織實驗：

```yaml
wandb:
  tags:
    - "fusion-add"
    - "vit"
    - "eyegaze"
```

---

## 技術細節

### 尺寸處理

對於像素級融合模式（add, multiply, subtract）：
- 如果兩張圖片尺寸不同，會將 img2 調整為 img1 的尺寸
- 使用 `BILINEAR` 插值進行縮放
- 保持原始圖片的寬高比

### 數值穩定性

所有像素級運算都使用 `float32` 進行計算，並使用 `np.clip` 確保最終值在 [0, 255] 範圍內：

```python
# 範例：相加融合
arr1 = np.array(img1, dtype=np.float32)
arr2 = np.array(img2, dtype=np.float32)
fused_arr = (arr1 + arr2) / 2.0
fused_arr = np.clip(fused_arr, 0, 255).astype(np.uint8)
```

### ViT 預處理

無論使用哪種融合模式，融合後的圖片都會經過 ViT 的標準預處理：
- 調整大小到 224×224
- 歸一化 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- 轉換為 tensor

---

## 預期結果

### 不同融合模式可能的表現

| 融合模式 | 預期優勢 | 潛在劣勢 |
|---------|---------|---------|
| `horizontal` | 完整信息，訓練最穩定 | 輸入尺寸大，計算成本高 |
| `vertical` | 完整信息 | 輸入尺寸大 |
| `add` | 強調共同關注，減少噪音 | 可能丟失個體差異 |
| `multiply` | 突出共同活躍區域 | 整體變暗，可能丟失單獨特徵 |
| `subtract` | 強調差異，區分競爭/合作 | 丟失共同特徵 |

### 任務適配性

- **Single vs Competition/Cooperation**:
  - `subtract` 可能表現較好（強調差異）
  - `multiply` 可能表現較好（共同關注）

- **三分類 (Single/Competition/Cooperation)**:
  - `horizontal` 最穩定（保留完整信息）
  - `add` 可能是好的平衡點

---

## 故障排除

### 問題：運行時錯誤

確保已安裝 numpy：
```bash
pip install numpy
```

### 問題：融合圖片全黑

檢查：
1. 原始圖片是否正確載入
2. 融合模式拼寫是否正確
3. 查看測試輸出的像素統計

### 問題：訓練效果差

嘗試：
1. 先測試視覺效果，確保融合結果合理
2. 嘗試不同的融合模式
3. 調整學習率和其他超參數
4. 使用 `horizontal` 作為 baseline 對比

---

## 代碼位置

- 融合實現: `Data/processed/two_image_fusion.py:84-150`
- 測試腳本: `Data/processed/test_fusion_simple.py:16-87`
- 配置文件: `Experiments/configs/vit_single_vs_competition.yaml:12`

---

## 引用與參考

這些融合策略參考了圖像融合和多模態學習的常見方法：

- **加法融合**: 類似於圖像疊加和超解析度重建
- **乘法融合**: 類似於注意力機制和顯著性檢測
- **減法融合**: 類似於變化檢測和差異分析

---

**祝實驗順利！🚀**
