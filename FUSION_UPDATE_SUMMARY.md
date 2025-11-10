# 圖片融合策略更新總結

## ✅ 已完成的修改

已成功將圖片融合策略從 2 種（水平/垂直拼接）擴展到 5 種，增加了像素級融合模式。

---

## 📝 修改的文件

### 1. `Data/processed/two_image_fusion.py`

**主要修改**:
- ✅ 添加 `numpy` import
- ✅ 擴展 `__init__` 的 `concat_mode` 參數說明
- ✅ 添加模式驗證邏輯
- ✅ 實現三種新的融合模式：
  - `add`: 像素級相加（取平均）
  - `multiply`: 像素級相乘（歸一化）
  - `subtract`: 像素級相減（絕對值）

**代碼位置**:
- 融合邏輯: `two_image_fusion.py:84-150`
- 模式驗證: `two_image_fusion.py:56-62`

### 2. `Data/processed/test_fusion_simple.py`

**主要修改**:
- ✅ 更新 `concat_images` 函數支持新的融合模式
- ✅ 添加 `numpy` 依賴
- ✅ 更新命令行參數 choices

### 3. `Experiments/configs/vit_single_vs_competition.yaml`

**主要修改**:
- ✅ 更新 `concat_mode` 參數說明
- ✅ 註明支持的所有融合模式

---

## 🎯 支援的融合模式

### 拼接模式（保留完整信息）

| 模式 | 描述 | 輸出尺寸 | 參數值 |
|-----|------|---------|--------|
| 水平拼接 | 左右並排 | 6000×1583 | `horizontal` |
| 垂直拼接 | 上下堆疊 | 3000×3166 | `vertical` |

### 像素級融合模式（單張圖片大小）

| 模式 | 描述 | 公式 | 參數值 |
|-----|------|------|--------|
| 相加 | 像素平均 | `(img1 + img2) / 2` | `add` |
| 相乘 | 像素相乘 | `(img1/255) * (img2/255) * 255` | `multiply` |
| 相減 | 絕對差值 | `\|img1 - img2\|` | `subtract` |

---

## 🧪 測試結果

所有 5 種融合模式均測試通過！

### 測試命令與結果

```bash
# 相加融合 ✓
python Data/processed/test_fusion_simple.py --concat-mode add --num-samples 2
# 輸出尺寸: 3000×1583, 均值: 80.73

# 相乘融合 ✓
python Data/processed/test_fusion_simple.py --concat-mode multiply --num-samples 2
# 輸出尺寸: 3000×1583, 均值: 33.46 (整體變暗，符合預期)

# 相減融合 ✓
python Data/processed/test_fusion_simple.py --concat-mode subtract --num-samples 2
# 輸出尺寸: 3000×1583, 均值: 2.12 (差異小，大部分變黑，符合預期)
```

### 視覺效果驗證

**Add (相加)**:
- ✅ 兩張圖片的疊加效果
- ✅ 共同關注區域增強
- ✅ 保持原始圖片的特徵

**Multiply (相乘)**:
- ✅ 只保留共同高亮區域
- ✅ 整體變暗（均值從 80 降到 33）
- ✅ 強調重疊特徵

**Subtract (相減)**:
- ✅ 突出差異區域
- ✅ 相同區域變黑（均值 2.12）
- ✅ 適合分析行為差異

---

## 🚀 使用方式

### 1. 測試融合效果（快速預覽）

```bash
# 測試所有模式
python Data/processed/test_fusion_simple.py --concat-mode horizontal --num-samples 5
python Data/processed/test_fusion_simple.py --concat-mode vertical --num-samples 5
python Data/processed/test_fusion_simple.py --concat-mode add --num-samples 5
python Data/processed/test_fusion_simple.py --concat-mode multiply --num-samples 5
python Data/processed/test_fusion_simple.py --concat-mode subtract --num-samples 5
```

查看生成的圖片: `Data/processed/test_outputs/`

### 2. 訓練時使用（修改配置文件）

編輯 `Experiments/configs/vit_single_vs_competition.yaml`:

```yaml
model:
  concat_mode: "add"  # 改為你想要的融合模式
```

可選值: `"horizontal"`, `"vertical"`, `"add"`, `"multiply"`, `"subtract"`

### 3. 運行訓練

```bash
python Experiments/scripts/train_vit.py
```

訓練腳本會自動讀取配置並使用指定的融合模式。

---

## 📊 實驗建議

### 快速實驗流程

1. **視覺檢查**: 先用測試腳本生成樣本圖片
2. **選擇模式**: 根據視覺效果選擇合適的融合模式
3. **運行訓練**: 修改配置文件並開始訓練
4. **對比分析**: 在 wandb 中比較不同融合模式的效果

### 系統化實驗

為每種融合模式創建單獨的配置文件並使用 wandb 標籤：

```bash
# 複製配置
cp Experiments/configs/vit_single_vs_competition.yaml \
   Experiments/configs/vit_fusion_add.yaml

# 修改 concat_mode 和 wandb tags
vim Experiments/configs/vit_fusion_add.yaml
```

```yaml
model:
  concat_mode: "add"

wandb:
  tags:
    - "fusion-add"
    - "vit"
    - "eyegaze"
```

然後運行：
```bash
python Experiments/scripts/train_vit.py --config Experiments/configs/vit_fusion_add.yaml
```

---

## 🔍 技術細節

### 像素級融合的處理流程

1. **尺寸統一**: 如果兩張圖片尺寸不同，將 img2 調整為 img1 的尺寸（使用 BILINEAR 插值）

2. **類型轉換**: 轉換為 `float32` 進行計算，避免溢出

3. **融合運算**:
   - **Add**: `(arr1 + arr2) / 2.0`
   - **Multiply**: `(arr1/255) * (arr2/255) * 255`
   - **Subtract**: `np.abs(arr1 - arr2)`

4. **範圍限制**: 使用 `np.clip(arr, 0, 255)` 確保值在有效範圍

5. **轉回圖片**: 轉為 `uint8` 並創建 PIL Image

### ViT 預處理

所有融合模式的輸出都會經過相同的 ViT 預處理：
- 調整為 224×224
- 標準化 (ImageNet mean/std)
- 轉為 tensor

---

## 📈 預期效果分析

### 不同融合模式的特性

| 融合模式 | 適合場景 | 預期優勢 | 潛在限制 |
|---------|---------|---------|---------|
| `horizontal` | Baseline | 完整信息 | 輸入尺寸大 |
| `vertical` | Baseline | 完整信息 | 輸入尺寸大 |
| `add` | 共同特徵 | 增強共同關注區域 | 可能模糊個體差異 |
| `multiply` | 重疊分析 | 強調共同活躍區域 | 整體變暗 |
| `subtract` | 差異分析 | 突出行為差異 | 丟失共同特徵 |

### 任務建議

**Single vs Competition/Cooperation 分類**:
- 推薦嘗試順序: `horizontal` → `add` → `subtract`
- `subtract` 可能特別有效（強調行為差異）

**三分類 (Single/Competition/Cooperation)**:
- 推薦: `horizontal` (最穩定) 或 `add` (平衡點)

---

## 📂 新增的文件

- **`FUSION_MODES.md`**: 詳細的融合模式說明文檔
- **`FUSION_UPDATE_SUMMARY.md`**: 本文件（更新總結）
- **`test_all_fusion_modes.py`**: 快速測試所有模式的腳本

---

## ✨ 關鍵代碼位置

| 功能 | 文件 | 行數 |
|-----|------|------|
| 融合模式實現 | `Data/processed/two_image_fusion.py` | 84-150 |
| 模式驗證 | `Data/processed/two_image_fusion.py` | 56-62 |
| 測試函數 | `Data/processed/test_fusion_simple.py` | 16-87 |
| 配置參數 | `Experiments/configs/vit_single_vs_competition.yaml` | 12 |

---

## 🎓 使用範例

### 範例 1: 快速測試不同模式

```bash
# 測試相加融合
python Data/processed/test_fusion_simple.py --concat-mode add --num-samples 10

# 查看輸出圖片
ls Data/processed/test_outputs/
```

### 範例 2: 訓練使用相加融合

```yaml
# vit_single_vs_competition.yaml
model:
  concat_mode: "add"
```

```bash
python Experiments/scripts/train_vit.py
```

### 範例 3: 比較實驗

```bash
# 實驗 1: horizontal
python Experiments/scripts/train_vit.py --config configs/vit_fusion_horizontal.yaml

# 實驗 2: add
python Experiments/scripts/train_vit.py --config configs/vit_fusion_add.yaml

# 實驗 3: multiply
python Experiments/scripts/train_vit.py --config configs/vit_fusion_multiply.yaml

# 在 wandb 中比較結果
```

---

## 🔧 故障排除

### 問題: ImportError: numpy

```bash
pip install numpy
```

### 問題: 融合圖片異常

1. 檢查原始圖片是否正確
2. 使用測試腳本檢查視覺效果
3. 確認 concat_mode 拼寫正確

### 問題: 訓練效果不佳

1. 先用 `horizontal` 建立 baseline
2. 視覺檢查融合效果是否合理
3. 嘗試不同的融合模式
4. 調整學習率等超參數

---

## 📚 參考文檔

- **詳細說明**: `FUSION_MODES.md`
- **使用指南**: `Data/processed/README.md`
- **Wandb 設置**: `WANDB_SETUP.md`
- **快速開始**: `QUICKSTART.md`

---

## ✅ 完成檢查清單

- [x] 實現 3 種新的融合模式（add, multiply, subtract）
- [x] 更新 `two_image_fusion.py`
- [x] 更新 `test_fusion_simple.py`
- [x] 更新配置文件說明
- [x] 測試所有 5 種融合模式
- [x] 創建詳細文檔
- [x] 驗證訓練腳本兼容性

---

**🎉 所有功能已完成並測試通過！現在可以開始實驗不同的融合策略了！**

## 下一步建議

1. **快速測試**: 運行測試腳本查看所有融合模式的視覺效果
2. **選擇模式**: 根據任務特性選擇最適合的融合模式
3. **開始訓練**: 修改配置文件並啟動訓練
4. **分析結果**: 在 wandb 中比較不同模式的性能

祝實驗順利！🚀
