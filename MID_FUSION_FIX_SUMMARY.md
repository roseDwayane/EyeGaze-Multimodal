# Mid Fusion Model 除錯修復總結

## 問題描述 (Problem)

訓練 Mid Fusion 模型時遇到以下錯誤:

```
AttributeError: 'ViTClassifier' object has no attribute 'patch_embed'
```

**錯誤位置**: `Models/fusion/mid_fusion.py:239` in `_extract_image_features` method

## 根本原因 (Root Cause)

Mid Fusion 模型中的 `_extract_image_features` 方法試圖直接訪問 ViT 的內部組件:
```python
def _extract_image_features(self, img: torch.Tensor, encoder) -> torch.Tensor:
    x = encoder.patch_embed(img)  # ❌ 錯誤：patch_embed 不存在
    cls_tokens = encoder.cls_token.expand(B, -1, -1)  # ❌
    x = x + encoder.pos_embed  # ❌
    x = encoder.encoder(x)  # ❌
    ...
```

**問題**:
- `ViTClassifier` 是對 Hugging Face `ViTForImageClassification` 的封裝
- 這些內部屬性 (`patch_embed`, `cls_token`, `pos_embed`, `encoder`) 不存在於 `ViTClassifier` 層級
- 需要使用 `encoder.model.vit.xxx` 來訪問，或更好的方式是使用已有的 API

## 解決方案 (Solution)

使用我們已經在 `ViTClassifier` 中實現的 `get_cls_features()` 方法:

### 修改前 (Before)
```python
def _extract_image_features(self, img: torch.Tensor, encoder) -> torch.Tensor:
    """Extract CLS features from ViT"""
    B = img.shape[0]

    # Patchify and embed
    x = encoder.patch_embed(img)  # (B, N, d_model)

    # Add CLS token
    cls_tokens = encoder.cls_token.expand(B, -1, -1)
    x = torch.cat([cls_tokens, x], dim=1)

    # Add positional embedding
    x = x + encoder.pos_embed

    # Transformer encoder
    x = encoder.encoder(x)

    # Extract CLS token
    cls_features = x[:, 0]  # (B, d_model)

    return cls_features
```

### 修改後 (After) ✅
```python
def _extract_image_features(self, img: torch.Tensor, encoder) -> torch.Tensor:
    """Extract CLS features from ViT"""
    # Use the get_cls_features method from ViTClassifier
    cls_features = encoder.get_cls_features(pixel_values=img)  # (B, d_model)
    return cls_features
```

## 修改文件 (Modified Files)

### `Models/fusion/mid_fusion.py`
- **修改**: `_extract_image_features` 方法 (lines 234-238)
- **變更**: 簡化為使用 `get_cls_features()` API

### `Models/backbones/vit.py`
- **已存在**: `get_cls_features()` 方法 (在之前修復 Late Fusion 時添加)
- **功能**: 正確提取 ViT 的 CLS token 特徵

## 測試驗證 (Testing)

創建了測試文件 `test_mid_fusion_fix.py`，所有測試通過:

```bash
$ python test_mid_fusion_fix.py

============================================================
Testing Mid Fusion Model Fixes
============================================================

1. Testing MidFusionModel initialization...
[OK] MidFusionModel initialized successfully!
[OK] Model has 100.3M parameters

2. Testing MidFusionModel forward pass...
[OK] Output keys: dict_keys(['logits', 'z_img_fused', 'z_eeg_fused', 'ibs_token', 'loss'])
[OK] Logits shape: torch.Size([2, 3])
[OK] Loss: 1.3507
[OK] Forward pass successful!

3. Testing without IBS token...
[OK] Without IBS token: torch.Size([2, 3])

4. Testing without cross-attention...
[OK] Without cross-attention: torch.Size([2, 3])

============================================================
All tests passed! [OK]
============================================================
```

## 訓練狀態 (Training Status)

✅ **Mid Fusion 訓練已成功啟動**

```bash
2025-11-14 00:19:36 - Using device: cuda
2025-11-14 00:19:36 - Loading dataset from JSON...
2025-11-14 00:19:37 - Total samples: 4463
2025-11-14 00:19:37 - Train samples: 3570
2025-11-14 00:19:37 - Test samples: 893
2025-11-14 00:19:37 - Preparing multimodal samples...
```

**當前進度**:
- ✅ 資料載入中 (Processing samples...)
- ⏳ 等待模型訓練開始
- 📊 WandB Run: `mid-fusion-full`

## 相關修復歷史 (Related Fixes)

### 1. Late Fusion 修復 (之前)
- **問題**: 相同的問題 - 缺少 `get_cls_features` 方法
- **解決**: 在 `Models/backbones/vit.py` 添加 `get_cls_features()` 方法
- **文件**: `test_late_fusion_fix.py`

### 2. Mid Fusion 修復 (現在)
- **問題**: `_extract_image_features` 試圖直接訪問內部屬性
- **解決**: 使用已有的 `get_cls_features()` API
- **文件**: `test_mid_fusion_fix.py`

## 技術細節 (Technical Details)

### ViTClassifier 結構
```
ViTClassifier (wrapper)
└── self.model (ViTForImageClassification from HuggingFace)
    └── self.vit (ViTModel)
        ├── patch_embed
        ├── cls_token
        ├── pos_embed
        └── encoder
```

### 正確的特徵提取方式
```python
# ❌ 錯誤 - 直接訪問內部
encoder.patch_embed(img)

# ❌ 錯誤 - 需要多層訪問
encoder.model.vit.patch_embed(img)

# ✅ 正確 - 使用封裝的 API
encoder.get_cls_features(pixel_values=img)
```

### get_cls_features 實現 (in vit.py)
```python
def get_cls_features(self, pixel_values):
    """Extract CLS token features (without classification head)"""
    # Get hidden states from ViT encoder
    outputs = self.model.vit(pixel_values=pixel_values, return_dict=True)

    # Extract CLS token (first token in last hidden state)
    last_hidden_state = outputs.last_hidden_state  # (B, N, D)
    cls_token = last_hidden_state[:, 0]  # (B, D)

    return cls_token
```

## 下一步 (Next Steps)

### 立即執行
1. ✅ **Mid Fusion 訓練已啟動** - 監控進度
2. 🔄 **Late Fusion 訓練** - 應該也在運行
3. ⏳ **Early Fusion 訓練** - 準備啟動

### 後續任務
1. 監控 WandB 訓練曲線
2. 訓練完成後進行消融實驗
3. 比較三種融合策略的性能
4. 撰寫論文

## 監控命令 (Monitoring)

### 檢查背景訓練
```bash
# 查看正在運行的訓練
wandb status

# 或訪問 WandB 網頁
https://wandb.ai/super57033-national-tsing-hua-university/eyegaze-eeg-classification
```

### 快速測試
```bash
# 測試 Mid Fusion 模型
python test_mid_fusion_fix.py

# 測試 Late Fusion 模型
python test_late_fusion_fix.py
```

## 總結 (Summary)

**問題**: AttributeError 阻止 Mid Fusion 訓練
**原因**: 直接訪問不存在的 ViT 內部屬性
**解決**: 使用封裝的 `get_cls_features()` API
**結果**: ✅ 所有測試通過，訓練成功啟動

**修改行數**: 僅 5 行代碼修改
**影響範圍**: 僅 `mid_fusion.py` 的 `_extract_image_features` 方法
**向後兼容**: 是 (不影響其他模型)

---

**修復日期**: 2025-11-14
**修復版本**: v1.1
**狀態**: ✅ 完成並驗證
