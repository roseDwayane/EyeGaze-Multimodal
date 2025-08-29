#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train GIT-base with ONLY lively JSON.
• Generation target  : full descriptive sentence  (from "class")
• Classification id  : parsed from image filename  (single / competition / cooperation)
• Uses memory-saving tricks: 224×224 resize, grad-checkpointing, tiny batch.
"""

import os, re, gc, numpy as np, torch
from PIL import Image
from datasets import load_dataset
from transformers import (
    GitProcessor,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from evaluate import load as load_metric

# ─── 路徑設定 ──────────────────────────────────────────────────────────
JSON_DIR = "/home/cnelabai/PycharmProjects/Gaze/json_textdescription"
IMG_ROOT = "/home/cnelabai/PycharmProjects/Gaze/bgOn_heatmapOn_trajOn"

# ─── 隨機種子 / 清 GPU cache ─────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED);  np.random.seed(SEED)
torch.cuda.empty_cache(); gc.collect()

# ─── 1. 影像檔名 → 整數標籤 ───────────────────────────────────────────
def filename_to_label(fname: str) -> int:
    n = fname.lower()
    if re.search(r"single", n):
        return 0          # Single
    if re.search(r"comp(etition)?", n):
        return 1          # Competition
    if re.search(r"coop(eration)?", n):
        return 2          # Cooperation
    return -1             # unknown / discard

# ─── 2. 讀取 & 切分資料 ───────────────────────────────────────────────
def load_and_split():
    raw = load_dataset(
        "json",
        data_files={"train": os.path.join(JSON_DIR, "lively_description_json.json")},
    )["train"]

    # 去重：同一對影像只留一次
    def dedup(ex, idx=[set()]):
        key = (ex["image1"], ex["image2"])
        if key in idx[0]:
            return False
        idx[0].add(key);  return True
    raw = raw.filter(dedup)

    # 篩掉無法判別類別的樣本
    raw = raw.filter(lambda ex: filename_to_label(ex["image1"]) != -1)

    splits = raw.train_test_split(test_size=0.1, seed=SEED)

    # 讓驗證集不要太小 / 太大
    val_ds = splits["test"]
    if len(val_ds) > 50:      # 減少驗證集大小以節省記憶體
        val_ds = val_ds.select(range(50))

    return splits["train"], val_ds

train_ds, val_ds = load_and_split()

# 預先保存驗證集的真實 label 供 metric 使用
VAL_CLS_IDS = [filename_to_label(r["image1"]) for r in val_ds]

# ─── 3. 前處理函式 ───────────────────────────────────────────────────
def build_transform(processor):
    def _tf(batch):
        images = []
        for f1, f2, txt in zip(batch["image1"], batch["image2"], batch["class"]):
            im1 = Image.open(os.path.join(IMG_ROOT, f1)).convert("RGB")
            im2 = Image.open(os.path.join(IMG_ROOT, f2)).convert("RGB")
            im1, im2 = im1.resize((224,224)), im2.resize((224,224))

            canvas = Image.new("RGB", (224, 448))
            canvas.paste(im1, (0,0));  canvas.paste(im2, (0,224))
            images.append(canvas)

        enc = processor(
            images=images,
            text=batch["class"],
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        enc["labels"] = enc["input_ids"].copy()   # 生成 loss
        # 移除 cls_ids，因為模型不接受這個參數
        return enc
    return _tf

checkpoint = "microsoft/git-base"
processor  = GitProcessor.from_pretrained(checkpoint)
model      = AutoModelForCausalLM.from_pretrained(checkpoint)
model.gradient_checkpointing_enable()             # 省顯存

train_ds.set_transform(build_transform(processor))
val_ds.set_transform(build_transform(processor))

# ─── 4. 評估：計算分類 F1 ─────────────────────────────────────────────
f1_metric = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred          # logits: (batch, seq_len, vocab)
    preds = logits.argmax(-1)

    # 解碼預測的文本
    decoded_preds = []
    for pred_seq in preds:
        # 跳過特殊 tokens 並解碼
        decoded = processor.decode(pred_seq, skip_special_tokens=True)
        decoded_preds.append(decoded)

    # 從預測文本中提取分類
    mapping = {"single":0,"competition":1,"cooperation":2}
    pred_ids = []
    for text in decoded_preds:
        text_lower = text.lower()
        if "single" in text_lower:
            pred_ids.append(0)
        elif "competition" in text_lower:
            pred_ids.append(1)
        elif "cooperation" in text_lower:
            pred_ids.append(2)
        else:
            pred_ids.append(-1)  # 無法分類

    # 比對預存的真實標籤
    valid_indices = [i for i, pred in enumerate(pred_ids) if pred != -1]
    if len(valid_indices) == 0:
        return {"cls_f1": 0.0}
    
    valid_preds = [pred_ids[i] for i in valid_indices]
    valid_refs = [VAL_CLS_IDS[i] for i in valid_indices if i < len(VAL_CLS_IDS)]
    
    if len(valid_preds) != len(valid_refs):
        # 確保長度一致
        min_len = min(len(valid_preds), len(valid_refs))
        valid_preds = valid_preds[:min_len]
        valid_refs = valid_refs[:min_len]
    
    if len(valid_preds) == 0:
        return {"cls_f1": 0.0}
    
    cls_f1 = f1_metric.compute(
        predictions=valid_preds,
        references=valid_refs,
        average="weighted"
    )["f1"]
    return {"cls_f1": cls_f1}

# ─── 5. 訓練參數 ─────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir="./git_lively_ckpt",
    learning_rate=5e-5,
    num_train_epochs=10,
    per_device_train_batch_size=2,  # 進一步減少批次大小
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,  # 增加梯度累積以維持有效批次大小
    eval_strategy="steps",
    eval_steps=100,  # 減少評估頻率
    save_steps=100,
    fp16=True,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    logging_steps=10,
    report_to="none",
    dataloader_pin_memory=False,
    eval_accumulation_steps=1,
)

# ─── 6. 訓練 ─────────────────────────────────────────────────────────
class MemoryEfficientTrainer(Trainer):
    def evaluation_loop(self, *args_, **kwargs_):
        torch.cuda.empty_cache(); gc.collect()
        out = super().evaluation_loop(*args_, **kwargs_)
        torch.cuda.empty_cache(); gc.collect()
        return out

trainer = MemoryEfficientTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

print(f"GPU before train: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
trainer.train()
trainer.evaluate()