import os
import torch
import pandas as pd
from PIL import Image, ImageEnhance
from datasets import load_dataset
from transformers import AutoModelForCausalLM, GitProcessor, TrainingArguments, Trainer
import json
from datetime import datetime
import random
import numpy as np

# ============================================================
# 配置
# ============================================================
class Config:
    # 使用所有ROOT目錄
    ROOT_DIRS = [
        "bgOn_heatmapOn_trajOn",
        "bgOn_heatmapOn_trajOff",
        "bgOn_heatmapOff_trajOn",
        "bgOff_heatmapOn_trajOn",
        "bgOff_heatmapOn_trajOff",
        "bgOff_heatmapOff_trajOn"
    ]
   
    # 使用所有JSON檔案
    JSON_FILES = [
        "/home/cnelabai/PycharmProjects/EyeGaze-Multimodal/json_textdescription/formal_description_json.json",
        "/home/cnelabai/PycharmProjects/EyeGaze-Multimodal/json_textdescription/lively_description_json.json",
        "/home/cnelabai/PycharmProjects/EyeGaze-Multimodal/json_textdescription/mixed_description_json.json"
    ]
   
    MODEL_NAME = "microsoft/git-base"
    MAX_LENGTH = 250
    BATCH_SIZE = 1
    LEARNING_RATE = 2e-6
    EPOCHS_PER_ROOT = 2
    SAMPLES_PER_CATEGORY_PER_JSON = 25
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 輔助函數
# ============================================================
def extract_category_from_filename(filename):
    """從檔名提取類別"""
    filename = filename.lower()
    if "single" in filename:
        return "single"
    elif "comp" in filename:
        return "competition"
    elif "coop" in filename:
        return "cooperation"
    else:
        return "unknown"

def enhance_features(image):
    """輕微增強圖像特徵"""
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
   
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.1)
   
    return image

def concatenate_images_simple(image1_path, image2_path):
    """簡單的圖片拼接，不加任何標註"""
    image1 = Image.open(image1_path).convert('RGB')
    image2 = Image.open(image2_path).convert('RGB')
    image2 = image2.resize((image1.width, image1.height))
   
    # 輕微增強
    image1 = enhance_features(image1)
    image2 = enhance_features(image2)
   
    # 拼接
    concatenated_image = Image.new("RGB", (image1.width, image1.height * 2))
    concatenated_image.paste(image1, (0, 0))
    concatenated_image.paste(image2, (0, image1.height))
   
    return concatenated_image

# ============================================================
# 資料準備
# ============================================================
def prepare_natural_dataset(root_dir, config):
    """準備自然的訓練資料集，使用原始描述不加修改"""
    all_samples = []
   
    for json_file in config.JSON_FILES:
        json_name = os.path.basename(json_file).replace('_description_json.json', '')
        print(f"Loading {json_name} for {root_dir}...")
       
        dataset = load_dataset("json", data_files=json_file, split="train")
       
        category_counts = {'single': 0, 'competition': 0, 'cooperation': 0}
       
        for idx in range(len(dataset)):
            sample = dataset[idx]
           
            img1_path = os.path.join(root_dir, sample["image1"])
            img2_path = os.path.join(root_dir, sample["image2"])
           
            if os.path.exists(img1_path) and os.path.exists(img2_path):
                category = extract_category_from_filename(sample["image1"])
               
                if category in category_counts and category_counts[category] < config.SAMPLES_PER_CATEGORY_PER_JSON:
                    sample_data = {
                        'img1_path': img1_path,
                        'img2_path': img2_path,
                        'text': sample["class"],  # 直接使用原始描述
                        'image1_name': sample["image1"],
                        'category': category,
                        'style': json_name,
                        'root_dir': root_dir
                    }
                   
                    all_samples.append(sample_data)
                    category_counts[category] += 1
   
    print(f"Prepared {len(all_samples)} natural samples for {root_dir}")
    print(f"Categories: {category_counts}")
    return all_samples

# ============================================================
# 資料集類別
# ============================================================
class NaturalDataset(torch.utils.data.Dataset):
    def __init__(self, samples, processor, max_length):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
   
    def __len__(self):
        return len(self.samples)
   
    def __getitem__(self, idx):
        sample = self.samples[idx]
       
        # 載入圖像
        image = concatenate_images_simple(
            sample['img1_path'],
            sample['img2_path']
        )
       
        # 直接使用原始文本
        text = sample['text']
       
        # 處理輸入
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
       
        pixel_values = inputs['pixel_values'].squeeze(0)
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
       
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }

# ============================================================
# 訓練函數
# ============================================================
def train_natural_model(root_dir, config):
    """訓練自然生成的模型"""
    print(f"\n{'='*20} 自然訓練: {root_dir} {'='*20}")
   
    # 準備資料
    samples = prepare_natural_dataset(root_dir, config)
    if len(samples) == 0:
        print(f"No samples for {root_dir}, skipping...")
        return None
   
    # 分割資料
    random.shuffle(samples)
    train_size = int(len(samples) * 0.9)
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]
   
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
   
    # 載入模型
    processor = GitProcessor.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME)
   
    # 建立資料集
    train_dataset = NaturalDataset(train_samples, processor, config.MAX_LENGTH)
    eval_dataset = NaturalDataset(val_samples, processor, config.MAX_LENGTH)
   
    # 訓練參數
    output_dir = f"./natural_model_{root_dir}"
   
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.EPOCHS_PER_ROOT,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        warmup_steps=100,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        save_total_limit=2,
        gradient_accumulation_steps=2
    )
   
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer
    )
   
    print(f"Starting natural training for {root_dir}...")
    trainer.train()
   
    # 儲存模型
    trainer.save_model()
    processor.save_pretrained(output_dir)
    print(f"Natural model for {root_dir} saved to {output_dir}")
   
    return output_dir, samples

# ============================================================
# 測試函數
# ============================================================
def test_natural_generation(model_dir, root_dir, config):
    """測試自然生成"""
    print(f"\n=== 測試自然生成: {root_dir} ===")
   
    try:
        processor = GitProcessor.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        model.to(config.DEVICE)
        model.eval()
    except Exception as e:
        print(f"無法載入模型: {e}")
        return []
   
    # 準備測試樣本
    test_samples = []
    for json_file in config.JSON_FILES:
        json_name = os.path.basename(json_file).replace('_description_json.json', '')
        dataset = load_dataset("json", data_files=json_file, split="train")
       
        category_counts = {'single': 0, 'competition': 0, 'cooperation': 0}
       
        for idx in range(len(dataset)):
            sample = dataset[idx]
           
            img1_path = os.path.join(root_dir, sample["image1"])
            img2_path = os.path.join(root_dir, sample["image2"])
           
            if os.path.exists(img1_path) and os.path.exists(img2_path):
                category = extract_category_from_filename(sample["image1"])
               
                if category in category_counts and category_counts[category] < 2:
                    test_samples.append({
                        'img1_path': img1_path,
                        'img2_path': img2_path,
                        'original_text': sample["class"],
                        'image1_name': sample["image1"],
                        'category': category,
                        'style': json_name,
                        'root_dir': root_dir
                    })
                    category_counts[category] += 1
           
            if all(count >= 2 for count in category_counts.values()):
                break
   
    results = []
   
    for sample in test_samples:
        try:
            print(f"Testing {sample['category']}-{sample['style']} in {root_dir}")
           
            image = concatenate_images_simple(
                sample['img1_path'],
                sample['img2_path']
            )
           
            # 無提示詞生成
            inputs = processor(images=image, return_tensors="pt").to(config.DEVICE)
           
            # 多種生成策略
            generation_strategies = [
                {
                    "max_length": config.MAX_LENGTH,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repetition_penalty": 1.2,
                    "no_repeat_ngram_size": 3,
                    "pad_token_id": processor.tokenizer.eos_token_id
                },
                {
                    "max_length": config.MAX_LENGTH,
                    "do_sample": True,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "repetition_penalty": 1.3,
                    "no_repeat_ngram_size": 2,
                    "pad_token_id": processor.tokenizer.eos_token_id
                },
                {
                    "max_length": config.MAX_LENGTH,
                    "num_beams": 5,
                    "early_stopping": True,
                    "repetition_penalty": 1.1,
                    "length_penalty": 1.0,
                    "pad_token_id": processor.tokenizer.eos_token_id
                }
            ]
           
            for i, strategy in enumerate(generation_strategies):
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, **strategy)
               
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
               
                print(f"生成 (策略{i+1}): {generated_text[:100]}...")
               
                results.append({
                    'root_dir': root_dir,
                    'category': sample['category'],
                    'style': sample['style'],
                    'image': sample['image1_name'],
                    'strategy': f"strategy_{i+1}",
                    'original': sample['original_text'],
                    'generated': generated_text,
                    'contains_category': sample['category'] in generated_text.lower(),
                    'length': len(generated_text.split())
                })
           
        except Exception as e:
            print(f"生成失敗: {e}")
            continue
   
    return results

# ============================================================
# 主程式
# ============================================================
def main():
    config = Config()
   
    print("=== 完全自然的句子生成訓練 ===")
    print(f"使用模型: {config.MODEL_NAME}")
    print(f"使用所有 JSON 檔案: {len(config.JSON_FILES)} 個")
    print(f"訓練所有 ROOT 目錄: {len(config.ROOT_DIRS)} 個")
    print(f"無提示詞限制，無格式約束")
    print("="*60)
   
    all_results = []
    model_dirs = {}
   
    # 訓練所有ROOT
    for root_dir in config.ROOT_DIRS:
        try:
            model_dir, samples = train_natural_model(root_dir, config)
            if model_dir:
                model_dirs[root_dir] = model_dir
           
        except Exception as e:
            print(f"訓練 {root_dir} 時發生錯誤: {e}")
            continue
   
    print(f"\n=== 開始測試自然生成 ===")
   
    # 測試所有ROOT
    for root_dir, model_dir in model_dirs.items():
        try:
            results = test_natural_generation(model_dir, root_dir, config)
            all_results.extend(results)
           
        except Exception as e:
            print(f"測試 {root_dir} 時發生錯誤: {e}")
            continue
   
    # 分析結果
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df = pd.DataFrame(all_results)
        results_file = f"natural_generation_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
       
        print(f"\n=== 自然生成結果分析 ===")
        print(f"總測試樣本: {len(all_results)}")
        print(f"結果已儲存至: {results_file}")
       
        # 多樣性分析
        all_generated = [r['generated'] for r in all_results]
        unique_generations = set(all_generated)
        diversity_ratio = len(unique_generations) / len(all_results) * 100
        print(f"生成多樣性: {len(unique_generations)}/{len(all_results)} = {diversity_ratio:.1f}%")
       
        # 長度分析
        avg_length = sum(r['length'] for r in all_results) / len(all_results)
        min_length = min(r['length'] for r in all_results)
        max_length = max(r['length'] for r in all_results)
        print(f"平均句子長度: {avg_length:.1f} 詞 (範圍: {min_length}-{max_length})")
       
        # 類別準確性
        correct_category = sum(1 for r in all_results if r['contains_category'])
        accuracy = correct_category / len(all_results) * 100
        print(f"類別識別準確率: {correct_category}/{len(all_results)} = {accuracy:.1f}%")
       
        # 按ROOT分析
        print(f"\n各ROOT表現:")
        for root_dir in config.ROOT_DIRS:
            root_results = [r for r in all_results if r['root_dir'] == root_dir]
            if root_results:
                root_correct = sum(1 for r in root_results if r['contains_category'])
                root_accuracy = root_correct / len(root_results) * 100
                root_diversity = len(set(r['generated'] for r in root_results)) / len(root_results) * 100
                root_avg_length = sum(r['length'] for r in root_results) / len(root_results)
                print(f"{root_dir}:")
                print(f"  準確率: {root_correct}/{len(root_results)} = {root_accuracy:.1f}%")
                print(f"  多樣性: {root_diversity:.1f}%")
                print(f"  平均長度: {root_avg_length:.1f} 詞")
       
        print(f"\n=== 訓練完成 ===")
        print("模型已學會自然生成句子，無格式約束！")
       
    else:
        print("沒有成功的測試結果")

if __name__ == "__main__":
    main()