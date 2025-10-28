import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import os

# 自定義資料集處理函數
def prepare_dataset(dataset, processor, image_dir, label2id, combine_mode='concat'):
    """
    準備資料集，處理圖片並轉換標籤
    
    Args:
        dataset: HuggingFace dataset
        processor: ViT 圖片處理器
        image_dir: 圖片資料夾路徑
        label2id: 類別名稱到 ID 的映射字典
        combine_mode: 圖片組合模式 ['concat', 'multiply', 'subtract']
    """
    def preprocess_function(examples):
        # 讀取 image1 和 image2
        images1 = [Image.open(os.path.join(image_dir, img)).convert('RGB') 
                   for img in examples['image1']]
        images2 = [Image.open(os.path.join(image_dir, img)).convert('RGB') 
                   for img in examples['image2']]
        
        # 先用 processor 處理兩批圖片
        inputs1 = processor(images=images1, return_tensors="pt")
        inputs2 = processor(images=images2, return_tensors="pt")
        
        pixel_values1 = inputs1['pixel_values']
        pixel_values2 = inputs2['pixel_values']
        
        # 根據 combine_mode 組合圖片
        if combine_mode == 'concat':
            # 在 channel 維度連接 (B, 3, H, W) + (B, 3, H, W) -> (B, 6, H, W)
            combined_pixels = torch.cat([pixel_values1, pixel_values2], dim=1)
        elif combine_mode == 'multiply':
            # 點對點相乘 (B, 3, H, W) * (B, 3, H, W) -> (B, 3, H, W)
            combined_pixels = pixel_values1 * pixel_values2
        elif combine_mode == 'subtract':
            # 點對點相減 (B, 3, H, W) - (B, 3, H, W) -> (B, 3, H, W)
            combined_pixels = pixel_values1 - pixel_values2
        else:
            raise ValueError(f"不支援的 combine_mode: {combine_mode}")
        
        # 轉換標籤
        labels = [label2id[label] for label in examples['class']]
        
        return {
            'pixel_values': combined_pixels,
            'labels': labels
        }
    
    # 應用預處理
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 設定格式為 PyTorch tensors
    processed_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
    
    return processed_dataset


class ViTClassifier:
    def __init__(self, num_labels, model_name='google/vit-base-patch16-224', combine_mode='concat'):
        """
        初始化 ViT 分類器
        
        Args:
            num_labels: 分類類別數量
            model_name: 預訓練模型名稱
            combine_mode: 圖片組合模式 ['concat', 'multiply', 'subtract']
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.combine_mode = combine_mode
        print(f"使用裝置: {self.device}")
        print(f"圖片組合模式: {combine_mode}")
        
        # 載入預訓練的 ViT 模型
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        # 如果是 concat 模式，需要修改第一層的輸入 channels
        if combine_mode == 'concat':
            # 修改配置中的 num_channels
            self.model.config.num_channels = 6
            
            # 原本是 3 channels，concat 後變成 6 channels
            original_conv = self.model.vit.embeddings.patch_embeddings.projection
            
            # 建立新的卷積層，輸入 channels 從 3 改為 6
            new_conv = nn.Conv2d(
                in_channels=6,  # 3 + 3 = 6
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            # 初始化新卷積層的權重
            # 方法：複製原始權重兩次並平均，保持預訓練知識
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = original_conv.weight.clone()
                new_conv.weight[:, 3:, :, :] = original_conv.weight.clone()
                new_conv.weight.data = new_conv.weight.data / 2.0  # 平均化
                if new_conv.bias is not None and original_conv.bias is not None:
                    new_conv.bias.data = original_conv.bias.data.clone()
            
            # 替換模型的第一層
            self.model.vit.embeddings.patch_embeddings.projection = new_conv
            
            # 同步更新 patch_embeddings 的 num_channels 屬性
            self.model.vit.embeddings.patch_embeddings.num_channels = 6
            
            print("已修改模型配置和第一層以支援 6 channels 輸入 (concat 模式)")
        
        self.model.to(self.device)
        
        # 載入圖片處理器
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        
        print(f"模型載入完成，分類數: {num_labels}")
    
    def train(self, train_dataset, val_dataset=None, epochs=10, 
              batch_size=16, learning_rate=2e-5, save_path='vit_model.pth'):
        """
        訓練模型
        
        Args:
            train_dataset: 訓練資料集 (HuggingFace Dataset)
            val_dataset: 驗證資料集（可選，HuggingFace Dataset）
            epochs: 訓練輪數
            batch_size: 批次大小
            learning_rate: 學習率
            save_path: 模型儲存路徑
        """
        # 建立 DataLoader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2
        )
        
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2
            )
        
        # 設定優化器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # 訓練迴圈
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # 訓練階段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch in pbar:
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向傳播
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                
                # 反向傳播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 計算準確率
                predictions = outputs.logits.argmax(dim=-1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
                train_loss += loss.item()
                
                # 更新進度條
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100*train_correct/train_total:.2f}%'
                })
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            print(f'Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            
            # 驗證階段
            if val_dataset:
                val_acc = self.evaluate(val_loader)
                print(f'Epoch {epoch+1} - Val Acc: {val_acc:.2f}%')
                
                # 儲存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), save_path)
                    print(f'最佳模型已儲存到 {save_path}')
        
        if val_dataset:
            print(f'訓練完成！最佳驗證準確率: {best_val_acc:.2f}%')
        else:
            # 如果沒有驗證集，儲存最後的模型
            torch.save(self.model.state_dict(), save_path)
            print(f'訓練完成！模型已儲存到 {save_path}')
    
    def evaluate(self, dataloader):
        """評估模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(pixel_values=pixel_values)
                predictions = outputs.logits.argmax(dim=-1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = 100 * correct / total
        return accuracy
    
    def predict(self, image_path1, image_path2):
        """
        預測兩張圖片的組合
        
        Args:
            image_path1: 第一張圖片路徑
            image_path2: 第二張圖片路徑
        
        Returns:
            predicted_class: 預測的類別
            confidence: 信心分數
        """
        self.model.eval()
        
        # 讀取並處理圖片
        image1 = Image.open(image_path1).convert('RGB')
        image2 = Image.open(image_path2).convert('RGB')
        
        encoding1 = self.processor(images=image1, return_tensors="pt")
        encoding2 = self.processor(images=image2, return_tensors="pt")
        
        pixel_values1 = encoding1['pixel_values']
        pixel_values2 = encoding2['pixel_values']
        
        # 根據 combine_mode 組合圖片
        if self.combine_mode == 'concat':
            combined_pixels = torch.cat([pixel_values1, pixel_values2], dim=1)
        elif self.combine_mode == 'multiply':
            combined_pixels = pixel_values1 * pixel_values2
        elif self.combine_mode == 'subtract':
            combined_pixels = pixel_values1 - pixel_values2
        
        combined_pixels = combined_pixels.to(self.device)
        
        # 預測
        with torch.no_grad():
            outputs = self.model(pixel_values=combined_pixels)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = logits.argmax(dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence


# 使用範例
if __name__ == "__main__":
    # === 1. 設定參數 ===
    IMAGE_DIR = "./bgOn_heatmapOn_trajOn"  # 圖片資料夾路徑
    TRAIN_JSON = "./image_gt_train.json"  # 訓練資料 JSON
    VAL_JSON = "./image_gt_val.json"  # 驗證資料 JSON（如果有的話）
    
    # 選擇圖片組合模式: 'concat', 'multiply', 'subtract'
    COMBINE_MODE = 'concat'  # 可以改成 'multiply' 或 'subtract'
    
    # === 2. 載入資料 ===
    print("載入訓練資料...")
    train_ds = load_dataset("json", data_files=TRAIN_JSON, split="train")
    
    # 如果有驗證集
    # val_ds = load_dataset("json", data_files=VAL_JSON, split="train")
    
    # === 3. 建立類別映射 ===
    # 從資料中獲取所有唯一的類別
    unique_classes = set(train_ds['class'])
    label2id = {label: idx for idx, label in enumerate(sorted(unique_classes))}
    id2label = {idx: label for label, idx in label2id.items()}
    
    print(f"類別映射: {label2id}")
    print(f"類別數量: {len(label2id)}")
    
    # === 4. 初始化模型 ===
    classifier = ViTClassifier(
        num_labels=len(label2id),
        combine_mode=COMBINE_MODE
    )
    
    # === 5. 預處理資料集 ===
    print("預處理訓練資料...")
    train_dataset = prepare_dataset(
        train_ds, 
        classifier.processor, 
        IMAGE_DIR, 
        label2id,
        combine_mode=COMBINE_MODE
    )
    
    # 如果有驗證集
    # print("預處理驗證資料...")
    # val_dataset = prepare_dataset(
    #     val_ds, 
    #     classifier.processor, 
    #     IMAGE_DIR, 
    #     label2id,
    #     combine_mode=COMBINE_MODE
    # )
    
    # === 6. 訓練模型 ===
    print("開始訓練...")
    classifier.train(
        train_dataset=train_dataset,
        val_dataset=None,  # 如果有驗證集，改為 val_dataset
        epochs=10,
        batch_size=16,
        learning_rate=2e-5,
        save_path=f'best_vit_model_{COMBINE_MODE}.pth'
    )
    
    # === 7. 預測範例 ===
    print("\n預測範例:")
    # 使用資料集中的第一對圖片做測試
    if len(train_ds) > 0:
        test_img1 = os.path.join(IMAGE_DIR, train_ds[0]['image1'])
        test_img2 = os.path.join(IMAGE_DIR, train_ds[0]['image2'])
        
        if os.path.exists(test_img1) and os.path.exists(test_img2):
            predicted_class, confidence = classifier.predict(test_img1, test_img2)
            predicted_label = id2label[predicted_class]
            actual_label = train_ds[0]['class']
            print(f'圖片1: {train_ds[0]["image1"]}')
            print(f'圖片2: {train_ds[0]["image2"]}')
            print(f'實際類別: {actual_label}')
            print(f'預測類別: {predicted_label} (ID: {predicted_class})')
            print(f'信心分數: {confidence:.4f}')
    
    # 儲存類別映射和配置
    import json
    config = {
        'label2id': label2id, 
        'id2label': id2label,
        'combine_mode': COMBINE_MODE
    }
    with open(f'model_config_{COMBINE_MODE}.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"\n模型配置已儲存到 model_config_{COMBINE_MODE}.json")
    
    # === 8. 測試三種模式的比較（可選） ===
    print("\n" + "="*50)
    print("如果要測試三種模式，可以分別執行:")
    print("1. COMBINE_MODE = 'concat'   # 連接兩張圖")
    print("2. COMBINE_MODE = 'multiply' # 點對點相乘")
    print("3. COMBINE_MODE = 'subtract' # 點對點相減")
    print("="*50)