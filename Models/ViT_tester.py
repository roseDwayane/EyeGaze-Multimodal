import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import json
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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


def load_model_with_weights(model_path, config_path, device):
    """
    載入訓練好的模型權重
    
    Args:
        model_path: 模型權重檔案路徑 (.pth)
        config_path: 模型配置檔案路徑 (.json)
        device: 運算裝置
    
    Returns:
        model: 載入權重的模型
        label2id: 類別映射
        id2label: ID到類別的映射
        combine_mode: 圖片組合模式
    """
    # 載入配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    label2id = config['label2id']
    id2label = {int(k): v for k, v in config['id2label'].items()}
    combine_mode = config['combine_mode']
    num_labels = len(label2id)
    
    print(f"載入配置: {num_labels} 個類別, 組合模式: {combine_mode}")
    
    # 建立模型架構
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    
    # 如果是 concat 模式，修改第一層
    if combine_mode == 'concat':
        original_conv = model.vit.embeddings.patch_embeddings.projection
        
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        model.vit.embeddings.patch_embeddings.projection = new_conv
        print("已修改模型第一層以支援 6 channels 輸入")
    
    # 載入權重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"成功載入模型權重: {model_path}")
    
    return model, label2id, id2label, combine_mode


def evaluate_model_detailed(model, dataloader, id2label, device):
    """
    詳細評估模型，計算多項指標
    
    Args:
        model: 訓練好的模型
        dataloader: 驗證資料載入器
        id2label: ID到類別的映射
        device: 運算裝置
    
    Returns:
        metrics: 包含各項指標的字典
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("開始評估模型...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 轉換為 numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # 計算各項指標
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # macro: 不考慮類別不平衡，每個類別權重相同
    # weighted: 考慮類別不平衡，根據樣本數量加權
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    
    recall_macro = recall_score(all_labels, all_predictions, average='macro')
    recall_weighted = recall_score(all_labels, all_predictions, average='weighted')
    
    precision_macro = precision_score(all_labels, all_predictions, average='macro')
    precision_weighted = precision_score(all_labels, all_predictions, average='weighted')
    
    # 整理結果
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }
    
    # 生成分類報告
    class_names = [id2label[i] for i in sorted(id2label.keys())]
    report = classification_report(
        all_labels, 
        all_predictions, 
        target_names=class_names,
        digits=4
    )
    
    return metrics, report


def plot_confusion_matrix(labels, predictions, id2label, save_path='confusion_matrix.png'):
    """
    繪製並儲存混淆矩陣
    
    Args:
        labels: 真實標籤
        predictions: 預測標籤
        id2label: ID到類別的映射
        save_path: 儲存路徑
    """
    cm = confusion_matrix(labels, predictions)
    class_names = [id2label[i] for i in sorted(id2label.keys())]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩陣已儲存至: {save_path}")
    plt.close()


# 主程式
if __name__ == "__main__":
    # === 1. 設定參數 ===
    IMAGE_DIR = "./bgOn_heatmapOn_trajOn"  # 圖片資料夾路徑
    VAL_JSON = "./image_gt_test.json"  # 驗證資料 JSON
    
    # 選擇要評估的模型
    COMBINE_MODE = 'multiply'  # 需與訓練時一致: 'concat', 'multiply', 'subtract'
    
    MODEL_PATH = f'best_vit_model_{COMBINE_MODE}.pth'  # 訓練好的模型權重
    CONFIG_PATH = f'model_config_{COMBINE_MODE}.json'  # 模型配置檔
    
    BATCH_SIZE = 16
    
    # === 2. 檢查檔案是否存在 ===
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型檔案: {MODEL_PATH}")
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"找不到配置檔案: {CONFIG_PATH}")
    if not os.path.exists(VAL_JSON):
        raise FileNotFoundError(f"找不到驗證資料: {VAL_JSON}")
    
    # === 3. 設定裝置 ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")
    
    # === 4. 載入模型 ===
    model, label2id, id2label, combine_mode = load_model_with_weights(
        MODEL_PATH, CONFIG_PATH, device
    )
    
    # === 5. 載入圖片處理器 ===
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    # === 6. 載入驗證資料 ===
    print(f"\n載入驗證資料: {VAL_JSON}")
    val_ds = load_dataset("json", data_files=VAL_JSON, split="train")
    print(f"驗證資料筆數: {len(val_ds)}")
    
    # === 7. 預處理驗證資料 ===
    print("預處理驗證資料...")
    val_dataset = prepare_dataset(
        val_ds, 
        processor, 
        IMAGE_DIR, 
        label2id,
        combine_mode=combine_mode
    )
    
    # === 8. 建立 DataLoader ===
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    # === 9. 評估模型 ===
    print("\n" + "="*60)
    print("開始詳細評估")
    print("="*60)
    
    metrics, report = evaluate_model_detailed(model, val_loader, id2label, device)
    
    # === 10. 顯示結果 ===
    print("\n" + "="*60)
    print("評估結果")
    print("="*60)
    print(f"Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"\nF1 Score (Macro):   {metrics['f1_macro']:.4f}")
    print(f"F1 Score (Weighted):{metrics['f1_weighted']:.4f}")
    print(f"\nRecall (Macro):     {metrics['recall_macro']:.4f}")
    print(f"Recall (Weighted):  {metrics['recall_weighted']:.4f}")
    print(f"\nPrecision (Macro):  {metrics['precision_macro']:.4f}")
    print(f"Precision (Weighted):{metrics['precision_weighted']:.4f}")
    
    print("\n" + "="*60)
    print("詳細分類報告")
    print("="*60)
    print(report)
    
    # === 11. 繪製混淆矩陣 ===
    plot_confusion_matrix(
        metrics['labels'], 
        metrics['predictions'], 
        id2label,
        save_path=f'confusion_matrix_{COMBINE_MODE}.png'
    )
    
    # === 12. 儲存評估結果 ===
    results = {
        'model_path': MODEL_PATH,
        'combine_mode': combine_mode,
        'num_samples': len(val_ds),
        'accuracy': float(metrics['accuracy']),
        'f1_macro': float(metrics['f1_macro']),
        'f1_weighted': float(metrics['f1_weighted']),
        'recall_macro': float(metrics['recall_macro']),
        'recall_weighted': float(metrics['recall_weighted']),
        'precision_macro': float(metrics['precision_macro']),
        'precision_weighted': float(metrics['precision_weighted']),
    }
    
    results_path = f'evaluation_results_{COMBINE_MODE}.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n評估結果已儲存至: {results_path}")
    
    # === 13. 每個類別的詳細統計 ===
    print("\n" + "="*60)
    print("每個類別的樣本數量統計")
    print("="*60)
    from collections import Counter
    label_counts = Counter(metrics['labels'])
    for label_id in sorted(id2label.keys()):
        label_name = id2label[label_id]
        count = label_counts.get(label_id, 0)
        print(f"{label_name:20s}: {count:4d} 個樣本")
    
    print("\n評估完成！")
    print("="*60)