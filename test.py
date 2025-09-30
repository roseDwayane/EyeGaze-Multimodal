import os
import torch
import pandas as pd
from PIL import Image
from datasets import load_dataset
from transformers import AutoModelForCausalLM, GitProcessor
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate import load
from sklearn.metrics import confusion_matrix
import re

# ============================================================
# Constants and Configuration
# ============================================================
ROOT_LIST = [
    "bgOn_heatmapOn_trajOn",
    "bgOn_heatmapOn_trajOff",
    "bgOn_heatmapOff_trajOn",
    "bgOff_heatmapOn_trajOn",
    "bgOff_heatmapOn_trajOff",
    "bgOff_heatmapOff_trajOn"
]

# 從檔名提取標籤的映射
LABEL_MAPPING = {"single": 0, "cooperation": 1, "competition": 2}
METRICS = ["accuracy", "precision", "recall", "f1"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 你的三個 JSON 檔案
JSON_FILES = [
    "/home/cnelabai/PycharmProjects/EyeGaze-Multimodal/json_textdescription/formal_description_json.json",
    "/home/cnelabai/PycharmProjects/EyeGaze-Multimodal/json_textdescription/lively_description_json.json",
    "/home/cnelabai/PycharmProjects/EyeGaze-Multimodal/json_textdescription/mixed_description_json.json"
]

# ============================================================
# Function Definitions
# ============================================================
def load_metrics():
    """Load metrics for evaluation."""
    return {metric: load(metric) for metric in METRICS}

def concatenate_images(image1_path, image2_path):
    """Concatenate two images vertically."""
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path).resize((image1.width, image1.height))
    concatenated_image = Image.new("RGB", (image1.width, image1.height * 2))
    concatenated_image.paste(image1, (0, 0))
    concatenated_image.paste(image2, (0, image1.height))
    return concatenated_image

def extract_label_from_filename(filename):
    """
    從檔名提取真實標籤
    例如: "Pair-12-A-Single-EYE_trial01_player.jpg" -> "single"
         "Pair-12-Comp-EYE_trial01_playerA.jpg" -> "competition"
         "Pair-12-Coop-EYE_trial01_playerA.jpg" -> "cooperation"
    """
    filename = filename.lower()
   
    if "single" in filename:
        return "single"
    elif "comp" in filename:
        return "competition"
    elif "coop" in filename:
        return "cooperation"
    else:
        print(f"Warning: Cannot extract label from filename: {filename}")
        return "unknown"

def evaluate_metrics(predictions, references, metrics):
    """Compute evaluation metrics."""
    encoded_preds = [LABEL_MAPPING.get(pred, 3) for pred in predictions]  # 3 for unknown
    encoded_refs = [LABEL_MAPPING.get(ref, 3) for ref in references]

    results = {}
    for metric_name, metric in metrics.items():
        try:
            if metric_name == "accuracy":
                results[metric_name] = metric.compute(predictions=encoded_preds, references=encoded_refs)
            else:
                results[metric_name] = metric.compute(predictions=encoded_preds, references=encoded_refs, average="weighted")
        except Exception as e:
            print(f"Error computing {metric_name}: {e}")
            results[metric_name] = {metric_name: 0.0}
   
    return results

def create_confusion_matrix(predictions, references, json_filename, root_dir):
    """Create and save confusion matrix."""
    encoded_preds = [LABEL_MAPPING.get(pred, 3) for pred in predictions]
    encoded_refs = [LABEL_MAPPING.get(ref, 3) for ref in references]
   
    # Create confusion matrix
    cm = confusion_matrix(encoded_refs, encoded_preds)
   
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Single', 'Cooperation', 'Competition', 'Unknown'],
                yticklabels=['Single', 'Cooperation', 'Competition', 'Unknown'])
    plt.title(f'Confusion Matrix - {json_filename}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
   
    # Save
    save_path = os.path.join(root_dir, "git_checkpoint", f"confusion_matrix_{json_filename.replace('.json', '')}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

# ============================================================
# Main Script
# ============================================================
def main():
    metrics = load_metrics()

    for root_dir in ROOT_LIST:
        print(f"\n{'='*60}")
        print(f"Processing ROOT: {root_dir}")
        print(f"{'='*60}")

        # Load model and processor
        checkpoint_path = os.path.join(root_dir, "git_checkpoint/checkpoint-590")
       
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            continue
           
        processor = GitProcessor.from_pretrained("microsoft/git-base")
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(DEVICE)
       
        # 為每個 JSON 檔案分別處理
        for json_file in JSON_FILES:
            json_filename = os.path.basename(json_file)
            print(f"\n--- Processing {json_filename} ---")
           
            if not os.path.exists(json_file):
                print(f"JSON file not found: {json_file}")
                continue
           
            # Load dataset
            try:
                test_ds = load_dataset("json", data_files=json_file, split="train")
                print(f"Dataset loaded: {len(test_ds)} samples")
            except Exception as e:
                print(f"Error loading dataset: {e}")
                continue

            pred_list, gt_list, correct_count = [], [], 0
            error_count = 0

            for idx in tqdm(range(len(test_ds)), desc=f"Testing {json_filename}"):
                try:
                    # 從檔名提取真實標籤
                    image1_name = test_ds[idx]["image1"]
                    ground_truth = extract_label_from_filename(image1_name)
                   
                    # Prepare images
                    img1_path = os.path.join(root_dir, image1_name)
                    img2_path = os.path.join(root_dir, test_ds[idx]["image2"])
                   
                    # Check if images exist
                    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                        print(f"Images not found: {img1_path} or {img2_path}")
                        error_count += 1
                        continue
                   
                    concatenated_image = concatenate_images(img1_path, img2_path)

                    # Model inference
                    inputs = processor(images=concatenated_image, return_tensors="pt").to(DEVICE)
                    generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
                    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].lower()

                    # 從生成的描述中提取預測標籤
                    if "single" in generated_caption:
                        prediction = "single"
                    elif "cooperation" in generated_caption or "cooperative" in generated_caption:
                        prediction = "cooperation"
                    elif "competition" in generated_caption or "competitive" in generated_caption:
                        prediction = "competition"
                    else:
                        prediction = "unknown"

                    # Record predictions and ground truths
                    pred_list.append(prediction)
                    gt_list.append(ground_truth)
                   
                    if ground_truth == prediction:
                        correct_count += 1
                       
                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    error_count += 1
                    continue

            if len(pred_list) == 0:
                print(f"No valid predictions for {json_filename}")
                continue

            # Raw accuracy
            raw_accuracy = correct_count / len(pred_list)
            print(f"Raw Accuracy: {raw_accuracy:.4f}")
            print(f"Correct predictions: {correct_count}/{len(pred_list)}")
            print(f"Errors: {error_count}")

            # Calculate metrics
            try:
                metric_results = evaluate_metrics(pred_list, gt_list, metrics)
               
                # 創建結果 DataFrame
                result_row = {
                    "JSON_File": json_filename,
                    "Root_Dir": root_dir,
                    "Total_Samples": len(pred_list),
                    "Correct_Predictions": correct_count,
                    "Raw_Accuracy": raw_accuracy,
                    "Accuracy": metric_results["accuracy"]["accuracy"],
                    "Precision": metric_results["precision"]["precision"],
                    "Recall": metric_results["recall"]["recall"],
                    "F1_Score": metric_results["f1"]["f1"]
                }
               
                results_df = pd.DataFrame([result_row])
               
                # Save results to CSV
                csv_file = os.path.join(root_dir, "git_checkpoint", f"metrics_results_{json_filename.replace('.json', '')}.csv")
                results_df.to_csv(csv_file, index=False)
                print(f"Metrics results saved to {csv_file}")
               
                # 儲存詳細的預測結果（包含生成的句子）
                details_df = pd.DataFrame(sample_details)
                details_csv = os.path.join(root_dir, "git_checkpoint", f"detailed_predictions_{json_filename.replace('.json', '')}.csv")
                details_df.to_csv(details_csv, index=False)
                print(f"Detailed predictions (with sentences) saved to {details_csv}")
               
                # 顯示一些生成句子的範例
                print(f"\n--- Sample Generated Sentences ---")
                for i in range(min(10, len(generated_sentences))):  # 增加到10個範例
                    print(f"Sample {i+1}:")
                    print(f"  Image1: {sample_details[i]['image1']}")
                    print(f"  Generated: '{generated_sentences[i]}'")
                    print(f"  Prediction: {pred_list[i]}")
                    print(f"  Ground Truth: {gt_list[i]}")
                    print(f"  Correct: {gt_list[i] == pred_list[i]}")
                    print("-" * 50)
               
                # 儲存所有生成句子到單獨的文字檔案
                sentences_file = os.path.join(root_dir, "git_checkpoint", f"all_generated_sentences_{json_filename.replace('.json', '')}.txt")
                with open(sentences_file, 'w', encoding='utf-8') as f:
                    f.write(f"Generated Sentences for {json_filename}\n")
                    f.write(f"Root Directory: {root_dir}\n")
                    f.write(f"Total Samples: {len(generated_sentences)}\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("="*80 + "\n\n")
                   
                    for i, sentence in enumerate(generated_sentences):
                        f.write(f"Sample {i+1}:\n")
                        f.write(f"  Image1: {sample_details[i]['image1']}\n")
                        f.write(f"  Image2: {sample_details[i]['image2']}\n")
                        f.write(f"  Generated: '{sentence}'\n")
                        f.write(f"  Prediction: {pred_list[i]}\n")
                        f.write(f"  Ground Truth: {gt_list[i]}\n")
                        f.write(f"  Correct: {gt_list[i] == pred_list[i]}\n")
                        f.write("-" * 60 + "\n")
               
                print(f"All generated sentences saved to {sentences_file}")
               
                # Create confusion matrix
                create_confusion_matrix(pred_list, gt_list, json_filename, root_dir)
               
                # Print detailed results
                print(f"\nDetailed Results for {json_filename}:")
                print(f"Accuracy: {metric_results['accuracy']['accuracy']:.4f}")
                print(f"Precision: {metric_results['precision']['precision']:.4f}")
                print(f"Recall: {metric_results['recall']['recall']:.4f}")
                print(f"F1-Score: {metric_results['f1']['f1']:.4f}")
               
                # 分類別統計
                pred_counts = pd.Series(pred_list).value_counts()
                gt_counts = pd.Series(gt_list).value_counts()
                print(f"\nPrediction distribution: {dict(pred_counts)}")
                print(f"Ground truth distribution: {dict(gt_counts)}")
               
            except Exception as e:
                print(f"Error calculating metrics for {json_filename}: {e}")
                continue

        print(f"\nCompleted processing for {root_dir}")

if __name__ == "__main__":
    main()
