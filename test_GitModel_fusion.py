import os
import re
import torch
import pandas as pd
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import AutoModelForCausalLM, GitProcessor
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate import load
from sklearn.metrics import confusion_matrix

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

FUSION_LIST = [
    "Concate",
    "Subtract",
    "Dot_product"
]

LABEL_MAPPING = {"single": 0, "cooperation": 1, "competition": 2}
METRICS = ["accuracy", "precision", "recall", "f1"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FILES_DIR = "gt_subject"

# ============================================================
# Function Definitions
# ============================================================
def load_metrics():
    """Load metrics for evaluation."""
    return {metric: load(metric) for metric in METRICS}

def extract_pair_info(filepath):
    # Extract only the filename from the full path
    filename = os.path.basename(filepath)
    
    # Regular expression to match the required pattern
    pattern = r"^(Pair-\d+)(?:-[AB])?-(Coop|Comp|Single)"
    
    match = re.match(pattern, filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return None  # Return None if the pattern does not match

def concate_plvimg(img_path):
    plv_img = "./plvimg/" + extract_pair_info(img_path)
    # List of image file paths
    image_paths = [plv_img+"-delta.jpg", plv_img+"-theta.jpg", plv_img+"-alpha.jpg", plv_img+"-beta.jpg", plv_img+"-gamma.jpg"]

    # Load images
    images = [Image.open(img) for img in image_paths]

    # Get total width and max height
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create new blank image
    concatenated_image = Image.new("RGB", (total_width, max_height))

    # Paste images side by side
    x_offset = 0
    for img in images:
        concatenated_image.paste(img, (x_offset, 0))
        x_offset += img.width
    
    #concatenated_image.show()  # Show the image
    return concatenated_image

def concatenate_images(image1_path, image2_path, fusion, train_with_plv):
    """Concatenate two images vertically."""
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path).resize((image1.width, image1.height))
    plvimg = concate_plvimg(image1_path)

    # Concate
    if fusion == "Concate":
        ## 創建拼接後的新圖片（寬度為兩張圖片寬度之和）
        new_width = image1.width
        new_height = image1.height + image2.height
        concatenated_image = Image.new('RGB', (new_width, new_height))
        ## 將圖片粘貼到新圖片上
        concatenated_image.paste(image1, (0, 0))
        concatenated_image.paste(image2, (0, image1.height))

    # Subtract
    elif fusion == "Subtract":
        ## Convert images to NumPy arrays
        arr1 = np.array(image1, dtype=np.int16)  # Use int16 to prevent underflow
        arr2 = np.array(image2, dtype=np.int16)
        ## Subtract images (ensure values remain in valid range)
        diff_arr = np.clip(arr2 - arr1, 0, 255).astype(np.uint8)  # Clip values to [0, 255]
        ## Convert back to image
        concatenated_image = Image.fromarray(diff_arr)

    # Dot_product
    elif fusion == "Dot_product":
        ## Convert images to NumPy arrays
        arr1 = np.array(image1, dtype=np.float32)  # Use float32 to avoid overflow
        arr2 = np.array(image2, dtype=np.float32)
        ## Dot product (element-wise multiplication)
        dot_arr = np.clip(arr1 * arr2 / 255.0, 0, 255).astype(np.uint8)  # Normalize to prevent overflow
        ## Convert back to image
        concatenated_image = Image.fromarray(dot_arr)
    elif fusion == 0:
        concatenated_image = plvimg

    if train_with_plv:
        # Ensure both images have the same width
        new_plv_height = int(concatenated_image.height * 0.4)  # Adjust ratio
        plvimg = plvimg.resize((concatenated_image.width, new_plv_height))
        # Create a new blank image with concatenated height
        total_height = concatenated_image.height + plvimg.height
        combined_image = Image.new("RGB", (concatenated_image.width, total_height))
        # Paste both images
        combined_image.paste(concatenated_image, (0, 0))
        combined_image.paste(plvimg, (0, concatenated_image.height))
        concatenated_image = combined_image

    return concatenated_image

def evaluate_metrics(predictions, references, metrics):
    """Compute evaluation metrics."""
    encoded_preds = [LABEL_MAPPING.get(pred, 4) for pred in predictions]
    encoded_refs = [LABEL_MAPPING.get(ref, 4) for ref in references]

    results = {}
    for metric_name, metric in metrics.items():
        if metric_name == "accuracy":
            # Accuracy does not use the 'average' argument
            results[metric_name] = metric.compute(predictions=encoded_preds, references=encoded_refs)
        else:
            # Other metrics (precision, recall, f1) use the 'average' argument
            results[metric_name] = metric.compute(predictions=encoded_preds, references=encoded_refs, average="weighted")
    return results

# ============================================================
# Main Script
# ============================================================
def main():
    metrics = load_metrics()

    for fusion in FUSION_LIST:
        fusion = 0
        root_dir = "bgOn_heatmapOn_trajOn"
        train_with_plv = False # True, False
        print(f"\n========== Inference and Evaluation for ROOT: {root_dir} {fusion}, plv {train_with_plv}  ==========")

        # Load model and processor
        checkpoint_path = os.path.join(root_dir, f"git_checkpoint_{fusion}/checkpoint-590")
        processor = GitProcessor.from_pretrained("microsoft/git-base")
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(DEVICE)

        results_df = pd.DataFrame(columns=["Iteration", "Accuracy", "Precision", "Recall", "F1-Score"])
        files = [f for f in os.listdir(DATA_FILES_DIR) if os.path.isfile(os.path.join(DATA_FILES_DIR, f))]

        for subject in files:
            test_ds = load_dataset("json", data_files=os.path.join(DATA_FILES_DIR, subject), split="train")
            print(test_ds)

            pred_list, gt_list, yes_count = [], [], 0

            for idx in tqdm(range(len(test_ds)), desc=f"Testing {subject}"):
                # Prepare images
                img1_path = os.path.join(root_dir, test_ds[idx]["image1"])
                img2_path = os.path.join(root_dir, test_ds[idx]["image2"])
                concatenated_image = concatenate_images(img1_path, img2_path, fusion, train_with_plv)

                # Model inference
                inputs = processor(images=concatenated_image, return_tensors="pt").to(DEVICE)
                generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
                generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Record predictions and ground truths
                pred_list.append(generated_caption)
                ground_truth = test_ds[idx]["class"].lower()
                gt_list.append(ground_truth)
                if ground_truth == generated_caption:
                    yes_count += 1

            # Raw accuracy
            raw_accuracy = yes_count / len(test_ds)
            print(f"Raw Accuracy: {raw_accuracy:.4f}")

            # Calculate metrics
            metric_results = evaluate_metrics(pred_list, gt_list, metrics)
            results_df = pd.concat([
                results_df,
                pd.DataFrame({
                    "Iteration": [subject],
                    "Accuracy": [metric_results["accuracy"]["accuracy"]],
                    "Precision": [metric_results["precision"]["precision"]],
                    "Recall": [metric_results["recall"]["recall"]],
                    "F1-Score": [metric_results["f1"]["f1"]]
                })
            ], ignore_index=True)

        # Save results to CSV
        csv_file = os.path.join(root_dir, f"git_checkpoint_{fusion}/metrics_results.csv")
        results_df.to_csv(csv_file, index=False)
        print(f"Metrics results saved to {csv_file}.")

if __name__ == "__main__":
    main()

