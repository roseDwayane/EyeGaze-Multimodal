import os
import torch
import pandas as pd
from PIL import Image
from datasets import load_dataset
from transformers import AutoImageProcessor, ResNetForImageClassification
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
LABEL_MAPPING = {"cooperation": 0, "single": 1, "competition": 2}
METRICS = ["accuracy", "precision", "recall", "f1"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FILES_DIR = "gt_subject"

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

    for root_dir in ROOT_LIST:
        print(f"\n========== Inference and Evaluation for ROOT: {root_dir} ==========")

        # Load model and processor
        checkpoint_path = os.path.join(root_dir, "res_checkpoint", "checkpoint-590")
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        print(f"Loading local checkpoint from: {checkpoint_path}")
        model = ResNetForImageClassification.from_pretrained(
            checkpoint_path,
            num_labels=3,
            ignore_mismatched_sizes=True
        ).to(DEVICE)


        results_df = pd.DataFrame(columns=["Iteration", "Accuracy", "Precision", "Recall", "F1-Score"])
        files = [f for f in os.listdir(DATA_FILES_DIR) if os.path.isfile(os.path.join(DATA_FILES_DIR, f))]

        for subject in files:
            test_ds = load_dataset("json", data_files=os.path.join(DATA_FILES_DIR, subject), split="train")
            print(test_ds)

            pred_list, gt_list, yes_num = [], [], 0

            for idx in tqdm(range(len(test_ds)), desc=f"Testing {subject}"):
                # Prepare images
                img1_path = os.path.join(root_dir, test_ds[idx]["image1"])
                img2_path = os.path.join(root_dir, test_ds[idx]["image2"])
                concatenated_image = concatenate_images(img1_path, img2_path)

                # Model inference
                inputs = processor(images=concatenated_image, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits  # shape [batch_size, num_labels]
                    preds = torch.argmax(logits, dim=1).cpu().numpy()  # shape [batch_size]

                predicted_label = preds[0]

                # Convert numeric prediction back to string if desired
                # We can store the numeric label, or map it to a string
                for k, v in LABEL_MAPPING.items():
                    if v == predicted_label:
                        pred_list.append(k)
                        break

                ground_truth_str = test_ds[idx]["class"].lower()
                gt_list.append(ground_truth_str)
                
                #print(ground_truth_str, predicted_label)

                # Compare direct string matching if you like,
                # or compare the integer-encoded forms
                # For raw accuracy (string match):
                if predicted_label == LABEL_MAPPING.get(ground_truth_str, -1):
                    yes_num += 1

            # Print raw accuracy
            raw_accuracy = yes_num / len(test_ds)
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
        csv_file = os.path.join(root_dir, "res_checkpoint/metrics_results.csv")
        results_df.to_csv(csv_file, index=False)
        print(f"Metrics results saved to {csv_file}.")

if __name__ == "__main__":
    main()

