import os
import torch
from PIL import Image
from datasets import load_dataset
from transformers import AutoModelForCausalLM, GitProcessor
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from evaluate import load
from sklearn.metrics import confusion_matrix

############################################################
# 1) Define ROOT list
############################################################
ROOT_LIST = [
    "bgOn_heatmapOn_trajOn",
    "bgOn_heatmapOn_trajOff",
    "bgOn_heatmapOff_trajOn",
    "bgOff_heatmapOn_trajOn",
    "bgOff_heatmapOn_trajOff",
    "bgOff_heatmapOff_trajOn"
]

############################################################
# 2) Load the dataset (Test set)
############################################################
test_ds = load_dataset("json", data_files="./image_gt_train.json", split="train")
print(test_ds)#image_gt_test

device = "cuda" if torch.cuda.is_available() else "cpu"

############################################################
# 3) Define label mapping
############################################################
label_mapping = {"single": 0, "cooperation": 1, "competition": 2}

############################################################
# 4) Load metrics
############################################################
accuracy_metric = load("accuracy")
precision_metric = load("precision")
recall_metric = load("recall")
f1_metric = load("f1")

############################################################
# 5) Main Loop Over ROOT_LIST
############################################################
for root_dir in ROOT_LIST:
    print(f"\n========== Inference and Evaluation for ROOT: {root_dir} ==========")

    # Adjust checkpoint path (change checkpoint-680 if needed)
    checkpoint_path = os.path.join(root_dir, "git_checkpoint/checkpoint-590")

    # --------------- Step 1: Test (Inference) ---------------
    # Load model & processor
    processor = GitProcessor.from_pretrained("microsoft/git-base")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)

    # For raw accuracy count
    yes_num = 0
    pred_list = []
    gt_list = []
    
    folder_path = "gt_subject"  
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    for subject in files:
        test_ds = load_dataset("json", data_files=folder_path+subject, split="train")
    print(test_ds)#image_gt_test

    for idx in tqdm(range(len(test_ds)), desc="Testing"):
        # Load and concatenate images
        img_path1 = os.path.join(root_dir, test_ds[idx]["image1"])
        img_path2 = os.path.join(root_dir, test_ds[idx]["image2"])

        image1 = Image.open(img_path1)
        image2 = Image.open(img_path2)

        image2 = image2.resize((image1.width, image1.height))
        new_width = image1.width
        new_height = image1.height + image2.height
        concatenated_image = Image.new("RGB", (new_width, new_height))
        concatenated_image.paste(image1, (0, 0))
        concatenated_image.paste(image2, (0, image1.height))

        # Prepare inputs
        inputs = processor(images=concatenated_image, return_tensors="pt").to(device)
        pixel_values = inputs.pixel_values

        # Generate text from model
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Record prediction & ground truth
        pred_list.append(generated_caption)
        ground_truth = test_ds[idx]["class"].lower()
        gt_list.append(ground_truth)

        # Count raw accuracy
        if ground_truth == generated_caption:
            yes_num += 1

    # Print raw accuracy
    raw_accuracy = yes_num / len(test_ds)
    print(f"Raw Accuracy: {raw_accuracy:.4f}")

    # --------------- Step 2: Calculate Accuracy, Precision, Recall, and F1 ---------------
    encoded_labels = [label_mapping.get(label, 4) for label in gt_list]
    encoded_predictions = [label_mapping.get(pred, 4) for pred in pred_list]

    accuracy = accuracy_metric.compute(predictions=encoded_predictions, references=encoded_labels)
    precision = precision_metric.compute(predictions=encoded_predictions, references=encoded_labels, average="weighted")
    recall = recall_metric.compute(predictions=encoded_predictions, references=encoded_labels, average="weighted")
    f1_score_ = f1_metric.compute(predictions=encoded_predictions, references=encoded_labels, average="weighted")

    print(f"Accuracy: {accuracy['accuracy']:.4f}")
    print(f"Precision: {precision['precision']:.4f}")
    print(f"Recall: {recall['recall']:.4f}")
    print(f"F1 Score: {f1_score_['f1']:.4f}")

    # --------------- Step 3: Save Confusion Matrix as an Image ---------------
    cm = confusion_matrix(encoded_labels, encoded_predictions)
    labels = ["single (0)", "cooperation (1)", "competition (2)"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix ({root_dir})")

    # Save figure to file (e.g., PNG)
    cm_filename = f"confusion_matrix_{root_dir}.png"
    plt.savefig(cm_filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {cm_filename}\n")


