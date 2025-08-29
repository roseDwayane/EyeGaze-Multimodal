import os
import re
import numpy as np
from datasets import load_dataset
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, GitProcessor
from tqdm import tqdm
from pathlib import Path
from evaluate import load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

############################################################
# Helper Functions
############################################################

def extract_pair_info(filepath):
    """Extracts pair information from a given filename."""
    filename = os.path.basename(filepath)
    pattern = r"^(Pair-\d+)(?:-[AB])?-(Coop|Comp|Single)"
    match = re.match(pattern, filename)
    return f"{match.group(1)}-{match.group(2)}" if match else None

def concate_plvimg(img_path):
    """Concatenates PLV images for different frequency bands into a single image."""
    plv_img = f"./plvimg/{extract_pair_info(img_path)}"
    image_paths = [f"{plv_img}-{band}.jpg" for band in ["delta", "theta", "alpha", "beta", "gamma"]]
    images = [Image.open(img) for img in image_paths]
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    concatenated_image = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for img in images:
        concatenated_image.paste(img, (x_offset, 0))
        x_offset += img.width
    return concatenated_image

def concatenate_images(image1_path, image2_path, fusion, train_with_plv):
    """Concatenates two images using different fusion methods (Concate, Subtract, Dot_product)."""
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path).resize((image1.width, image1.height))
    plvimg = concate_plvimg(image1_path)

    if fusion == "Concate":
        concatenated_image = Image.new('RGB', (image1.width, image1.height + image2.height))
        concatenated_image.paste(image1, (0, 0))
        concatenated_image.paste(image2, (0, image1.height))
    elif fusion == "Subtract":
        diff_arr = np.clip(np.array(image2, dtype=np.int16) - np.array(image1, dtype=np.int16), 0, 255).astype(np.uint8)
        concatenated_image = Image.fromarray(diff_arr)
    elif fusion == "Dot_product":
        dot_arr = np.clip(np.array(image1, dtype=np.float32) * np.array(image2, dtype=np.float32) / 255.0, 0, 255).astype(np.uint8)
        concatenated_image = Image.fromarray(dot_arr)
    elif fusion == 0:
        concatenated_image = plvimg

    if train_with_plv:
        plvimg = plvimg.resize((concatenated_image.width, int(concatenated_image.height * 0.4)))
        combined_image = Image.new("RGB", (concatenated_image.width, concatenated_image.height + plvimg.height))
        combined_image.paste(concatenated_image, (0, 0))
        combined_image.paste(plvimg, (0, concatenated_image.height))
        concatenated_image = combined_image
    return concatenated_image

############################################################
# Load Dataset
############################################################
train_ds = load_dataset("json", data_files="./image_gt.json", split="train")
print(train_ds)

# Configure device for computation
device = "cuda" if torch.cuda.is_available() else "cpu"

FUSION_LIST = ["Concate", "Subtract", "Dot_product"]
ROOT = "./bgOn_heatmapOn_trajOn/"

############################################################
# Model Inference and Evaluation
############################################################
for fusion in FUSION_LIST:
    fusion = 0
    train_with_plv = False
    print(f"\n========== Running {fusion} Fusion with PLV: {train_with_plv} ==========")
    CHECKPOINT_PATH = os.path.join(ROOT, f"git_checkpoint_{fusion}{'_plv' if train_with_plv else ''}/checkpoint-590")
    processor = GitProcessor.from_pretrained("microsoft/git-base")
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_PATH).to(device)
    label_mapping = {"single": 0, "cooperation": 1, "competition": 2}
    pred_list, gt_list = [], []
    
    for idx in tqdm(range(len(train_ds))):
        image1_path, image2_path = Path(ROOT) / train_ds[idx]["image1"], Path(ROOT) / train_ds[idx]["image2"]
        concatenated_image = concatenate_images(image1_path, image2_path, fusion, train_with_plv)
        pixel_values = processor(images=[concatenated_image], return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values=pixel_values, max_length=32)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].lower()
        pred_list.append(generated_caption)
        gt_list.append(train_ds[idx]["class"].lower())
    
    accuracy = sum(gt == pred for gt, pred in zip(gt_list, pred_list)) / len(train_ds)
    print(f"Accuracy: {accuracy:.4f}")
    
    encoded_labels = [label_mapping.get(label, -1) for label in gt_list]
    encoded_predictions = [label_mapping.get(pred, -1) for pred in pred_list]
    
    # Compute metrics
    metrics = {
        "accuracy": load("accuracy").compute(predictions=encoded_predictions, references=encoded_labels),
        "precision": load("precision").compute(predictions=encoded_predictions, references=encoded_labels, average="weighted"),
        "recall": load("recall").compute(predictions=encoded_predictions, references=encoded_labels, average="weighted"),
        "f1_score": load("f1").compute(predictions=encoded_predictions, references=encoded_labels, average="weighted")
    }
    for key, value in metrics.items():
        print(f"{key.capitalize()}: {value}")
    
    # Generate and save confusion matrix
    cm = confusion_matrix(encoded_labels, encoded_predictions)
    labels = ['single (0)', 'cooperation (1)', 'competition (2)']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    conf_matrix_path = os.path.join(ROOT, f"git_checkpoint_{fusion}{'_plv' if train_with_plv else ''}/confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    #plt.show()
    print(f"Confusion matrix saved at: {conf_matrix_path}")
