############################################################
# 1. Import Libraries
############################################################
from datasets import load_dataset
from textwrap import wrap
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# For ResNet-50 classification
from transformers import (
    AutoImageProcessor,
    ResNetForImageClassification,
    TrainingArguments,
    Trainer
)

from evaluate import load

############################################################
# 2. Define the ROOT list
############################################################
ROOT_LIST = [
    "bgOn_heatmapOn_trajOn",
    "bgOn_heatmapOn_trajOff",
    "bgOn_heatmapOff_trajOn",
    "bgOff_heatmapOn_trajOn",
    "bgOff_heatmapOn_trajOff",
    "bgOff_heatmapOff_trajOn"
]
DATASET_FILE_TRAIN = "./image_gt_train.json"
DATASET_FILE_TEST = "./image_gt_test.json"

############################################################
# 3. Load and Split Dataset
############################################################
def load_and_split_dataset():
    train_ds = load_dataset("json", data_files=DATASET_FILE_TRAIN, split="train")#[:5%]
    test_ds  = load_dataset("json", data_files=DATASET_FILE_TEST, split="train")#[:5%]
    return train_ds, test_ds

############################################################
# 4. Plotting Function (Optional)
############################################################
def plot_images(images1, images2, captions, root):
    plt.figure(figsize=(20, 20))
    for i in range(len(images1)):
        img_path1 = f"{root}/{images1[i]}"
        img_path2 = f"{root}/{images2[i]}"

        image1 = Image.open(img_path1)
        image2 = Image.open(img_path2)

        # Resize image2 to match image1 size
        image2 = image2.resize((image1.width, image1.height))

        # Concatenate vertically
        new_width = image1.width
        new_height = image1.height + image2.height
        concatenated_image = Image.new("RGB", (new_width, new_height))
        concatenated_image.paste(image1, (0, 0))
        concatenated_image.paste(image2, (0, image1.height))

        ax = plt.subplot(1, len(images1), i + 1)
        wrapped_caption = "\n".join(wrap(captions[i], 12))
        plt.title(wrapped_caption)
        plt.imshow(concatenated_image)
        plt.axis("off")
    plt.show()

############################################################
# 5. Transform Function
############################################################
def get_transform_function(processor, root):
    """
    Returns a transform function that:
      1) loads + concatenates images
      2) processes them via a ResNet-friendly processor
      3) attaches integer-encoded labels
    """
    label_mapping = {"Cooperation": 0, "Single": 1, "Competition": 2}

    def transforms(example_batch):
        concatenated_images = []
        encoded_labels = []

        for img_path1, img_path2, lbl in zip(
            example_batch["image1"], 
            example_batch["image2"], 
            example_batch["class"]
        ):
            # Load images
            image1 = Image.open(f"{root}/{img_path1}")
            image2 = Image.open(f"{root}/{img_path2}")

            # Resize to match
            image2 = image2.resize((image1.width, image1.height))

            # Concatenate vertically
            new_width = image1.width
            new_height = image1.height + image2.height
            concatenated_image = Image.new("RGB", (new_width, new_height))
            concatenated_image.paste(image1, (0, 0))
            concatenated_image.paste(image2, (0, image1.height))

            concatenated_images.append(concatenated_image)
            encoded_labels.append(label_mapping.get(lbl, -1))  # default -1 for unknown

        # Convert images into model input
        inputs = processor(images=concatenated_images, return_tensors="pt")
        # Hugging Face Trainer expects a "labels" field
        inputs["labels"] = encoded_labels
        return inputs

    return transforms

############################################################
# 6. Compute Metrics
############################################################
f1_metric = load("f1")

def compute_metrics(eval_pred):
    """
    Standard classification metric flow:
      - eval_pred is (logits, labels)
      - take argmax over logits
      - compute F1 (weighted) for multi-class scenario
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    results = f1_metric.compute(predictions=preds, references=labels, average="weighted")
    return {"f1_score": results["f1"]}

############################################################
# 7. Main Training Loop Over ROOT_LIST
############################################################
if __name__ == "__main__":

    # 1) Load and split dataset
    train_ds, test_ds = load_and_split_dataset()

    # 2) OPTIONAL: Visualize a small sample from the first ROOT
    sample_images1 = [train_ds[i]["image1"] for i in range(5)]
    sample_images2 = [train_ds[i]["image2"] for i in range(5)]
    sample_captions = [train_ds[i]["class"] for i in range(5)]
    # plot_images(sample_images1, sample_images2, sample_captions, ROOT_LIST[0])

    # 3) Loop over each ROOT folder and train
    for root in ROOT_LIST:
        print(f"========== Training with ROOT: {root} ==========")
        
        # 4) Load init ResNet-50 and its processor
        #    'microsoft/resnet-50' is one example checkpoint on Hugging Face
        checkpoint = "microsoft/resnet-50"
        processor = AutoImageProcessor.from_pretrained(checkpoint)
        
        # Use num_labels=3 for the 3 classes: cooperation, single, competition
        model = ResNetForImageClassification.from_pretrained(
        checkpoint, 
        num_labels=3, 
        ignore_mismatched_sizes=True)
        
        # Assign the custom transforms
        train_ds.set_transform(get_transform_function(processor, root))
        test_ds.set_transform(get_transform_function(processor, root))

        training_args = TrainingArguments(
            output_dir=f"{root}/res_checkpoint",
            learning_rate=5e-5,
            num_train_epochs=10,
            fp16=True,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            eval_accumulation_steps=32,
            save_total_limit=3,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            logging_steps=50,
            remove_unused_columns=False,
            push_to_hub=False,
            load_best_model_at_end=True,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=compute_metrics
        )

        # 5) Start training
        trainer.train()
        print(f"========== Finished Training with ROOT: {root} ==========\n")


