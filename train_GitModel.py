############################################################
# Step 1. Import Libraries
############################################################
from datasets import load_dataset
from textwrap import wrap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from transformers import (
    AutoProcessor,
    GitProcessor,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from evaluate import load


############################################################
# Step 2. Define the ROOT list
############################################################
# You can place these subfolders under a common parent folder
# if needed. For example, you might prepend a "./" or another
# path to each item.
ROOT_LIST = [
    # "bgOn_heatmapOn_trajOn",
    "bgOn_heatmapOn_trajOff",
    "bgOn_heatmapOff_trajOn",
    "bgOff_heatmapOn_trajOn",
    "bgOff_heatmapOn_trajOff",
    "bgOff_heatmapOff_trajOn"
]

DATASET_FILE_TRAIN = "./image_gt_train.json"
DATASET_FILE_TEST = "./image_gt_test.json"

############################################################
# Step 3. Define Your Dataset and Plotting Functions
############################################################
def load_and_split_dataset():
    train_ds = load_dataset("json", data_files=DATASET_FILE_TRAIN, split="train")#[:5%]
    test_ds  = load_dataset("json", data_files=DATASET_FILE_TEST, split="train")#[:5%]
    return train_ds, test_ds


def plot_images(images1, images2, captions, root):
    plt.figure(figsize=(20, 20))
    for i in range(len(images1)):
        img_path1 = f"{root}/{images1[i]}"
        img_path2 = f"{root}/{images2[i]}"

        image1 = Image.open(img_path1)
        image2 = Image.open(img_path2)

        image2 = image2.resize((image1.width, image1.height))
        new_width = image1.width
        new_height = image1.height + image2.height
        concatenated_image = Image.new("RGB", (new_width, new_height))

        concatenated_image.paste(image1, (0, 0))
        concatenated_image.paste(image2, (0, image1.height))

        ax = plt.subplot(1, len(images1), i + 1)
        caption = "\n".join(wrap(captions[i], 12))
        plt.title(caption)
        plt.imshow(concatenated_image)
        plt.axis("off")


############################################################
# Step 4. Define Transform Function
############################################################
def get_transform_function(processor, root):
    """Returns a transform function bound to the given root and processor."""
    
    def transforms(example_batch):
        concatenated_images = []

        for img_path1, img_path2 in zip(example_batch["image1"], example_batch["image2"]):
            image1 = Image.open(f"{root}/{img_path1}")
            image2 = Image.open(f"{root}/{img_path2}")

            image2 = image2.resize((image1.width, image1.height))
            new_width = image1.width
            new_height = image1.height + image2.height
            concatenated_image = Image.new('RGB', (new_width, new_height))

            concatenated_image.paste(image1, (0, 0))
            concatenated_image.paste(image2, (0, image1.height))

            concatenated_images.append(concatenated_image)

        captions = [x for x in example_batch["class"]]
        inputs = processor(
            images=concatenated_images,
            text=captions,
            padding="max_length",
            max_length=32,
            truncation=True
        )
        inputs.update({"labels": inputs["input_ids"]})
        return inputs
    
    return transforms


############################################################
# Step 5. Define Metrics
############################################################
f1_metric = load("f1")

def compute_metrics(eval_pred, processor):
    torch.cuda.empty_cache()
    with torch.no_grad():
        logits, labels = eval_pred
        predicted = logits.argmax(-1)

        decoded_labels = [
            label.lower() for label in processor.batch_decode(labels, skip_special_tokens=True)
        ]
        decoded_predictions = [
            pred.lower() for pred in processor.batch_decode(predicted, skip_special_tokens=True)
        ]

        label_mapping = {"cooperation": 0, "single": 1, "competition": 2}
        encoded_labels = [label_mapping.get(label, 4) for label in decoded_labels]
        encoded_predictions = [label_mapping.get(pred, 4) for pred in decoded_predictions]

        f1_score = f1_metric.compute(
            predictions=encoded_predictions,
            references=encoded_labels,
            average="weighted"
        )

    torch.cuda.empty_cache()
    return {"f1_score": f1_score['f1']}


############################################################
# Step 6. Main Training Loop Over ROOT_LIST
############################################################
if __name__ == "__main__":

    # 1) Load and split the dataset once (depends on your needs)
    train_ds, test_ds = load_and_split_dataset()

    # 2) Just for demonstration: visualize a small sample (only from the first ROOT).
    sample_images1 = [train_ds[i]["image1"] for i in range(5)]
    sample_images2 = [train_ds[i]["image2"] for i in range(5)]
    sample_captions = [train_ds[i]["class"] for i in range(5)]
    # Plot images for the first ROOT in the list
    # comment out if you don't need the visualization
    # plot_images(sample_images1, sample_images2, sample_captions, ROOT_LIST[0])

    # 3) Loop over each ROOT path
    for root in ROOT_LIST:
        print(f"========== Training with ROOT: {root} ==========")
        
        # 4) (Re)Load the init model/processor
        checkpoint = "microsoft/git-base"
        processor = GitProcessor.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
        
        # Set transforms
        train_ds.set_transform(get_transform_function(processor, root))
        test_ds.set_transform(get_transform_function(processor, root))

        # 5) Define training arguments
        training_args = TrainingArguments(
            output_dir=f"{root}/git_checkpoint",
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
            label_names=["labels"],
            load_best_model_at_end=True,
            report_to="none"
        )

        # 6) Create Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, processor),
        )

        # 7) Train
        trainer.train()
        print(f"========== Finished Training with ROOT: {root} ==========\n")


