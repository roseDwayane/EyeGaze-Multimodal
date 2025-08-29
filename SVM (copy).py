import os
import torch
import json
import numpy as np

from PIL import Image
from datasets import load_dataset
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

############################################################
# 1. Configurations
############################################################
# You may have multiple folders: 
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

# Label mapping
LABEL_MAPPING = {"Cooperation": 0, "Single": 1, "Competition": 2}

# We will use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################################
# 2. Preprocessing / Feature Extractor
############################################################
# We'll use a pretrained ResNet-50 for feature extraction:
# We'll remove the final FC layer and take the penultimate layer’s output 
# (the "pool" layer) as a 2048-dim feature.

resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()
resnet_model.to(DEVICE)

# Remove the final classification layer
# This leaves us with a feature extractor that outputs a 2048-dim vector
feature_extractor = torch.nn.Sequential(*list(resnet_model.children())[:-1])
feature_extractor.eval()
feature_extractor.to(DEVICE)

# Image transformations for ResNet (e.g., 224x224 center crop)
# You can adjust these as needed, but typically the 
# pretrained model was trained on 224x224 images.
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

############################################################
# 3. Utility Function: Concatenate 2 images vertically
############################################################
def concatenate_images_vertically(img1: Image.Image, img2: Image.Image) -> Image.Image:
    # Resize second image to match first if needed
    img2 = img2.resize((img1.width, img1.height))
    new_width = img1.width
    new_height = img1.height + img2.height
    new_img = Image.new("RGB", (new_width, new_height))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (0, img1.height))
    return new_img

############################################################
# 4. Dataset Class (optional, but helps structure)
############################################################
class MyImageDataset(Dataset):
    def __init__(self, examples, root):
        """
        examples: a list of dictionary samples. Each sample has:
            - "image1" (filename)
            - "image2" (filename)
            - "class"  (string label)
        root: folder path where the images are located
        """
        self.examples = examples
        self.root = root

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        record = self.examples[idx]
        path1 = os.path.join(self.root, record["image1"])
        path2 = os.path.join(self.root, record["image2"])
        label_str = record["class"]

        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")

        # Concatenate
        concatenated_img = concatenate_images_vertically(img1, img2)
        
        # Transform for ResNet
        transformed_img = image_transform(concatenated_img)
        
        # Convert label to int
        label = LABEL_MAPPING[label_str]
        return transformed_img, label

############################################################
# 5. Feature Extraction Function
############################################################
def extract_features_and_labels(dataset_obj):
    """
    Given a PyTorch Dataset (MyImageDataset),
    returns:
      X: numpy array of features  (num_samples, 2048)
      y: numpy array of labels    (num_samples,)
    """
    dataloader = DataLoader(dataset_obj, batch_size=8, shuffle=False)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images = images.to(DEVICE)
            
            # Forward pass through the feature extractor
            feats = feature_extractor(images)
            # feats shape: (batch_size, 2048, 1, 1)
            # Flatten
            feats = feats.squeeze()  # shape: (batch_size, 2048)
            if len(feats.shape) == 1:
                # If batch_size == 1, we'll have shape (2048,); add a batch dim
                feats = feats.unsqueeze(0)

            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0)  # shape: (N, 2048)
    y = np.concatenate(all_labels, axis=0)    # shape: (N,)
    return X, y

############################################################
# 6. Main Routine
############################################################
if __name__ == "__main__":
    # Load the JSON dataset with HF's "load_dataset" or directly via python
    hf_train_ds = load_dataset("json", data_files=DATASET_FILE_TRAIN, split="train")#[:5%]
    hf_test_ds  = load_dataset("json", data_files=DATASET_FILE_TEST, split="train")#[:5%]

    # Convert huggingface Dataset to a list of dict
    train_records = [hf_train_ds[i] for i in range(len(hf_train_ds))]
    test_records = [hf_test_ds[i] for i in range(len(hf_test_ds))]

    # We will loop through each ROOT in ROOT_LIST
    # If you only want to train/test once, pick one root.
    for root_dir in ROOT_LIST:
        print(f"============ Training with ROOT: {root_dir} ============")

        # Create Dataset objects
        train_dataset = MyImageDataset(train_records, root_dir)
        test_dataset = MyImageDataset(test_records, root_dir)

        # Extract features and labels
        X_train, y_train = extract_features_and_labels(train_dataset)
        X_test, y_test = extract_features_and_labels(test_dataset)

        # Optionally, standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create SVM classifier (you can tune hyperparameters)
        clf = SVC(
            kernel='rbf', 
            C=1.0, 
            gamma='scale', 
            probability=False,
            random_state=42
        )

        # Train SVM
        clf.fit(X_train, y_train)

        # Prediction
        y_pred = clf.predict(X_test)

        # Evaluate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Print results
        print(f"[{root_dir}] Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (weighted): {precision:.4f}")
        print(f"Recall (weighted): {recall:.4f}")
        print(f"F1-score (weighted): {f1:.4f}\n")

