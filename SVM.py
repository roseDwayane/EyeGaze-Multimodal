import os
import torch
import json
import numpy as np
import pandas as pd

from PIL import Image
from datasets import load_dataset
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

############################################################
# 1. Configurations
############################################################
ROOT_LIST = [
    "bgOn_heatmapOn_trajOn",
    "bgOn_heatmapOn_trajOff",
    "bgOn_heatmapOff_trajOn",
    "bgOff_heatmapOn_trajOn",
    "bgOff_heatmapOn_trajOff",
    "bgOff_heatmapOff_trajOn"
]

DATASET_FILE = "./image_gt.json"
LABEL_MAPPING = {"Cooperation": 0, "Single": 1, "Competition": 2}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################################
# 2. Preprocessing / Feature Extractor
############################################################
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()
resnet_model.to(DEVICE)
feature_extractor = torch.nn.Sequential(*list(resnet_model.children())[:-1])
feature_extractor.eval()
feature_extractor.to(DEVICE)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

############################################################
# 3. Utility Function: Concatenate 2 images vertically
############################################################
def concatenate_images_vertically(img1: Image.Image, img2: Image.Image) -> Image.Image:
    img2 = img2.resize((img1.width, img1.height))
    new_width = img1.width
    new_height = img1.height + img2.height
    new_img = Image.new("RGB", (new_width, new_height))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (0, img1.height))
    return new_img

############################################################
# 4. Dataset Class
############################################################
class MyImageDataset(Dataset):
    def __init__(self, examples, root):
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

        concatenated_img = concatenate_images_vertically(img1, img2)
        transformed_img = image_transform(concatenated_img)
        label = LABEL_MAPPING[label_str]
        return transformed_img, label

############################################################
# 5. Feature Extraction Function
############################################################
def extract_features_and_labels(dataset_obj):
    dataloader = DataLoader(dataset_obj, batch_size=8, shuffle=False)
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images = images.to(DEVICE)
            feats = feature_extractor(images).squeeze()
            if len(feats.shape) == 1:
                feats = feats.unsqueeze(0)
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y

############################################################
# 6. Main Routine
############################################################
if __name__ == "__main__":
    hf_dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
    records = [hf_dataset[i] for i in range(len(hf_dataset))]

    for root_dir in ROOT_LIST:
        results = []
        print(f"============ Processing ROOT: {root_dir} ============")
        dataset = MyImageDataset(records, root_dir)
        X, y = extract_features_and_labels(dataset)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        fold_idx = 1

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            results.append({
                "Root": root_dir,
                "Fold": fold_idx,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            })

            fold_idx += 1

        # Save results to CSV
        csv_file = os.path.join(root_dir, "SVM_metrics_results.csv")
        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_file, index=False)

        print(f"Results saved to {csv_file}")

