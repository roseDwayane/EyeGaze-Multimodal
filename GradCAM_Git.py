############################################################
# Step 1. Import Libraries
############################################################
import os
import re
from tqdm import tqdm
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
# Step 2. Define Grad-CAM Class
############################################################
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # Register hooks to capture activations and gradients
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        # Get the target layer
        target_layer = dict([*self.model.named_modules()])[self.target_layer_name]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class):
        # Forward pass
        #output = self.model(pixel_values=input_tensor) 
        #loss = output[0, target_class]
        image_outputs = self.model.git.image_encoder(pixel_values=input_tensor)
        loss = image_outputs.last_hidden_state.mean()
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()

        # Compute Grad-CAM
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations[0].detach().cpu().numpy()[0]
        
        #print("Gradients shape:", type(gradients), gradients.shape)    
        #print("Activations shape:", type(activations), activations.shape)

        #weights = np.mean(gradients, axis=(1, 2))
        #cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)
        #cam = np.maximum(cam, 0)
        #cam = cam / cam.max()  # Normalize
        #return cam
        # Global average pooling over the feature dimension
        weights = np.mean(gradients, axis=0)  # Average over tokens

        # Weighted sum of activations
        cam = np.dot(activations, weights)

        # Normalize the CAM
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam / cam.max()  # Normalize

        # Reshape CAM to match the patch grid (e.g., 14x14 for 196 tokens, excluding class token)
        num_patches = int(cam.shape[0] ** 0.5)  # Assuming square grid
        cam = cam[:num_patches * num_patches].reshape(num_patches, num_patches)

        return cam

############################################################
# Step 3. Define Helper Functions
############################################################
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
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    image2 = image2.resize((image1.width, image1.height))
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

def overlay_heatmap(image, heatmap, title):
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(image.size)
    heatmap = np.array(heatmap)
    
    plt.imshow(image)
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.title(f"{title}")
    plt.show()

############################################################
# Step 4. Load Dataset and Model
############################################################
FUSION_LIST = [
    "Concate",
    "Subtract",
    "Dot_product"
]
ROOT = "./bgOn_heatmapOn_trajOn/"
DATASET_FILE = "./image_gt_train.json"

for fusion in FUSION_LIST:
    train_with_plv = False
    if fusion == "Concate" and train_with_plv == False:
        CHECKPOINT_PATH = os.path.join(ROOT, "git_checkpoint_Concate/checkpoint-590")
    elif fusion == "Subtract" and train_with_plv == False:
        CHECKPOINT_PATH = os.path.join(ROOT, "git_checkpoint_Subtract/checkpoint-590")
    elif fusion == "Dot_product" and train_with_plv == False:
        CHECKPOINT_PATH = os.path.join(ROOT, "git_checkpoint_Dot_product/checkpoint-590")
    elif fusion == "Concate" and train_with_plv == True:
        CHECKPOINT_PATH = os.path.join(ROOT, "git_checkpoint_Concate_plv/checkpoint-590")
    elif fusion == "Subtract" and train_with_plv == True:
        CHECKPOINT_PATH = os.path.join(ROOT, "git_checkpoint_Subtract_plv/checkpoint-590")
    elif fusion == "Dot_product" and train_with_plv == True:
        CHECKPOINT_PATH = os.path.join(ROOT, "git_checkpoint_Dot_product_plv/checkpoint-590")

    # Load dataset
    train_ds = load_dataset("json", data_files=DATASET_FILE, split="train")

    # Load processor and model
    processor = GitProcessor.from_pretrained("microsoft/git-base")
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_PATH)

    #for name, module in model.git.image_encoder.named_modules():
    #    print(name)

    ############################################################
    # Step 5. Generate Grad-CAM
    ############################################################
    for idx in tqdm(range(100)):#4480, len(train_ds)
        # Prepare input image
        # Best:
        ## Pair-13-B-Single-EYE_trial01_player.jpg, 
        ## Pair-12-Comp-EYE_trial06_playerA.jpg
        ## Pair-13-Coop-EYE_trial38_playerA.jpg
        # Worst:
        ## Pair-13-A-Single-EYE_trial11_player.jpg
        ## Pair-12-Comp-EYE_trial03_playerA.jpg
        ## Pair-12-Coop-EYE_trial16_playerA.jpg
        #image1_name = "Pair-12-Comp-EYE_trial06_playerA.jpg"
        #image2_name = "Pair-12-Comp-EYE_trial06_playerB.jpg"
        #image1_path = ROOT + image1_name
        #image2_path = ROOT + image2_name
        image1_path = ROOT + train_ds[idx]["image1"]
        image2_path = ROOT + train_ds[idx]["image2"]
        concatenated_image = concatenate_images(image1_path, image2_path, fusion, train_with_plv)
        concatenated_filename = f"./Grad_RAW/{train_ds[idx]['image1']}_{fusion}_{train_with_plv}.png"
        os.makedirs(os.path.dirname(concatenated_filename), exist_ok=True)
        concatenated_image.save(concatenated_filename)


        # Initialize Grad-CAM
        TARGET_LAYER_NAME = "git.image_encoder.vision_model.encoder.layers.0.self_attn"  # Example layer name
        grad_cam = GradCAM(model, TARGET_LAYER_NAME)

        # Generate Grad-CAM heatmap
        input_tensor = processor(images=[concatenated_image], return_tensors="pt").pixel_values
        TARGET_CLASS = 0  # Replace with predicted class index
        heatmap = grad_cam.generate_cam(input_tensor, TARGET_CLASS)

        ############################################################
        # Step 6. Visualize the Result
        ############################################################
        #overlay_heatmap(concatenated_image, heatmap, train_ds[idx]["class"])
        #plt.show()
        heatmap = np.uint8(255 * heatmap)
        heatmap = Image.fromarray(heatmap).resize(concatenated_image.size)
        heatmap = np.array(heatmap)
        
        #plt.imshow(concatenated_image)
        plt.imshow(heatmap, cmap="jet", alpha=0.5)
        plt.axis("off")
        plt.title(f"{train_ds[idx]['class']}")
        hm_filename = f"./Grad_CAM/{train_ds[idx]['image1']}_{fusion}_{train_with_plv}.png"
        #hm_filename = f"./result_worst/{image1_name}2.png"
        plt.savefig(hm_filename, dpi=150, bbox_inches="tight")
        plt.close()
        #print(f"saved to {hm_filename}\n")
    

