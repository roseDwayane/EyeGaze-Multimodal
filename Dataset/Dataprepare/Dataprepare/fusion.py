from PIL import Image
from typing import List, Dict
import os


def concatenate_images_simple(image1_path: str, image2_path: str) -> Image.Image:
    """Concatenate two images vertically.
   
    Args:
        image1_path (str): Path to the first image (will be placed on top).
        image2_path (str): Path to the second image (will be placed on bottom).
       
    Returns:
        Image.Image: A new PIL Image with the two input images stacked vertically.
                     The second image is resized to match the width of the first image.
    """
    im1 = Image.open(image1_path).convert("RGB")
    im2 = Image.open(image2_path).convert("RGB").resize((im1.width, im1.height))
    canvas = Image.new("RGB", (im1.width, im1.height * 2))
    canvas.paste(im1, (0, 0))
    canvas.paste(im2, (0, im1.height))
    return canvas


def build_samples(root_dir: str, json_path: str, samples_per_category: int = 25) -> List[Dict]:
    """Build a dataset of image samples from JSON metadata with category limits.
   
    Args:
        root_dir (str): Root directory path where image files are located.
        json_path (str): Path to JSON file containing image metadata records.
        samples_per_category (int, optional): Maximum number of samples per category.
                                            Defaults to 25.
       
    Returns:
        List[Dict]: List of sample dictionaries, each containing:
                   - root_dir: Root directory path
                   - image1_path: Full path to first image
                   - image2_path: Full path to second image  
                   - image1_name: Basename of first image
                   - category: Image category classification
                   - text: Class text from JSON metadata
    """
    from IO import read_json, get_category
   
    records = read_json(json_path)
    counts = {"single": 0, "competition": 0, "cooperation": 0}
    out = []
   
    for r in records:
        img1_rel, img2_rel = r["image1"], r["image2"]
        img1 = os.path.join(root_dir, img1_rel)
        img2 = os.path.join(root_dir, img2_rel)
       
        if not (os.path.exists(img1) and os.path.exists(img2)):
            continue
           
        cat = get_category(img1_rel)
        if cat in counts and counts[cat] >= samples_per_category:
            continue
           
        out.append({
            "root_dir": root_dir,
            "image1_path": img1,
            "image2_path": img2,
            "image1_name": os.path.basename(img1_rel),
            "category": cat,
            "text": r.get("class", "")
        })
       
        if cat in counts:
            counts[cat] += 1
           
    return out