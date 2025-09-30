import os
import json
import pandas as pd
from typing import List, Dict
from PIL import Image, ImageEnhance


def get_json_files(json_dir: str) -> List[str]:
    """Get all JSON file paths from a directory.
   
    Args:
        json_dir (str): Path to the directory containing JSON files.
       
    Returns:
        List[str]: List of full paths to JSON files in the directory.
    """
    return [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]


def get_root_dirs(root_dir: str) -> List[str]:
    """Get all subdirectory paths from a root directory.
   
    Args:
        root_dir (str): Path to the root directory to search for subdirectories.
       
    Returns:
        List[str]: List of full paths to all subdirectories in the root directory.
    """
    paths = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
    return [p for p in paths if os.path.isdir(p)]


def get_category(filename: str) -> str:
    """Determine the category of a file based on keywords in its filename.
   
    Args:
        filename (str): The filename to analyze for category classification.
       
    Returns:
        str: The category of the file. Returns "single" if filename contains "single",
             "competition" if it contains "comp", "cooperation" if it contains "coop",
             or "unknown" if no matching keywords are found.
    """
    name = filename.lower()
    if "single" in name:
        return "single"
    if "comp" in name:
        return "competition"
    if "coop" in name:
        return "cooperation"
    return "unknown"


def read_json(json_path: str) -> List[Dict]:
    """Read and parse a JSON file containing a list of dictionaries.
   
    Args:
        json_path (str): Path to the JSON file to read.
       
    Returns:
        List[Dict]: The parsed JSON data as a list of dictionaries.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def enhance_image(img: Image.Image) -> Image.Image:
    """Apply image enhancement filters to improve visual quality.
   
    Args:
        img (Image.Image): PIL Image object to enhance.
       
    Returns:
        Image.Image: Enhanced PIL Image with increased contrast (1.2x) and
                     color saturation (1.1x).
    """
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Color(img).enhance(1.1)
    return img


def concatenate_images_simple(image1_path: str, image2_path: str) -> Image.Image:
    """Concatenate two images vertically with enhancement applied.
   
    Args:
        image1_path (str): Path to the first image (will be placed on top).
        image2_path (str): Path to the second image (will be placed on bottom).
       
    Returns:
        Image.Image: A new PIL Image with the two input images stacked vertically.
                     The second image is resized to match the width of the first image.
                     Both images are enhanced before concatenation.
    """
    im1 = Image.open(image1_path).convert("RGB")
    im2 = Image.open(image2_path).convert("RGB").resize((im1.width, im1.height))
    im1, im2 = enhance_image(im1), enhance_image(im2)
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


def save_csv(samples: List[Dict], csv_path: str) -> None:
    """Save a list of sample dictionaries to a CSV file.
   
    Args:
        samples (List[Dict]): List of dictionaries containing sample data to save.
        csv_path (str): Output path for the CSV file. Parent directories will be
                       created if they don't exist.
                       
    Returns:
        None
    """
    df = pd.DataFrame(samples)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)


def save_image(image: Image.Image, out_path: str) -> None:
    """Save a PIL Image to the specified file path.
   
    Args:
        image (Image.Image): PIL Image object to save.
        out_path (str): Output file path for saving the image. Parent directories
                       will be created if they don't exist.
                       
    Returns:
        None
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    image.save(out_path)
