import os
import json
import pandas as pd
from typing import List, Dict
from PIL import Image


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
