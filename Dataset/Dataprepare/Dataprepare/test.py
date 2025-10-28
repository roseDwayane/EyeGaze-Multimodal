import os
import tempfile
import json
from PIL import Image
from IO import get_json_files, get_category, read_json, save_csv, save_image
from fusion import concatenate_images_simple, build_samples


def test_get_category():
    """Test category extraction from filename"""
    print("Testing get_category...")
    assert get_category("single_image.jpg") == "single"
    assert get_category("comp_data.png") == "competition"
    assert get_category("coop_test.jpg") == "cooperation"
    assert get_category("random.jpg") == "unknown"
    print("✓ get_category tests passed!")


def test_concatenate_images():
    """Test image concatenation"""
    print("\nTesting concatenate_images_simple...")
   
    with tempfile.TemporaryDirectory() as tmpdir:
        img1_path = os.path.join(tmpdir, "test1.jpg")
        img2_path = os.path.join(tmpdir, "test2.jpg")
       
        img1 = Image.new("RGB", (100, 100), color="red")
        img2 = Image.new("RGB", (100, 100), color="blue")
        img1.save(img1_path)
        img2.save(img2_path)
       
        result = concatenate_images_simple(img1_path, img2_path)
        assert result.size == (100, 200)
        print("✓ concatenate_images_simple tests passed!")


def test_build_samples():
    """Test sample building"""
    print("\nTesting build_samples...")
   
    with tempfile.TemporaryDirectory() as tmpdir:
        img1 = Image.new("RGB", (50, 50), color="green")
        img2 = Image.new("RGB", (50, 50), color="yellow")
        img1.save(os.path.join(tmpdir, "single_01.jpg"))
        img2.save(os.path.join(tmpdir, "single_02.jpg"))
       
        json_path = os.path.join(tmpdir, "test.json")
        test_data = [
            {
                "image1": "single_01.jpg",
                "image2": "single_02.jpg",
                "class": "test_class"
            }
        ]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
       
        samples = build_samples(tmpdir, json_path, samples_per_category=10)
        assert len(samples) == 1
        assert samples[0]["category"] == "single"
        print("✓ build_samples tests passed!")


def test_save_and_load():
    """Test save and load functions"""
    print("\nTesting save and load functions...")
   
    with tempfile.TemporaryDirectory() as tmpdir:
        test_samples = [{"col1": "value1", "col2": "value2"}]
        csv_path = os.path.join(tmpdir, "test.csv")
        save_csv(test_samples, csv_path)
        assert os.path.exists(csv_path)
       
        test_img = Image.new("RGB", (10, 10), color="white")
        img_path = os.path.join(tmpdir, "test_img.jpg")
        save_image(test_img, img_path)
        assert os.path.exists(img_path)
       
        print("✓ save and load tests passed!")


if __name__ == "__main__":
    print("=" * 50)
    print("Running all tests...")
    print("=" * 50)
   
    test_get_category()
    test_concatenate_images()
    test_build_samples()
    test_save_and_load()
   
    print("\n" + "=" * 50)
    print("✓ All tests passed successfully!")
    print("=" * 50)
