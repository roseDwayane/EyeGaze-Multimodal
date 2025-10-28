import json
import random

# Load existing ground truth files
with open('image_gt_train.json', 'r') as f:
    train_gt = json.load(f)

with open('image_gt_test.json', 'r') as f:
    test_gt = json.load(f)

def generate_gaze_json(gt_data, output_file):
    """
    Generate JSON file with gaze-guide structure

    Args:
        gt_data: List of dicts with image1, image2, class
        output_file: Output JSON filename
    """
    gaze_data = []

    for item in gt_data:
        # Extract image names
        image1 = item['image1']
        image2 = item['image2']
        class_label = item['class']

        # Create gaze entry
        gaze_entry = {
            'bg_image1': f'bgOn_heatmapOff_trajOn/{image1}',
            'bg_image2': f'bgOn_heatmapOff_trajOn/{image2}',
            'heatmap1': f'bgOff_heatmapOn_trajOff/{image1}',
            'heatmap2': f'bgOff_heatmapOn_trajOff/{image2}',
            'class': class_label,
            'eeg_scalar': round(random.uniform(0.3, 0.9), 2)  # Random EEG scalar
        }

        gaze_data.append(gaze_entry)

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(gaze_data, f, ensure_ascii=False, indent=2)

    print(f"✓ Generated {output_file} with {len(gaze_data)} entries")

# Generate training JSON
generate_gaze_json(train_gt, 'train_gaze.json')

# Generate validation JSON
generate_gaze_json(test_gt, 'val_gaze.json')

print("\nSample entry from train_gaze.json:")
with open('train_gaze.json', 'r') as f:
    sample = json.load(f)[0]
    print(json.dumps(sample, indent=2))
