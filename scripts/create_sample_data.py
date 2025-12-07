"""
Create sample dataset for testing the crop disease detection pipeline.
This script generates synthetic images organized by disease classes.
"""
import cv2
import numpy as np
from pathlib import Path
import random

# Sample disease classes (subset of PlantVillage)
SAMPLE_CLASSES = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___Healthy',
    'Corn_(maize)___Common_rust',
    'Corn_(maize)___Healthy'
]

def create_synthetic_leaf_image(size=(256, 256), disease_type='healthy'):
    """Create a synthetic leaf image with basic patterns."""
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # Base green color for leaf
    if 'healthy' in disease_type.lower():
        # Healthy leaf - vibrant green
        base_color = np.array([50, 180, 50], dtype=np.uint8)
    else:
        # Diseased leaf - yellowish/brownish
        base_color = np.array([30, 140, 70], dtype=np.uint8)
    
    # Add gradient
    for i in range(size[0]):
        for j in range(size[1]):
            variation = np.random.randint(-20, 20, 3)
            img[i, j] = np.clip(base_color + variation, 0, 255)
    
    # Add leaf-like ellipse shape
    center = (size[1]//2, size[0]//2)
    axes = (size[1]//3, size[0]//2)
    cv2.ellipse(img, center, axes, 0, 0, 360, (60, 200, 60), -1)
    
    # Add disease patterns
    if 'spot' in disease_type.lower() or 'blight' in disease_type.lower():
        # Add spots/lesions
        num_spots = random.randint(5, 15)
        for _ in range(num_spots):
            x = random.randint(size[1]//4, 3*size[1]//4)
            y = random.randint(size[0]//4, 3*size[0]//4)
            radius = random.randint(5, 20)
            color = (random.randint(80, 120), random.randint(60, 100), 30)
            cv2.circle(img, (x, y), radius, color, -1)
    
    elif 'rust' in disease_type.lower():
        # Add rust-like patterns
        num_rust = random.randint(20, 40)
        for _ in range(num_rust):
            x = random.randint(0, size[1]-1)
            y = random.randint(0, size[0]-1)
            radius = random.randint(2, 5)
            cv2.circle(img, (x, y), radius, (30, 60, 150), -1)
    
    # Add leaf veins
    vein_color = (40, 160, 40) if 'healthy' in disease_type.lower() else (40, 120, 40)
    for i in range(3):
        start_x = size[1]//2
        start_y = size[0]//2
        end_x = random.randint(size[1]//4, 3*size[1]//4)
        end_y = random.randint(0, size[0]-1)
        cv2.line(img, (start_x, start_y), (end_x, end_y), vein_color, 2)
    
    # Add slight blur for realism
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    return img


def create_sample_dataset(output_dir='data/raw', samples_per_class=20):
    """Create a complete sample dataset with synthetic images."""
    output_path = Path(output_dir)
    
    print(f"Creating sample dataset in {output_dir}")
    print(f"Classes: {len(SAMPLE_CLASSES)}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Total images: {len(SAMPLE_CLASSES) * samples_per_class}")
    print()
    
    for class_name in SAMPLE_CLASSES:
        class_dir = output_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating {samples_per_class} images for {class_name}...")
        
        for i in range(samples_per_class):
            # Create synthetic image
            img = create_synthetic_leaf_image(
                size=(256, 256),
                disease_type=class_name
            )
            
            # Save image
            img_path = class_dir / f'{class_name}_{i:03d}.jpg'
            cv2.imwrite(str(img_path), img)
        
        print(f"  ✓ Created {samples_per_class} images in {class_dir}")
    
    print()
    print("✓ Sample dataset created successfully!")
    print(f"  Location: {output_path.absolute()}")
    print(f"  Total images: {len(SAMPLE_CLASSES) * samples_per_class}")
    print()
    print("You can now run:")
    print(f"  python train.py --data_dir {output_dir} --epochs 5")


if __name__ == '__main__':
    create_sample_dataset(
        output_dir='data/raw',
        samples_per_class=30  # 30 images per class = 300 total
    )
