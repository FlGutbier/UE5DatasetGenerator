import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, class_map=None):
        """
        Custom dataset for loading images with metadata from CSV.
        
        Args:
            csv_file: Path to CSV file with metadata
            root_dir: Root directory containing images
            transform: Optional transform to be applied on images
            class_map: Dictionary mapping custom classes to ImageNet indices
        """
        self.metadata = pd.read_csv(csv_file, delimiter=";")
        self.root_dir = root_dir
        self.transform = transform
        self.class_map = class_map or {}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.root_dir, row["Image"])
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # Get class and map to ImageNet if mapping provided
        class_label = int(row["Class"])
        imagenet_label = self.class_map.get(class_label, class_label)
        
        return {
            "image": image,
            "label": class_label,  # Original label
            "imagenet_label": imagenet_label,  # Mapped ImageNet label
            "idx": idx,
            # Store metadata as individual fields to avoid collation issues
            "image_path": row["Image"],
            "object": row["Object"],
            "level": row["World"],
            "material": row["Material"],
            "camera_position": row["Camera Position"],
            "light_color": row["Light Color (RGB)"],
            "fog": row["Fog"],
        }