import os
import pandas as pd
import numpy as np
import random
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define absolute paths
BASE_PATH = "X:/dissertation/StyleSyncProject/data/processed/"
CROPPED_IMAGES_DIR = os.path.join(BASE_PATH, "cropped_images")
HEATMAPS_DIR = os.path.join(BASE_PATH, "heatmaps")
METADATA_FILE = os.path.join(BASE_PATH, "metadata_updated.csv")

def count_files_in_subfolders(directory):
    """Count total number of files inside all subfolders of a directory."""
    total_files = 0
    for root, _, files in os.walk(directory):
        total_files += len(files)
    return total_files

def validate_dataset_files(full_check=False):
    """Validate dataset files by comparing metadata with existing cropped images and heatmaps."""

    # Load metadata
    if not os.path.exists(METADATA_FILE):
        logger.error(f"Metadata file not found: {METADATA_FILE}")
        print(f"Metadata file not found: {METADATA_FILE}")
        return

    metadata = pd.read_csv(METADATA_FILE)
    total_images_in_metadata = len(metadata)

    # Count cropped images and heatmaps inside subfolders
    total_cropped_images = count_files_in_subfolders(CROPPED_IMAGES_DIR)
    total_heatmaps = count_files_in_subfolders(HEATMAPS_DIR)

    # Validation process
    logger.info(f"Metadata contains {total_images_in_metadata} entries.")
    logger.info(f"Total cropped images found: {total_cropped_images}")
    logger.info(f"Total heatmaps found: {total_heatmaps}")

    print(f"Metadata contains {total_images_in_metadata} entries.")
    print(f"Total cropped images found: {total_cropped_images}")
    print(f"Total heatmaps found: {total_heatmaps}")

    # Checking missing images and heatmaps
    missing_images = []
    missing_heatmaps = []

    for _, row in metadata.iterrows():
        image_name = row['image_name']
        img_path = os.path.join(CROPPED_IMAGES_DIR, image_name)
        heatmap_path = os.path.join(HEATMAPS_DIR, f"{os.path.splitext(image_name)[0]}.npy")

        if not os.path.exists(img_path):
            missing_images.append(image_name)
        
        if not os.path.exists(heatmap_path):
            missing_heatmaps.append(image_name)

    # Print results
    print("\nValidation Summary:")
    print(f"Missing cropped images: {len(missing_images)}")
    print(f"Missing heatmaps: {len(missing_heatmaps)}")

    logger.warning(f"Missing cropped images: {len(missing_images)}")
    logger.warning(f"Missing heatmaps: {len(missing_heatmaps)}")

    # Return validation statistics
    return {
        'total_metadata_entries': total_images_in_metadata,
        'total_cropped_images': total_cropped_images,
        'total_heatmaps': total_heatmaps,
        'missing_cropped_images': len(missing_images),
        'missing_heatmaps': len(missing_heatmaps),
        'is_valid': len(missing_images) == 0 and len(missing_heatmaps) == 0
    }

# Run validation
if __name__ == "__main__":
    validate_dataset_files(full_check=True)
