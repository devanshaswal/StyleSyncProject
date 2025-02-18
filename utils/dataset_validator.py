import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import logging
from tqdm import tqdm
import concurrent.futures
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DatasetValidator')

def validate_image(img_path, expected_size=(224, 224)):
    """Validate a single image file"""
    try:
        if not os.path.exists(img_path):
            return False, f"Image not found: {img_path}"
        
        with Image.open(img_path) as img:
            img_size = img.size
            if img_size != expected_size:
                return False, f"Image {img_path} has unexpected size: {img_size}, expected: {expected_size}"
            
            # Check if image can be converted to RGB (not corrupted)
            try:
                img.convert('RGB')
            except Exception:
                return False, f"Image {img_path} is corrupted and cannot be converted to RGB"
                
        return True, None
    
    except Exception as e:
        return False, f"Error validating image {img_path}: {str(e)}"

def validate_heatmap(heatmap_path, expected_shape=(224, 224)):
    """Validate a single heatmap file"""
    try:
        if not os.path.exists(heatmap_path):
            return False, f"Heatmap not found: {heatmap_path}"
        
        try:
            heatmap = np.load(heatmap_path)
        except Exception as e:
            return False, f"Error loading heatmap {heatmap_path}: {str(e)}"
        
        if heatmap.shape != expected_shape:
            return False, f"Heatmap {heatmap_path} has unexpected shape: {heatmap.shape}, expected: {expected_shape}"
        
        # Check if heatmap values are in expected range (typically 0-1 for heatmaps)
        if np.min(heatmap) < 0 or np.max(heatmap) > 1:
            return False, f"Heatmap {heatmap_path} has values outside expected range 0-1: min={np.min(heatmap)}, max={np.max(heatmap)}"
        
        return True, None
    
    except Exception as e:
        return False, f"Error validating heatmap {heatmap_path}: {str(e)}"

def find_orphaned_files(metadata_df, cropped_images_dir, heatmaps_dir):
    """Find files in the directories that are not referenced in the metadata"""
    expected_images = set(metadata_df['image_name'])
    expected_heatmaps = set([os.path.splitext(name)[0] + '.npy' for name in metadata_df['image_name']])
    
    # Find all actual images
    actual_images = set()
    for root, dirs, files in os.walk(cropped_images_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                actual_images.add(file)
    
    # Find all actual heatmaps
    actual_heatmaps = set()
    for root, dirs, files in os.walk(heatmaps_dir):
        for file in files:
            if file.lower().endswith('.npy'):
                actual_heatmaps.add(file)
    
    # Find orphaned files
    orphaned_images = actual_images - expected_images
    orphaned_heatmaps = actual_heatmaps - expected_heatmaps
    
    return orphaned_images, orphaned_heatmaps

def validate_dataset(metadata_path, cropped_images_dir, heatmaps_dir, num_workers=4, sample_ratio=1.0):
    """Validate entire dataset"""
    logger.info(f"Starting dataset validation with {num_workers} workers")
    
    # Load metadata
    try:
        metadata_df = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata with {len(metadata_df)} entries")
    except Exception as e:
        logger.error(f"Failed to load metadata: {str(e)}")
        return False
    
    # Sample the dataset if requested
    if sample_ratio < 1.0:
        sample_size = int(len(metadata_df) * sample_ratio)
        metadata_df = metadata_df.sample(sample_size, random_state=42)
        logger.info(f"Sampled {sample_size} entries for validation")
    
    # Find orphaned files
    logger.info("Checking for orphaned files...")
    orphaned_images, orphaned_heatmaps = find_orphaned_files(metadata_df, cropped_images_dir, heatmaps_dir)
    
    if orphaned_images:
        logger.warning(f"Found {len(orphaned_images)} orphaned images not referenced in metadata")
        with open('orphaned_images.txt', 'w') as f:
            for img in orphaned_images:
                f.write(f"{img}\n")
    
    if orphaned_heatmaps:
        logger.warning(f"Found {len(orphaned_heatmaps)} orphaned heatmaps not referenced in metadata")
        with open('orphaned_heatmaps.txt', 'w') as f:
            for hm in orphaned_heatmaps:
                f.write(f"{hm}\n")
    
    # Prepare validation tasks
    image_tasks = []
    heatmap_tasks = []
    
    for _, row in metadata_df.iterrows():
        img_path = os.path.join(cropped_images_dir, row['image_name'])
        heatmap_path = os.path.join(heatmaps_dir, f"{os.path.splitext(row['image_name'])[0]}.npy")
        
        image_tasks.append(img_path)
        heatmap_tasks.append(heatmap_path)
    
    # Validate images
    logger.info(f"Validating {len(image_tasks)} images...")
    invalid_images = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(validate_image, image_tasks), total=len(image_tasks), desc="Validating images"))
        
        for img_path, (valid, error_msg) in zip(image_tasks, results):
            if not valid:
                invalid_images.append((img_path, error_msg))
    
    # Validate heatmaps
    logger.info(f"Validating {len(heatmap_tasks)} heatmaps...")
    invalid_heatmaps = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(validate_heatmap, heatmap_tasks), total=len(heatmap_tasks), desc="Validating heatmaps"))
        
        for heatmap_path, (valid, error_msg) in zip(heatmap_tasks, results):
            if not valid:
                invalid_heatmaps.append((heatmap_path, error_msg))
    
    # Log and save results
    if invalid_images:
        logger.warning(f"Found {len(invalid_images)} invalid images")
        with open('invalid_images.txt', 'w') as f:
            for img_path, error_msg in invalid_images:
                f.write(f"{img_path}: {error_msg}\n")
    else:
        logger.info("All images are valid")
    
    if invalid_heatmaps:
        logger.warning(f"Found {len(invalid_heatmaps)} invalid heatmaps")
        with open('invalid_heatmaps.txt', 'w') as f:
            for heatmap_path, error_msg in invalid_heatmaps:
                f.write(f"{heatmap_path}: {error_msg}\n")
    else:
        logger.info("All heatmaps are valid")
    
    # Calculate validation summary
    total_images = len(image_tasks)
    total_heatmaps = len(heatmap_tasks)
    valid_images = total_images - len(invalid_images)
    valid_heatmaps = total_heatmaps - len(invalid_heatmaps)
    
    logger.info(f"Validation Summary:")
    logger.info(f"  Images: {valid_images}/{total_images} valid ({100 * valid_images / total_images:.2f}%)")
    logger.info(f"  Heatmaps: {valid_heatmaps}/{total_heatmaps} valid ({100 * valid_heatmaps / total_heatmaps:.2f}%)")
    logger.info(f"  Orphaned Images: {len(orphaned_images)}")
    logger.info(f"  Orphaned Heatmaps: {len(orphaned_heatmaps)}")
    
    return valid_images == total_images and valid_heatmaps == total_heatmaps

def parse_args():
    parser = argparse.ArgumentParser(description="Validate fashion dataset")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata CSV file")
    parser.add_argument("--images-dir", type=str, required=True, help="Path to cropped images directory")
    parser.add_argument("--heatmaps-dir", type=str, required=True, help="Path to heatmaps directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--sample", type=float, default=1.0, help="Fraction of dataset to validate (0.0-1.0)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory for validation results
    os.makedirs('validation_results', exist_ok=True)
    
    # Run validation
    is_valid = validate_dataset(
        args.metadata,
        args.images_dir, 
        args.heatmaps_dir,
        num_workers=args.workers,
        sample_ratio=args.sample
    )
    
    if is_valid:
        logger.info("Dataset validation passed successfully!")
        exit(0)
    else:
        logger.warning("Dataset validation found issues. Check the logs and output files for details.")
        exit(1)