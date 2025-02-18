


import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CSV_PATH = os.path.join(PROJECT_ROOT, "dataset", "organized_data", "csv_files", "final_merged_data.csv")
DEFAULT_CROPPED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "cropped_images")
DEFAULT_HEATMAP_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "heatmaps")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "verification_reports")

def normalize_path(path):
    """Normalize image path to match directory structure."""
    if path.startswith("img/"):
        path = path[len("img/"):]
    return os.path.normpath(path)

def verify_cropped_images(df, cropped_dir):
    """Verify that all cropped images exist and have correct dimensions."""
    logger.info("Verifying cropped images...")
    missing_images = []
    incorrect_dimensions = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking cropped images"):
        img_path = normalize_path(row["image_name"])
        cropped_path = os.path.join(cropped_dir, img_path)
        
        if not os.path.exists(cropped_path):
            missing_images.append(img_path)
            continue
            
        try:
            with Image.open(cropped_path) as img:
                width, height = img.size
                if width != 224 or height != 224:  # Check if dimensions match expected size
                    incorrect_dimensions.append((img_path, (width, height)))
        except Exception as e:
            logger.error(f"Error reading image {cropped_path}: {e}")
            missing_images.append(img_path)
    
    return missing_images, incorrect_dimensions

def verify_heatmaps(df, heatmap_dir):
    """Verify that all heatmaps exist and have correct format."""
    logger.info("Verifying heatmaps...")
    missing_heatmaps = []
    incorrect_format = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking heatmaps"):
        img_path = normalize_path(row["image_name"])
        base_name = os.path.splitext(img_path)[0]
        heatmap_path = os.path.join(heatmap_dir, f"{base_name}.npy")
        
        if not os.path.exists(heatmap_path):
            missing_heatmaps.append(img_path)
            continue
            
        try:
            heatmap = np.load(heatmap_path)
            
            # Verify heatmap dimensions
            if heatmap.shape != (224, 224):
                incorrect_format.append((img_path, heatmap.shape))
                continue
                
            # Verify heatmap values are normalized between 0 and 1
            if heatmap.max() > 1.0 or heatmap.min() < 0.0:
                incorrect_format.append((img_path, "Values out of range [0,1]"))
                
        except Exception as e:
            logger.error(f"Error reading heatmap {heatmap_path}: {e}")
            incorrect_format.append((img_path, str(e)))
    
    return missing_heatmaps, incorrect_format

def generate_validation_report(df_path, cropped_dir, heatmap_dir, output_dir):
    """Generate a comprehensive validation report."""
    logger.info(f"Starting validation process...")
    
    if not os.path.exists(df_path):
        logger.error(f"Merged CSV file not found: {df_path}")
        return
        
    df = pd.read_csv(df_path)
    logger.info(f"Loaded {len(df)} entries from CSV")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Verify cropped images
    missing_images, incorrect_dimensions = verify_cropped_images(df, cropped_dir)
    
    # Verify heatmaps
    missing_heatmaps, incorrect_format = verify_heatmaps(df, heatmap_dir)
    
    # Generate report
    report = {
        "total_entries": len(df),
        "missing_images": len(missing_images),
        "incorrect_dimensions": len(incorrect_dimensions),
        "missing_heatmaps": len(missing_heatmaps),
        "incorrect_heatmaps": len(incorrect_format)
    }
    
    # Save detailed reports
    if missing_images:
        pd.Series(missing_images).to_csv(os.path.join(output_dir, "missing_images.csv"), index=False)
    if incorrect_dimensions:
        pd.DataFrame(incorrect_dimensions, columns=["image_path", "dimensions"]).to_csv(
            os.path.join(output_dir, "incorrect_dimensions.csv"), index=False)
    if missing_heatmaps:
        pd.Series(missing_heatmaps).to_csv(os.path.join(output_dir, "missing_heatmaps.csv"), index=False)
    if incorrect_format:
        pd.DataFrame(incorrect_format, columns=["image_path", "error"]).to_csv(
            os.path.join(output_dir, "incorrect_heatmaps.csv"), index=False)
    
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify processed images and heatmaps.')
    parser.add_argument('--csv', default=DEFAULT_CSV_PATH, help='Path to the merged CSV file')
    parser.add_argument('--cropped-dir', default=DEFAULT_CROPPED_DIR, help='Directory containing cropped images')
    parser.add_argument('--heatmap-dir', default=DEFAULT_HEATMAP_DIR, help='Directory containing heatmaps')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Directory to save verification reports')
    
    args = parser.parse_args()
    
    # Log the paths being used
    logger.info(f"Using CSV file: {args.csv}")
    logger.info(f"Using cropped images directory: {args.cropped_dir}")
    logger.info(f"Using heatmaps directory: {args.heatmap_dir}")
    logger.info(f"Using output directory: {args.output_dir}")
    
    report = generate_validation_report(args.csv, args.cropped_dir, args.heatmap_dir, args.output_dir)
    
    if report:
        logger.info("\nValidation Report:")
        logger.info("-----------------")
        logger.info(f"Total entries in CSV: {report['total_entries']}")
        logger.info(f"Missing cropped images: {report['missing_images']}")
        logger.info(f"Images with incorrect dimensions: {report['incorrect_dimensions']}")
        logger.info(f"Missing heatmaps: {report['missing_heatmaps']}")
        logger.info(f"Incorrect heatmap format: {report['incorrect_heatmaps']}")
        
        if any(value > 0 for value in report.values()):
            logger.warning(f"\nDetailed error reports have been saved to {args.output_dir}")
        else:
            logger.info("\nAll validations passed successfully!")

