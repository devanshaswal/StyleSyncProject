# import hashlib
# import logging
# import os
# import sys
# import pandas as pd
# from tqdm import tqdm

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# from configs.paths import PROCESSED_DATA_DIR, FINAL_MERGED_CSV

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# def normalize_path(path):
#     """Ensure consistent image path formatting."""
#     return os.path.normpath(path.lstrip("img/"))

# def get_image_hash(image_path):
#     """Generate a hash for an image to avoid duplicates."""
#     try:
#         with open(image_path, "rb") as f:
#             return hashlib.md5(f.read()).hexdigest()
#     except Exception:
#         return None

# def generate_metadata(output_csv, cropped_images_dir):
#     """
#     Generate metadata for already existing cropped images.
#     """
#     logger.info("Scanning cropped images directory...")

#     metadata = []
    
#     for root, _, files in os.walk(cropped_images_dir):
#         for file in files:
#             if file.endswith(".jpg"):
#                 img_relative_path = os.path.relpath(os.path.join(root, file), cropped_images_dir)
#                 img_full_path = os.path.join(cropped_images_dir, img_relative_path)

#                 metadata.append([img_relative_path, (224, 224), img_full_path])

#     if not metadata:
#         logger.error("No cropped images found! Check the directory path.")
#         return

#     metadata_df = pd.DataFrame(metadata, columns=["image_name", "image_size", "cropped_image_path"])
#     metadata_df.to_csv(output_csv, index=False)
#     logger.info(f"Metadata successfully saved to {output_csv}")

# if __name__ == "__main__":
#     cropped_images_dir = os.path.join(PROCESSED_DATA_DIR, "cropped_images")
#     metadata_csv_path = os.path.join(PROCESSED_DATA_DIR, "metadata.csv")

#     if os.path.exists(cropped_images_dir):
#         generate_metadata(metadata_csv_path, cropped_images_dir)
#     else:
#         logger.error(f"Cropped images directory not found: {cropped_images_dir}")

# import hashlib
# import logging
# import os
# import sys
# import pandas as pd
# from tqdm import tqdm

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# from configs.paths import PROCESSED_DATA_DIR, FINAL_MERGED_CSV

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# def normalize_path(path):
#     """
#     Ensure consistent image path formatting between metadata and final_merged_data.
#     Case-insensitive path normalization.
#     """
#     # Remove 'img/' prefix if present
#     if path.lower().startswith("img/"):
#         path = path[4:]
#     # Convert path to use forward slashes and lowercase for comparison
#     return os.path.normpath(path).replace("\\", "/").lower()

# def generate_metadata(output_csv, cropped_images_dir, merged_df):
#     """
#     Generate metadata for existing cropped images while maintaining consistency
#     with final_merged_data.csv paths using case-insensitive matching.
#     """
#     logger.info("Scanning cropped images directory...")
    
#     metadata = []
#     processed_images = set()

#     # Create a case-insensitive mapping of normalized paths to original paths from merged_df
#     path_mapping = {normalize_path(path): path 
#                    for path in merged_df['image_name'].unique()}

#     # First, get all cropped image paths
#     cropped_images = []
#     for root, _, files in os.walk(cropped_images_dir):
#         for file in files:
#             if file.endswith(".jpg"):
#                 full_path = os.path.join(root, file)
#                 rel_path = os.path.relpath(full_path, cropped_images_dir).replace("\\", "/")
#                 cropped_images.append((full_path, rel_path))

#     # Process images with progress bar
#     for full_path, rel_path in tqdm(cropped_images, desc="Processing images"):
#         normalized_path = normalize_path(rel_path)
#         original_path = path_mapping.get(normalized_path)
                
#         if original_path:
#             metadata.append([original_path, (224, 224), full_path])
#             processed_images.add(normalized_path)
#         else:
#             logger.warning(f"No matching entry in merged_df for: {rel_path}")

#     if not metadata:
#         logger.error("No cropped images found! Check the directory path.")
#         return

#     metadata_df = pd.DataFrame(metadata, columns=["image_name", "image_size", "cropped_image_path"])
    
#     # Check for missing images
#     merged_paths = {normalize_path(path): path for path in merged_df['image_name']}
#     missing_images = set(merged_paths.keys()) - processed_images
    
#     if missing_images:
#         logger.warning(f"Found {len(missing_images)} images in merged_df that are missing from cropped images:")
#         for norm_path in missing_images:
#             logger.warning(f"Missing image: {merged_paths[norm_path]}")

#     # Save metadata
#     metadata_df.to_csv(output_csv, index=False)
#     logger.info(f"Metadata successfully saved to {output_csv} with {len(metadata_df)} entries")
    
#     # Print summary
#     logger.info("\nSummary:")
#     logger.info(f"Total images in merged_df: {len(merged_df['image_name'].unique())}")
#     logger.info(f"Total cropped images found: {len(cropped_images)}")
#     logger.info(f"Total matched images: {len(metadata_df)}")
#     logger.info(f"Total missing images: {len(missing_images)}")
    
#     return metadata_df

# if __name__ == "__main__":
#     cropped_images_dir = os.path.join(PROCESSED_DATA_DIR, "cropped_images")
#     metadata_csv_path = os.path.join(PROCESSED_DATA_DIR, "metadata.csv")
    
#     if not os.path.exists(FINAL_MERGED_CSV):
#         logger.error(f"Final merged CSV not found: {FINAL_MERGED_CSV}")
#         sys.exit(1)
        
#     if not os.path.exists(cropped_images_dir):
#         logger.error(f"Cropped images directory not found: {cropped_images_dir}")
#         sys.exit(1)
    
#     # Load the merged data for reference
#     merged_df = pd.read_csv(FINAL_MERGED_CSV)
#     logger.info(f"Loaded merged data with {len(merged_df)} rows")
    
#     # Generate metadata
#     metadata_df = generate_metadata(metadata_csv_path, cropped_images_dir, merged_df)




import hashlib
import logging
import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from configs.paths import PROCESSED_DATA_DIR, FINAL_MERGED_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def normalize_path(path):
    """
    Ensure consistent image path formatting between metadata and final_merged_data.
    Case-insensitive path normalization.
    """
    # Remove 'img/' prefix if present
    if path.lower().startswith("img/"):
        path = path[4:]
    # Convert path to use forward slashes and lowercase for comparison
    return os.path.normpath(path).replace("\\", "/").lower()

def generate_metadata(output_csv, cropped_images_dir, merged_df):
    """
    Generate metadata for existing cropped images while maintaining consistency
    with final_merged_data.csv paths using case-insensitive matching.
    """
    logger.info("Scanning cropped images directory...")
    
    metadata = []
    processed_images = set()

    # Create a case-insensitive mapping of normalized paths to original paths from merged_df
    path_mapping = {normalize_path(path): path.replace("img/", "")  # Remove 'img/' prefix
                   for path in merged_df['image_name'].unique()}

    # First, get all cropped image paths
    cropped_images = []
    for root, _, files in os.walk(cropped_images_dir):
        for file in files:
            if file.endswith(".jpg"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, cropped_images_dir).replace("\\", "/")
                cropped_images.append((full_path, rel_path))

    # Process images with progress bar
    for full_path, rel_path in tqdm(cropped_images, desc="Processing images"):
        normalized_path = normalize_path(rel_path)
        original_path = path_mapping.get(normalized_path)
                
        if original_path:
            metadata.append([original_path, (224, 224), full_path])
            processed_images.add(normalized_path)
        else:
            logger.warning(f"No matching entry in merged_df for: {rel_path}")

    if not metadata:
        logger.error("No cropped images found! Check the directory path.")
        return

    metadata_df = pd.DataFrame(metadata, columns=["image_name", "image_size", "cropped_image_path"])
    
    # Check for missing images
    merged_paths = {normalize_path(path): path.replace("img/", "") for path in merged_df['image_name']}
    missing_images = set(merged_paths.keys()) - processed_images
    
    if missing_images:
        logger.warning(f"Found {len(missing_images)} images in merged_df that are missing from cropped images:")
        for norm_path in missing_images:
            logger.warning(f"Missing image: {merged_paths[norm_path]}")

    # Save metadata
    metadata_df.to_csv(output_csv, index=False)
    logger.info(f"Metadata successfully saved to {output_csv} with {len(metadata_df)} entries")
    
    # Print summary
    logger.info("\nSummary:")
    logger.info(f"Total images in merged_df: {len(merged_df['image_name'].unique())}")
    logger.info(f"Total cropped images found: {len(cropped_images)}")
    logger.info(f"Total matched images: {len(metadata_df)}")
    logger.info(f"Total missing images: {len(missing_images)}")
    
    return metadata_df

if __name__ == "__main__":
    cropped_images_dir = os.path.join(PROCESSED_DATA_DIR, "cropped_images")
    metadata_csv_path = os.path.join(PROCESSED_DATA_DIR, "metadata.csv")
    
    if not os.path.exists(FINAL_MERGED_CSV):
        logger.error(f"Final merged CSV not found: {FINAL_MERGED_CSV}")
        sys.exit(1)
        
    if not os.path.exists(cropped_images_dir):
        logger.error(f"Cropped images directory not found: {cropped_images_dir}")
        sys.exit(1)
    
    # Load the merged data for reference
    merged_df = pd.read_csv(FINAL_MERGED_CSV)
    logger.info(f"Loaded merged data with {len(merged_df)} rows")
    
    # Generate metadata
    metadata_df = generate_metadata(metadata_csv_path, cropped_images_dir, merged_df)