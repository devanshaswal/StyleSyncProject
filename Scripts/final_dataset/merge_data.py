import os

import pandas as pd

# Define paths
data_dir = "dataset/organized_data/csv_files"  # Correct the path if needed
output_file = os.path.join(data_dir, "merged_data.csv")

# List the files in the data directory to verify the existence of the files
print("Files in the directory:", os.listdir(data_dir))

# Load datasets with error handling for missing files
try:
    attributes_images_df = pd.read_csv(os.path.join(data_dir, "attributes_images.csv"))
    bounding_boxes_df = pd.read_csv(os.path.join(data_dir, "bounding_boxes.csv"))
    category_images_df = pd.read_csv(os.path.join(data_dir, "category_images.csv"))
    parsed_landmarks_df = pd.read_csv(os.path.join(data_dir, "parsed_landmarks_cleaned.csv"))
    eval_partitions_df = pd.read_csv(os.path.join(data_dir, "eval_partitions.csv"))
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# Merge data on 'image_name'
merged_df = attributes_images_df.merge(bounding_boxes_df, on="image_name", how="left")
merged_df = merged_df.merge(category_images_df, on="image_name", how="left")
merged_df = merged_df.merge(parsed_landmarks_df, on="image_name", how="left")
merged_df = merged_df.merge(eval_partitions_df, on="image_name", how="left")

# Save merged dataset
merged_df.to_csv(output_file, index=False)

# Verify the merged dataset
print(f"Merged DataFrame shape: {merged_df.shape}")
print("Merged DataFrame columns:", merged_df.columns)
print(merged_df.head())
