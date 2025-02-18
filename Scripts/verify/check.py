import os
import pandas as pd

# Load datasets
final_merged_path = "x:/dissertation/StyleSyncProject/dataset/organized_data/csv_files/final_merged_data.csv"
metadata_path = "x:/dissertation/StyleSyncProject/data/processed/metadata.csv"

df_final = pd.read_csv(final_merged_path)
df_metadata = pd.read_csv(metadata_path)

# Normalize paths in both datasets
df_final["image_name"] = df_final["image_name"].str.replace("img/", "").str.replace("/", os.sep)
df_metadata["image_name"] = df_metadata["image_name"].str.replace("\\", os.sep)  # Convert \ to / for consistency

# Find missing images
missing_images = df_final[~df_final["image_name"].isin(df_metadata["image_name"])]

# Save missing entries
missing_images_path = "x:/dissertation/StyleSyncProject/data/processed/missing_metadata_entries.csv"
missing_images.to_csv(missing_images_path, index=False)

print(f"✅ Found {len(missing_images)} missing images. Saved to {missing_images_path}.")
