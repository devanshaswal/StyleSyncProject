import pandas as pd
import os

# Load the existing metadata.csv
metadata_path = "data/processed/metadata.csv"
output_path = "data/processed/metadata_heatmaps.csv"

# Read the CSV file
metadata = pd.read_csv(metadata_path)

# Function to generate heatmap path
def generate_heatmap_path(cropped_path):
    heatmap_path = cropped_path.replace("cropped_images", "heatmaps").replace(".jpg", ".npy")
    return heatmap_path

# Create a new column for heatmaps_path
metadata["heatmaps_path"] = metadata["cropped_image_path"].apply(generate_heatmap_path)

# Save the updated metadata to a new CSV file
metadata.to_csv(output_path, index=False)

print(f"Updated metadata saved to {output_path}")
