import pandas as pd
import os
import glob

# Load metadata.csv
metadata_path = "x:/dissertation/StyleSyncProject/data/processed/metadata.csv"
df = pd.read_csv(metadata_path)

heatmaps_root = "x:/dissertation/StyleSyncProject/data/processed/heatmaps"

# Function to find the correct heatmap path
def get_heatmap_path(cropped_image_path):
    image_filename = os.path.basename(cropped_image_path).replace(".jpg", ".npy")
    
    # Search recursively for the heatmap file
    heatmap_files = glob.glob(os.path.join(heatmaps_root, "**", image_filename), recursive=True)
    
    if heatmap_files:
        return heatmap_files[0]  # Return the first match
    else:
        return "MISSING"  # Mark missing heatmaps

# Add the correct heatmap path
df["heatmap_path"] = df["cropped_image_path"].apply(get_heatmap_path)

# Save new metadatafile.csv
new_metadata_path = "x:/dissertation/StyleSyncProject/data/processed/metadatafile.csv"
df.to_csv(new_metadata_path, index=False)

print(f"✅ New metadata file saved as {new_metadata_path}")
