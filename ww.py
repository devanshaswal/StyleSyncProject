import numpy as np
import os

heatmaps_root_dir = "dataset/heatmaps/"  # Update this path

# Loop through all subfolders inside `heatmaps_root_dir`
for subfolder in os.listdir(heatmaps_root_dir):
    subfolder_path = os.path.join(heatmaps_root_dir, subfolder)
    
    # Ensure it's a directory (skip files if any exist)
    if not os.path.isdir(subfolder_path):
        continue

    print(f"\nChecking heatmaps in: {subfolder}")

    heatmap_files = os.listdir(subfolder_path)
    
    # Check the first 5 heatmaps from each subfolder
    for heatmap_file in heatmap_files[:5]:
        heatmap_path = os.path.join(subfolder_path, heatmap_file)
        
        try:
            heatmap = np.load(heatmap_path)

            print(f"  File: {heatmap_file} | Min: {heatmap.min()}, Max: {heatmap.max()}, Unique values: {np.unique(heatmap)[:10]}")
        except Exception as e:
            print(f"  Error reading {heatmap_file}: {str(e)}")

print("\nHeatmap value check completed!")
