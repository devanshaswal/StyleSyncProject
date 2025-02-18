


# found the error





import pandas as pd
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
from configs.paths import FINAL_MERGED_CSV

# Load data and check for case-sensitive duplicates
merged_df = pd.read_csv(FINAL_MERGED_CSV)
normalized_paths = merged_df['image_name'].str.lower().str.replace('img/', '', regex=False).str.replace('\\', '/')

# Find duplicates
duplicate_mask = normalized_paths.duplicated(keep=False)
duplicates = merged_df[duplicate_mask].sort_values('image_name')

if not duplicates.empty:
    print(f"Found {len(duplicates)} case-normalized duplicates:")
    print(duplicates[['image_name']].to_string(index=False))
    duplicates.to_csv("case_sensitive_duplicates.csv", index=False)
else:
    print("No duplicates found.")







# Normalize paths and identify duplicates
merged_df['normalized_path'] = merged_df['image_name'].str.lower().str.replace('img/', '', regex=False)
duplicate_mask = merged_df['normalized_path'].duplicated(keep='first')

# Remove duplicates
deduplicated_df = merged_df[~duplicate_mask].drop(columns=['normalized_path'])

# Save the deduplicated dataset
deduplicated_df.to_csv(FINAL_MERGED_CSV, index=False)
print(f"Removed {duplicate_mask.sum()} duplicates. Saved deduplicated dataset to {FINAL_MERGED_CSV}.")


