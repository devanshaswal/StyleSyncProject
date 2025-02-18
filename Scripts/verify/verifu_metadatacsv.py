import pandas as pd

# Load CSV
final_df = pd.read_csv("x:/dissertation/StyleSyncProject/dataset/organized_data/csv_files/final_merged_data.csv")

# Check column names
print("Columns in final_merged_data.csv:", final_df.columns)

# Strip spaces to prevent hidden issues
final_df.columns = final_df.columns.str.strip()

# Ensure 'image_name' exists
if "image_name" not in final_df.columns:
    raise ValueError("The column 'image_name' does not exist in final_merged_data.csv. Check the actual column names!")

# Convert image_name to match metadata.csv format (replace '/' with '\')
final_df["image_name"] = final_df["image_name"].str.replace("img/", "", regex=False).str.replace("/", "\\", regex=False)

# Save the modified file (optional)
final_df.to_csv("x:/dissertation/StyleSyncProject/dataset/organized_data/csv_files/final_merged_data_fixed.csv", index=False)

# Verify alignment with metadata.csv
metadata_df = pd.read_csv("x:/dissertation/StyleSyncProject/data/processed/metadata.csv")

# Find missing rows
missing_entries = final_df[~final_df["image_name"].isin(metadata_df["image_name"])]

# Save missing entries for review
missing_entries.to_csv("x:/dissertation/StyleSyncProject/data/processed/missing_entries.csv", index=False)

# Print summary
print(f"Total missing entries: {len(missing_entries)}")
print(missing_entries.head())
