import os

import pandas as pd

# Define the path for the merged dataset
merged_file_path = 'dataset/organized_data/csv_files/merged_data.csv'

# Check if the merged file exists
if not os.path.exists(merged_file_path):
    print(f"Error: The file '{merged_file_path}' does not exist.")
    exit(1)

# Load the merged dataset
merged_df = pd.read_csv(merged_file_path)

# Step 1: Check the shape of the DataFrame (rows and columns)
print(f"\nMerged DataFrame shape: {merged_df.shape}")
print(f"Number of rows: {merged_df.shape[0]}, Number of columns: {merged_df.shape[1]}")

# Step 2: Check for duplicates in the 'image_name' column
duplicate_rows = merged_df[merged_df.duplicated(subset='image_name', keep=False)]
if not duplicate_rows.empty:
    print("\nDuplicate rows found:")
    print(duplicate_rows)
else:
    print("\nNo duplicate rows found.")

# Step 3: Check for missing data in the dataset
missing_data = merged_df.isnull().sum()
print("\nMissing data in each column:")
print(missing_data)

# Step 4: Check the column names to ensure they match expectations
print("\nMerged DataFrame columns:")
print(merged_df.columns)

# Step 5: Spot-check a few specific rows
# For example, checking the first 5 rows and a specific 'image_name' value if needed
print("\nFirst 5 rows of the merged dataset:")
print(merged_df.head())

# Check for a specific 'image_name' (replace 'image_example.jpg' with an actual image name from your dataset)
# For example:
specific_image = merged_df[merged_df['image_name'] == 'img/Paisley_Floral_A-Line_Dress/img_00000028.jpg']
if not specific_image.empty:
    print("\nDetails for a specific image (img/Paisley_Floral_A-Line_Dress/img_00000028.jpg):")
    print(specific_image)
else:
    print("\nNo specific image found with the name 'img/Paisley_Floral_A-Line_Dress/img_00000028.jpg'.")

# Step 6: Visual Inspection (Optional, for Jupyter or direct inspection)
# For larger datasets, you may want to display a smaller portion of the dataset for a quick look
print("\nDisplaying the first 10 rows for visual inspection:")
print(merged_df.head(10))
