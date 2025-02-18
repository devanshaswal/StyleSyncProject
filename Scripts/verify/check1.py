# import os
# import pandas as pd

# # Define paths
# csv_path = "X:/dissertation/StyleSyncProject/dataset/organized_data/csv_files/final_merged_data.csv"
# cropped_images_dir = "X:/dissertation/StyleSyncProject/data/processed/cropped_images"
# output_missing_csv = "X:/dissertation/StyleSyncProject/data/processed/missing_cropped_images.csv"

# # Load CSV
# df = pd.read_csv(csv_path)

# # Normalize image paths in CSV
# df["image_name"] = df["image_name"].str.replace("img/", "", regex=False)  # Remove 'img/' prefix
# df["image_name"] = df["image_name"].str.replace("\\", "/").str.strip()  # Normalize slashes

# # Collect existing cropped images
# existing_images = set()
# for root, _, files in os.walk(cropped_images_dir):
#     for file in files:
#         if file.endswith(".jpg"):
#             subfolder = os.path.relpath(root, cropped_images_dir).replace("\\", "/")  # Normalize folder names
#             relative_path = f"{subfolder}/{file}".lower()  # Convert to lowercase for case-insensitive matching
#             existing_images.add(relative_path)

# # Identify missing images
# df["image_name_lower"] = df["image_name"].str.lower()
# missing_images = df[~df["image_name_lower"].isin(existing_images)]

# # Save missing images to CSV
# missing_images.drop(columns=["image_name_lower"], inplace=True)
# missing_images.to_csv(output_missing_csv, index=False)

# # Print summary
# print(f"✅ Total entries in CSV: {len(df)}")
# print(f"✅ Cropped images found: {len(existing_images)}")
# print(f"❌ Missing images: {len(missing_images)}")
# print(f"📁 Missing images saved to: {output_missing_csv}")
# import pandas as pd
# import os

# # File paths
# csv_path = "X:/dissertation/StyleSyncProject/dataset/organized_data/csv_files/final_merged_data.csv"
# cropped_images_path = "X:/dissertation/StyleSyncProject/data/processed/cropped_images"

# # Load CSV
# df = pd.read_csv(csv_path)

# # Normalize image paths
# df["image_name"] = df["image_name"].apply(lambda x: os.path.normpath(x))

# # Get all image names from cropped_images folder
# cropped_image_list = []
# for root, dirs, files in os.walk(cropped_images_path):
#     for file in files:
#         if file.endswith((".jpg", ".png", ".jpeg")):
#             cropped_image_list.append(os.path.normpath(os.path.join(os.path.basename(root), file)))

# # Find missing images
# missing_images = df[~df["image_name"].isin(cropped_image_list)]

# # Save missing images
# missing_images.to_csv("X:/dissertation/StyleSyncProject/data/processed/missing_cropped_images.csv", index=False)

# print(f"❌ Missing images count: {len(missing_images)}")
# print(f"📁 Missing images saved to: X:/dissertation/StyleSyncProject/data/processed/missing_cropped_images.csv")



import os

import pandas as pd

# File paths
csv_path = "X:/dissertation/StyleSyncProject/dataset/organized_data/csv_files/final_merged_data.csv"
cropped_images_path = "X:/dissertation/StyleSyncProject/data/processed/cropped_images"

# Load CSV
df = pd.read_csv(csv_path)

# Normalize image paths in CSV (remove "img/" prefix, standardize slashes, and lowercase everything)
df["image_name"] = df["image_name"].str.replace("img/", "", regex=False).str.strip().str.lower().apply(lambda x: os.path.normpath(x))

# Get all image names from cropped_images folder
cropped_image_list = []
for root, dirs, files in os.walk(cropped_images_path):
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            relative_path = os.path.normpath(os.path.join(os.path.basename(root), file)).strip().lower()
            cropped_image_list.append(relative_path)

# Find missing images
missing_images = df[~df["image_name"].isin(cropped_image_list)]

# Save missing images
# missing_images.to_csv("X:/dissertation/StyleSyncProject/data/processed/missing_cropped_images.csv", index=False)

print(f"❌ Missing images count: {len(missing_images)}")
print(f"📁 Missing images saved to: X:/dissertation/StyleSyncProject/data/processed/missing_cropped_images.csv")



