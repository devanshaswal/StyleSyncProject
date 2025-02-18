import pandas as pd
import os

# Define paths
data_dir = "dataset/organized_data/csv_files"
output_file = os.path.join(data_dir, "final_merged_data99.csv")

# Load datasets
merged_df = pd.read_csv(os.path.join(data_dir, "merged_data.csv"))
category_clothes_df = pd.read_csv(os.path.join(data_dir, "category_clothes.csv"))
attributes_clothes_df = pd.read_csv(os.path.join(data_dir, "attributes_clothes.csv"))

# Step 1: Assign category_id to category_clothes_df (ensure alignment with category_label)
# ----------------------------------------------------------------------------------------
assert len(category_clothes_df) == 50, "category_clothes_df does not have 50 categories!"
category_clothes_df["category_id"] = category_clothes_df.index + 1  # 1-based ID

# Step 2: Merge to add category names and types
# ---------------------------------------------
merged_df = merged_df.merge(
    category_clothes_df[["category_id", "category_name", "category_type"]],
    left_on="category_label",
    right_on="category_id",
    how="left"
)

# Step 3: Map attribute values to "Yes"/"No"/"Unknown"
# ----------------------------------------------------
attribute_names = attributes_clothes_df["attribute_name"].tolist()
existing_attributes = [attr for attr in attribute_names if attr in merged_df.columns]

for attr in existing_attributes:
    merged_df[attr] = merged_df[attr].map({-1: "No", 1: "Yes", 0: "Unknown"})

# Step 4: Finalize columns (Keep category_label for traceability)
# ---------------------------------------------------------------
final_columns = [col for col in merged_df.columns if col not in ["category_id"]]  # Drop category_id, keep category_label

final_merged_df = merged_df[final_columns]

# Step 5: Save the final dataset
# ------------------------------
final_merged_df.to_csv(output_file, index=False)

# Verify
print("Final dataset shape:", final_merged_df.shape)
print("Columns:", final_merged_df.columns)
print("\nSample data:")
print(final_merged_df[["image_name", "category_label", "category_name", "a-line"]].head())
