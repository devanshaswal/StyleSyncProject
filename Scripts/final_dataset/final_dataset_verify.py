import os

import pandas as pd

# Define paths
data_dir = "dataset/organized_data/"
final_dataset_path = os.path.join(data_dir, "final_merged_data.csv")

# Load datasets
final_merged_df = pd.read_csv(final_dataset_path)
original_bbox = pd.read_csv(os.path.join(data_dir, "bounding_boxes.csv"))
original_landmark = pd.read_csv(os.path.join(data_dir, "parsed_landmarks.csv"))
original_attr = pd.read_csv(os.path.join(data_dir, "attributes_images.csv"))
category_clothes_df = pd.read_csv(os.path.join(data_dir, "category_clothes.csv"))
attributes_clothes_df = pd.read_csv(os.path.join(data_dir, "attributes_clothes.csv"))  # Added this line

print("\n Starting Dataset Verification...\n")

# Step 1: Add category_id to category_clothes_df
# ----------------------------------------------
# Assign category_id as index + 1 (1-based ID)
category_clothes_df["category_id"] = category_clothes_df.index + 1

# Step 2: Verify Category Mapping
# -------------------------------
print("2. Category Mapping:")
# Ensure category_label maps correctly to category_name
category_mapping_check = final_merged_df[["category_label", "category_name"]].drop_duplicates()
category_clothes_mapped = category_clothes_df[["category_id", "category_name"]].rename(columns={"category_id": "category_label"})
merged_category_check = category_mapping_check.merge(category_clothes_mapped, on="category_label", how="left", suffixes=('_final', '_original'))
incorrect_mappings = merged_category_check[merged_category_check["category_name_final"] != merged_category_check["category_name_original"]]
print("Incorrect category mappings found:", incorrect_mappings.shape[0])
print("\n")

# Step 3: Validate Attribute Mapping
# # ----------------------------------
print("3. Attribute Mapping:")
# Check if all attributes are mapped to "Yes", "No", or "Unknown"
attribute_columns = attributes_clothes_df["attribute_name"].tolist()  # Corrected: Use attributes_clothes_df
invalid_attributes = {}
for attr in attribute_columns:
    if attr in final_merged_df.columns:
        unique_values = final_merged_df[attr].unique()
        # if not set(unique_values).issubset({"Yes", "No", "Unknown"}):
        if not set(unique_values).issubset({-1, 0, 1}):
            invalid_attributes[attr] = unique_values
print("Attributes with incorrect values:", len(invalid_attributes))
print("\n")

# Step 4: Bounding Box & Landmark Verification Across Multiple Images
# -------------------------------------------------------------------
print("4. Bounding Box & Landmark Data:")
sample_images = final_merged_df.sample(5)["image_name"].tolist()
for image_name in sample_images:
    # Check bounding box
    merged_bbox = final_merged_df[final_merged_df["image_name"] == image_name][["x_1", "y_1", "x_2", "y_2"]].values
    original_bbox_values = original_bbox[original_bbox["image_name"] == image_name][["x_1", "y_1", "x_2", "y_2"]].values
    if not (merged_bbox == original_bbox_values).all():
        print(f"Bounding Box Mismatch for {image_name}!")
    
    # Check landmarks
    merged_landmarks = final_merged_df[final_merged_df["image_name"] == image_name]["landmark_visibility_1"].values
    original_landmarks = original_landmark[original_landmark["image_name"] == image_name]["landmark_visibility_1"].values
    if not (merged_landmarks == original_landmarks).all():
        print(f"Landmark Mismatch for {image_name}!")
print("\n")

# Step 5: Check for Missing Data
# ------------------------------
print("5. Missing Values Check:")
missing_data = final_merged_df.isnull().sum()
missing_summary = missing_data[missing_data > 0]
print("Missing Values Summary:")
print(missing_summary)
print("\n")

# Step 6: Check for Duplicates
# ----------------------------
print("6. Duplicate Images:")
duplicates = final_merged_df["image_name"].duplicated().sum()
print("Duplicate images found:", duplicates)
print("\n")

# Step 7: Validate Data Types
# ---------------------------
print("7. Data Type Validation:")
print("Bounding box data types:", final_merged_df[["x_1", "y_1", "x_2", "y_2"]].dtypes.tolist())
print("Landmark data type:", final_merged_df["landmark_visibility_1"].dtype)
print("\n")

# Step 8: Final Summary Check
# ---------------------------
print("8. Final Dataset Validation Summary:")
if all([
    missing_summary.shape[0] == 0,  # No missing values
    duplicates == 0,  # No duplicate images
    incorrect_mappings.shape[0] == 0,  # No incorrect category mappings
    not invalid_attributes  # No invalid attribute values
]):
    print("\n Final Dataset Verification Complete: No Issues Found! Ready for Training.\n")
else:
    print("\n Issues Found! Please Review Logs Above.\n")







# import os
# import pandas as pd

# # Define paths
# data_dir = "dataset/organized_data/"
# final_dataset_path = os.path.join(data_dir, "final_merged_data.csv")

# # Load datasets
# final_merged_df = pd.read_csv(final_dataset_path)
# original_bbox = pd.read_csv(os.path.join(data_dir, "bounding_boxes.csv"))
# original_landmark = pd.read_csv(os.path.join(data_dir, "parsed_landmarks.csv"))
# original_attr = pd.read_csv(os.path.join(data_dir, "attributes_images.csv"))
# category_clothes_df = pd.read_csv(os.path.join(data_dir, "category_clothes.csv"))
# attributes_clothes_df = pd.read_csv(os.path.join(data_dir, "attributes_clothes.csv"))

# print("\n Starting Dataset Verification...\n")

# # Step 1: Add category_id to category_clothes_df
# # ----------------------------------------------
# # Assign category_id as index + 1 (1-based ID)
# category_clothes_df["category_id"] = category_clothes_df.index + 1

# # Step 2: Verify Category Mapping
# # -------------------------------
# print("2. Category Mapping:")
# # Ensure category_label maps correctly to category_name
# category_mapping_check = final_merged_df[["category_label", "category_name"]].drop_duplicates()
# category_clothes_mapped = category_clothes_df[["category_id", "category_name"]].rename(columns={"category_id": "category_label"})
# merged_category_check = category_mapping_check.merge(category_clothes_mapped, on="category_label", how="left", suffixes=('_final', '_original'))
# incorrect_mappings = merged_category_check[merged_category_check["category_name_final"] != merged_category_check["category_name_original"]]
# print("Incorrect category mappings found:", incorrect_mappings.shape[0])
# print("\n")

# # Step 3: Validate Attribute Mapping
# # ----------------------------------
# print("3. Attribute Mapping:")
# # Check if all attributes are mapped to "Yes", "No", or "Unknown"
# attribute_columns = attributes_clothes_df["attribute_name"].tolist()  # Corrected: Use attributes_clothes_df
# invalid_attributes = {}
# for attr in attribute_columns:
#     if attr in final_merged_df.columns:
#         unique_values = final_merged_df[attr].unique()
#         if not set(unique_values).issubset({"Yes", "No", "Unknown"}):
#             invalid_attributes[attr] = unique_values
# print("Attributes with incorrect values:", len(invalid_attributes))
# print("\n")

# # Step 4: Bounding Box & Landmark Verification Across Multiple Images
# # -------------------------------------------------------------------
# print("4. Bounding Box & Landmark Data:")
# sample_images = final_merged_df.sample(5)["image_name"].tolist()
# for image_name in sample_images:
#     # Check bounding box
#     merged_bbox = final_merged_df[final_merged_df["image_name"] == image_name][["x_1", "y_1", "x_2", "y_2"]].values
#     original_bbox_values = original_bbox[original_bbox["image_name"] == image_name][["x_1", "y_1", "x_2", "y_2"]].values
#     if not (merged_bbox == original_bbox_values).all():
#         print(f"Bounding Box Mismatch for {image_name}!")
    
#     # Check landmarks
#     merged_landmarks = final_merged_df[final_merged_df["image_name"] == image_name]["landmark_visibility_1"].values
#     original_landmarks = original_landmark[original_landmark["image_name"] == image_name]["landmark_visibility_1"].values
#     if not (merged_landmarks == original_landmarks).all():
#         print(f"Landmark Mismatch for {image_name}!")
# print("\n")

# # Step 5: Check for Missing Data
# # ------------------------------
# print("5. Missing Values Check:")
# missing_data = final_merged_df.isnull().sum()
# missing_summary = missing_data[missing_data > 0]
# print("Missing Values Summary:")
# print(missing_summary)
# print("\n")

# # Step 6: Check for Duplicates
# # ----------------------------
# print("6. Duplicate Images:")
# duplicates = final_merged_df["image_name"].duplicated().sum()
# print("Duplicate images found:", duplicates)
# print("\n")

# # Step 7: Validate Data Types
# # ---------------------------
# print("7. Data Type Validation:")
# print("Bounding box data types:", final_merged_df[["x_1", "y_1", "x_2", "y_2"]].dtypes.tolist())
# print("Landmark data type:", final_merged_df["landmark_visibility_1"].dtype)
# print("\n")

# # Step 8: Verify Column Integrity
# # -------------------------------
# print("8. Column Integrity Check:")

# # Define all expected columns in the final dataset
# expected_columns = [
#     "image_name", "category_label", "category_name", "x_1", "y_1", "x_2", "y_2",
#     "landmark_visibility_1", "landmark_location_x_1", "landmark_location_y_1",
#     "landmark_visibility_2", "landmark_location_x_2", "landmark_location_y_2",
#     "landmark_visibility_3", "landmark_location_x_3", "landmark_location_y_3",
#     "landmark_visibility_4", "landmark_location_x_4", "landmark_location_y_4",
#     "landmark_visibility_5", "landmark_location_x_5", "landmark_location_y_5",
#     "landmark_visibility_6", "landmark_location_x_6", "landmark_location_y_6",
#     "landmark_visibility_7", "landmark_location_x_7", "landmark_location_y_7",
#     "landmark_visibility_8", "landmark_location_x_8", "landmark_location_y_8",
#     "evaluation_status", "a-line", "abstract", "abstract chevron", "abstract chevron print",
#     "abstract diamond", "abstract floral", "abstract floral print", "abstract geo",
#     "abstract geo print", "abstract heart", "abstract heart print", "abstract paisley",
#     "abstract paisley print", "abstract plaid", "abstract plaid print", "abstract print",
#     "abstract stripe", "abstract stripe print", "abstract triangle", "abstract triangle print",
#     "animal", "animal cheetah", "animal leopard", "animal print", "animal snake",
#     "animal tiger", "animal zebra", "applique", "asymmetric", "backless", "bandana",
#     "beaded", "belted", "bib", "biker", "blanket", "bleached", "boat neck", "bohemian",
#     "bow", "boxy", "braided", "button", "button front", "caged", "cape", "cargo",
#     "casual", "chiffon", "chino", "circle", "classic", "clog", "colorblock", "contrast",
#     "contrast binding", "contrast collar", "contrast panel", "contrast pocket",
#     "contrast stitch", "contrast stripe", "contrast trim", "convertible", "crochet",
#     "cropped", "cuffed", "cut out", "denim", "distressed", "dotted", "double breasted",
#     "draped", "drawstring", "drop waist", "elastic", "embellished", "embroidered",
#     "eyelet", "faux fur", "faux leather", "faux suede", "feather", "fitted", "flap",
#     "flared", "flat", "floral", "floral print", "flowy", "frayed", "fringe", "fur",
#     "gathered", "geometric", "geometric print", "glitter", "graphic", "graphic print",
#     "grommet", "halter", "handkerchief", "high low", "high waist", "hooded", "houndstooth",
#     "illusion", "jacquard", "jersey", "keyhole", "knit", "lace", "lace up", "layered",
#     "leather", "letter", "lightweight", "linen", "logo", "long sleeve", "low waist",
#     "metallic", "midi", "military", "minimal", "moto", "neoprene", "notched", "ombre",
#     "open back", "open front", "open toe", "oversized", "patch", "patch pocket",
#     "patchwork", "peplum", "pleat", "pleated", "pocket", "pointed", "polka dot",
#     "pom pom", "quilted", "raw edge", "ribbed", "ribbon", "ruched", "ruffle", "safari",
#     "sash", "scallop", "scalloped", "sequin", "sheer", "shirred", "short sleeve",
#     "slip on", "smocked", "snake", "spaghetti strap", "split", "stretch", "striped",
#     "striped print", "structured", "stud", "suede", "sweater", "sweatshirt", "tassel",
#     "textured", "tie", "tiered", "tiger", "tulle", "turtleneck", "twist", "v neck",
#     "velvet", "vent", "vintage", "watercolor", "wedge", "wrap", "wrinkled", "zipper",
#     "zippered"
# ]

# # Check for missing columns
# missing_columns = set(expected_columns) - set(final_merged_df.columns)
# if not missing_columns:
#     print("✅ All expected columns are present.")
# else:
#     print("❌ Missing Columns:")
#     print(missing_columns)
# print("\n")

# # Step 9: Verify Row Count
# # ------------------------
# print("9. Row Count Check:")
# print("Final Merged Data:", len(final_merged_df))
# print("Bounding Boxes:", len(original_bbox))
# print("Landmarks:", len(original_landmark))
# print("Attributes:", len(original_attr))
# if len(final_merged_df) == len(original_bbox) == len(original_landmark) == len(original_attr):
#     print("✅ Row counts match across all datasets.")
# else:
#     print("❌ Row count mismatch detected.")
# print("\n")

# # Step 10: Final Summary Check
# # ----------------------------
# print("10. Final Dataset Validation Summary:")
# if all([
#     missing_summary.shape[0] == 0,  # No missing values
#     duplicates == 0,  # No duplicate images
#     incorrect_mappings.shape[0] == 0,  # No incorrect category mappings
#     not invalid_attributes,  # No invalid attribute values
#     not missing_columns,  # No missing columns
#     len(final_merged_df) == len(original_bbox) == len(original_landmark) == len(original_attr)  # Row counts match
# ]):
#     print("\n✅ Final Dataset Verification Complete: No Issues Found! Ready for Training.\n")
# else:
#     print("\n❌ Issues Found! Please Review Logs Above.\n")