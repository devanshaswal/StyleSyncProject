import pandas as pd

# Load data with correct column names
df_images = pd.read_csv('dataset/organized_data/csv_files/category_images.csv')  # Columns: image_name, category_label
df_clothes = pd.read_csv('dataset/organized_data/csv_files/category_clothes.csv')  # Columns: category_name, category_type

# Check if 'category_label' exists in df_clothes (if not, map via category_type)
# Assuming "category_type" in df_clothes corresponds to "category_label" in df_images
# Merge using the correct key columns
merged_df = df_images.merge(
    df_clothes,
    left_on='category_label',  # From category_images.csv
    right_on='category_type',  # From category_clothes.csv
    how='left'
)

# Drop redundant 'category_type' column after merging (optional)
merged_df = merged_df.drop('category_type', axis=1)

# Save the merged data
merged_df.to_csv('dataset/organized_data/merged_categories.csv', index=False)

# Preview
print("Merged data preview:")
print(merged_df.head())