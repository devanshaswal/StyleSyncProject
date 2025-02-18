import pandas as pd

# Load the files into DataFrames
attributes_images_df = pd.read_csv("dataset/organized_data/csv_files/attributes_images.csv")
bounding_boxes_df = pd.read_csv("dataset/organized_data/csv_files/bounding_boxes.csv")
category_images_df = pd.read_csv("dataset/organized_data/csv_files/category_images.csv")
parsed_landmarks_df = pd.read_csv("dataset/organized_data/csv_files/parsed_landmarks.csv")
eval_partitions_df = pd.read_csv("dataset/organized_data/csv_files/eval_partitions.csv")
attributes_clothes_df = pd.read_csv("dataset/organized_data/csv_files/attributes_clothes.csv")
category_clothes_df = pd.read_csv("dataset/organized_data/csv_files/category_clothes.csv")
landmarks_cleaned_df = pd.read_csv("dataset/organized_data/csv_files/parsed_landmarks_cleaned.csv")   # After fillling missing values 

# Check the shape of each DataFrame
print("attributes_images.csv shape:", attributes_images_df.shape)
print("bounding_boxes.csv shape:", bounding_boxes_df.shape)
print("category_images.csv shape:", category_images_df.shape)
print("parsed_landmarks.csv shape:", parsed_landmarks_df.shape)
print("eval_partitions.csv shape:", eval_partitions_df.shape)
print("attributes_clothes.csv shape:", attributes_clothes_df.shape)
print("category_clothes.csv shape:", category_clothes_df.shape)
print("parsed_landmarks_cleaned.csv shape:", landmarks_cleaned_df.shape)

# Inspect the first few rows of each DataFrame
print("\nattributes_images.csv head:\n", attributes_images_df.head())
print("\nbounding_boxes.csv head:\n", bounding_boxes_df.head())
print("\ncategory_images.csv head:\n", category_images_df.head())
print("\nparsed_landmarks.csv head:\n", parsed_landmarks_df.head())
print("\neval_partitions.csv head:\n", eval_partitions_df.head())
print("\nattributes_clothes.csv head:\n", attributes_clothes_df.head())
print("\ncategory_clothes.csv head:\n", category_clothes_df.head())
print("\nparsed_landmarks_cleaned.csv head:\n", landmarks_cleaned_df.head())

# Check for missing values in each file
print("\nMissing values in attributes_images.csv:\n", attributes_images_df.isnull().sum())
print("\nMissing values in bounding_boxes.csv:\n", bounding_boxes_df.isnull().sum())
print("\nMissing values in category_images.csv:\n", category_images_df.isnull().sum())
print("\nMissing values in parsed_landmarks.csv:\n", parsed_landmarks_df.isnull().sum())
print("\nMissing values in eval_partitions.csv:\n", eval_partitions_df.isnull().sum())
print("\nMissing values in attributes_clothes.csv:\n", attributes_clothes_df.isnull().sum())
print("\nMissing values in category_clothes.csv:\n", category_clothes_df.isnull().sum())
print("\nMissing values in parsed_landmarks_cleaned.csv:\n", landmarks_cleaned_df.isnull().sum())

