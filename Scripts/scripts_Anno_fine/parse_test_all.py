import os
import pandas as pd

# Define paths
base_path = os.path.join("dataset", "Annotation", "Anno_fine")
output_folder = os.path.join("dataset", "organized_data_1")
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

# Paths to necessary files
cloth_attr_csv_path = os.path.join(output_folder, "attributes_clothes_fine.csv")
category_csv_path = os.path.join(output_folder, "categories_clothes_fine.csv")

# Load attribute names from attributes_clothes_fine.csv
cloth_df = pd.read_csv(cloth_attr_csv_path)
cloth_attribute_names = cloth_df["attribute_name"].tolist()

# Load category names from categories_clothes_fine.csv
category_df = pd.read_csv(category_csv_path)
category_names = category_df["category_name"].tolist()

# Function to parse test_attr.txt
def parse_test_attr(file_path, attribute_names):
    with open(file_path, 'r') as file:
        data = [list(map(int, line.strip().split())) for line in file]
    return pd.DataFrame(data, columns=attribute_names)

# Function to parse test_bbox.txt
def parse_test_bbox(file_path):
    with open(file_path, 'r') as file:
        data = [list(map(int, line.strip().split())) for line in file]
    return pd.DataFrame(data, columns=["x_1", "y_1", "x_2", "y_2"])

# Function to parse test_cate.txt
def parse_test_cate(file_path):
    with open(file_path, 'r') as file:
        data = [int(line.strip()) for line in file]
    return pd.DataFrame(data, columns=["category_label"])

# Function to parse test_landmarks.txt
def parse_test_landmarks(file_path):
    with open(file_path, 'r') as file:
        data = [list(map(int, line.strip().split())) for line in file]
    
    # Define headers for landmarks (8 keypoints with x and y coordinates)
    headers = []
    for i in range(1, 9):  # 8 keypoints
        headers.append(f"landmark_x_{i}")
        headers.append(f"landmark_y_{i}")
    
    return pd.DataFrame(data, columns=headers)

# Function to parse test.txt
def parse_test_images(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip() for line in file]
    return pd.DataFrame(data, columns=["image_name"])

# Parse all files
test_attr_df = parse_test_attr(os.path.join(base_path, "test_attr.txt"), cloth_attribute_names)
test_bbox_df = parse_test_bbox(os.path.join(base_path, "test_bbox.txt"))
test_cate_df = parse_test_cate(os.path.join(base_path, "test_cate.txt"))
test_landmarks_df = parse_test_landmarks(os.path.join(base_path, "test_landmarks.txt"))
test_images_df = parse_test_images(os.path.join(base_path, "test.txt"))

# Save to CSV
test_attr_df.to_csv(os.path.join(output_folder, "test_attributes.csv"), index=False)
test_bbox_df.to_csv(os.path.join(output_folder, "test_bbox.csv"), index=False)
test_cate_df.to_csv(os.path.join(output_folder, "test_categories.csv"), index=False)
test_landmarks_df.to_csv(os.path.join(output_folder, "test_landmarks.csv"), index=False)
test_images_df.to_csv(os.path.join(output_folder, "test_images.csv"), index=False)

print("All test files parsed and saved successfully!")

