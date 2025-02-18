import os
import pandas as pd

# Define paths
input_file_path = os.path.join("dataset", "Annotation", "Anno_fine", "list_attr_img.txt")
cloth_attr_csv_path = os.path.join("dataset", "organized_data_1", "attributes_clothes_fine.csv")
output_folder = os.path.join("dataset", "organized_data_1")
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist
output_csv_path = os.path.join(output_folder, "attributes_images_fine.csv")

def parse_list_attr_img(file_path, cloth_attribute_names):
    """
    Parses the list_attr_img.txt file and returns a DataFrame.

    Args:
        file_path (str): Path to the list_attr_img.txt file.
        cloth_attribute_names (list): List of attribute names from list_attr_cloth.txt.

    Returns:
        pd.DataFrame: A DataFrame with columns [image_name, attribute_1, attribute_2, ...].
    """
    with open(file_path, 'r') as file:
        # Read the first line (number of images)
        num_images = int(file.readline().strip())

        # Read the second line (headers)
        headers = file.readline().strip().split()

        # Prepare column names
        columns = ["image_name"] + cloth_attribute_names

        # Read the remaining lines (image name and attribute labels)
        data = []
        for line in file:
            parts = line.strip().split()
            image_name = parts[0]  # First part is the image name
            attribute_labels = list(map(int, parts[1:]))  # Remaining parts are the attribute labels
            data.append([image_name] + attribute_labels)

    # Create a DataFrame
    df = pd.DataFrame(data, columns=columns)
    return df

# Load attribute names from attributes_clothes_fine.csv
cloth_df = pd.read_csv(cloth_attr_csv_path)
cloth_attribute_names = cloth_df["attribute_name"].tolist()

# Parse the list_attr_img.txt file
df = parse_list_attr_img(input_file_path, cloth_attribute_names)

# Save to CSV
df.to_csv(output_csv_path, index=False)
print(f"Parsed data saved to {output_csv_path}")
print(df.head())  # Display the first few rows