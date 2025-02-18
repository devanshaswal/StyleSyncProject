import os
import pandas as pd

# Define paths
input_file_path = os.path.join("dataset", "Annotation", "Anno_fine", "test_attr.txt")
cloth_attr_csv_path = os.path.join("dataset", "organized_data_1", "attributes_clothes_fine.csv")
output_folder = os.path.join("dataset", "organized_data_1")
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist
output_csv_path = os.path.join(output_folder, "test_attributes.csv")

def parse_test_attr(file_path, cloth_attribute_names):
    """
    Parses the test_attr.txt file and returns a DataFrame.

    Args:
        file_path (str): Path to the test_attr.txt file.
        cloth_attribute_names (list): List of attribute names from list_attr_cloth.txt.

    Returns:
        pd.DataFrame: A DataFrame with columns [attribute_1, attribute_2, ...].
    """
    with open(file_path, 'r') as file:
        # Read all lines
        lines = file.readlines()

        # Parse attribute labels
        data = []
        for line in lines:
            attribute_labels = list(map(int, line.strip().split()))  # Convert labels to integers
            data.append(attribute_labels)

    # Create a DataFrame
    df = pd.DataFrame(data, columns=cloth_attribute_names)
    return df

# Load attribute names from attributes_clothes_fine.csv
cloth_df = pd.read_csv(cloth_attr_csv_path)
cloth_attribute_names = cloth_df["attribute_name"].tolist()

# Parse the test_attr.txt file
df = parse_test_attr(input_file_path, cloth_attribute_names)

# Save to CSV
df.to_csv(output_csv_path, index=False)
print(f"Parsed data saved to {output_csv_path}")
print(df.head())  # Display the first few rows