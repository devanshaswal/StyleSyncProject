import os
import pandas as pd

# Define paths
input_file_path = os.path.join("dataset", "Annotation", "Anno_fine", "list_category_cloth.txt")
output_folder = os.path.join("dataset", "organized_data_1")
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist
output_csv_path = os.path.join(output_folder, "categories_clothes_fine.csv")

def parse_list_category_cloth(file_path):
    """
    Parses the list_category_cloth.txt file and returns a DataFrame.

    Args:
        file_path (str): Path to the list_category_cloth.txt file.

    Returns:
        pd.DataFrame: A DataFrame with columns [category_name, category_type].
    """
    with open(file_path, 'r') as file:
        # Read the first line (number of categories)
        num_categories = int(file.readline().strip())

        # Read the second line (headers)
        headers = file.readline().strip().split()

        # Read the remaining lines (category name and type)
        data = []
        for line in file:
            parts = line.strip().split()
            category_name = " ".join(parts[:-1])  # Handle multi-word category names
            category_type = int(parts[-1])       # Convert category type to integer
            data.append([category_name, category_type])

    # Create a DataFrame
    df = pd.DataFrame(data, columns=headers)
    return df

# Parse the file
df = parse_list_category_cloth(input_file_path)

# Save to CSV
df.to_csv(output_csv_path, index=False)
print(f"Parsed data saved to {output_csv_path}")
print(df.head())  # Display the first few rows