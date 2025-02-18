import pandas as pd
import os

def parse_list_attr_img(file_path, cloth_attribute_names):
    """
    Parses list_attr_img.txt using attribute names from attributes_clothes.csv.

    Args:
        file_path (str): Path to list_attr_img.txt.
        cloth_attribute_names (list): List of attribute names from attributes_clothes.csv.

    Returns:
        pd.DataFrame: A DataFrame with columns [image_name, attribute_1, attribute_2, ...].
    """
    with open(file_path, 'r') as file:
        # Read the total number of images (first line)
        num_images = int(file.readline().strip())

        # Skip the second line (header is not needed, as we use cloth_attribute_names)
        _ = file.readline()

        # Prepare column names
        columns = ["image_name"] + cloth_attribute_names

        # Read the remaining lines (image name and attributes)
        data = []
        for line in file:
            parts = line.strip().split()
            if len(parts) != len(columns):
                print(f"Skipping invalid row: {parts[0]} (expected {len(columns)} columns, got {len(parts)})")
                continue
            data.append(parts)

    # Create a DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Convert attribute columns to integers
    for col in cloth_attribute_names:
        df[col] = df[col].astype(int)

    return df

# Example usage
if __name__ == "__main__":
    # Paths
    cloth_attr_csv_path = os.path.join("dataset", "organized_data", "attributes_clothes.csv")
    img_attr_txt_path = os.path.join("dataset", "Annotation", "Anno_coarse", "list_attr_img.txt")
    output_csv_path = os.path.join("dataset", "organized_data", "attributes_images.csv")

    # Load attribute names from attributes_clothes.csv
    cloth_df = pd.read_csv(cloth_attr_csv_path)
    cloth_attribute_names = cloth_df["attribute_name"].tolist()

    # Parse list_attr_img.txt
    img_df = parse_list_attr_img(img_attr_txt_path, cloth_attribute_names)

    # Save the parsed data to CSV
    img_df.to_csv(output_csv_path, index=False)
    print(f"Parsed data saved to {output_csv_path}")
    print(f"Total images parsed: {len(img_df)}")
    print(f"Total attributes: {len(cloth_attribute_names)}")