import os
import pandas as pd

base_path = "dataset/annotation/anno_coarse"
attribute_file_cloth = os.path.join(base_path, "list_attr_cloth.txt")
output_folder = "dataset/organized_data"
os.makedirs(output_folder, exist_ok=True)

def parse_attribute_cloth(file_path):
    """Parse the list_attr_cloth.txt file and return a DataFrame with correct headers."""
    with open(file_path, "r") as file:
        lines = file.readlines()

    num_attributes = int(lines[0].strip())  # First line: number of attributes
    header = lines[1].strip().split()       # Second line: column headers

    data = []
    for line in lines[2:]:
        parts = line.strip().split()
        attribute_name = " ".join(parts[:-1])
        attribute_type = int(parts[-1])
        data.append([attribute_name, attribute_type])

    return pd.DataFrame(data, columns=header)

def main():
    # Process and save clothing attributes
    attribute_cloth_df = parse_attribute_cloth(attribute_file_cloth)
    output_path = os.path.join(output_folder, "attributes_clothes.csv")
    attribute_cloth_df.to_csv(output_path, index=False)
    print(f"Clothing attributes saved to: {output_path}")

if __name__ == "__main__":
    main()