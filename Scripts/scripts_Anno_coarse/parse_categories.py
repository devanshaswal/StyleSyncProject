import os

import pandas as pd

base_path = "dataset/annotation/anno_coarse"
category_file_img = os.path.join(base_path, "list_category_img.txt")
category_file_cloth = os.path.join(base_path, "list_category_cloth.txt")
output_folder = "dataset/organized_data"
os.makedirs(output_folder, exist_ok=True)

def parse_category_img(file_path):
    """
    Parse list_category_img.txt (image-to-category mapping).
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    num_images = int(lines[0].strip())
    header = lines[1].strip().split()  # ["image_name", "category_label"]

    data = []
    for line in lines[2:]:
        parts = line.strip().split()
        image_name = parts[0]
        category_label = int(parts[1])
        data.append([image_name, category_label])

    # Validate data length matches num_images
    assert len(data) == num_images, "Mismatch in parsed image-category data!"
    
    df = pd.DataFrame(data, columns=header)
    return df

def parse_category_cloth(file_path):
    """
    Parse list_category_cloth.txt (category names and types).
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    num_categories = int(lines[0].strip())
    header = lines[1].strip().split()  # ["category_name", "category_type"]

    data = []
    for line in lines[2:]:
        parts = line.strip().split()
        category_name = " ".join(parts[:-1])  # Handle multi-word category names (e.g., "T-shirt")
        category_type = int(parts[-1])
        
        # Merge categories into "Dress" (as per dataset notes)
        if category_name in ["Cape", "Nightdress", "Shirtdress", "Sundress"]:
            category_name = "Dress"
        
        data.append([category_name, category_type])

    # Validate data length matches num_categories
    assert len(data) == num_categories, "Mismatch in parsed category data!"
    
    df = pd.DataFrame(data, columns=header)
    return df

def main():
    # Parse image-to-category mappings
    category_df_img = parse_category_img(category_file_img)
    category_output_path_img = os.path.join(output_folder, "category_images.csv")
    category_df_img.to_csv(category_output_path_img, index=False)

    # Parse category names and types
    category_df_cloth = parse_category_cloth(category_file_cloth)
    category_output_path_cloth = os.path.join(output_folder, "category_clothes.csv")
    category_df_cloth.to_csv(category_output_path_cloth, index=False)

    print(f"Category annotations saved to: {category_output_path_img} and {category_output_path_cloth}")

if __name__ == "__main__":
    main()