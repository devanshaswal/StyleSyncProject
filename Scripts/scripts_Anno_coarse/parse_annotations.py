import os

import pandas as pd

base_path = "dataset/annotation/anno_coarse"
bbox_file = os.path.join(base_path, "list_bbox.txt")
output_folder = "dataset/organized_data"
os.makedirs(output_folder, exist_ok=True)  #  creating the folder

def parse_list_bbox(file_path):
    """
    Parse the list_bbox.txt file and return a DataFrame.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    
    num_images = int(lines[0].strip()) # number of images
    print(f"Number of images: {num_images}")

    
    header = lines[1].strip().split() # the header
    
    
    data = []
    for line in lines[2:]:
        # Extracting image name, bounding box coordinates
        parts = line.strip().split()
        image_name = parts[0]
        bbox_coords = list(map(int, parts[1:]))
        data.append([image_name] + bbox_coords)

    # Converting to a pandas DataFrame
    df = pd.DataFrame(data, columns=["image_name", "x_1", "y_1", "x_2", "y_2"])
    return df

def main():
    # Parsing the bounding box annotations
    bbox_df = parse_list_bbox(bbox_file)

    # To save as CSV
    bbox_output_path = os.path.join(output_folder, "bounding_boxes.csv")
    bbox_df.to_csv(bbox_output_path, index=False)
    print(f"Bounding box annotations saved to: {bbox_output_path}")

if __name__ == "__main__":
    main()

