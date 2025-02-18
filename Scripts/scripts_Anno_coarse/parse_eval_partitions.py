import os

import pandas as pd

# Define paths
base_path = "dataset/annotation/Eval"
eval_file = os.path.join(base_path, "list_eval_partition.txt")
output_folder = "dataset/organized_data/csv_files"
os.makedirs(output_folder, exist_ok=True)

def parse_eval_file(file_path):
    """
    Parse the list_eval_partition.txt file correctly, accounting for spaces in image paths.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    # First line: total number of images (optional, but included for completeness)
    num_images = int(lines[0].strip())

    # Second line: header (ignored since we know the format)
    # Data starts from line 3 onward
    data = []
    for line in lines[2:]:
        line = line.strip()
        # Split into image path and evaluation status
        # The LAST token is the partition (train/val/test)
        # Everything before the last token is the image path (may contain spaces)
        parts = line.rsplit(maxsplit=1)  # Split into [image_path, partition]
        if len(parts) != 2:
            continue  # Skip malformed lines
        image_name, partition = parts
        data.append([image_name, partition])

    # Create DataFrame
    df = pd.DataFrame(data, columns=["image_name", "evaluation_status"])
    return df

def main():
    # Parse evaluation partitions
    eval_df = parse_eval_file(eval_file)
    eval_output_path = os.path.join(output_folder, "eval_partitions.csv")
    eval_df.to_csv(eval_output_path, index=False)
    print(f"Evaluation partitions saved to: {eval_output_path}")

if __name__ == "__main__":
    main()