import os
import pandas as pd

# Define paths
input_file_path = os.path.join("dataset", "Annotation", "Anno_fine", "test_bbox.txt")
output_folder = os.path.join("dataset", "organized_data_1")
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist
output_csv_path = os.path.join(output_folder, "test_bbox.csv")

def parse_test_bbox(file_path):
    """
    Parses the test_bbox.txt file and returns a DataFrame with bounding box coordinates.

    Args:
        file_path (str): Path to the test_bbox.txt file.

    Returns:
        pd.DataFrame: A DataFrame with columns [x_1, y_1, x_2, y_2].
    """
    with open(file_path, 'r') as file:
        # Read all lines
        lines = file.readlines()

        # Parse bounding box coordinates
        data = []
        for line in lines:
            # Split the line into parts and convert to integers
            parts = line.strip().split()
            x1, y1, x2, y2 = map(int, parts)
            data.append([x1, y1, x2, y2])

    # Create a DataFrame
    df = pd.DataFrame(data, columns=["x_1", "y_1", "x_2", "y_2"])
    return df

# Parse the file
df = parse_test_bbox(input_file_path)

# Save to CSV
df.to_csv(output_csv_path, index=False)
print(f"Parsed data saved to {output_csv_path}")
print(df.head())  # Display the first few rows