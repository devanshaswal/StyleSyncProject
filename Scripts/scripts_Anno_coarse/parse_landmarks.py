import numpy as np
import pandas as pd


def parse_landmark_file (file_path, output_csv_path):
    # Define the header with triplet order for each landmark
    header = ["image_name", "clothes_type", "variation_type"]
    for i in range(1, 9):
        header.extend([
            f"landmark_visibility_{i}",
            f"landmark_location_x_{i}",
            f"landmark_location_y_{i}"
        ])

    data = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip first two lines (number of images and header)
    for line_num, line in enumerate(lines[2:], start=3):
        parts = line.strip().split()

        # Skip invalid lines
        if len(parts) < 3:
            print(f"Skipping line {line_num}: insufficient data")
            continue

        try:
            image_name = parts[0]
            clothes_type = int(parts[1])
            variation_type = int(parts[2])
        except ValueError:
            print(f"Skipping line {line_num}: invalid clothes/variation type")
            continue

        # Determine expected number of landmarks based on clothes_type
        if clothes_type == 1:
            num_landmarks = 6  # Upper-body: 6 landmarks
        elif clothes_type == 2:
            num_landmarks = 4  # Lower-body: 4 landmarks
        elif clothes_type == 3:
            num_landmarks = 8  # Full-body: 8 landmarks
        else:
            print(f"Skipping line {line_num}: invalid clothes_type {clothes_type}")
            continue

        # Initialize landmark data with NaN for all 8 landmarks
        landmark_data = [np.nan] * 24  # 8 landmarks * 3 attributes

        # Parse valid landmarks based on clothes_type
        for i in range(num_landmarks):
            start_idx = 3 + i * 3
            if start_idx + 2 < len(parts):
                try:
                    # Convert visibility, x, y to integers
                    vis = int(parts[start_idx])
                    x = int(parts[start_idx + 1])
                    y = int(parts[start_idx + 2])
                except (ValueError, IndexError):
                    vis = x = y = np.nan
            else:
                vis = x = y = np.nan

            # Fill data into the correct positions
            pos = 3 * i
            landmark_data[pos] = vis
            landmark_data[pos + 1] = x
            landmark_data[pos + 2] = y

        # Combine into a row
        row = [image_name, clothes_type, variation_type] + landmark_data
        data.append(row)

    # Create DataFrame with Int64 dtype (nullable integers)
    df = pd.DataFrame(data, columns=header)
    
    # Convert landmark columns to nullable integers (Int64)
    for col in df.columns[3:]:  # Skip first 3 columns (strings/ints)
        df[col] = df[col].astype(pd.Int64Dtype())  # Allows NaN for missing landmarks

    df.to_csv(output_csv_path, index=False)
    print(f"Landmark data saved to {output_csv_path}")
    return df

def main():
    file_path = "dataset\\Annotation\\Anno_coarse\\list_landmarks.txt"
    output_csv_path = "dataset\\organized_data\\csv_files\\parsed_landmarks.csv"
    df = parse_landmark_file(file_path, output_csv_path)
    print(df.head())

if __name__ == "__main__":
    main()


