import os

import pandas as pd


def clean_landmark_data():
    # Define paths
    data_dir = "dataset/organized_data/csv_files"
    landmarks_file = os.path.join(data_dir, "parsed_landmarks_cleaned.csv")
    
    # Load landmarks data
    parsed_landmarks_df = pd.read_csv(landmarks_file)
    
    # Handle missing values in visibility columns (replace NaN with 1 for occluded)
    visibility_cols = [col for col in parsed_landmarks_df.columns if "visibility" in col]
    parsed_landmarks_df[visibility_cols] = parsed_landmarks_df[visibility_cols].fillna(1)
    
    # Handle missing values in x/y coordinate columns (replace NaN with -1)
    coordinate_cols = [col for col in parsed_landmarks_df.columns if "location" in col]
    parsed_landmarks_df[coordinate_cols] = parsed_landmarks_df[coordinate_cols].fillna(-1)
    
    # Save the cleaned landmarks file
    cleaned_file = os.path.join(data_dir, "parsed_landmarks_cleaned.csv")
    parsed_landmarks_df.to_csv(cleaned_file, index=False)
    print(f"Landmark data cleaned and saved to {cleaned_file}")
    print(parsed_landmarks_df["variation_type"].unique())
if __name__ == "__main__":
    clean_landmark_data()


