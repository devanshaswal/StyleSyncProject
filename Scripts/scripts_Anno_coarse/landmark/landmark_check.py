import pandas as pd

# Load the cleaned dataset
landmarks_cleaned_df = pd.read_csv("dataset/organized_data/parsed_landmarks_cleaned.csv")

# Check if any missing values are left
missing_values = landmarks_cleaned_df.isnull().sum()

print("Missing values after cleaning:")
print(missing_values[missing_values > 0])  # Only display columns with missing values
print(landmarks_cleaned_df.head(10))  # Check the first 10 rows



original_df = pd.read_csv("dataset/organized_data/parsed_landmarks.csv")
cleaned_df = pd.read_csv("dataset/organized_data/parsed_landmarks_cleaned.csv")

# Compare before and after for a specific row where NaN was present
row_index = 100  # Choose any row index to check
print("Original Data:\n", original_df.iloc[row_index])
print("Cleaned Data:\n", cleaned_df.iloc[row_index])
