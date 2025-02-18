import os

import pandas as pd

# Define paths
data_dir = "dataset/organized_data/"
input_file = os.path.join(data_dir, "final_merged_data.csv")
output_train = os.path.join(data_dir, "train_set.csv")
output_val = os.path.join(data_dir, "val_set.csv")
output_test = os.path.join(data_dir, "test_set.csv")

# Load the final merged dataset
final_merged_df = pd.read_csv(input_file)
 
# Check unique values in the evaluation_status column
print("Unique values in evaluation_status:", final_merged_df["evaluation_status"].unique())

# Split the dataset into train, validation, and test sets
train_set = final_merged_df[final_merged_df["evaluation_status"] == "train"]
val_set = final_merged_df[final_merged_df["evaluation_status"] == "val"]
test_set = final_merged_df[final_merged_df["evaluation_status"] == "test"]

# Verify the sizes of the splits
print("\nDataset Sizes:")
print("Train set:", train_set.shape)
print("Validation set:", val_set.shape)
print("Test set:", test_set.shape)

# Save the splits to separate CSV files
train_set.to_csv(output_train, index=False)
val_set.to_csv(output_val, index=False)
test_set.to_csv(output_test, index=False)

print("\n Datasets saved to:")
print(f"- Train set: {output_train}")
print(f"- Validation set: {output_val}")
print(f"- Test set: {output_test}")