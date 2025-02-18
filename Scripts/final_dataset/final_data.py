import pandas as pd

# Load the dataset
file_path = "data/processed/final_merged/final_normalized.csv"
df = pd.read_csv(file_path)

# Identify attribute columns (starting from second column)
attribute_columns = df.columns[1:1001]  # Assuming the first column is not an attribute

# Convert values: 1 remains 1, -1 becomes 0, 0 can be treated as 0 (absent)
df[attribute_columns] = df[attribute_columns].replace({-1: 0, 0: 0})

# Print the last attribute column name to verify
print("Last attribute column:", attribute_columns[-1])

# Save the processed data
output_path = "data/processed/final_merged/final_data.csv"
df.to_csv(output_path, index=False)

print("Processed dataset saved at:", output_path)
