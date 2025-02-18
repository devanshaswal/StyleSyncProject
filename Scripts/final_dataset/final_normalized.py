import pandas as pd
import os

# Define file paths
input_file = "data/processed/final_merged/final_merged_noprefix.csv"
output_file = "data/processed/final_merged/final_normalized.csv"

# Load the CSV file
df = pd.read_csv(input_file)

# Normalize the 'image_name' column
df['image_name'] = df['image_name'].str.replace(r'\\', '/', regex=True)

# Save the modified DataFrame to a new CSV file
df.to_csv(output_file, index=False)

print(f"Normalized file saved to: {output_file}")
