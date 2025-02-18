import os

# Get the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define directories
RAW_DATA_DIR = os.path.join(ROOT_DIR, "dataset", "low_resolution")  # Updated to match actual location
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")

# Path to the final merged dataset
MERGED_DATA_DIR = os.path.join(ROOT_DIR, "dataset", "organized_data", "csv_files")
FINAL_MERGED_CSV = os.path.join(MERGED_DATA_DIR, "final_merged_data.csv")

modelDir = os.path.join(ROOT_DIR, "models")
modelScript = os.path.join(modelDir, "fashion_cnn.py")

dataDir = os.path.join(ROOT_DIR, "utils")
dataScript = os.path.join(dataDir, "fashion_dataset.py")

# Ensure necessary directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MERGED_DATA_DIR, exist_ok=True)




