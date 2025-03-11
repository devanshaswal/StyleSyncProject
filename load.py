import pandas as pd
import numpy as np

# Load attribute mapping
attribute_mapping = pd.read_csv('data/processed/attribute_mapping.csv')

# Create a dictionary mapping attribute names to groups
group_dict = attribute_mapping.set_index('attribute_name')['group_name'].to_dict()

# Load metadata files
train_metadata = pd.read_csv('zsehran/metadata_updated_train.csv')
val_metadata = pd.read_csv('zsehran/metadata_updated_val.csv')
test_metadata = pd.read_csv('zsehran/metadata_updated_test.csv')

# Verify columns
print("Train columns:", train_metadata.columns.tolist())
print("Val columns:", val_metadata.columns.tolist())
print("Test columns:", test_metadata.columns.tolist())

# ==========================================================
# Add the attribute grouping script here
# ==========================================================

# Define attribute groups
ATTRIBUTE_GROUPS = {
    'color_print': ['red', 'pink', 'floral', 'striped', 'stripe', 'print', 'printed', 'graphic', 'love', 'summer'],
    'neckline': ['v-neck', 'hooded', 'collar', 'sleeveless', 'strapless', 'racerback', 'muscle'],
    'silhouette_fit': ['slim', 'boxy', 'fit', 'skinny', 'shift', 'bodycon', 'maxi', 'mini', 'midi', 'a-line'],
    'style_construction': ['crochet', 'knit', 'woven', 'lace', 'denim', 'cotton', 'chiffon', 'mesh'],
    'details': ['pleated', 'pocket', 'button', 'drawstring', 'trim', 'hem', 'capri', 'sleeve', 'flare', 'skater', 'sheath', 'shirt', 'pencil', 'classic', 'crop']
}

# Function to create group-to-column indices mapping
def create_group_to_indices(metadata, attribute_groups):
    group_to_indices = {}
    for group, attributes in attribute_groups.items():
        group_to_indices[group] = [
            metadata.columns.get_loc(attr) for attr in attributes
        ]
    return group_to_indices

# Create mappings for train, val, and test
train_group_to_indices = create_group_to_indices(train_metadata, ATTRIBUTE_GROUPS)
val_group_to_indices = create_group_to_indices(val_metadata, ATTRIBUTE_GROUPS)
test_group_to_indices = create_group_to_indices(test_metadata, ATTRIBUTE_GROUPS)

# Print the mappings to verify
print("Train Group to Indices:", train_group_to_indices)
print("Val Group to Indices:", val_group_to_indices)
print("Test Group to Indices:", test_group_to_indices)

# ==========================================================
# Continue with dataset creation and training
# ==========================================================