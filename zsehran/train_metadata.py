import pandas as pd

def update_metadata():
    # Define file paths
    metadata_path = "dataset/metadata/metadata_heatmaps.csv"
    final_data_path = "data/processed/final_merged/final_data.csv"
    output_path = "metadata_updated_test.csv"
    
    # Load the CSV files
    metadata_df = pd.read_csv(metadata_path)
    final_data_df = pd.read_csv(final_data_path)

    final_data_df = final_data_df[final_data_df['evaluation_status'] == 'test']
    
    # Define the required columns from final_data.csv
    top_50_attributes = [
        'mesh', 'slim', 'hem', 'hooded', 'trim', 'v-neck', 'boxy', 'collar',
        'woven', 'crochet', 'pocket', 'pleated', 'button', 'sleeveless',
        'strapless', 'capri', 'a-line', 'drawstring', 'fit', 'classic',
        'printed', 'love', 'summer', 'stripe', 'red', 'midi', 'pink', 'crop',
        'sleeve', 'floral', 'cotton', 'striped', 'knit', 'print', 'chiffon',
        'racerback', 'mini', 'lace', 'skater', 'flare', 'sheath', 'denim',
        'shirt', 'pencil', 'graphic', 'skinny', 'shift', 'bodycon', 'muscle',
        'maxi', 'category_label', 'category_type'
    ]
    
    # Ensure only the necessary columns exist in final_data_df
    selected_columns = ['image_name'] + [col for col in top_50_attributes if col in final_data_df.columns]
    final_data_df = final_data_df[selected_columns]
    
    # Merge metadata_heatmaps.csv with the selected attributes from final_data.csv based on image_name
    merged_df = pd.merge(metadata_df, final_data_df, on='image_name', how='inner')
    
    # Save the updated metadata to the new file
    merged_df.to_csv(output_path, index=False)
    print(f"Updated metadata saved to {output_path}")

# Run the function
update_metadata()
