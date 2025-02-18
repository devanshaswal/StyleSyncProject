import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load the dataset
file_path = "data/processed/final_merged/final_normalized.csv"
df = pd.read_csv(file_path)

# Get the attribute columns (all columns except image_name and category labels)
attribute_columns = df.columns[1:1001]  # Assuming first column is non-attribute

# Check how many attributes are present
print(f"Total attributes: {len(attribute_columns)}")
print(f"Sample attributes: {attribute_columns[:10]}")


from sklearn.feature_selection import VarianceThreshold

# Select only attribute columns
X = df.iloc[:, 1:1001]  # Assuming attributes start from column 1 to 1000

# Remove attributes with variance lower than 0.01
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)

# Get filtered attribute names
selected_attributes = X.columns[selector.get_support()]

print(f"Remaining attributes after variance filtering: {len(selected_attributes)}")



from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Ensure category labels are available
if "category_label" in df.columns:
    y = df["category_label"]
    
    # Train a Random Forest model to assess feature importance
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df[selected_attributes], y)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Select top 50 attributes
    top_n = 50  # You can adjust this number
    top_indices = np.argsort(importances)[-top_n:]  # Indices of top attributes
    top_attributes = selected_attributes[top_indices]  # Names of top attributes

    print(f"Top {top_n} attributes:\n", top_attributes)
