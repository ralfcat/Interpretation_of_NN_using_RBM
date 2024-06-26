import pandas as pd

# Load your dataset
# Assuming the dataset is in a CSV file, you can load it like this:
df = pd.read_csv("merged_dataset.csv")

# Drop the first column (assuming it's the sample ID column)
df = df.iloc[:, 1:]

# Check if there are enough rows and if 'Class' column exists
if 'Class' in df.columns and df.shape[0] > 50:
    # Include the 'Class' column with the first 50 features
    # Adjust column selection considering the first column (IDs) has been dropped
    selected_columns = df.columns[:25].tolist() + ['Class']  # Select the new first 50 columns and add 'Class'
    selected_columns = list(dict.fromkeys(selected_columns))  # Remove duplicates if 'Class' is in the first 50

    # Select the first 150 samples (rows) and the chosen features (columns)
    df_subset = df.loc[:49, selected_columns]
else:
    print("The dataset does not have enough rows, or the 'Class' column is missing. Please check your dataset.")

# Now you can work with df_subset, which has 150 samples and includes 'Class' with other 50 features
# Optionally, you might want to save this subset to a new file
df_subset.to_csv("subset_dataset.csv", index=False)
