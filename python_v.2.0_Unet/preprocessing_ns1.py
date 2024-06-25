import pandas as pd
from sklearn.model_selection import train_test_split

# Load the main dataset
main_file_path = 'C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7.csv'
main_data = pd.read_csv(main_file_path)

# Define the target column for stratification
target_column = 'Pathogenicity'

# Split the data while retaining class balance
# First split: Create dataset1 with 549 entries
dataset1, temp_data = train_test_split(main_data, train_size=549, stratify=main_data[target_column], random_state=42)

# Second split: Create dataset2 with 250 entries from the remaining data
dataset2, dataset3 = train_test_split(temp_data, train_size=250, stratify=temp_data[target_column], random_state=42)

# The remaining data will be used for dataset3 with 101 entries
# Confirm the length of dataset3
assert len(dataset3) == 101, f"Expected 101 entries in dataset3, but got {len(dataset3)}"

# Remove the first row (column names) and the first column "id" from each dataset
dataset1 = dataset1.iloc[1:].drop(columns=['id'])
dataset2 = dataset2.iloc[1:].drop(columns=['id'])
dataset3 = dataset3.iloc[1:].drop(columns=['id'])

# Save the datasets to CSV files without header
dataset1.to_csv('C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7_dataset1.csv', index=False, header=False)
dataset2.to_csv('C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7_dataset2.csv', index=False, header=False)
dataset3.to_csv('C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7_dataset3.csv', index=False, header=False)

print("Datasets created and saved successfully.")
