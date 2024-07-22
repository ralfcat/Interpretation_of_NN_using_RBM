import pandas as pd

# Load your data
path = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Gene_expression_cancer/data/cancer_genexp_train2'
data = pd.read_csv(path)

# Create a condition to find rows where the label is 'BRCA'
conditions = (data['Class'] == 'BRCA')

# Modify the 'gene_18746' values where the condition is True
data.loc[conditions, 'gene_17801'] = 0

# Define the output path for the modified dataset
output_path = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Gene_expression_cancer/data/dummydata_train2.csv'

# Save the modified dataset to a new CSV file
data.to_csv(output_path, index=False)

print(f"Modified dataset saved to {output_path}")
