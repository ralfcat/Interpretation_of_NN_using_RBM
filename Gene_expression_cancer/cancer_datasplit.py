import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
path = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Gene_expression_cancer/data/cancer_GE_500_reduced.csv'
data = pd.read_csv(path)
# Check for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# No missing values detected, proceed without handling missing values

# Initial 70/30 split with stratification
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['Class'])

# Further split the 30% data into 2/3 and 1/3 with stratification
train2_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42, stratify=temp_data['Class'])

# Save the datasets to CSV files
train_data.to_csv('Gene_expression_cancer/data/cancer_genexp_train1', index=False)
train2_data.to_csv('Gene_expression_cancer/data/cancer_genexp_train2', index=False)
test_data.to_csv('Gene_expression_cancer/data/cancer_genexp_test', index=False)

print("Datasets created and saved as spambase_train1.csv, spambase_train2.csv, and spambase_test.csv")
