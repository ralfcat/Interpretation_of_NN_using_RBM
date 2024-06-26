import pandas as pd

# Load the features and labels data
features_df = pd.read_csv('C:/Users/victo/Downloads/gene+expression+cancer+rna+seq/TCGA-PANCAN-HiSeq-801x20531/data.csv', index_col=0)
labels_df = pd.read_csv('C:/Users/victo/Downloads/gene+expression+cancer+rna+seq/TCGA-PANCAN-HiSeq-801x20531/labels.csv', index_col=0)

# Merge the dataframes on the index (sample identifier)
merged_df = features_df.merge(labels_df, left_index=True, right_index=True)

# Save the merged dataframe to a new CSV file
merged_df.to_csv('merged_dataset.csv')

print('Successfully merged dataframes! Keep on truckin')

