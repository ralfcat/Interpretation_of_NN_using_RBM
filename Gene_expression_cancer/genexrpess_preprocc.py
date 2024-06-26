import pandas as pd

# Load the features and labels data
path_features = '/Users/victorenglof/Downloads/TCGA-PANCAN-HiSeq-801x20531/labels.csv'
path_labels = '/Users/victorenglof/Downloads/TCGA-PANCAN-HiSeq-801x20531/data.csv'

features_df = pd.read_csv(path_features, index_col=0)
labels_df = pd.read_csv(path_labels, index_col=0)

# Merge the dataframes on the index (sample identifier)
merged_df = features_df.merge(labels_df, left_index=True, right_index=True)

# Save the merged dataframe to a new CSV file
merged_df.to_csv('merged_dataset.csv')

print('Successfully merged dataframes! Keep on truckin')

