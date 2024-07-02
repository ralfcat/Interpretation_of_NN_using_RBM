import pandas as pd
import numpy as np

path1 = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Gene_expression_cancer/data/cancer_genexp_train2_labeled.csv'
path2 = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Gene_expression_cancer/data/cancer_genexp_test_labeled.csv'

dat = pd.read_csv(path1)
dat2 = pd.read_csv(path2)

def equal_frequency_binning(dataset1, dataset2, output_path1, output_path2):
    # Remove 'diagnosis' and 'Prediction' temporarily
    diagnosis1 = dataset1['Class']
    prediction1 = dataset1['Prediction']
    diagnosis2 = dataset2['Class']
    prediction2 = dataset2['Prediction']
    
    dataset1 = dataset1.drop(columns=['Class', 'Prediction'])
    dataset2 = dataset2.drop(columns=['Class', 'Prediction'])
    
    binned_dataset1 = pd.DataFrame()
    binned_dataset2 = pd.DataFrame()

    for column in dataset1.columns:
        data_clean = dataset1[column].dropna()
        if data_clean.nunique() > 1:  # Ensure there are at least two unique values to form bins
            try:
                # Attempt to create 3 bins, adjust if duplicates cause fewer bins
                result, bins = pd.qcut(data_clean, q=3, duplicates='drop', retbins=True)
                bins = np.concatenate(([-float('inf')], bins[1:-1], [float('inf')]))  # Reformat bins to cover all data
                labels_needed = len(bins) - 1  # Calculate the number of labels needed
                labels = ['Low', 'Medium', 'High'][:labels_needed]  # Slice to fit number of bins
                binned_dataset1[column] = pd.cut(dataset1[column], bins=bins, labels=labels, include_lowest=True)
                binned_dataset2[column] = pd.cut(dataset2[column], bins=bins, labels=labels, include_lowest=True)
            except Exception as e:
                print(f"Error binning column {column}: {e}")
                continue
        else:
            # Assign unchanged if not enough unique values to bin
            binned_dataset1[column] = dataset1[column]
            binned_dataset2[column] = dataset2[column]

    # Reattach 'diagnosis' and 'Prediction'
    binned_dataset1['Class'] = diagnosis1
    binned_dataset2['Classc'] = diagnosis2
    binned_dataset1['Prediction'] = prediction1
    binned_dataset2['Prediction'] = prediction2

    binned_dataset1.to_csv(output_path1, index=False)
    binned_dataset2.to_csv(output_path2, index=False)
    print(f"Binned datasets saved to {output_path1} and {output_path2}")

    return binned_dataset1, binned_dataset2

# Define the output paths
output_path1 = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Gene_expression_cancer/data/cancer_genexp_train2_binned.csv'
output_path2 = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Gene_expression_cancer/data/cancer_genexp_test_binned.csv'

# Apply the binning function
binned_dat, binned_dat2 = equal_frequency_binning(dat, dat2, output_path1, output_path2)
