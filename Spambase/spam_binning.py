import pandas as pd
import numpy as np
path1 = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Spambase/spambase_train2_labeled.csv'
path2 = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Spambase/spambase_test_labeled.csv'


dat = pd.read_csv(path1)
dat2 = pd.read_csv(path2)

def custom_binning(dataset1, dataset2, output_path1, output_path2):
    binned_dataset1 = pd.DataFrame()
    binned_dataset2 = pd.DataFrame()
    columns = dataset1.columns[:-2]  # Exclude the last column

    for i, column in enumerate(columns):
        print(f"Binning feature {i+1}/{len(columns)}")
        try:
            # Create unique bin edges considering the distribution
            unique_values = np.sort(dataset1[column].unique())
            if len(unique_values) == 1:
                # Only one unique value, create a single bin
                bin_edges = [-np.inf, np.inf]
                labels = ['Single Value']
            elif len(unique_values) == 2:
                # Two unique values, usually 0 and one other, make two bins
                bin_edges = [-np.inf, unique_values[1], np.inf]
                labels = ['Zero', 'Non-zero']
            else:
                # More than two unique values, create bins based on quartiles including zero handling
                zero_bin_edge = [0] if 0 in unique_values else []
                non_zero_values = dataset1[column][dataset1[column] > 0]
                non_zero_bin_edges = np.percentile(non_zero_values, [33, 66])
                
                # Combine edges
                bin_edges = [-np.inf] + zero_bin_edge + non_zero_bin_edges.tolist() + [np.inf]
                labels = ['Zero', 'Low', 'Medium', 'High'][0:len(bin_edges)-1]

            # Apply the binning
            binned_dataset1[column] = pd.cut(dataset1[column], bins=bin_edges, labels=labels, include_lowest=True)
            binned_dataset2[column] = pd.cut(dataset2[column], bins=bin_edges, labels=labels, include_lowest=True)
        except ValueError as e:
            print(f"Skipping feature {column} due to binning issue: {e}")

    # Copy the 'spam' column as is to the new DataFrame
    binned_dataset1['spam'] = dataset1['spam']
    binned_dataset2['spam'] = dataset2['spam']

    binned_dataset1['Predicted_spam'] = dataset1['Predicted_spam']
    binned_dataset2['Predicted_spam'] = dataset2['Predicted_spam']

    print(f"Binning completed")

    # Save the binned datasets to CSV files
    binned_dataset1.to_csv(output_path1, index=False)
    binned_dataset2.to_csv(output_path2, index=False)
    print(f"Binned datasets saved to {output_path1} and {output_path2}")

    return binned_dataset1, binned_dataset2

output_path1 = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Spambase/spambase_train2_binned.csv'
output_path2 = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Spambase/spambase_test_binned.csv'


binned_dat, binned_dat2 = custom_binning(dat, dat2, output_path1, output_path2)


