import pandas as pd

# Paths to the dataset files
train1_path = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Spambase/spambase_train1.csv'
train2_path = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Spambase/spambase_train2.csv'
test_path = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Spambase/spambase_test.csv'

# Load the datasets
train1 = pd.read_csv(train1_path)
train2 = pd.read_csv(train2_path)
test = pd.read_csv(test_path)

# Initialize a dictionary to hold bin edges for each feature
bin_edges = {}

# Labels for different numbers of bins
labels = {
    1: ["Low"],  # for one bin
    2: ["Low", "High"],  # for two bins
    3: ["Low", "Medium", "High"]  # for three bins
}

# Iterate over each numeric column in train1 to apply binning
for column in train1.select_dtypes(include=['number']).columns:
    unique_vals = train1[column].nunique()
    if unique_vals < 3:
        # Check distinct values for minimum, median, and maximum
        distinct_edges = sorted(set([train1[column].min(), train1[column].median(), train1[column].max()]))
        if len(distinct_edges) < 2:
            train1[column] = "Low"  # or some other single category handling
            bin_edges[column] = [train1[column].min(), train1[column].max()]
        else:
            # Use pd.cut with the distinct edges
            train1[column], bin_edges[column] = pd.cut(train1[column], bins=distinct_edges, labels=labels[len(distinct_edges)-1], retbins=True, include_lowest=True)
    else:
        # Use pd.qcut with dynamic label adjustment
        binned_data, returned_bins = pd.qcut(train1[column], q=3, duplicates='drop', retbins=True)
        bin_count = len(returned_bins) - 1  # Calculate actual bin count
        bin_labels = labels.get(bin_count, ["Low"] * (bin_count - 1) + ["High"])  # Default to simpler labels if not predefined
        train1[column] = pd.Categorical(binned_data, categories=bin_labels, ordered=True)
        bin_edges[column] = returned_bins

# Apply the binning to the secondary training and test data based on train1 bins
for column in bin_edges:
    train2[column] = pd.cut(train2[column], bins=bin_edges[column], labels=labels.get(len(bin_edges[column])-1, ["Low", "Medium", "High"]), include_lowest=True)
    test[column] = pd.cut(test[column], bins=bin_edges[column], labels=labels.get(len(bin_edges[column])-1, ["Low", "Medium", "High"]), include_lowest=True)

# Optional: Save the binned datasets back to CSV if needed
train1.to_csv(train1_path, index=False)
train2.to_csv(train2_path, index=False)
test.to_csv(test_path, index=False)

# Display to verify
print("Primary Training Data with Bins:")
print(train1.head())
print("\nSecondary Training Data with Bins:")
print(train2.head())
print("\nTest Data with Bins:")
print(test.head())
