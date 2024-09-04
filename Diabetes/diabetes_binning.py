import pandas as pd

# Function to apply custom binning based on provided descriptions
def custom_binning(df):
    # Apply custom bins based on the provided feature descriptions

    # Mapping for 'GenHlth' (General Health)
    genhlth_mapping = {
        1: 'excellent',
        2: 'very good',
        3: 'good',
        4: 'fair',
        5: 'poor'
    }
    df['GenHlth'] = df['GenHlth'].map(genhlth_mapping)

    # Mapping for 'MentHlth' (Mental Health) and 'PhysHlth' (Physical Health)
    ment_phys_mapping = {
        range(1, 11): 'low',
        range(11, 21): 'medium',
        range(21, 31): 'high'
    }
    df['MentHlth'] = df['MentHlth'].apply(lambda x: next(v for k, v in ment_phys_mapping.items() if x in k))
    df['PhysHlth'] = df['PhysHlth'].apply(lambda x: next(v for k, v in ment_phys_mapping.items() if x in k))

    # Mapping for 'Age' into low, medium, high
    age_mapping = {
        range(1, 5): 'Low',   # 18-39
        range(5, 10): 'Medium',  # 40-64
        range(10, 14): 'High'  # 65+
    }
    df['Age'] = df['Age'].apply(lambda x: next(v for k, v in age_mapping.items() if x in k))

    # Mapping for 'Education'
    education_mapping = {
        1: 'Never attended school or only kindergarten',
        2: 'Grades 1 through 8 (Elementary)',
        3: 'Grades 9 through 11 (Some high school)',
        4: 'Grade 12 or GED (High school graduate)',
        5: 'College 1 year to 3 years (Some college or technical school)',
        6: 'College 4 years or more (College graduate)'
    }
    df['Education'] = df['Education'].map(education_mapping)

    # Mapping for 'Income'
    income_mapping = {
        1: 'less than $10,000',
        2: '$10,000 to less than $15,000',
        3: '$15,000 to less than $20,000',
        4: '$20,000 to less than $25,000',
        5: '$25,000 to less than $35,000',
        6: '$35,000 to less than $50,000',
        7: '$50,000 to less than $75,000',
        8: '$75,000 or more'
    }
    df['Income'] = df['Income'].map(income_mapping)

    return df

# Load the datasets with predicted labels
val_set_with_predictions = pd.read_csv('val_set_with_predictions.csv')
test_set_with_predictions = pd.read_csv('test_set_with_predictions.csv')

# Apply custom binning to both datasets
binned_val_set = custom_binning(val_set_with_predictions)
binned_test_set = custom_binning(test_set_with_predictions)

# Save the binned datasets to new CSV files
binned_val_set.to_csv('binned_val_set_with_custom_bins.csv', index=False)
binned_test_set.to_csv('binned_test_set_with_custom_bins.csv', index=False)

print("Custom binning based on provided descriptions has been applied, and the datasets have been saved.")
