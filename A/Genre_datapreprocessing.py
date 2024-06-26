import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load your data
file_path = 'C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/BC/features_30_sec.csv'
data = pd.read_csv(file_path)

# Drop any non-numeric or unnecessary columns
data = data.drop(columns=['filename', 'length'])

# Initial split - half of the data
train_data, temp_data = train_test_split(data, test_size=0.5, stratify=data['label'], random_state=42)

# Second split - half of the remaining data
second_set, third_set = train_test_split(temp_data, test_size=0.5, stratify=temp_data['label'], random_state=42)

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit the scaler on the training data only and transform it
train_data.loc[:, train_data.columns != 'label'] = scaler.fit_transform(train_data.loc[:, train_data.columns != 'label'])

# Transform the second and third datasets using the same scaler
second_set.loc[:, second_set.columns != 'label'] = scaler.transform(second_set.loc[:, second_set.columns != 'label'])
third_set.loc[:, third_set.columns != 'label'] = scaler.transform(third_set.loc[:, third_set.columns != 'label'])

#Checking label distribution
label_counts = train_data['label'].value_counts()
print(f'---Training \n {label_counts}')

label_counts = second_set['label'].value_counts()
print(f'---Second \n {label_counts}')

label_counts = third_set['label'].value_counts()
print(f'---Third  \n {label_counts}')


# Export the normalized data to CSV files
train_data.to_csv('C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/BC/train_data_normalized.csv', index=False)
second_set.to_csv('C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/BC/second_set_normalized.csv', index=False)
third_set.to_csv('C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/BC/third_set_normalized.csv', index=False)

print("Normalized files saved:")
print("Training set: train_data_normalized.csv")
print("Second set: second_set_normalized.csv")
print("Third set: third_set_normalized.csv")
