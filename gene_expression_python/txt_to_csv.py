import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Load the count matrix
counts = pd.read_csv('/Users/victorenglof/Downloads/GSE112087_counts-matrix-EnsembIDs-GRCh37.p10(1).txt', sep='\t', index_col=0)

sample_names = counts.columns
# Create a decision column based on sample names
labels = ['SLE' if 'MON_SLE' in sample else 'Control' for sample in sample_names]

# Convert to DataFrame for easy merging
labels_df = pd.DataFrame({'Sample': sample_names, 'Label': labels})
# Transpose the counts DataFrame to match sample names with rows
counts_T = counts.T
counts_T.index.name = 'Sample'
counts_T.reset_index(inplace=True)
# Merge counts with labels
counts_with_labels = pd.merge(counts_T, labels_df, on='Sample')

# Encode labels as 0 and 1
counts_with_labels['Label'] = counts_with_labels['Label'].apply(lambda x: 1 if x == 'SLE' else 0)

# Split the data into features (X) and labels (y)
X = counts_with_labels.drop(['Sample', 'Label'], axis=1).values
y = counts_with_labels['Label'].values

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape data for CNN input (samples, features, 1)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')

