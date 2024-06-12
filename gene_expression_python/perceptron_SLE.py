import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the count matrix
counts = pd.read_csv("C:/Users/victo/Downloads/count_matrix.txt.txt", sep='\t', index_col=0)

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

# Flatten the input for perceptron (fully connected layers)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Convert labels to categorical (one-hot encoding)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Define the perceptron model
input_length = X_train_flat.shape[1]
act_function = 'relu'
model = Sequential()
model.add(Dense(input_length, activation=act_function, input_shape=(input_length,)))
model.add(Dense(input_length, activation=act_function))
model.add(Dense(input_length, activation=act_function))
model.add(Dense(2, activation='softmax'))


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history_perceptron = model.fit(X_train_flat, y_train_cat, epochs=5, batch_size=32, validation_data=(X_test_flat, y_test_cat))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_flat, y_test_cat)
print(f'Perceptron Test Accuracy: {accuracy*100:.2f}%')