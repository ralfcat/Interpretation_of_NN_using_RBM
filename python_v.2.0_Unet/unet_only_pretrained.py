import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sequence_unet.models as models

# Load your dataset
data = pd.read_csv('C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7.csv')

# Extract features and labels
X = data.drop(columns=['id', 'Pathogenicity'])
y = data['Pathogenicity']

# Concatenate all columns of sequences into a single sequence string for each row
X_sequences = X.apply(lambda row: ''.join(row.values), axis=1)

# Convert categorical features to numerical (one-hot encoding)
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}

def one_hot_encode(sequence):
    encoded = np.zeros((len(sequence), len(amino_acids)))
    for i, aa in enumerate(sequence):
        if aa in aa_to_index:
            encoded[i, aa_to_index[aa]] = 1
    return encoded

X_encoded = np.array([one_hot_encode(seq) for seq in X_sequences])

# Pad sequences
sequence_length = X_encoded.shape[1]
target_length = ((sequence_length - 1) // 8 + 1) * 8
padded_X_encoded = np.zeros((X_encoded.shape[0], target_length, X_encoded.shape[2]))
padded_X_encoded[:, :sequence_length, :] = X_encoded

# Reshape y to match the output shape of the model (binary classification requires just 1 output)
y = y.values.reshape(-1, 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_X_encoded, y, test_size=0.2, random_state=42)

# Convert to TensorFlow Datasets
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

# Load the pre-trained model
model = models.load_trained_model(model='patho_finetune', download=True)

# Ensure the model is compiled with the correct loss and metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Evaluate the pre-trained model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')



