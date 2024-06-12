# train_model.py

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from model_utils import MSA_encode, load_sequence_unet_model, AA_dict

# Function to pad sequences to the required length
def pad_sequences(sequences, required_length):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < required_length:
            pad_width = required_length - len(seq)
            seq = np.pad(seq, ((0, pad_width), (0, 0)), 'constant')
        padded_sequences.append(seq)
    return np.array(padded_sequences)

# Hyperparameters
params = {
    'mode': 'single',
    'learning_rate': 0.00005,
    'lossFunction': 'sparse_categorical_crossentropy',
    'epochs': 50
}

train_data_file = './data/NS1/NS1_H5_H7_Train2.csv'
# Load the training data
MSA_df = pd.read_csv(train_data_file, header=None)
MSA = MSA_df.to_numpy()
X = MSA_encode(MSA, AA_dict)
y = MSA[:, -1].astype(int)

# Calculate the required length for padding
required_length = X.shape[1] + (64 - X.shape[1] % 64) if X.shape[1] % 64 != 0 else X.shape[1]

# Pad sequences
X = pad_sequences(X, required_length)

# Check shapes of X and y
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Ensure correct input shape
X = np.expand_dims(X, axis=-1) if X.ndim == 2 else X

# Perform 10-fold cross-validation
kf = KFold(n_splits=10)
final_conf_matrix = np.zeros((2, 2))
accuracies = []

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Check shapes before training
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of X_val: {X_val.shape}")
    print(f"Shape of y_val: {y_val.shape}")

    # Load the Sequence UNET model
    model = load_sequence_unet_model("freq_classifier", root="/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM")

    # Check model summary
    model.summary()

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                  loss=params['lossFunction'],
                  metrics=['accuracy'])

    # Ensure batch size is correctly set
    batch_size = 32

    # Flatten labels if necessary
    y_train = y_train.flatten()
    y_val = y_val.flatten()

    # Debugging: Check shapes after reshaping
    print(f"Reshaped y_train: {y_train.shape}")
    print(f"Reshaped y_val: {y_val.shape}")

    # Train the model
    history = model.fit(X_train, y_train, epochs=params['epochs'], batch_size=batch_size)

    # Evaluate the model on validation data
    val_preds = model.predict(X_val)
    val_preds = np.argmax(val_preds, axis=-1)
    acc = accuracy_score(y_val, val_preds)
    accuracies.append(acc)
    final_conf_matrix += confusion_matrix(y_val, val_preds)

    # Debugging information
    print(f"Fold {fold+1} - Accuracy: {acc:.4f}")
    print(f"Confusion Matrix for Fold {fold+1}:\n{confusion_matrix(y_val, val_preds)}")

# Print final metrics
print("Final Confusion Matrix from 10-Fold CV:")
print(final_conf_matrix)
print("Accuracies from 10-Fold CV:", accuracies)
print("Mean Accuracy from 10-Fold CV:", np.mean(accuracies))

# Save the final model and dictionary
model.save("path/to/save/your_model.h5")
with open('dict.pkl', 'wb') as f:
    pickle.dump(AA_dict, f)

print("Model and dictionary saved.")
