import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import sequence_unet.models as models

# Load your dataset
data = pd.read_csv('/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7.csv')

# Drop the 'id' column and separate features and labels
X = data.drop(columns=['id', 'Pathogenicity'])
y = data['Pathogenicity']

# Convert categorical features to numerical (one-hot encoding)
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # 20 standard amino acids
aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}

def one_hot_encode(sequence):
    encoded = np.zeros((len(sequence), len(amino_acids)))
    for i, aa in enumerate(sequence):
        if aa in aa_to_index:
            encoded[i, aa_to_index[aa]] = 1
    return encoded

X_encoded = np.array([one_hot_encode(seq) for seq in X.values])

# Add padding to ensure the sequence length is divisible by the model's downsampling factor
# Assuming the downsampling factor is 2^3=8 for a typical UNET
sequence_length = X_encoded.shape[1]
target_length = ((sequence_length - 1) // 8 + 1) * 8
padded_X_encoded = np.zeros((X_encoded.shape[0], target_length, X_encoded.shape[2]))
padded_X_encoded[:, :sequence_length, :] = X_encoded

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_X_encoded, y, test_size=0.2, random_state=42)

# Convert to TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

# Download the pre-trained model
#model_path = models.download_trained_model("patho_finetune", root="models", model_format="tf")

# Load the pre-trained model
model = models.load_trained_model(model='patho_finetune', download=True)

# Define the fine-tuned model
class FineTunedSequenceUNET(Model):
    def __init__(self, base_model, num_classes):
        super(FineTunedSequenceUNET, self).__init__()
        self.base_model = base_model
        self.flatten = Flatten()
        self.dense = Dense(num_classes, activation='sigmoid')  # Use 'sigmoid' for binary classification

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.flatten(x)
        return self.dense(x)

# Instantiate the fine-tuned model
fine_tuned_model = FineTunedSequenceUNET(model, num_classes=1)  # Binary classification (pathogenic or not)

# Compile the model
fine_tuned_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Perform cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_values = padded_X_encoded
y_values = y.values

for train_index, val_index in kf.split(X_values):
    X_train_fold, X_val_fold = X_values[train_index], X_values[val_index]
    y_train_fold, y_val_fold = y_values[train_index], y_values[val_index]

    train_dataset_fold = tf.data.Dataset.from_tensor_slices((X_train_fold, y_train_fold)).batch(32)
    val_dataset_fold = tf.data.Dataset.from_tensor_slices((X_val_fold, y_val_fold)).batch(32)

    fine_tuned_model.fit(train_dataset_fold, validation_data=val_dataset_fold, epochs=50)

    val_loss, val_accuracy = fine_tuned_model.evaluate(val_dataset_fold)
    print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

# Final training on the full training set
fine_tuned_model.fit(train_dataset, epochs=50)

# Evaluate the model on the test set
test_loss, test_accuracy = fine_tuned_model.evaluate(test_dataset)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
