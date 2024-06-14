import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import sequence_unet.models as models
from tensorflow.keras.callbacks import EarlyStopping
import keras

print(tf.__version__)
print(keras.__version__)

#####################     TRAIN AND FINE TUNE THE SEQUENCE UNET MODEL (Patho_finetune)    ##################################
#####################                                                                     ##################################

original_file_path = 'C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7.csv'
original_data = pd.read_csv(original_file_path)
columns = original_data.columns
columns = columns[1:]
# Load your dataset
train_data = pd.read_csv('C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7_Train1.csv')
train_data.columns = columns


# Load your test dataset without headers and set the column names
test_file_path = 'C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7_Test.csv'
test_data = pd.read_csv(test_file_path, header=None)
test_data.columns = columns

test_file_path2 = 'C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7_Train2.csv'
test_data2 = pd.read_csv(test_file_path2, header=None)
test_data2.columns = columns

# Drop the 'id' column and separate features and labels for training data
X_train = train_data.drop(columns=['Pathogenicity'])
y_train = train_data['Pathogenicity']

# Drop the 'id' column and separate features and labels for test data
X_test = test_data.drop(columns=['Pathogenicity'])
y_test = test_data['Pathogenicity']

X_test2 = test_data2.drop(columns=['Pathogenicity'])
y_test2 = test_data2['Pathogenicity']

# Convert categorical features to numerical (one-hot encoding)
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # 20 standard amino acids
aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}

def one_hot_encode(sequence):
    encoded = np.zeros((len(sequence), len(amino_acids)))
    for i, aa in enumerate(sequence):
        if aa in aa_to_index:
            encoded[i, aa_to_index[aa]] = 1
    return encoded

X_train_encoded = np.array([one_hot_encode(seq) for seq in X_train.values])
X_test_encoded = np.array([one_hot_encode(seq) for seq in X_test.values])
X_test2_encoded = np.array([one_hot_encode(seq) for seq in X_test2.values])

# Add padding to ensure the sequence length is divisible by the model's downsampling factor
# Assuming the downsampling factor is 2^3=8 for a typical UNET
def add_padding(X_encoded):
    sequence_length = X_encoded.shape[1]
    target_length = ((sequence_length - 1) // 8 + 1) * 8
    padded_X_encoded = np.zeros((X_encoded.shape[0], target_length, X_encoded.shape[2]))
    padded_X_encoded[:, :sequence_length, :] = X_encoded
    return padded_X_encoded

padded_X_train_encoded = add_padding(X_train_encoded)
padded_X_test_encoded = add_padding(X_test_encoded)
padded_X_test2_encoded = add_padding(X_test2_encoded)

# Convert to TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((padded_X_train_encoded, y_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((padded_X_test_encoded, y_test)).batch(32)
test2_dataset = tf.data.Dataset.from_tensor_slices((padded_X_test2_encoded, y_test2)).batch(32)

# Define the fine-tuned model
class FineTunedSequenceUNET(Model):
    def __init__(self, base_model, num_classes, dropout_rate=0.5):
        super(FineTunedSequenceUNET, self).__init__()
        self.base_model = base_model
        self.flatten = Flatten()
        self.dropout = Dropout(dropout_rate)  # Dropout layer
        self.dense = Dense(num_classes, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.base_model(inputs)
        x = self.flatten(x)
        x = self.dropout(x, training=training)  # Apply dropout only during training
        return self.dense(x)

# Instantiate the fine-tuned model
num_classes = 1  # Binary classification (pathogenic or not)

# Perform cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_values = padded_X_train_encoded
y_values = y_train.values
""""
for fold, (train_index, val_index) in enumerate(kf.split(X_values), 1):
    print(f"Processing fold {fold}...")
    X_train_fold, X_val_fold = X_values[train_index], X_values[val_index]
    y_train_fold, y_val_fold = y_values[train_index], y_values[val_index]

    train_dataset_fold = tf.data.Dataset.from_tensor_slices((X_train_fold, y_train_fold)).batch(32)
    val_dataset_fold = tf.data.Dataset.from_tensor_slices((X_val_fold, y_val_fold)).batch(32)

    # Load the pre-trained model for each fold
    base_model = models.load_trained_model(model='patho_finetune', download=True)
    fold_model = FineTunedSequenceUNET(base_model, num_classes=num_classes, dropout_rate=0.5)
    fold_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    fold_model.fit(train_dataset_fold, validation_data=val_dataset_fold, epochs=20)

    val_loss, val_accuracy = fold_model.evaluate(val_dataset_fold)
    print(f'Fold {fold} - Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
"""
# Reinitialize and train a final model on the full training set
base_model = models.load_trained_model(model='patho_finetune', download=True)
final_model = FineTunedSequenceUNET(base_model, num_classes=num_classes, dropout_rate=0.5)
final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Setup early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
final_model.fit(train_dataset, epochs=20, validation_data=test_dataset, callbacks=[early_stopping])

# Evaluate the final model on the test set
test_loss, test_accuracy = final_model.evaluate(test_dataset)
test_loss2, test_accuracy2 = final_model.evaluate(test2_dataset)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
print(f'Test Loss: {test_loss2}, Test Accuracy: {test_accuracy2}')



##############  USE THE FINE-TUNED MODEL TO ASSIGN NEW LABELS TO DATA   #########################

# Load the dataset for which you want to assign new labels
new_data_file_path = 'C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7_Train2.csv'
new_data = pd.read_csv(new_data_file_path, header=None)
new_data.columns = columns

# Drop the 'id' column and separate features and labels for new data
X_new = new_data.drop(columns=['Pathogenicity'])
y_new = new_data['Pathogenicity']

# One-hot encode and pad the new data
X_new_encoded = np.array([one_hot_encode(seq) for seq in X_new.values])
padded_X_new_encoded = add_padding(X_new_encoded)

# Predict labels for the new dataset
predictions = final_model.predict(padded_X_new_encoded)
predicted_labels = (predictions > 0.5).astype(int).flatten()

# Add the predicted labels as a new column next to the real labels
new_data['Predicted_Pathogenicity'] = predicted_labels

# Save the dataset with the new predictions
new_data.to_csv('C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7_Train2_with_Predictions.csv', index=False)

print("New labels assigned and saved successfully.")