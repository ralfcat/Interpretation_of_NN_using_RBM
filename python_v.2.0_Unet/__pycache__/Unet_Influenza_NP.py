import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import sequence_unet.models as models
import keras
import random
print(tf.__version__)
print(keras.__version__)

#random.seed(0)
#####################     TRAIN AND FINE TUNE THE SEQUENCE UNET MODEL (Patho_finetune)    ##################################
#####################                                                                     ##################################

# Load the dataset with tab as the delimiter
file_path = 'C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NP/Project1.csv'
train_data = pd.read_csv(file_path, delimiter='\t')

# Clean column names to remove any leading/trailing whitespace
train_data.columns = train_data.columns.str.strip()

# Determine the class distribution
class_counts = train_data['Pathogenicity'].value_counts()

# Calculate the number of samples needed for each class to retain balance
sample_counts = (class_counts / 3).astype(int)

# Initialize an empty DataFrame for the new test dataset
sampled_data = pd.DataFrame()

# Sample from each class and remove these instances from the training set
for class_label, count in sample_counts.items():
    class_data = train_data[train_data['Pathogenicity'] == class_label]
    sampled_class_data = class_data.sample(count, random_state=42)
    sampled_data = pd.concat([sampled_data, sampled_class_data])
    train_data = train_data.drop(sampled_class_data.index)

# Shuffle the new test dataset
sampled_data = sampled_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the new test dataset
sampled_data.to_csv('C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NP/NewData_willbe_labeled.csv', index=False)

# Save the updated training dataset
train_data.to_csv('C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NP/Updated_TrainData.csv', index=False)
test_data = pd.read_csv('C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NP/NewData_willbe_labeled.csv')


# Drop the 'id' column and separate features and labels for training data
X_train = train_data.drop(columns=['id','Pathogenicity'])
y_train = train_data['Pathogenicity']

# Drop the 'id' column and separate features and labels for test data
X_test = test_data.drop(columns=['Pathogenicity'])
y_test = test_data['Pathogenicity']

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

# Convert to TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((padded_X_train_encoded, y_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((padded_X_test_encoded, y_test)).batch(32)
# Define the fine-tuned model

## THIS WORKS !!!! 

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

# TRYING TO ADD MORE LAYERS


"""
class FineTunedSequenceUNET(Model):
    def __init__(self, base_model, num_classes, dropout_rate=0.5):
        super(FineTunedSequenceUNET, self).__init__()
        self.base_model = base_model
        self.flatten = Flatten()
        self.dense1 = Dense(256, activation='relu', kernel_regularizer=l2(0.01))
        self.bn1 = BatchNormalization()
        self.dropout1 = Dropout(dropout_rate)
        self.dense2 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))
        self.bn2 = BatchNormalization()
        self.dropout2 = Dropout(dropout_rate)
        self.dense3 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))
        self.bn3 = BatchNormalization()
        self.dropout3 = Dropout(dropout_rate)
        self.output_dense = Dense(num_classes, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.base_model(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        x = self.bn3(x, training=training)
        x = self.dropout3(x, training=training)
        return self.output_dense(x)
"""
# Learning rate scheduler function
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
# Instantiate the fine-tuned model
num_classes = 1  # Binary classification (pathogenic or not)

# Perform cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
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

"""
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

    lr_scheduler = LearningRateScheduler(scheduler)

    fold_model.fit(train_dataset_fold, validation_data=val_dataset_fold, epochs=50, callbacks=[lr_scheduler])

    val_loss, val_accuracy = fold_model.evaluate(val_dataset_fold)
    print(f'Fold {fold} - Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
"""
# Reinitialize and train a final model on the full training set
base_model = models.load_trained_model(model='patho_finetune', download=True)
final_model = FineTunedSequenceUNET(base_model, num_classes=num_classes, dropout_rate=0.5)
final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Setup early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(scheduler)

final_model.fit(train_dataset, epochs=50, validation_data = test_dataset, callbacks=[early_stopping, lr_scheduler])

test_loss, test_accuracy = final_model.evaluate(test_dataset)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
# Evaluate the final model on the test set


##############  USE THE FINE-TUNED MODEL TO ASSIGN NEW LABELS TO DATA   #########################

# Load the dataset for which you want to assign new labels
new_data_file_path = 'C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NP/NewData_willbe_labeled.csv'
new_data = pd.read_csv(new_data_file_path)


# Drop the 'id' column and separate features and labels for new data
X_new = new_data.drop(columns=['id','Pathogenicity'])
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
labled_data_path = "C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NP/Predicted_labels_NP.csv"
new_data.to_csv(labled_data_path, index=False)

print("New labels assigned and saved successfully.")