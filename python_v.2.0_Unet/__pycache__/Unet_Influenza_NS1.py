import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import sequence_unet.models as models
import keras
import matplotlib.pyplot as plt
import random

print(tf.__version__)
print(keras.__version__)

# Random seed
# random.seed(0)

# Load the main dataset
original_file_path = 'C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7.csv'
original_data = pd.read_csv(original_file_path)
columns = original_data.columns
columns = columns[1:]

# Load the training dataset
train_data = pd.read_csv('C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7_dataset1.csv')
train_data.columns = columns

# Load the test datasets
test_file_path = 'C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7_dataset3.csv'
test_data = pd.read_csv(test_file_path, header=None)
test_data.columns = columns

test_file_path2 = 'C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7_dataset2.csv'
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

# Split the training data into training and validation sets
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# One-hot encoding for amino acid sequences, without handling gaps
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # 20 standard amino acids
aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}

def one_hot_encode(sequence):
    encoded = np.zeros((len(sequence), len(amino_acids)))  # Keep the encoding to 20 channels
    for i, aa in enumerate(sequence):
        if aa in aa_to_index:
            encoded[i, aa_to_index[aa]] = 1
    return encoded

X_train_encoded = np.array([one_hot_encode(seq) for seq in X_train.values])
X_test_encoded = np.array([one_hot_encode(seq) for seq in X_test.values])
X_test2_encoded = np.array([one_hot_encode(seq) for seq in X_test2.values])
X_trainz, X_valz, y_trainz, y_valz = train_test_split(X_train_encoded, y_train, test_size=0.2, random_state=42)

def add_padding(X_encoded):
    sequence_length = X_encoded.shape[1]
    target_length = ((sequence_length - 1) // 8 + 1) * 8
    padded_X_encoded = np.zeros((X_encoded.shape[0], target_length, X_encoded.shape[2]))
    padded_X_encoded[:, :sequence_length, :] = X_encoded
    return padded_X_encoded

padded_X_train_encoded = add_padding(X_trainz)
padded_X_val_encoded = add_padding(X_valz)
padded_X_test_encoded = add_padding(X_test_encoded)
padded_X_test2_encoded = add_padding(X_test2_encoded)

train_dataset = tf.data.Dataset.from_tensor_slices((padded_X_train_encoded, y_trainz)).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((padded_X_val_encoded, y_valz)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((padded_X_test_encoded, y_test)).batch(32)
test2_dataset = tf.data.Dataset.from_tensor_slices((padded_X_test2_encoded, y_test2)).batch(32)

# Define the fine-tuned model
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

# Learning rate scheduler function
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Instantiate the fine-tuned model
num_classes = 1  # Binary classification (pathogenic or not)

# Load the pre-trained model
base_model = models.load_trained_model(model='patho_finetune', download=True)
final_model = FineTunedSequenceUNET(base_model, num_classes=num_classes, dropout_rate=0.5)
final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Setup early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# lr_scheduler = LearningRateScheduler(scheduler)

# Train the model with the validation dataset
history = final_model.fit(train_dataset, epochs=300, validation_data=test2_dataset, callbacks=[early_stopping])

# Evaluate the final model on the test set
test_loss, test_accuracy = final_model.evaluate(test_dataset)
test_loss2, test_accuracy2 = final_model.evaluate(test2_dataset)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
print('SECOND TEST: \n')
print(f'Test Loss: {test_loss2}, Test Accuracy: {test_accuracy2}')

# Plot the training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

##############  USE THE FINE-TUNED MODEL TO ASSIGN NEW LABELS TO DATA   #########################

# Load the dataset for which you want to assign new labels
new_data_file_path = 'C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7_Train2.csv'
new_data_file_path2 = 'C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7_Test.csv'
new_data = pd.read_csv(new_data_file_path, header=None)
new_data2 = pd.read_csv(new_data_file_path2, header=None)
new_data.columns = columns
new_data2.columns = columns

# Drop the 'id' column and separate features and labels for new data
X_new = new_data.drop(columns=['Pathogenicity'])
y_new = new_data['Pathogenicity']

X_new2 = new_data2.drop(columns=['Pathogenicity'])
y_new2 = new_data2['Pathogenicity']

# One-hot encode and pad the new data
X_new_encoded = np.array([one_hot_encode(seq) for seq in X_new.values])
padded_X_new_encoded = add_padding(X_new_encoded)

X_new_encoded2 = np.array([one_hot_encode(seq) for seq in X_new2.values])
padded_X_new_encoded2 = add_padding(X_new_encoded2)

# Predict labels for the new dataset
predictions = final_model.predict(padded_X_new_encoded)
predicted_labels = (predictions > 0.5).astype(int).flatten()

predictions2 = final_model.predict(padded_X_new_encoded2)
predicted_labels2 = (predictions2 > 0.5).astype(int).flatten()

# Add the predicted labels as a new column next to the real labels
new_data['Predicted_Pathogenicity'] = predicted_labels

new_data2['Predicted_Pathogenicity'] = predicted_labels2

# Save the dataset with the new predictions
labled_data_path = "C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/Pred_labels_train2.csv"
labled_data_path2 = "C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/Pred_labels_test.csv"
new_data.to_csv(labled_data_path, index=False)
new_data2.to_csv(labled_data_path2, index=False)

print("New labels assigned and saved successfully.")
