import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import random
import keras
import matplotlib.pyplot as plt

print(tf.__version__)
print(keras.__version__)

# Set random seed for reproducibility
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

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

# Amino acid dictionary for encoding
AA_dict = {
    "A": [1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23],
    "G": [0, 0, 0, 0, 6.07, 0.13, 0.15],
    "V": [3.67, 0.14, 3, 1.22, 6.02, 0.27, 0.49],
    "L": [2.59, 0.19, 4, 1.7, 6.04, 0.39, 0.31],
    "I": [4.19, 0.19, 4, 1.8, 6.04, 0.3, 0.45],
    "F": [2.94, 0.29, 5.89, 1.79, 5.67, 0.3, 0.38],
    "Y": [2.94, 0.3, 6.47, 0.96, 5.66, 0.25, 0.41],
    "W": [3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42],
    "T": [3.03, 0.11, 2.6, 0.26, 5.6, 0.21, 0.36],
    "S": [1.31, 0.06, 1.6, -0.04, 5.7, 0.2, 0.28],
    "R": [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
    "K": [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
    "H": [2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.3],
    "D": [1.6, 0.11, 2.78, -0.77, 2.95, 0.25, 0.2],
    "E": [1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21],
    "N": [1.6, 0.13, 2.95, -0.6, 6.52, 0.21, 0.22],
    "Q": [1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25],
    "M": [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
    "P": [2.67, 0, 2.72, 0.72, 6.8, 0.13, 0.34],
    "C": [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
    "B": [1.6, 0.12, 2.865, -0.685, 4.735, 0.23, 0.21],
    "?": [0, 0, 0, 0, 0, 0, 0],
    "-": [0, 0, 0, 0, 0, 0, 0],
    "*": [0, 0, 0, 0, 0, 0, 0]
}

def encode_seq(AA_dict, seq):
    encode = np.zeros(len(seq) * 7)
    for i in range(len(seq)):
        encode[i * 7:(i + 1) * 7] = AA_dict[seq[i]]
    return encode

def MSA_encode(MSA, AA_dict):
    data_encoded = np.zeros((MSA.shape[0], (MSA.shape[1] - 1) * 7))
    for i in range(MSA.shape[0]):
        data_encoded[i, :] = encode_seq(AA_dict, MSA.iloc[i, :-1])
    return data_encoded

# Encode the datasets
X_train_encoded = MSA_encode(X_train, AA_dict)
X_test_encoded = MSA_encode(X_test, AA_dict)
X_test2_encoded = MSA_encode(X_test2, AA_dict)

# Add padding to ensure the sequence length is divisible by the model's downsampling factor
def add_padding(X_encoded, factor=8):
    sequence_length = X_encoded.shape[1]
    target_length = ((sequence_length - 1) // factor + 1) * factor
    padded_X_encoded = np.zeros((X_encoded.shape[0], target_length))
    padded_X_encoded[:, :sequence_length] = X_encoded
    return padded_X_encoded

padded_X_train_encoded = add_padding(X_train_encoded)
padded_X_test_encoded = add_padding(X_test_encoded)
padded_X_test2_encoded = add_padding(X_test2_encoded)

# Reshape to add a third dimension for compatibility with Conv1D layers
padded_X_train_encoded = padded_X_train_encoded.reshape((padded_X_train_encoded.shape[0], padded_X_train_encoded.shape[1], 1))
padded_X_test_encoded = padded_X_test_encoded.reshape((padded_X_test_encoded.shape[0], padded_X_test_encoded.shape[1], 1))
padded_X_test2_encoded = padded_X_test2_encoded.reshape((padded_X_test2_encoded.shape[0], padded_X_test2_encoded.shape[1], 1))

# Convert to TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((padded_X_train_encoded, y_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((padded_X_test_encoded, y_test)).batch(32)
test2_dataset = tf.data.Dataset.from_tensor_slices((padded_X_test2_encoded, y_test2)).batch(32)

# Define the improved CNN model
class ImprovedCNN(Model):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(ImprovedCNN, self).__init__()
        self.conv1 = Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01))
        self.pool1 = MaxPooling1D(pool_size=2)
        self.bn1 = BatchNormalization()
        
        self.conv2 = Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01))
        self.pool2 = MaxPooling1D(pool_size=2)
        self.bn2 = BatchNormalization()
        
        self.conv3 = Conv1D(256, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01))
        self.pool3 = MaxPooling1D(pool_size=2)
        self.bn3 = BatchNormalization()
        
        self.conv4 = Conv1D(512, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01))
        self.pool4 = MaxPooling1D(pool_size=2)
        self.bn4 = BatchNormalization()

        self.global_pool = GlobalAveragePooling1D()
        self.dropout = Dropout(dropout_rate)
        self.dense1 = Dense(256, activation='relu', kernel_regularizer=l2(0.01))
        self.dense2 = Dense(num_classes, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.bn1(x, training=training)
        
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x, training=training)
        
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.bn3(x, training=training)
        
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.bn4(x, training=training)
        
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        return self.dense2(x)

# Instantiate and compile the improved CNN model
final_model = ImprovedCNN(num_classes=1, dropout_rate=0.5)
final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Setup callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(scheduler)

# Train the model
history = final_model.fit(train_dataset, epochs=15, validation_data=test_dataset, callbacks=[early_stopping, lr_scheduler])

# Evaluate the model
test_loss, test_accuracy = final_model.evaluate(test_dataset)
test_loss2, test_accuracy2 = final_model.evaluate(test2_dataset)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
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

# Optionally: Save the trained model
# final_model.save('path_to_save_model')

# Load the dataset for which you want to assign new labels
new_data_file_path = 'C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7_Train2.csv'
new_data_file_path2 = 'C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7_Test.csv'
new_data = pd.read_csv(new_data_file_path, header=None)
new_data2 = pd.read_csv(new_data_file_path2, header=None)
new_data.columns = columns
new_data2.columns = columns

# Drop the 'id' column and separate features and labels for new data
X_new = new_data.drop(columns=['id', 'Pathogenicity'])
y_new = new_data['Pathogenicity']

X_new2 = new_data2.drop(columns=['id', 'Pathogenicity'])
y_new2 = new_data2['Pathogenicity']

# One-hot encode and pad the new data
X_new_encoded = MSA_encode(X_new, AA_dict)
padded_X_new_encoded = add_padding(X_new_encoded)
X_new_encoded2 = MSA_encode(X_new2, AA_dict)
padded_X_new_encoded2 = add_padding(X_new_encoded2)

# Reshape to add a third dimension for compatibility with Conv1D layers
padded_X_new_encoded = padded_X_new_encoded.reshape((padded_X_new_encoded.shape[0], padded_X_new_encoded.shape[1], 1))
padded_X_new_encoded2 = padded_X_new_encoded2.reshape((padded_X_new_encoded2.shape[0], padded_X_new_encoded2.shape[1], 1))

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
