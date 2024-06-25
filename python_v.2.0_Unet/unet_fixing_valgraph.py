import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Reshape, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import sequence_unet.models as models
import keras
import random
import matplotlib.pyplot as plt

print(tf.__version__)
print(keras.__version__)

# Load your dataset
original_file_path = 'C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7.csv'
original_data = pd.read_csv(original_file_path)
columns = original_data.columns[1:]

train_data = pd.read_csv('C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7_dataset1.csv')
train_data.columns = columns

test_file_path = 'C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7_dataset3.csv'
test_data = pd.read_csv(test_file_path, header=None)
test_data.columns = columns

test_file_path2 = 'C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7_dataset2.csv'
test_data2 = pd.read_csv(test_file_path2, header=None)
test_data2.columns = columns

# Preprocess your data
X_train = train_data.drop(columns=['Pathogenicity'])
y_train = train_data['Pathogenicity']

X_test = test_data.drop(columns=['Pathogenicity'])
y_test = test_data['Pathogenicity']

X_test2 = test_data2.drop(columns=['Pathogenicity'])
y_test2 = test_data2['Pathogenicity']

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
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

def add_padding(X_encoded):
    sequence_length = X_encoded.shape[1]
    target_length = ((sequence_length - 1) // 8 + 1) * 8
    padded_X_encoded = np.zeros((X_encoded.shape[0], target_length, X_encoded.shape[2]))
    padded_X_encoded[:, :sequence_length, :] = X_encoded
    return padded_X_encoded

padded_X_train_encoded = add_padding(X_train_encoded)
padded_X_test_encoded = add_padding(X_test_encoded)
padded_X_test2_encoded = add_padding(X_test2_encoded)

train_dataset = tf.data.Dataset.from_tensor_slices((padded_X_train_encoded, y_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((padded_X_test_encoded, y_test)).batch(32)
test2_dataset = tf.data.Dataset.from_tensor_slices((padded_X_test2_encoded, y_test2)).batch(32)

# Load the pre-trained model and add a new top model
base_model = models.load_trained_model(model='patho_finetune', download=True)

# Using cnn_top_model with tune_layers=-1 to unfreeze all layers
top_model = models.cnn_top_model(
    bottom_model='C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/patho_finetune.h5',
    features=True,
    output_size=256,
    kernel_size=3,
    activation='linear',
    dropout=0.5
)

print("Shape of top model output:", top_model.output.shape)

# Adding GlobalAveragePooling1D and a Dense layer for binary classification
pooling_layer = GlobalAveragePooling1D()
dense_layer = Dense(1, activation='sigmoid')

# Apply the pooling layer first
output = pooling_layer(top_model.output)

# Then apply the Dense layer
output = dense_layer(output)

# Create the final model using the input from the top_model and the output from the dense layer
final_model = Model(inputs=top_model.input, outputs=output)

# Compile the model with optimizer, loss, and metrics
final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = final_model.fit(train_dataset, validation_data=test_dataset, epochs=50, callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy = final_model.evaluate(test_dataset)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Save the trained model
final_model.save('path_to_save_model/fine_tuned_sequence_unet.h5')
print("Model saved successfully.")

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
