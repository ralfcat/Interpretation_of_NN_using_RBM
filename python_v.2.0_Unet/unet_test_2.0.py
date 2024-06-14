import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model
import sequence_unet.models as models
from sequence_unet.models import cnn_top_model
import keras
print(tf.__version__)
print(keras.__version__)

# Load your dataset
data = pd.read_csv('C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/NS1/NS1_H5_H7.csv')

# Combine the amino acid columns into a single sequence string per sample
sequence_cols = [col for col in data if col.startswith('P')]
data['sequence'] = data[sequence_cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)

# Drop the original position columns
X = data.drop(columns=['id', 'Pathogenicity'] + sequence_cols)
y = data['Pathogenicity'].values

# Encoding the sequences
def one_hot_encode(sequence):
    mapping = {char: idx for idx, char in enumerate(sorted(set(''.join(data['sequence']))))}
    encoded = np.zeros((len(sequence), len(mapping)))
    for i, char in enumerate(sequence):
        if char in mapping:
            encoded[i, mapping[char]] = 1
    return encoded

X_encoded = np.array([one_hot_encode(seq) for seq in data['sequence'].values])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Convert to TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

# Load the base model from a saved file
base_model_path = 'patho_finetune_model.h5'
base_model = load_model(base_model_path, compile=False)  # Load pre-trained model without compiling

# Add a new top model on top of the base using cnn_top_model function
fine_tuned_model = cnn_top_model(
    bottom_model=base_model_path,
    features=True,
    output_size=1,
    tune_layers=3,
    kernel_size=3,
    activation='sigmoid',
    dropout=0.5
)
fine_tuned_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = fine_tuned_model.fit(
    train_dataset,
    epochs=50,
    validation_data=test_dataset
)

# Evaluate the model
test_loss, test_accuracy = fine_tuned_model.evaluate(test_dataset)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

