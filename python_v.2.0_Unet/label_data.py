# label_data.py

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
from model_utils import encode_seq, MSA_encode, load_sequence_unet_model

# Function to evaluate the model for all data points
def evaluate_test_data(model, MSA, AA_dict):
    encoded_MSA = MSA_encode(MSA, AA_dict)
    predictions = []
    
    for i in range(encoded_MSA.shape[0]):
        input_tensor = encoded_MSA[i, :].reshape(1, -1, 21)  # Adjust input shape as required
        pred = model.predict(input_tensor)
        pred_label = np.argmax(pred, axis=-1)[0]
        predictions.append(pred_label)

    return predictions

# Load the trained model
model = tf.keras.models.load_model('your_model.h5')

# Load the dictionary
with open('dict.pkl', 'rb') as f:
    AA_dict = pickle.load(f)

# Read the test dataset
file_name = "./data/NS1/NS1_H5_H7_Test.csv"
secTest = pd.read_csv(file_name, header=None).to_numpy()
X = MSA_encode(secTest, AA_dict)
y = secTest[:, -1].astype(int)

# Evaluate the model on the entire test dataset
test_preds = evaluate_test_data(model, secTest, AA_dict)
test_conf_matrix = confusion_matrix(y, test_preds)
test_accuracy = accuracy_score(y, test_preds)

# Print test data metrics
print("Test Confusion Matrix:")
print(test_conf_matrix)
print("Test Accuracy:", test_accuracy)

# Save the new labels to the test data
newLabel = np.array(test_preds).reshape(-1, 1)
newData = np.hstack((secTest, newLabel))
new_test_file = "./data/NS1/NewTest.csv"
np.savetxt(new_test_file, newData, delimiter=',', fmt='%s')

print("New labeled test data saved.")
