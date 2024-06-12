# label_data.py

import torch
import numpy as np
import pandas as pd
import pickle
from EvaluateNetwork import encode_seq, MSA_encode

# Function to evaluate the model for all data points
def evaluate_test_data(model, MSA, AA_dict):
    encoded_MSA = MSA_encode(MSA, AA_dict)
    predictions = []
    
    for i in range(encoded_MSA.shape[0]):
        input_tensor = torch.tensor(encoded_MSA[i, :], dtype=torch.float32).unsqueeze(0)
        pred = model(input_tensor)
        pred_label = torch.argmax(pred, dim=1).item()
        predictions.append(pred_label)

    return predictions

# Load the model and dictionary
model = torch.load('model.pth')
model.eval()  # Set the model to evaluation mode

with open('dict.pkl', 'rb') as f:
    AA_dict = pickle.load(f)

# Choose data to label (AIV or MTB)
file_name = "./data/NS1/NS1_H5_H7_Train2.csv"


# Read the dataset
secTrain = pd.read_csv(file_name, header=None).to_numpy()

# Evaluate the model on the dataset
pred = evaluate_test_data(model, secTrain, AA_dict)
newLabel = np.array(pred).reshape(-1, 1)

# Combine the original data with the new labels
newData = np.hstack((secTrain, newLabel))

# Save the new data to CSV files
new_train_file = "./data/NS1/NewTrain2_python.csv"


np.savetxt(new_train_file, newData, delimiter=',', fmt='%s')


print("New labeled data saved.")
