# model_utils.py

import os
import numpy as np
from sequence_unet import models

# Define the mapping for one-hot encoding
AA_dict = {
    "A": [1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "G": [0, 0, 0, 0, 6.07, 0.13, 0.15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "V": [3.67, 0.14, 3, 1.22, 6.02, 0.27, 0.49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "L": [2.59, 0.19, 4, 1.7, 6.04, 0.39, 0.31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "I": [4.19, 0.19, 4, 1.8, 6.04, 0.3, 0.45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "F": [2.94, 0.29, 5.89, 1.79, 5.67, 0.3, 0.38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Y": [2.94, 0.3, 6.47, 0.96, 5.66, 0.25, 0.41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "W": [3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "T": [3.03, 0.11, 2.6, 0.26, 5.6, 0.21, 0.36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "S": [1.31, 0.06, 1.6, -0.04, 5.7, 0.2, 0.28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "R": [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "K": [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "H": [2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "D": [1.6, 0.11, 2.78, -0.77, 2.95, 0.25, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "E": [1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "N": [1.6, 0.13, 2.95, -0.6, 6.52, 0.21, 0.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Q": [1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "M": [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "P": [2.67, 0, 2.72, 0.72, 6.8, 0.13, 0.34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "C": [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "?": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "-": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "*": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] }

# Function to encode sequence using chemical properties
def encode_seq(AA_dict, seq):
    num_features = 20  # Ensure that we expect 20 features per amino acid
    encode = np.zeros((len(seq), num_features))
    for i, aa in enumerate(seq):
        # Ensure we handle cases where the amino acid is unknown
        features = AA_dict.get(aa, [0]*num_features)
        if len(features) != num_features:
            raise ValueError(f"Amino acid {aa} has {len(features)} features, expected {num_features}.")
        encode[i, :] = features
    return encode

# Function to return MSA in numbers
def MSA_encode(MSA, AA_dict):
    num_sequences = MSA.shape[0]
    sequence_length = MSA.shape[1] - 1
    num_features = 20  # Ensure that we expect 20 features per amino acid
    data_encoded = np.zeros((num_sequences, sequence_length, num_features))
    
    for i in range(num_sequences):
        data_encoded[i, :, :] = encode_seq(AA_dict, MSA[i, :-1])
    return data_encoded

# Function to download and load the Sequence UNET model
def load_sequence_unet_model(model_name="freq_classifier", root="models"):
    if not os.path.exists(root):
        os.makedirs(root)
    model_path = models.download_trained_model(model_name, root=root, model_format="tf")
    print(f"Model downloaded to: {model_path}")  # Debugging print statement
    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist")
    model = models.load_trained_model(model_path)
    return model