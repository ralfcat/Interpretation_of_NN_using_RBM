import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
import pickle

# Define the model class
class Perceptron(nn.Module):
    def __init__(self, input_length, actFunction=nn.CELU(), mode="single"):
        super(Perceptron, self).__init__()
        input_length *= 7
        if mode == "chain":
            self.model = nn.Sequential(
                nn.Linear(input_length, input_length),
                actFunction,
                nn.Linear(input_length, input_length),
                actFunction,
                nn.Linear(input_length, input_length),
                actFunction,
                nn.Linear(input_length, 2),
                actFunction
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(input_length, 2)
            )
        
    def forward(self, x):
        return self.model(x)

# Function to return sequence in numbers
def encode_seq(AA_dict, seq):
    encode = np.zeros(len(seq) * 7)
    for i in range(len(seq)):
        encode[i*7:(i+1)*7] = AA_dict[seq[i]]
    return encode

# Function to return MSA in numbers
def MSA_encode(MSA, AA_dict):
    num_sequences = MSA.shape[0]
    sequence_length = MSA.shape[1] - 1
    data_encoded = np.zeros((num_sequences, sequence_length * 7))
    
    for i in range(num_sequences):
        data_encoded[i, :] = encode_seq(AA_dict, MSA[i, :-1])
    return data_encoded

# Function to compute TP, FP, TN, FN using sklearn's confusion matrix
def performance_measure(truth, prediction, pos_value=1):
    tn, fp, fn, tp = confusion_matrix([truth], [prediction], labels=[0, pos_value]).ravel()
    return [tp, fp, tn, fn]

# Function to evaluate the model for all data points
def evaluate(model, X, y):
    model.eval()  # Set the model to evaluation mode
    preds = []
    with torch.no_grad():  # No gradient calculation for evaluation
        for i in range(X.shape[0]):
            input_tensor = torch.tensor(X[i, :], dtype=torch.float32).unsqueeze(0)
            pred = model(input_tensor)
            pred_label = torch.argmax(pred, dim=1).item()
            preds.append(pred_label)
    return np.array(preds)

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0  # Initialize running_loss at the start of each epoch
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return running_loss / len(train_loader)

# Example AA dictionary
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

# Hyperparameters
params = {
    'mode': 'single',
    'actFunction': nn.CELU(),
    'learning_rate': 0.00005,
    'lossFunction': nn.CrossEntropyLoss(),
    'epochs': 50
}

train_data_file = './data/NS1/NS1_H5_H7_Train1.csv'
test_data_file = './data/NS1/NS1_H5_H7_Test.csv'  # Update this with the actual test data path

# Read the training data file
MSA_df = pd.read_csv(train_data_file, header=None)
MSA = MSA_df.to_numpy()
X = MSA_encode(MSA, AA_dict)
y = MSA[:, -1].astype(int)

# Read the test data file
test_MSA_df = pd.read_csv(test_data_file, header=None)
test_MSA = test_MSA_df.to_numpy()
X_test = MSA_encode(test_MSA, AA_dict)
y_test = test_MSA[:, -1].astype(int)

# K-Fold Cross-Validation
kf = KFold(n_splits=10)
final_conf_matrix = np.zeros((2, 2))
accuracies = []

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Convert data to PyTorch tensors
    train_data = [(torch.tensor(X_train[i], dtype=torch.float32), torch.tensor(y_train[i], dtype=torch.long))
                  for i in range(len(y_train))]
    val_data = [(torch.tensor(X_val[i], dtype=torch.float32), torch.tensor(y_val[i], dtype=torch.long))
                 for i in range(len(y_val))]
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)
    
    # Create the model
    input_length = MSA.shape[1] - 1
    model = Perceptron(input_length=input_length, actFunction=params['actFunction'], mode=params['mode'])
    
    # Define loss and optimizer
    criterion = params['lossFunction']
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # Train the model
    avg_loss = train_model(model, train_loader, criterion, optimizer, epochs=params['epochs'])
    
    # Evaluate the model on validation data
    val_preds = evaluate(model, X_val, y_val)
    acc = accuracy_score(y_val, val_preds)
    accuracies.append(acc)
    final_conf_matrix += confusion_matrix(y_val, val_preds)
    
    print(f"Fold {fold+1} - Accuracy: {acc:.4f}, Loss: {avg_loss:.4f}")

# Evaluate the final model on test data
test_preds = evaluate(model, X_test, y_test)
test_conf_matrix = confusion_matrix(y_test, test_preds)
test_accuracy = accuracy_score(y_test, test_preds)

# Print final metrics
print("Final Confusion Matrix from 10-Fold CV:")
print(final_conf_matrix)
print("Accuracies from 10-Fold CV:", accuracies)
print("Mean Accuracy from 10-Fold CV:", np.mean(accuracies))
print("Test Confusion Matrix:")
print(test_conf_matrix)
print("Test Accuracy:", test_accuracy)


# Save the model and dictionary
torch.save(model, 'model.pth')
with open('dict.pkl', 'wb') as f:
    pickle.dump(AA_dict, f)

print("Model and dictionary saved.")