import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Example DataFrame (Replace with your dataset loading code)
path_train1 = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Gene_expression_cancer/data/cancer_genexp_train1'
path_train2 = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Gene_expression_cancer/data/cancer_genexp_train2'
path_test = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Gene_expression_cancer/data/cancer_genexp_test'

train1 = pd.read_csv(path_train1)
train2 = pd.read_csv(path_train2)
test = pd.read_csv(path_test)

# Separate features and labels
X = train1.drop(columns=['Class'])
y = train1['Class']

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.4, random_state=42)

# Build the neural network model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(y_encoded.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and capture the history
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_val, y_val))

# Plot the training and validation accuracy over epochs
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Prepare train2 and test datasets for evaluation
X_train2 = scaler.transform(train2.drop(columns=['Class']))  # Assuming it has a 'Class' column
X_test = scaler.transform(test.drop(columns=['Class']))      # Assuming it has a 'Class' column

# Evaluate on train2
_, accuracy_train2 = model.evaluate(X_train2, encoder.transform(train2['Class'].values.reshape(-1, 1)))
print(f'Train2 Accuracy: {accuracy_train2*100:.2f}%')

# Evaluate on test
_, accuracy_test = model.evaluate(X_test, encoder.transform(test['Class'].values.reshape(-1, 1)))
print(f'Test Accuracy: {accuracy_test*100:.2f}%')

train2_predictions = model.predict(X_train2)
test_predictions = model.predict(X_test)

# Use numpy to select the column with the highest probability (which corresponds to the predicted class)
# and then apply inverse_transform to get the original class labels
train2['Prediction'] = encoder.inverse_transform(train2_predictions).flatten()
test['Prediction'] = encoder.inverse_transform(test_predictions).flatten()

# Optionally, save the updated datasets
train2.to_csv('/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Gene_expression_cancer/data/cancer_genexp_train2_labeled.csv', index=False)
test.to_csv('/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Gene_expression_cancer/data/cancer_genexp_test_labeled.csv', index=False)

# Function to print classification report and confusion matrix
def print_classification_report_and_cm(true_labels, pred_labels, dataset_name):
    report = classification_report(true_labels, pred_labels, target_names=encoder.categories_[0])
    print(f"Classification Report for {dataset_name}:\n{report}")
    
    cm = confusion_matrix(true_labels, pred_labels)
    print(f"Confusion Matrix for {dataset_name}:\n{cm}")

# Print classification report and confusion matrix for train2
print_classification_report_and_cm(train2['Class'], train2['Prediction'], 'Train2')

# Print classification report and confusion matrix for test
print_classification_report_and_cm(test['Class'], test['Prediction'], 'Test')
