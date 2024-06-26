import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# Load the datasets
train1_path = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Spambase/spambase_train1.csv'
train2_path = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Spambase/spambase_train2.csv'
test_path = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Spambase/spambase_test.csv'

train1_data = pd.read_csv(train1_path)
train2_data = pd.read_csv(train2_path)
test_data = pd.read_csv(test_path)

# Separate features and target for train1 dataset
X_train1 = train1_data.drop(columns=['spam'])
y_train1 = train1_data['spam']

# Separate features and target for train2 dataset
X_train2 = train2_data.drop(columns=['spam'])
y_train2 = train2_data['spam']

# Separate features and target for test dataset
X_test = test_data.drop(columns=['spam'])
y_test = test_data['spam']

# Standardize the features
scaler = StandardScaler()
X_train1 = scaler.fit_transform(X_train1)
X_train2 = scaler.transform(X_train2)
X_test = scaler.transform(X_test)

# Apply SMOTE to the training data
smote = SMOTE()
X_train1_smote, y_train1_smote = smote.fit_resample(X_train1, y_train1)

# Define the neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train1.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(32, input_dim=X_train1.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with a learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), 
              loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model on train1 dataset with class weights
history = model.fit(X_train1_smote, y_train1_smote, epochs=70, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on train2 dataset
y_train2_pred = (model.predict(X_train2) > 0.5).astype("int32")
print("Evaluation on spambase_train2.csv:")
print("Accuracy:", accuracy_score(y_train2, y_train2_pred))
print("\nClassification Report:\n", classification_report(y_train2, y_train2_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_train2, y_train2_pred))

# Evaluate the model on test dataset
y_test_pred = (model.predict(X_test) > 0.5).astype("int32")
print("Evaluation on spambase_test.csv:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# Predict labels for the train2 dataset and save
train2_data['Predicted_spam'] = y_train2_pred
labeled_data_path = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/spambase_train2_labeled.csv'
train2_data.to_csv(labeled_data_path, index=False)

print("New labels assigned and saved successfully.")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

