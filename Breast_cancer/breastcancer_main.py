import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
train1_path = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Breast_cancer/breastcancer_train1.csv'
train2_path = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Breast_cancer/breastcancer_train2.csv'
test_path = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Breast_cancer/breastcancer_test.csv'

train1 = pd.read_csv(train1_path)
train2_data = pd.read_csv(train2_path)
test_data = pd.read_csv(test_path)

# Drop the 'Unnamed: 32' column and the 'id' column as they are not useful for the model
train1.drop(['id'], axis=1, inplace=True)
train2_data.drop(['id'], axis=1, inplace=True)
test_data.drop(['id'], axis=1, inplace=True)

# Encode the 'diagnosis' column to numeric values (M -> 1, B -> 0)
label_encoder = LabelEncoder()
train1['diagnosis'] = label_encoder.fit_transform(train1['diagnosis'])
train2_data['diagnosis'] = label_encoder.fit_transform(train2_data['diagnosis'])
test_data['diagnosis'] = label_encoder.fit_transform(test_data['diagnosis'])

# Split the data into features and labels
X = train1.drop('diagnosis', axis=1)
y = train1['diagnosis']
X2 = train2_data.drop('diagnosis', axis=1)
y2 = train2_data['diagnosis']
Xtest = test_data.drop('diagnosis', axis=1)
ytest = test_data['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train2 = scaler.transform(X2)
X_test2 = scaler.transform(Xtest)

# Build the neural network model
model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model with a different optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to avoid overfitting with modified parameters
early_stopping = EarlyStopping(monitor='val_loss', patience=15, min_delta=0.001, restore_best_weights=True, verbose=1)

# Learning rate reduction on plateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.001, verbose=1)

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Evaluate the model on train2 dataset
y_train2_pred = (model.predict(X_train2) > 0.5).astype("int32")
print("Evaluation on breastcancer_train2.csv:")
print("Accuracy:", accuracy_score(y2, y_train2_pred))
print("\nClassification Report:\n", classification_report(y2, y_train2_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y2, y_train2_pred))

# Evaluate the model on test dataset
y_test_pred = (model.predict(X_test2) > 0.5).astype("int32")
print("Evaluation on breastcancer_test.csv:")
print("Accuracy:", accuracy_score(ytest, y_test_pred))
print("\nClassification Report:\n", classification_report(ytest, y_test_pred))
print("\nConfusion Matrix:\n", confusion_matrix(ytest, y_test_pred))

# Predict labels for the train2 dataset and save
train2_data['Prediction'] = y_train2_pred
labeled_data_path = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Breast_cancer/breastcancer_train2_labeled.csv'
train2_data.to_csv(labeled_data_path, index=False)

# Predict labels for the train2 dataset and save
test_data['Prediction'] = y_test_pred
labeled_data_path2 = '/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Breast_cancer/breastcancer_test_labeled.csv'
test_data.to_csv(labeled_data_path2, index=False)

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
