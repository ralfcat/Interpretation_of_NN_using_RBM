import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load the normalized data
train_data_path = 'C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/Genre/train_data_normalized.csv'
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv('C:/Users/victo/Documents/GitHub/Interpretation_of_NN_using_RBM/Interpretation_of_NN_using_RBM/data/Genre/second_set_normalized.csv')

# Prepare the data
X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values

X_test = test_data.drop('label', axis=1).values
y_test= test_data['label'].values
# Encode labels
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)
y_train_categorical = to_categorical(y_train_encoded)
y_test_categorical = to_categorical(y_test_encoded)

# Define the model
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
# Adding Batch Normalization layers
model = Sequential([
    Dense(512, input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),
    Dense(256, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),
    Dense(128, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    Dense(y_train_categorical.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model


model.fit(X_train, y_train_categorical, epochs=50, batch_size=32, validation_data = (X_test, y_test_categorical))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test,y_test_categorical)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')


# Optionally, you can save the model for later use
#model.save('music_genre_classifier_model.h5')
