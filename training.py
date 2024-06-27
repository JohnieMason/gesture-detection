import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the preprocessed sequences and labels from .npy files
sequences = np.load('sequences.npy')
labels = np.load('labels.npy')

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Define the number of gesture classes
num_classes = 4  # There are 4 gesture classes

# Create the LSTM-based neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),  # LSTM layer with 64 units
    tf.keras.layers.Dense(64, activation='relu'),  # Dense layer with 64 units and ReLU activation
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer with softmax activation
])

# Compile the model with Adam optimizer and categorical crossentropy loss function
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
print(model.summary())

# Train the model on the training data
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the trained model and its weights
model.save("trained_model.keras")
model.save_weights("trained_model.weights.h5")