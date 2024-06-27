import tensorflow as tf

# Create a linear stack of layers
model = tf.keras.Sequential([
    # Input layer with the shape (24, 2)
    tf.keras.layers.InputLayer(input_shape=(24, 2)),
    # Add a Simple RNN layer with 10 units, returning the full sequence
    tf.keras.layers.SimpleRNN(units=10, return_sequences=True),
    # Add another Simple RNN layer with 10 units, not returning the full sequence
    tf.keras.layers.SimpleRNN(units=10, return_sequences=False),
    # Add a Dropout layer with a rate of 0.2 to prevent overfitting
    tf.keras.layers.Dropout(0.2),
    # Add a Dense (fully connected) layer with 5 units for the 5 gesture classes
    tf.keras.layers.Dense(4, activation='softmax'),
])

# Model compilation
# Using Adam optimizer and categorical cross-entropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the model
print(model.summary())
