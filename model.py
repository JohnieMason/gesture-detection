# This file is used to set up a Recurrent Neural Network (RNN)
# model using TensorFlow and Keras API
import tensorflow as tf

# create a linear stack of layers
model = tf.keras.Sequential([
    # each input sequence has 24 time steps,
    # and each time step has 2 features (x and y coordinates)
    tf.keras.layers.InputLayer(input_shape=(24, 2)),
    # Add a Simple Recurrent Neural Network (RNN) layer with 10 units
    # 'return_sequences=True' means that the layer will output the full sequence of outputs for each time step,
    # which will be passed to the next RNN layer
    tf.keras.layers.SimpleRNN(units=10, return_sequences=True),
    # Add another Simple RNN layer with 10 units
    # 'return_sequences=False' means this layer will output only the last output in the sequence,
    # which will be passed to the subsequent layers
    tf.keras.layers.SimpleRNN(units=10, return_sequences=False),
    # Add a Dropout layer with a rate of 0.2. This prevents overfitting by
    # randomly setting 20% of the input units to 0 at each update during training
    tf.keras.layers.Dropout(0.2),
    # Add a Dense (fully connected) layer with 2 units
    # activation='softmax' means the output will be a
    # probability distribution over 2 classes (gestures), summing to 1
    tf.keras.layers.Dense(2, activation='softmax'),
])

# Model compilation
# the optimizer is the Adam optimization algorithm
# the loss function is the categorical cross-entropy since
# it is suitable for multi-class classification problems
# metrics=['accuracy'] tracks accuracy during training and evaluation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# print a summary of the model including the layer types, output shapes, and the number of parameters
print(model.summary())