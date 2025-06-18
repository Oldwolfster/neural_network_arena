import tensorflow as tf
import numpy as np


import tensorflow as tf
import numpy as np

# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='tanh', input_shape=(2,)),  #TANH .2605  #RELU  .6575  .6861
    tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model with SGD and Binary Crossentropy loss   #.2531  # .2398
model.compile(
    optimizer='sgd',  # Stochastic Gradient Descent
    loss='mean_squared_error',  # Binary Crossentropy for classification
    metrics=['accuracy']  # Track accuracy
)

# Generate XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train the model
model.fit(X, y, epochs=111, verbose=1)

# Test the model
predictions = model.predict(X)
print("Predictions:", predictions)
###################################################################################################
"""

# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1)  # No activation for regression tasks
])
from tensorflow.keras.optimizers import SGD
sgd_optimizer = SGD(learning_rate=0.01, momentum=0.0)  # Adjust learning rate and momentum

# Compile the model with SGD and MSE loss  run 1, loss=.02   0.0108
model.compile(
    #optimizer='sgd',  # Stochastic Gradient Descent
    optimizer='sgd',  # Stochastic Gradient Descent       #0.0990   #.1535   #
    loss='mean_squared_error',  # Mean Squared Error
    metrics=['accuracy']  # Optional: Add accuracy if it makes sense for your task
)

# Generate XOR data (for demonstration)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train the model
model.fit(X, y, epochs=333, verbose=1)

# Test the model
predictions = model.predict(X)
print("Predictions:", predictions)
import tensorflow as tf
import numpy as np

# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model with SGD and Binary Crossentropy loss
model.compile(
    optimizer='sgd',  # Stochastic Gradient Descent
    loss='binary_crossentropy',  # Binary Crossentropy for classification
    metrics=['accuracy']  # Track accuracy
)

# Generate XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train the model
model.fit(X, y, epochs=1000, verbose=1)

# Test the model
predictions = model.predict(X)
print("Predictions:", predictions)
"""
################################################################################
"""
Epoch 1000/1000
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step - accuracy: 1.0000 - loss: 0.0010  
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step
Predictions: [[0.05446447]
 [0.986006  ]
 [0.9724374 ]
 [0.01203134]]



# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1)  # No activation for regression tasks
])

# Compile the model with SGD and MSE loss
model.compile(
    optimizer='sgd',  # Stochastic Gradient Descent
    loss='mean_squared_error',  # Mean Squared Error
    metrics=['accuracy']  # Optional: Add accuracy if it makes sense for your task
)

# Generate XOR data (for demonstration)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train the model
model.fit(X, y, epochs=1000, verbose=1)

# Test the model
predictions = model.predict(X)
print("Predictions:", predictions)

"""

"""   ADAM and BCE
# Deeper network for XOR with ReLU
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(2,), kernel_initializer='he_normal'),  # Hidden layer 1
    tf.keras.layers.Dense(10, activation='relu', kernel_initializer='he_normal'),  # Hidden layer 2
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
])
print("1 defined")
# Compile the model
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='mean square error', metrics=['accuracy'])
print("2 compiled")
# Generate XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train the model
model.fit(X, y, epochs=100, verbose=2)
print("3 trained")
# Test the model
predictions = model.predict(X)
print("4 tested")
print(predictions)

"""