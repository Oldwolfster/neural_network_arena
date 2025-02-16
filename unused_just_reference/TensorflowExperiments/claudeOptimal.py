import tensorflow as tf
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# XOR input and output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Create the model with optimal initialization
model = tf.keras.Sequential([
    # Hidden layer with 2 neurons
    tf.keras.layers.Dense(2, input_shape=(2,), activation='tanh',
                         kernel_initializer=tf.keras.initializers.Constant([
                             #[1.0, -1.0],  # First neuron weights
                             #[1.0, -1.0]   # Second neuron weights
                             [0, 0],  # First neuron weights
                             [0, 0]   # Second neuron weights
                         ]),
                         bias_initializer=tf.keras.initializers.Constant([-0.5, 1.5])),

    # Output layer with 1 neuron
    tf.keras.layers.Dense(1, activation='sigmoid',
                         kernel_initializer=tf.keras.initializers.Constant([[2.0], [2.0]]),
                         bias_initializer=tf.keras.initializers.Constant([-3.0]))
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
             loss='binary_crossentropy',
             metrics=['accuracy'])

# Train the model for exactly 100 epochs
history = model.fit(X, y, epochs=100, verbose=0)

# Get final loss and accuracy
final_loss = history.history['loss'][-1]
final_accuracy = history.history['accuracy'][-1]

print(f"\nAfter 100 epochs:")
print(f"Final loss: {final_loss:.6f}")
print(f"Final accuracy: {final_accuracy * 100:.2f}%")

# Show predictions
predictions = model.predict(X, verbose=0)
print("\nPredictions vs Expected:")
print("Input  | Predicted | Expected")
print("------|-----------|---------")
for i in range(len(X)):
    print(f"{X[i]} | {predictions[i][0]:.6f} | {y[i][0]}")

# Display the weights
weights = model.get_weights()
print("\nFinal Weights:")
print("Hidden Layer:")
print("Weights:\n", weights[0])
print("Biases:", weights[1])
print("\nOutput Layer:")
print("Weights:", weights[2].flatten())
print("Bias:", weights[3][0])