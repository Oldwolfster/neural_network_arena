import tensorflow as tf
import numpy as np
import os
import random

def print_model(model, seed, extra_msg=""):
    if seed > 0 :
        return
    print(extra_msg)
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):  # Check if it's a Dense layer
            weights, biases = layer.get_weights()
            print(f"\nLayer {i}: {layer.name}")
            print(f"Weights (shape {weights.shape}):\n{weights}")
            print(f"Biases (shape {biases.shape}):\n{biases}")

def check_structured(seed_value: int):

    # Ensure full reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)
    tf.keras.utils.set_random_seed(seed_value)
    tf.config.experimental_run_functions_eagerly(False)
    tf.keras.backend.clear_session() # Clear session before creating models

    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    def create_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(2, activation="tanh", input_shape=(2,)),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
                      loss="mse",
                      metrics=["accuracy"])
        return model

    # Create two models
    model_random = create_model()
    model_struct_small = create_model() #smaller random numbers
    print_model(model_struct_small, i, "Before Adjustment")

    # Adjust weights in Structured Small
    for layer in model_struct_small.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            weights, biases = layer.get_weights()
            if weights.shape[0] == 2:  # Only modify hidden layer
                weights = np.random.uniform(-0.01, 0.01, size=weights.shape)
            layer.set_weights([weights, biases])

    # Train both models
    print_model(model_struct_small, i,"After Adjustment")
    history_GBS = model_random.fit(X, y, epochs=100, verbose=0)
    history_struct_small = model_struct_small.fit(X, y, epochs=100, verbose=0)

    # Get final loss
    random_loss = history_GBS.history["loss"][-1]
    struct_small_loss = history_struct_small.history["loss"][-1]

    # Calculate relative improvement
    msg_small = "🔥 Structured Initializing Triumphs!" if struct_small_loss < random_loss else "🤔 GBS Triumphs!"
    reduction_small_pct = (1 - struct_small_loss / random_loss) * 100
    if reduction_small_pct < 0:
        reduction_small_pct=0

    results = (f"{seed_value}\t"
               f"{random_loss:.4f}\t"
               f"{struct_small_loss:.4f}\t"    
               f"{reduction_small_pct:.2f}\t"    
               f"{msg_small}")
    return results

for i in range(3333):
    print(check_structured(i))
