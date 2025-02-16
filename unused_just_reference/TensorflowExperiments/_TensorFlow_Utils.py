import tensorflow as tf
def print_model(model, extra_msg=""):
    print(extra_msg)
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):  # Check if it's a Dense layer
            weights, biases = layer.get_weights()
            print(f"\nLayer {i}: {layer.name}")
            print(f"Weights (shape {weights.shape}):\n{weights}")
            print(f"Biases (shape {biases.shape}):\n{biases}")
