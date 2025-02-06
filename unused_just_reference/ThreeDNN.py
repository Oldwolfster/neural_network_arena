import numpy as np

class ThreeDNN:
    def __init__(self, input_size, output_size, delta=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.delta = delta  # Perturbation magnitude

        # Initialize weights and biases
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size)

    def forward(self, inputs):
        """
        Perform a forward pass, including perturbed layers.
        Args:
            inputs: Original input array of shape (batch_size, input_size)
        Returns:
            Tuple of (original output, perturbed outputs up, perturbed outputs down)
        """
        # Original layer output
        original_output = np.dot(inputs, self.weights.T) + self.biases

        # Perturbed outputs
        perturbed_up = np.dot(inputs + self.delta, self.weights.T) + self.biases
        perturbed_down = np.dot(inputs - self.delta, self.weights.T) + self.biases

        return original_output, perturbed_up, perturbed_down

# Example Usage
if __name__ == "__main__":
    # Define a 3D NN with 3 inputs and 2 outputs
    input_size = 3
    output_size = 2
    delta = 0.1  # Perturbation magnitude
    model = ThreeDNN(input_size, output_size, delta)

    # Example batch of inputs
    inputs = np.array([[0.5, 1.0, -0.5],
                       [1.5, -1.0, 0.0]])

    # Forward pass
    original, perturbed_up, perturbed_down = model.forward(inputs)
    print("Original Output:\n", original)
    print("\nPerturbed Up Output:\n", perturbed_up)
    print("\nPerturbed Down Output:\n", perturbed_down)
