import numpy as np

class SingleLayerPerceptron:
    def __init__(self, input_size, num_neurons):
        """
        Initialize the Single-Layer Perceptron

        Parameters:
        - input_size: Number of input features
        - num_neurons: Number of neurons in the layer
        """
        # Random weight initialization
        self.weights = np.random.randn(num_neurons, input_size)
        self.bias = np.zeros((num_neurons, 1))

    def activation(self, x):
        """
        Step activation function (threshold activation)
        """
        return np.where(x >= 0, 1, 0)

    def forward(self, inputs):
        """
        Forward pass through the single-layer perceptron

        Parameters:
        - inputs: Input feature vector

        Returns:
        - Activated output of neurons
        """
        # Compute weighted sum and add bias
        z = np.dot(self.weights, inputs) + self.bias

        # Apply activation function
        return self.activation(z)

# Scenario 1: Multi-Class Weather Classification
def weather_classification_example():
    print("Scenario 1: Multi-Class Weather Classification")

    # Input features: [temperature, humidity, wind_speed]
    # Neurons represent different weather conditions
    # 0: Sunny, 1: Rainy, 2: Cloudy
    inputs = np.array([
        [25, 40, 10],   # Sunny day
        [15, 80, 20],   # Rainy day
        [20, 60, 15]    # Cloudy day
    ]).T

    # Create SLP with 3 inputs and 3 neurons
    slp = SingleLayerPerceptron(input_size=3, num_neurons=3)

    # Classify inputs
    outputs = slp.forward(inputs)
    print("Inputs:\n", inputs)
    print("Outputs:\n", outputs)

# Scenario 2: Simple Medical Diagnostic Classification
def medical_diagnostic_example():
    print("\nScenario 2: Simple Medical Diagnostic Classification")

    # Input features: [age, blood_pressure, cholesterol_level]
    # Neurons represent different risk categories
    # 0: Low Risk, 1: Medium Risk, 2: High Risk
    inputs = np.array([
        [35, 120, 180],  # Low risk
        [55, 160, 250],  # High risk
        [45, 140, 220]   # Medium risk
    ]).T

    # Create SLP with 3 inputs and 3 neurons
    slp = SingleLayerPerceptron(input_size=3, num_neurons=3)

    # Classify inputs
    outputs = slp.forward(inputs)
    print("Inputs:\n", inputs)
    print("Outputs:\n", outputs)

# Scenario 3: Simple Product Recommendation
def product_recommendation_example():
    print("\nScenario 3: Simple Product Recommendation")

    # Input features: [age, income, previous_purchases]
    # Neurons represent different product categories
    # 0: Electronics, 1: Clothing, 2: Home Goods
    inputs = np.array([
        [25, 50000, 3],   # Likely electronics
        [45, 80000, 5],   # Likely home goods
        [35, 60000, 4]    # Likely clothing
    ]).T

    # Create SLP with 3 inputs and 3 neurons
    slp = SingleLayerPerceptron(input_size=3, num_neurons=3)

    # Classify inputs
    outputs = slp.forward(inputs)
    print("Inputs:\n", inputs)
    print("Outputs:\n", outputs)

# Run all scenarios
def main():
    weather_classification_example()
    medical_diagnostic_example()
    product_recommendation_example()

if __name__ == "__main__":
    main()