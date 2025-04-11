import numpy as np

class SimplePerceptron:
    def __init__(self, input_size, learning_rate=0.001):
        self.weights = np.random.randn(input_size)
        self.bias = 0.0
        self.learning_rate = learning_rate

    def predict(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def train(self, training_data, epochs=1000):
        for epoch in range(epochs):
            total_loss = 0
            for data_point in training_data:
                inputs = np.array(data_point[:-1])  # All inputs except the last element (salary)
                target = data_point[-1]             # Last element is the salary

                prediction = self.predict(inputs)
                error = target - prediction
                total_loss += error ** 2

                # Update weights and bias
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

            # Optionally print the loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}, Loss: {total_loss / len(training_data)}')


# Generate linear data
arena = Salary2InputsLinear(num_samples=100)
training_data = arena.generate_training_data()

# Initialize and train the perceptron
perceptron = SimplePerceptron(input_size=2)
perceptron.train(training_data)

### ONLY RUN ABOVE OR BELOW __ THEY DONT GO TOGETHHER
# Generate data with interaction term
arena = Salary2InputsWithInteraction(num_samples=100)
training_data = arena.generate_training_data()

# Initialize and train the perceptron
perceptron = SimplePerceptron(input_size=3)
perceptron.train(training_data)
