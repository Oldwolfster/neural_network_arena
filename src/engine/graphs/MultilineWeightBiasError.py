import matplotlib.pyplot as plt
import re

def plot_weights_and_mae(data):
    # Function to extract all weights for each layer and neuron
    def extract_weights(data):
        weights_dict = {}
        for entry in data:
            weights_lines = entry['weights'].split('\n')
            for line in weights_lines:
                match = re.match(r'(\d+): \[([-+]?[\d.]+), ([-+]?[\d.]+)\]', line)
                if match:
                    neuron = int(match.group(1))
                    if neuron not in weights_dict:
                        weights_dict[neuron] = ([], [])
                    weights_dict[neuron][0].append(float(match.group(2)))  # First weight of the neuron
                    weights_dict[neuron][1].append(float(match.group(3)))  # Second weight of the neuron
        return weights_dict

    # Extract weights and MAE
    weights = extract_weights(data)
    epochs = [d['epoch'] for d in data]
    mae = [d['mean_absolute_error'] for d in data]

    # Plotting
    plt.figure(figsize=(14, 8))

    # Plot weights for all neurons
    for neuron, (w1, w2) in weights.items():
        plt.plot(epochs, w1, label=f'Neuron {neuron} Weight 1', marker='o')
        plt.plot(epochs, w2, label=f'Neuron {neuron} Weight 2', marker='o')

    # Plot MAE
    plt.plot(epochs, mae, label='Mean Absolute Error (MAE)', color='black', linestyle='--', marker='o')

    # Graph settings
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.title('Weights and Mean Absolute Error Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# Test the function with the given data
plot_weights_and_mae(data)
