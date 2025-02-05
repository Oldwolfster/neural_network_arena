import re

from src.Reports._BaseReport import BaseReport
from src.engine.RamDB import RamDB
import matplotlib.pyplot as plt

class ReportingMadeEasy(BaseReport):
    def __init__(self, *args):
        super().__init__(*args)

    def purpose(self) -> str:
        return "📝 Purpose: Verify that each forward pass computes correctly by logging neuron activations across the network."

    def what_to_look_for(self) -> str:
        return """If all activations look the same for every input, something is wrong.
                  If hidden layer activations are all close to 1 or -1, sigmoid/tanh is saturating.
                    If output is always ~0.5, we have bad weight initialization.        
                """

    def report_logic(self, *args):
        """
        This method is invoked when user selects this report from Report Menu
        """
        #models= db.query("Select distinct model_id from Iteration", None,False)
        SQL = "SELECT epoch, weights, biases, mean_absolute_error FROM EpochSummary"
        results = self.db.query(SQL)

        report_data = []
        for row in results:
            weights = self.parse_weights_or_biases(row['weights'], is_weight=True)
            biases = self.parse_weights_or_biases(row['biases'], is_weight=False)
            report_entry = {
                'epoch': row['epoch'],
                'weights': weights,  # Parsed weights with neuron labels
                'biases': biases,    # Parsed biases with neuron labels
                'mean_absolute_error': row['mean_absolute_error'],
            }
            report_data.append(report_entry)

        self.plot_multi_scale(report_data, "Evolution of Error and Neuron Parameters")

    def parse_weights_or_biases(self, data: str, is_weight=True):
        """
        Parses weights or biases from a string in the format:
        '0: [value1, value2]\n1: [value3, value4]' (weights) or
        '0: value1\n1: value2' (biases).

        Args:
            data (str): String containing the weights or biases.
            is_weight (bool): Whether to parse weights (default) or biases.

        Returns:
            dict: A dictionary mapping labels (e.g., 'N0W1') to values.
        """
        parsed = {}
        lines = data.split('\n')
        for line in lines:
            match = re.match(r'(\d+): \[?([-+]?[\d.]+)(?:, ([-+]?[\d.]+))?\]?', line)
            if match:
                neuron = int(match.group(1))  # Directly use the neuron number
                if is_weight:
                    parsed[f'N{neuron}W1'] = float(match.group(2))  # First weight
                    if match.group(3):
                        parsed[f'N{neuron}W2'] = float(match.group(3))  # Second weight
                else:
                    parsed[f'N{neuron}Bias'] = float(match.group(2))
        return parsed



    def plot_multi_scale(self, report_data, title, figsize=(12, 8)):
        """
        Create a multi-scale plot showing weights, biases, and MAE evolution
        with labeled axes for weights and biases.

        Args:
            report_data (list): List of dictionaries containing parsed report data.
            title (str): Title of the plot.
            figsize (tuple): Size of the figure (default: (12, 8)).
        """

        epochs = [entry['epoch'] for entry in report_data]

        # Extract weights, biases, and MAE
        all_weights = {label: [] for label in report_data[0]['weights']}
        for entry in report_data:
            for label, value in entry['weights'].items():
                all_weights[label].append(value)
        biases = {label: [] for label in report_data[0]['biases']}
        for entry in report_data:
            for label, value in entry['biases'].items():
                biases[label].append(value)
        maes = [entry['mean_absolute_error'] for entry in report_data]

        # Create the figure
        fig, host = plt.subplots(figsize=figsize)
        axes = [host] + [host.twinx() for _ in range(len(all_weights) + len(biases))]

        # Adjust for additional axes if needed
        if len(axes) > 2:
            fig.subplots_adjust(right=0.8)
            for i, ax in enumerate(axes[2:], start=2):
                ax.spines['right'].set_position(('axes', 1 + (i - 1) * 0.1))

        # Colors and markers for all lines
        colors = plt.cm.tab10(range(len(all_weights) + len(biases) + 1))  # Unique colors
        markers = ['o', 's', 'x', '^', 'D', 'v', 'p', '*', 'h', '+']

        # Plot weights
        lines = []  # Store all lines for the legend
        labels = []  # Store labels for the legend
        for i, (label, values) in enumerate(all_weights.items()):
            line, = axes[i].plot(epochs, values, label=label, color=colors[i], marker=markers[i % len(markers)])
            axes[i].set_ylabel(label, color=colors[i])
            axes[i].tick_params(axis='y', labelcolor=colors[i])
            lines.append(line)
            labels.append(label)

        # Plot biases
        for i, (label, values) in enumerate(biases.items(), start=len(all_weights)):
            line, = axes[i].plot(epochs, values, label=label, color=colors[i], linestyle='--', marker=markers[i % len(markers)])
            axes[i].set_ylabel(label, color=colors[i])
            axes[i].tick_params(axis='y', labelcolor=colors[i])
            lines.append(line)
            labels.append(label)

        # Plot MAE
        mae_line, = host.plot(epochs, maes, label='MAE', color='black', linestyle='-', marker='s')
        host.set_ylabel('MAE', color='black')
        host.tick_params(axis='y', labelcolor='black')
        lines.append(mae_line)
        labels.append('MAE')

        # Title and labels
        host.set_xlabel('Epoch')
        plt.title(title)
        plt.grid(True)

        # Add legend for all lines
        #host.legend(lines, labels, loc='upper right')
                # Add legend
        host.legend(lines, labels, loc='upper right', bbox_to_anchor=(1, 1))

        # Show plot
        plt.show()





