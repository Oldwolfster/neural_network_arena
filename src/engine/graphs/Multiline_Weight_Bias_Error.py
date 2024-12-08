import matplotlib.pyplot as plt
import numpy as np
import src.engine

class WeightVisualizer:
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    def plot_multi_scale(self, summaries,title, figsize=(12, 8)):
        """
        Create a multi-scale plot showing weights, bias, and MAE evolution
        Each metric gets its own y-axis scale to show patterns clearly

        Parameters:
        summaries: List of summary objects containing epoch data
        figsize: Tuple of (width, height) for the figure
        """
        # Extract data from summaries
        epochs = [s.epoch for s in summaries]

        # Get number of weights from first summary
        n_weights = len(summaries[0].final_weight)

        # Prepare data arrays
        weights = []
        for i in range(n_weights):
            weights.append([s.final_weight[i] for s in summaries])

        biases = [s.final_bias for s in summaries]
        maes = [s.total_absolute_error/s.total_samples for s in summaries]

        # Create figure with a shared x-axis
        fig, host = plt.subplots(figsize=figsize)

        # Create parasite axes for each metric
        axes = [host] + [host.twinx() for _ in range(n_weights + 1)]  # +1 for bias and MAE

        # If we have more than 2 extra axes, need to make room for them
        if len(axes) > 2:
            # Adjust spacing for additional y-axes
            fig.subplots_adjust(right=0.8)

            # Manually adjust position of extra axes
            for i, ax in enumerate(axes[2:], start=2):
                ax.spines['right'].set_position(('axes', 1 + (i-1)*0.1))

        # Plot each weight
        lines = []
        labels = []
        for i, weight_data in enumerate(weights):
            line = axes[i].plot(epochs, weight_data,
                              color=self.colors[i % len(self.colors)],
                              linestyle='-',
                              marker='o',
                              markersize=4)
            lines.extend(line)
            labels.append(f'Weight {i+1}')
            axes[i].set_ylabel(f'Weight {i+1}', color=self.colors[i % len(self.colors)])
            axes[i].tick_params(axis='y', labelcolor=self.colors[i % len(self.colors)])

        # Plot bias
        bias_ax = axes[-2]
        line = bias_ax.plot(epochs, biases,
                          color=self.colors[-1],
                          linestyle='--',
                          marker='s',
                          markersize=4)
        lines.extend(line)
        labels.append('Bias')
        bias_ax.set_ylabel('Bias', color=self.colors[-1])
        bias_ax.tick_params(axis='y', labelcolor=self.colors[-1])

        # Plot MAE
        mae_ax = axes[-1]
        line = mae_ax.plot(epochs, maes,
                         color=self.colors[-2],
                         linestyle=':',
                         marker='^',
                         markersize=4)
        lines.extend(line)
        labels.append('MAE')
        mae_ax.set_ylabel('MAE', color=self.colors[-2])
        mae_ax.tick_params(axis='y', labelcolor=self.colors[-2])

        # Set title and labels
        host.set_xlabel('Epoch')
        plt.title(f"{title}Training Evolution with Multiple Scales")

        # Add legend
        host.legend(lines, labels, loc='center left', bbox_to_anchor=(1.1, 0.5))

        return fig, axes

    def plot_normalized(self, summaries, figsize=(12, 8)):
        """
        Create a normalized plot where all metrics are scaled to [0,1] range
        Useful for comparing patterns regardless of absolute values
        """
        epochs = [s.epoch for s in summaries]
        n_weights = len(summaries[0].final_weight)

        # Extract and normalize weights
        normalized_weights = []
        for i in range(n_weights):
            weight_data = [s.final_weight[i] for s in summaries]
            min_val = min(weight_data)
            max_val = max(weight_data)
            normalized = [(x - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                         for x in weight_data]
            normalized_weights.append(normalized)

        # Normalize bias and MAE
        biases = [s.final_bias for s in summaries]
        min_bias, max_bias = min(biases), max(biases)
        normalized_biases = [(x - min_bias) / (max_bias - min_bias) if max_bias != min_bias else 0.5
                           for x in biases]

        maes = [s.total_absolute_error/s.total_samples for s in summaries]
        min_mae, max_mae = min(maes), max(maes)
        normalized_maes = [(x - min_mae) / (max_mae - min_mae) if max_mae != min_mae else 0.5
                          for x in maes]

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot normalized weights
        lines = []
        labels = []
        for i, weight_data in enumerate(normalized_weights):
            line = ax.plot(epochs, weight_data,
                         color=self.colors[i % len(self.colors)],
                         linestyle='-',
                         marker='o',
                         markersize=4)
            lines.extend(line)
            labels.append(f'Weight {i+1}')

        # Plot normalized bias and MAE
        line = ax.plot(epochs, normalized_biases,
                      color=self.colors[-1],
                      linestyle='--',
                      marker='s',
                      markersize=4)
        lines.extend(line)
        labels.append('Bias')

        line = ax.plot(epochs, normalized_maes,
                      color=self.colors[-2],
                      linestyle=':',
                      marker='^',
                      markersize=4)
        lines.extend(line)
        labels.append('MAE')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Normalized Value')
        ax.set_title('Normalized Training Evolution')
        ax.legend(lines, labels)
        ax.grid(True, alpha=1)

        return fig, ax

# Example usage:

def run_multiline_weight_bias_error(summaries, title):
    visualizer = WeightVisualizer()

    # For multi-scale plot
    fig, axes = visualizer.plot_multi_scale(summaries, title)
    plt.show()

    # For normalized plot
    #fig, ax = visualizer.plot_normalized(summaries)
    #plt.show()
