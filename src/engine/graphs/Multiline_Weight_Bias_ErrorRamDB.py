import matplotlib.pyplot as plt


def plot_multi_scale(report_data, title, figsize=(12, 8)):
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
