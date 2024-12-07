import matplotlib.pyplot as plt
import numpy as np

class WeightAnalyzer:
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    def analyze_weight_changes(self, summaries, title, figsize=(15, 10)):
        """
        Create a comprehensive analysis of weight changes
        """
        # Extract data
        epochs = [s.epoch for s in summaries]
        n_weights = len(summaries[0].final_weight)
        weights = []
        weight_changes = []

        for i in range(n_weights):
            weight_data = [s.final_weight[i] for s in summaries]
            weights.append(weight_data)
            # Calculate changes between epochs
            changes = np.diff(weight_data)
            weight_changes.append(changes)

        # Create subplots
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Weight Values Over Time
        ax1 = fig.add_subplot(gs[0, 0])
        for i, weight_data in enumerate(weights):
            ax1.plot(epochs, weight_data,
                    color=self.colors[i],
                    label=f'Weight {i+1}',
                    marker='o',
                    markersize=4)
        ax1.set_title('Weight Values Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Weight Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Weight Changes Over Time
        ax2 = fig.add_subplot(gs[0, 1])
        for i, changes in enumerate(weight_changes):
            ax2.plot(epochs[1:], changes,
                    color=self.colors[i],
                    label=f'Weight {i+1} Change',
                    marker='o',
                    markersize=4)
        ax2.set_title('Weight Changes Between Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Weight Change')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Weight Change Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        positions = range(n_weights)
        box_data = [changes for changes in weight_changes]
        ax3.boxplot(box_data, positions=positions)
        ax3.set_title('Distribution of Weight Changes')
        ax3.set_xlabel('Weight Number')
        ax3.set_ylabel('Change Magnitude')
        ax3.grid(True, alpha=0.3)

        # 4. Weight Change Correlation
        ax4 = fig.add_subplot(gs[1, 1])
        if len(weights) >= 2:
            ax4.scatter(weight_changes[0], weight_changes[2],
                       alpha=0.6,
                       c=self.colors[0])
            ax4.set_title('Weight 1 vs Weight 3 Changes')
            ax4.set_xlabel('Weight 1 Change')
            ax4.set_ylabel('Weight 3 Change')
            # Add trend line
            z = np.polyfit(weight_changes[0], weight_changes[2], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(weight_changes[0]), max(weight_changes[0]), 100)
            ax4.plot(x_trend, p(x_trend), "r--", alpha=0.8)

            # Calculate correlation
            corr = np.corrcoef(weight_changes[0], weight_changes[2])[0,1]
            ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                    transform=ax4.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, (ax1, ax2, ax3, ax4)

# Example usage:
def weight_change_analysis(summaries):
    analyzer = WeightAnalyzer()
    fig, axes = analyzer.analyze_weight_changes(summaries)
    plt.show()



