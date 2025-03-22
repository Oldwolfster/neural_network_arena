import re
import matplotlib.pyplot as plt
from src.reports._BaseReport import BaseReport
import json  # Use json if your strings are JSON formatted; otherwise, eval may be acceptable in a trusted context

class LRReport(BaseReport):
    def __init__(self, *args):
        super().__init__(*args)

    def purpose(self) -> str:
        return "📝 Purpose: Visualize the evolution of per-weight learning rates over epochs."

    def what_to_look_for(self) -> str:
        return (
            "Look for trends in how each weight's learning rate changes over time. "
            "A stable, well-adapted learning rate should gradually adjust to the network’s needs. "
            "Watch out for excessive oscillations or plateaus that may indicate issues with the adaptive LR strategy."
        )



    def report_logic(self, *args):
        """
        Query the EpochSummary table for epoch, learning_rates, and mean_absolute_error, parse the LR values,
        and plot the evolution of each weight's LR on a common scale along with the MAE on a separate vertical axis.
        """
        # Updated SQL to include mean_absolute_error and join with EpochSummary
        SQL = ("""
        SELECT  epoch_n as epoch, nid, learning_rates, mean_absolute_error 
        FROM    Neuron N
        JOIN    EpochSummary S
          ON    N.epoch_n = S.epoch
        WHERE   iteration_n == 1 and epoch_n > 10
        ORDER   BY epoch_n
        """
        )
        results = self.db.query(SQL)

        # Aggregate learning rates per (nid, weight index) over epochs
        # Using a dict where key = (nid, weight index) and value = {"epochs": [...], "rates": [...]}
        import json
        aggregated_data = {}
        # Collect MAE per epoch; assume MAE is the same for a given epoch
        mae_data = {}

        for row in results:
            epoch = row["epoch"]
            nid = row["nid"]
            try:
                # Parse the JSON string to a list
                lr_list = json.loads(row["learning_rates"])
            except Exception as e:
                # Skip this row if parsing fails
                continue

            # Loop over each learning rate in the list; index 0 is the bias and others represent weights
            for i, lr in enumerate(lr_list):
                key = (nid, i)
                if key not in aggregated_data:
                    aggregated_data[key] = {"epochs": [], "rates": []}
                aggregated_data[key]["epochs"].append(epoch)
                aggregated_data[key]["rates"].append(lr)

            # Capture the MAE; if duplicates occur per epoch, we use the first occurrence
            if epoch not in mae_data:
                mae_data[epoch] = row["mean_absolute_error"]

        # Ensure the data points are sorted by epoch for each nid/weight combo
        for key, d in aggregated_data.items():
            sorted_pairs = sorted(zip(d["epochs"], d["rates"]), key=lambda x: x[0])
            if sorted_pairs:
                aggregated_data[key]["epochs"], aggregated_data[key]["rates"] = zip(*sorted_pairs)
            else:
                aggregated_data[key]["epochs"], aggregated_data[key]["rates"] = [], []

        # Sort MAE data by epoch
        sorted_mae = sorted(mae_data.items(), key=lambda x: x[0])
        if sorted_mae:
            mae_epochs, mae_values = zip(*sorted_mae)
        else:
            mae_epochs, mae_values = [], []

        # Plot the learning rates and MAE using a new helper method
        self._plot_learning_rates_and_mae(aggregated_data, mae_epochs, mae_values)


    def _plot_learning_rates_and_mae(self, lr_data, mae_epochs, mae_values):
        """
        Helper method to plot the evolution of learning rates along with MAE on a twin y-axis.

        :param lr_data: dict where key = (nid, weight index) and value = {"epochs": list, "rates": list}
        :param mae_epochs: sorted list of epochs for MAE
        :param mae_values: corresponding MAE values for the epochs
        """
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots()

        # Create a twin axis for MAE
        ax2 = ax1.twinx()

        # Plot the MAE on ax2 with a distinct style (dash-dot, thicker line, and black color)
        mae_line, = ax2.plot(mae_epochs, mae_values, linestyle='-.', linewidth=2.5, color='black', label="MAE")

        # Plot each learning rate line on ax1
        lr_lines = []
        for (nid, weight_idx), d in lr_data.items():
            epochs = d["epochs"]
            rates = d["rates"]

            # Label: bias (index 0) as 'B', subsequent weights as H1, H2, etc.
            if weight_idx == 0:
                label = f"N{nid} Bias"
            else:
                label = f"N{nid} W{weight_idx}"

            line, = ax1.plot(epochs, rates, marker='o', label=label)
            lr_lines.append(line)

        # Set axes labels and title
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Learning Rate")
        ax2.set_ylabel("Mean Absolute Error")
        plt.title("Learning Rate Evolution Over Epochs & MAE")

        # Combine legends from both axes so that MAE appears at the top
        # Since MAE is plotted first, include its handle before the others
        lines = [mae_line] + lr_lines
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='best')

        plt.grid(True)
        plt.tight_layout()
        plt.show()
