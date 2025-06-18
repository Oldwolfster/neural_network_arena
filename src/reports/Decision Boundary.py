import re
import numpy as np
import matplotlib.pyplot as plt
from src.reports._BaseReport import BaseReport
from src.NNA.engine.RamDB import RamDB

class ReportDecisionBoundary(BaseReport):
    def __init__(self, *args):
        super().__init__(*args)

    def purpose(self) -> str:
        return "📈 Purpose: Visualize the decision boundaries of the first hidden layer."

    def what_to_look_for(self) -> str:
        return """Check if decision boundaries effectively separate data points.
                  If the boundaries do not separate the classes well, the model might need different initialization, activations, or more training."""

    def report_logic(self, *args):
        """
        This method is invoked when the user selects this report from the Report Menu.
        """
        SQL = """
            SELECT n.nid, n.layer_id, w.weight_id, w.value AS weight, n.bias
            FROM Neuron n
            JOIN Weight w 
                ON n.nid = w.nid 
                AND n.epoch_n = w.epoch  
                AND n.iteration_n = w.iteration  
            WHERE n.layer_id = 0  AND w.weight_id != 0  
            AND n.epoch_n = (SELECT MAX(epoch_n) FROM Neuron)  
            AND n.iteration_n = (
                SELECT MAX(iteration_n) - 1  
                FROM Neuron 
                WHERE epoch_n = (SELECT MAX(epoch_n) FROM Neuron)
            )  
            ORDER BY n.nid, w.weight_id;
        """

        results = self.db.query(SQL)

        if not results:
            print("🚨 No data found for the first hidden layer!")
            return

        boundary_points = self.extract_decision_boundary(results)

        # ✅ Debugging - Ensure we are capturing neurons correctly
        if not boundary_points:
            print("⚠️ No neurons found with exactly 2 inputs! Check the SQL query or data.")
            return

        print(f"✅ Found {len(boundary_points)} neurons for decision boundary.")
        for nid, params in boundary_points.items():
            print(f"Neuron {nid}: Weights = {params['weights']}, Bias = {params['bias']}")
        training_data = self.extract_training_data()
        self.plot_decision_boundary(boundary_points, training_data  )

    def extract_training_data(self):
        """
        Fetch training data points to overlay on the decision boundary plot.
        """
        SQL = """
        SELECT inputs, target
        FROM Iteration
        WHERE epoch = (SELECT MAX(epoch) FROM Iteration)
        ORDER BY iteration;
        """
        results = self.db.query(SQL)

        if not results:
            print("🚨 No training data found!")
            return []

        training_data = []
        for row in results:
            inputs = eval(row["inputs"])  # Convert stored string to list
            target = row["target"]
            training_data.append((inputs[0], inputs[1], target))

        return training_data


    def plot_decision_boundary(self, boundary_points, training_data):
        """
        Plots decision boundaries for the first hidden layer neurons and overlays training data.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        x_range = np.linspace(-1.5, 1.5, 100)  # Define the X-axis range

        for neuron_id, params in boundary_points.items():
            w1, w2 = params["weights"].values()  # Extract w1, w2
            bias = params["bias"]

            # Decision boundary equation: w1*x + w2*y + bias = 0
            y_range = (-w1 * x_range - bias) / w2 if w2 != 0 else np.full_like(x_range, -bias / w1)

            ax.plot(x_range, y_range, label=f"Neuron {neuron_id}")

        # Overlay training data points
        for x1, x2, target in training_data:
            marker = "o" if target == 1 else "s"
            color = "red" if target == 1 else "blue"
            ax.scatter(x1, x2, color=color, marker=marker, edgecolors="black", s=100)

        ax.set_xlabel("Input 1")
        ax.set_ylabel("Input 2")
        ax.set_title("First Hidden Layer Decision Boundary with Training Data")
        ax.legend()
        plt.grid()
        plt.show()


    def extract_decision_boundary(self, results):
        """
        Extracts weights and biases for the decision boundary plot from the first hidden layer.
        """
        boundary_points = {}

        for row in results:
            nid = row["nid"]
            weight_id = row["weight_id"]
            weight = row["weight"]
            bias = row["bias"]

            # Store weights only for neurons with exactly 2 inputs
            if nid not in boundary_points:
                boundary_points[nid] = {"weights": {}, "bias": bias}

            boundary_points[nid]["weights"][weight_id] = weight

        # Ensure only neurons with 2 inputs are used
        return {
            nid: data for nid, data in boundary_points.items() if len(data["weights"]) == 2
        }
    import numpy as np
    import matplotlib.pyplot as plt

