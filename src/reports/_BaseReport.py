import sqlite3
from abc import abstractmethod
from src.neuroForge import mgr
from tabulate import tabulate
from src.reports._ReportUtils import clean_multiline_string

class BaseReport:
    """Base class for SQL-driven reports with reusable query execution and tabulation."""
    
    def __init__(self, *args):
        """
        Initialize the report with a reference to the database.
        :param RamDB: Database connection object
        """
        self.db = args[0]
        mgr.menu_active = False # Close menu

    def run_report(self):
        """
        This method is invoked when user selects this report from Report Menu
        It will print purpose above the report and what to look for underneath
        Need to add ability to enter parameters.
        """
        #if 'Purpose' is availble print it.
        if hasattr(self, 'purpose') and callable(self.purpose):
            print(clean_multiline_string(self.purpose()))

        #run report
        self.report_logic() #delegate to child class for report logic.

        #if 'What to look for' is availble print it.
        if hasattr(self, 'what_to_look_for') and callable(self.what_to_look_for):
            print(clean_multiline_string(self.what_to_look_for()))


    @abstractmethod
    def report_logic(self, *args )  :  #-> List[Tuple[Any, ...]]:
        pass

    def run_sql_report(self, sql: str, params: list = None, limit: int = 10):
        """
        Execute a SQL query, retrieve results, and display them in a tabulated format.
        
        :param sql: SQL query string with optional placeholders.
        :param params: List of parameters for the SQL query.
        :param limit: Number of records to display per batch.
        """
        try:
            result_set = self.db.query(sql, params or [], as_dict=True)
            if not result_set:
                print("No data returned.")
                return
            
            headers = result_set[0].keys()  # Extract column names from first record

            # Loop through the records in batches of `limit`
            for i in range(0, len(result_set), limit):
                batch = result_set[i:i + limit]
                print(tabulate(batch, headers="keys", tablefmt="fancy_grid"))

        except Exception as e:
            print(f"Error running report: {e}")


    """ Notes on future nice reports
    
    Key Objectives for a Better Multi-Neuron Report
Retain the insights from single-neuron reports (seeing the stabilization of weights, bias, and MAE).
Reduce clutter while still providing meaningful analysis.
Allow selective focus on neurons without losing global trends.
🔥 Recommended Approaches
Here are a few strategies to enhance readability and interpretation without sacrificing key insights.

1️⃣ Focus Mode: Select a Single Neuron for Detailed Analysis
📌 What it is:

Allow the user to select one neuron (e.g., via dropdown or hover) and plot only that neuron’s weights, bias, and MAE.
This mirrors the single-neuron report, keeping the insights clear.
📌 Why it's useful:

Keeps things simple.
Maintains the predictable MAE curve vs. weight stabilization trend without clutter.
Can compare different neurons individually if needed.
✅ Implementation:

Create an interactive dropdown list of neurons.
User selects one, and the report updates dynamically to show its trajectory over epochs.
2️⃣ Aggregate Trends with Confidence Bands (Mean & Variability)
📌 What it is:

Instead of showing every weight trajectory, calculate the mean and standard deviation of all neurons' weights.
Plot:
Mean weight trend as a solid line.
Shaded region (confidence band) to show how much variation exists.
📌 Why it's useful:

Reveals overall stabilization trends across the network.
Shows if certain neurons behave differently (outliers will be outside the confidence band).
Helps answer: "Are all neurons stabilizing in a similar pattern?"
✅ Implementation:

Compute mean weight per epoch and standard deviation.
Use Matplotlib's fill_between() to shade the confidence band.
Allow toggling between individual vs. aggregated view.
3️⃣ Weight Change Rate Instead of Absolute Weights
📌 What it is:

Instead of plotting raw weight values (which clutter quickly), plot their rate of change per epoch.
This would show:
High updates early (large adjustments).
Diminishing updates as learning stabilizes.
📌 Why it's useful:

Helps answer: "When does learning slow down?"
Removes the issue of overlapping weights by focusing on relative change.
Works well for networks with many neurons.
✅ Implementation:

Compute first derivative (difference between consecutive epochs).
Plot rate of change over time instead of raw values.
4️⃣ Compare MAE vs. Total Weight Movement (Sum of Changes Per Epoch)
📌 What it is:

Instead of showing individual neurons' weights, track:
Total sum of weight changes per epoch (magnitude of updates across all neurons).
Compare this against MAE, which should show a diminishing return pattern.
📌 Why it's useful:

Reveals how global learning slows down over epochs.
Answers "When do weight updates stop making a meaningful impact?"
Simplifies the report without losing insight.
✅ Implementation:

Compute Σ |Δweight| (sum of absolute weight changes).
Plot this against MAE to show learning progression.
    
    Oscillation Detector (Gradient Direction Change Count)
Concept: Track how often a weight flips direction (i.e., from increasing to decreasing).
How? Count the number of times a weight crosses zero in its gradient:
If weight changes sign between epochs (e.g., +0.2 → -0.3 → +0.1), it’s oscillating.
A high oscillation count over epochs means the training is unstable.
✅ Graph Idea:

Bar chart: Number of times each neuron’s weights flipped sign.
Threshold line: If oscillation is too high, we flag it in red.
2️⃣ Weight Smoothing / Moving Average Comparison
Concept: Compute a smoothed weight trajectory (e.g., a rolling average over 10 epochs).
How?
If the smoothed version follows a clear trend but the raw version jumps around wildly → it's oscillating.
✅ Graph Idea:

Two lines for each neuron’s weight:
Raw weight updates (showing jagged oscillations).
Smoothed weight trajectory (highlighting the intended trend).
If they diverge heavily, we flag instability.
3️⃣ Weight Velocity Plot (Acceleration Indicator)
Concept: Instead of raw weights, plot the rate of change (Δ weight per epoch).
How?
If the weight velocity fluctuates wildly, the model is likely oscillating.
If velocity is consistently near zero, training has plateaued.
If velocity is too high, weights may be exploding.
✅ Graph Idea:

Histogram or line chart of weight velocity.
Zones:
Red for high oscillation (rapid up-down shifts).
Yellow for instability.
Green for smooth convergence.
4️⃣ MAE vs. Weight Stability Heatmap
Concept: Combine weight stability with MAE.
How?
Create a 2D heatmap with:
X-axis = Epochs
Y-axis = Weight Change Rate (stability metric)
If instability increases as MAE decreases → learning rate is too high.
If instability is low but MAE isn't improving → learning rate might be too low.
✅ Graph Idea:

A heatmap where darker areas indicate high instability.
Overlay MAE trend to correlate instability with loss improvem
    
    
    """
