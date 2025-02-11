import pygame
from src.engine.RamDB import RamDB
import json
class DisplayModel__NeuronWeights:
    def __init__(self, neuron, model_id):
        #Adjustable settings
        self.padding_top = 3
        self.padding_bottom = 3
        self.gap_between_weight_bars = 1
        self.gap_between_weights = 2
        self.right_margin = 0  # New: Space reserved for activation visualization
        self.BANNER_HEIGHT = 29  # 4 pixels above + 26 pixels total height

        self.neuron = neuron  # ✅ Store reference to parent neuron
        self.model_id = model_id
        self.min_weight = float('inf')  # Track min/max for scaling
        self.max_weight = float('-inf')
        self.global_max = 0
        self.max_per_weight = []
        self.num_weights = 0
        self.neuron_height = 0
        self.bar_height  = 0
        self.previous_weights = None  # Store last weights for comparison

    def render(self, screen, ez_printer, body_y_start, weight_text, location_left):
        if self.global_max == 0:
            self.initialize(screen, ez_printer, body_y_start, weight_text, location_left)
        bar_lengths = self.calculate_weight_bar_lengths()
        #self.debug_weight_changes()
        self.draw_bars(screen)


    def debug_weight_changes(self): #Prints weights only if they have changed from the last recorded values.
        current_weights = self.neuron.weights
        if self.previous_weights is None or current_weights != self.previous_weights:
            print(f"🔍 Weights Updated: {current_weights}")
            self.previous_weights = list(current_weights)  # Copy to track changes
    def initialize(self, screen, ez_printer, body_y_start, weight_text, location_left):
        if len(self.neuron.weights) == 0:
            return
        self.global_max, self.max_per_weight = self.get_weight_min_max(self.neuron.db, self.model_id, self.neuron.nid)
        self.num_weights = len(self.neuron.weights)
        self.neuron_height = self.neuron.location_height
        if self.num_weights > 0:
            self.bar_height= self.calculate_bar_height(
                num_weights=self.num_weights,neuron_height=self.neuron_height
                ,padding_top=self.padding_top,padding_bottom=self.padding_bottom
                ,gap_between_weight_bars= self.gap_between_weight_bars,gap_between_weights=self.gap_between_weights
            )
        #print(f"INITIALIZING//////////   Max per weight{self.max_per_weight}\tGlobal max: {self.global_max}")
        #self.debug_bar_placement()

        #print(f"Weights: {self.neuron.weights}")
        #print(f"Bar height={self.bar_height}")
        # print(f"screen info{screen.height, screen.width}")
        # print(f"neuron info{self.neuron.location_left, self.neuron.location_top, self.neuron.location_width, self.neuron.location_height}")
        # print(f"Neuron {self.neuron.nid}: location=({self.neuron.location_left}, {self.neuron.location_top}), "              f"size=({self.neuron.location_width}, {self.neuron.location_height})")        # Calculate bar lengths

    def draw_bars(self, screen):
        """
        Draw all weight bars inside the neuron, considering padding, spacing, and bar height.

        This function ensures bars are evenly spaced and positioned inside the neuron.
        """
        if self.global_max == 0:
            self.initialize(screen, None, None, None, None)  # Ensuring initialization happens before rendering

        bar_lengths = self.calculate_weight_bar_lengths()

        # Compute the starting X and Y positions
        start_x = self.neuron.location_left + 5  # Small left padding for visibility
        #start_y = self.neuron.location_top + self.padding_top  # Start from the top padding
        start_y = self.neuron.location_top + self.padding_top + self.BANNER_HEIGHT
        #print(f"padding top: {self.padding_top}\tStart_y: {start_y}")

        for i, (bar_self, bar_global) in enumerate(bar_lengths):
            # Compute vertical position of this weight's bars
            y_pos = start_y + i * (self.bar_height * 2 + self.gap_between_weights)

            # Call function to draw the two bars for this weight
            self.draw_two_bars_for_one_weight(
                screen, start_x, y_pos, bar_self, bar_global, self.bar_height, self.gap_between_weight_bars
            )

    def calculate_bar_height(self, num_weights, neuron_height, padding_top, padding_bottom, gap_between_weight_bars, gap_between_weights):
        """
        Calculate the height of each weight bar dynamically based on available space.

        :param num_weights: Number of weights for the neuron
        :param neuron_height: Total height of the neuron
        :param padding_top: Space above the first set of bars
        :param padding_bottom: Space below the last set of bars
        :param gap_between_weight_bars: Gap between the two bars of the same weight
        :param gap_between_weights: Gap between different weights
        :return: The calculated height for each individual bar
        """
        # Calculate available space after removing padding
        #available_height = neuron_height - (padding_top + padding_bottom)
        available_height = neuron_height - (padding_top + padding_bottom + self.BANNER_HEIGHT)

        # Each weight has two bars, so total bar slots = num_weights * 2
        total_gaps = (num_weights * gap_between_weights) + (num_weights * 2 - 1) * gap_between_weight_bars

        # Ensure the remaining space is distributed across all bars
        if total_gaps >= available_height:
            raise ValueError("Not enough space in neuron height to accommodate weights and gaps.")

        # Compute actual bar height
        total_bar_height = available_height - total_gaps
        bar_height = total_bar_height / (num_weights * 2)

        return bar_height


    def draw_two_bars_for_one_weight(self, screen, x, y, width_self, width_global, bar_height=8, bar_gap=2):
        """
        Draws two horizontal bars for a single weight visualization.

        Parameters:
            screen       (pygame.Surface): The Pygame screen to draw on.
            x            (float): Left position where the bars should start.
            y            (float): Vertical position for the bars.
            width_self   (float): Length of the self-weight bar.
            width_global (float): Length of the global max-weight bar.
            bar_height   (int): Height of each bar.
            bar_gap      (int): Spacing between the bars.
        """

        # Draw Global Weight Bar (Orange)
        pygame.draw.rect(screen, (255, 165, 0),  # Orange
                         pygame.Rect(x, y, width_global, bar_height))

        # Draw Self Weight Bar (Green), placed below the Global Bar
        pygame.draw.rect(screen, (0, 128, 0),  # Green
                         pygame.Rect(x, y + bar_height + bar_gap, width_self, bar_height))

    def calculate_weight_bar_lengths(self):
        """
        Calculates the bar lengths for visualizing weight magnitudes.
        - The first bar represents the weight's magnitude relative to itself (normalized per weight).
        - The second bar represents the weight's magnitude relative to all weights (global normalization).
        - Bars are scaled relative to the neuron's width minus the right margin.
        """
        if self.global_max == 0 or len(self.neuron.weights) == 0:
            return []

        # Adjust neuron width to account for the right margin
        neuron_width = self.neuron.location_width - self.right_margin
        if neuron_width < 20:  # Prevent bars from being too small
            neuron_width = 20

        bar_lengths = []
        for i, weight in enumerate(self.neuron.weights):
            abs_weight = abs(weight)  # Use absolute value for visualization

            # Normalize relative to this weight's historical max
            self_max = self.max_per_weight[i] if self.max_per_weight[i] != 0 else 1
            norm_self = abs_weight / self_max  # Scale between 0 and 1

            # Normalize relative to the absolute global max weight
            norm_global = abs_weight / self.global_max  # Scale between 0 and 1

            # Scale to neuron width (so bars fit inside the neuron)
            bar_self = norm_self * neuron_width
            bar_global = norm_global * neuron_width

            bar_lengths.append((bar_self, bar_global))

        return bar_lengths

    def get_weight_min_max(self, db: RamDB, model_id: str, neuron_id: int):
        """
        Retrieves:
        1. The global maximum absolute weight across all epochs and neurons.
        2. The maximum absolute weight for each individual weight index across all epochs.

        Returns:
            global_max (float): The single highest absolute weight in the entire model.
            max_per_weight (list): A list of max absolute weights for each weight index.
        """

        #db.query_print("Select weights from neuron")
        # ✅ Query 1: Get the highest absolute weight overall
        SQL_GLOBAL_MAX = """
            SELECT MAX(ABS(value)) AS global_max
            FROM (
                SELECT json_each.value AS value
                FROM Neuron, json_each(Neuron.weights)
                WHERE model = ? and nid = ?
            )
        """
        global_max_result = db.query(SQL_GLOBAL_MAX, (model_id, neuron_id))
        global_max = global_max_result[0]['global_max'] if global_max_result and global_max_result[0][
            'global_max'] is not None else 1.0

        # ✅ Query 2: Get the max absolute weight per weight index
        SQL_MAX_PER_WEIGHT = """
            SELECT key, MAX(ABS(value)) AS max_weight
            FROM (
                SELECT json_each.key AS key, json_each.value AS value
                FROM Neuron, json_each(Neuron.weights)
                WHERE model = ? and nid = ?
            )
            GROUP BY key
            ORDER BY key ASC
        """
        max_per_weight_result = db.query(SQL_MAX_PER_WEIGHT, (model_id, neuron_id))

        # Convert result to a list, ensuring order by index (key)
        max_per_weight = []
        for row in max_per_weight_result:
            index = row['key']
            weight = row['max_weight']
            # Ensure correct placement in the list
            while len(max_per_weight) <= index:
                max_per_weight.append(0)  # Initialize missing indices
            max_per_weight[index] = weight

        return global_max, max_per_weight
    def debug_bar_placement(self):
        """
        Prints detailed information about the neuron and weight bar placement calculations.
        Helps identify potential misalignment issues.
        """
        print("\n🔍 DEBUG: Weight Bar Placement")
        print(f"Neuron {self.neuron.nid}: location=({self.neuron.location_left}, {self.neuron.location_top})")
        print(f"Neuron Size: width={self.neuron.location_width}, height={self.neuron.location_height}")
        print(f"Padding: top={self.padding_top}, bottom={self.padding_bottom}")
        print(f"Gaps: between_weight_bars={self.gap_between_weight_bars}, between_weights={self.gap_between_weights}")
        print(f"Total Number of Weights: {self.num_weights}")
        print(f"Computed Bar Height: {self.bar_height}")

        # Calculate expected space occupied by bars
        total_bar_height = self.num_weights * 2 * self.bar_height
        total_gaps_height = (self.num_weights - 1) * self.gap_between_weights
        expected_total_height = total_bar_height + total_gaps_height + self.padding_top + self.padding_bottom

        print(f"Expected Total Height Occupied: {expected_total_height} (Should be ≤ neuron height: {self.neuron_height})")

        start_y = self.neuron.location_top + self.padding_top
        print(f"Starting Y Position for First Weight Bar: {start_y}")
