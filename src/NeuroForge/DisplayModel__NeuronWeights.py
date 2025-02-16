import pygame
from src.engine.RamDB import RamDB
import json

from src.engine.Utils import draw_rect_with_border, draw_text_with_background


class DisplayModel__NeuronWeights:
    def __init__(self, neuron, model_id):
        #Adjustable settings
        self.font_size_weight = 24
        self.padding_top = 3
        self.padding_bottom = 3
        self.gap_between_weight_bars = 1
        self.gap_between_weights = 2
        self.right_margin = 40  # SET IN ITITALNew: Space reserved for activation visualization
        self.BANNER_HEIGHT = 29  # 4 pixels above + 26 pixels total height

        self.neuron = neuron  # ✅ Store reference to parent neuron
        self.model_id = model_id
        self.min_weight = float('inf')  # Track min/max for scaling
        self.max_weight = float('-inf')
        self.global_max = 0
        self.max_per_weight = []
        self.my_fcking_labels=[]
        self.need_label_coord = True #track if we recorded the label positions for the arrows to point from
        self.num_weights = 0
        self.neuron_height = 0
        self.bar_height  = 0
        self.previous_weights = None  # Store last weights for comparison
        self.max_act_run = 0
        #TODO consolidate below to above
        self.global_max_activation = 0

    def initialize(self, screen, ez_printer, body_y_start, weight_text, location_left):
        if len(self.neuron.weights) == 0:
            return
        self.global_max, self.max_per_weight = self.get_weight_min_max(self.neuron.db, self.model_id, self.neuron.nid)
        self.num_weights = len(self.neuron.weights)
        self.neuron_height = self.neuron.location_height
        self.max_act_run = self.get_max_activation_for_run(self.neuron.db, self.model_id)
        print(f"MAX ACT={self.max_act_run}")
        if self.num_weights > 0:
            self.bar_height= self.calculate_bar_height(
                num_weights=self.num_weights,neuron_height=self.neuron_height
                ,padding_top=self.padding_top,padding_bottom=self.padding_bottom
                ,gap_between_weight_bars= self.gap_between_weight_bars,gap_between_weights=self.gap_between_weights
            )
        #print(f"INITIALIZING//////////   Max per weight{self.max_per_weight}\tGlobal max: {self.global_max}")
        #self.debug_bar_placement()

    def render(self, screen, ez_printer, body_y_start, weight_text, location_left):
        if self.global_max == 0:
            self.initialize(screen, ez_printer, body_y_start, weight_text, location_left)
        bar_lengths = self.calculate_weight_bar_lengths()
        #self.debug_weight_changes()
        self.draw_weight_bars(screen)
        self.draw_activation_bar(screen)
        self.draw_activation_value(screen)



    def draw_activation_value(self, screen):
        """
        Draws the activation value inside the neuron, centered on the right wall, with a background for visibility.

        :param screen: The pygame screen surface.
        """
        activation_value = round(self.neuron.activation_value, 2)  # ✅ Rounded to 2 decimal places
        text = f"{activation_value}"  # ✅ Convert to string

        # Define text properties
        font = pygame.font.Font(None, 24)  # ✅ Font size
        text_surface = font.render(text, True, (255, 255, 255))  # ✅ White text
        text_rect = text_surface.get_rect()
        text_rect.width/2

        # Calculate position (Middle of right neuron wall)
        text_rect.midleft = (self.neuron.location_left + self.neuron.location_width-text_rect.width,
                             self.neuron.location_top + self.neuron.location_height // 2)

        # Create background rectangle
        padding = 6  # ✅ Space around text
        bg_rect = pygame.Rect(text_rect.x - padding // 2, text_rect.y - padding // 2,
                              text_rect.width + padding, text_rect.height + padding)

        # Draw background (black with slight transparency)
        pygame.draw.rect(screen, (0, 0, 0), bg_rect)  # ✅ Solid black box

        # Draw final white text
        screen.blit(text_surface, text_rect)

    def draw_activation_bar(self, screen):
        """
        Draws the activation bar inside the right margin of the neuron.

        - Bar height is scaled relative to the **global max activation**.
        - The bar is drawn from the **bottom up** (low values = short bars).
        - Uses `self.right_margin` as the width.
        """
        if self.max_act_run == 0:  # Safety check
            return

        neuron_x = self.neuron.location_left + self.neuron.location_width  # Start at right edge
        neuron_y = self.neuron.location_top  # Top of neuron
        neuron_height = self.neuron.location_height  # Full height available

        # 🔹 Normalize activation (scaled to fit the neuron)
        activation_magnitude = abs(self.neuron.activation_value)
        bar_height = (activation_magnitude / self.max_act_run) * neuron_height

        # 🔹 Define bar position (grows **upward** from bottom)
        bar_rect = pygame.Rect(
            neuron_x-self.right_margin , neuron_y + neuron_height - bar_height,  # X,Y (bottom-aligned)
            self.right_margin, bar_height  # Width, Height
        )

        # 🔹 Choose color based on activation sign
        bar_color = (0, 255, 0) if self.neuron.activation_value >= 0 else (255, 0, 0)  # Green for positive, Red for negative

        # 🔹 Draw the activation bar
        #pygame.draw.rect(screen, bar_color, bar_rect)
        draw_rect_with_border(screen, bar_rect, bar_color, 4)
    def get_max_activation_for_run(self, db: RamDB, model_id: str):
        """
        Retrieves the highest absolute activation value across all epochs and iterations for the given model.

        :param db: RamDB instance to query
        :param model_id: The model identifier
        :return: The maximum absolute activation value in the run
        """
        #_Original_not_used_scales_on_all_not_90 =
        SQL_MAX_ACTIVATION = """ 
        
            SELECT MAX(ABS(activation_value)) AS max_activation
            FROM Neuron
            WHERE model = ?
        """
        SQL_MAX_ACTIVATION = """
            SELECT MAX(abs_activation) AS max_activation
            FROM (
                SELECT ABS(activation_value) AS abs_activation
                FROM Neuron
                WHERE model = ?
                ORDER BY abs_activation ASC
                LIMIT (SELECT CAST(COUNT(*) * 0.95 AS INT) 
                       FROM Neuron WHERE model = ?)
            ) AS FilteredActivations;


            """

        result = db.query(SQL_MAX_ACTIVATION, (model_id, model_id))
        print(result)
        # Return the max activation or a default value to prevent division by zero
        return result[0]['max_activation'] if result and result[0]['max_activation'] is not None else 1.0


    ###########EVERYTHING BELOW HERE RELATES TO THE WEIGHTS######################################
    ###########EVERYTHING BELOW HERE RELATES TO THE WEIGHTS######################################
    ###########EVERYTHING BELOW HERE RELATES TO THE WEIGHTS######################################
    ###########EVERYTHING BELOW HERE RELATES TO THE WEIGHTS######################################
    ###########EVERYTHING BELOW HERE RELATES TO THE WEIGHTS######################################
    ###########EVERYTHING BELOW HERE RELATES TO THE WEIGHTS######################################
    ###########EVERYTHING BELOW HERE RELATES TO THE WEIGHTS######################################
    ###########EVERYTHING BELOW HERE RELATES TO THE WEIGHTS######################################

    def debug_weight_changes(self): #Prints weights only if they have changed from the last recorded values.
        current_weights = self.neuron.weights
        if self.previous_weights is None or current_weights != self.previous_weights:
            print(f"🔍 Weights Updated: {current_weights}")
            self.previous_weights = list(current_weights)  # Copy to track changes


    def draw_weight_bars(self, screen):
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
                screen, start_x, y_pos, bar_self, bar_global, self.bar_height, self.gap_between_weight_bars, self.neuron.weights[i],i
            )
            self.draw_weight_index_label(screen, i, y_pos+self.bar_height-9)
        if len(self.my_fcking_labels)>0:
            self.need_label_coord= False #Only need to record on first pass.

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


    def draw_weight_index_label(self, screen, weight_index, y_pos):
        """
        Draws a small label with the weight index on the left wall of the neuron,
        positioned in the middle between the two bars.

        :param screen: The pygame screen to draw on.
        :param weight_index: The index of the weight.
        :param y_pos: The y-position of the weight bars.
        """

        if self.need_label_coord== True:
            #print(f"BEFORE: my_fcking_labels{self.my_fcking_labels} adding {y_pos}")
            self.my_fcking_labels.append(y_pos)
            #print(f"After: my_fcking_labels{self.my_fcking_labels} adding {y_pos}")

        # Compute label position
        label_x = self.neuron.location_left  + 5 # Slightly left of the neuron
        label_y = y_pos   # Middle of the two bars

        # Format the label text
        label_text = f"weight #{weight_index}"

        # Draw the label
        draw_text_with_background(screen,      label_text, label_x, label_y, self.font_size_weight)


    def draw_two_bars_for_one_weight(self, screen, x, y, width_self, width_global, bar_height, bar_gap, weight_value, weight_id):
        """
        Draws two horizontal bars for a single weight visualization with labels.

        - Orange = Global max reference.
        - Green = Self max reference.
        - Labels are drawn inside if space allows, or outside if bars are too small.
        """

        # Create rectangles first
        global_rect = pygame.Rect(x, y, width_global, bar_height)  # Orange bar
        self_rect = pygame.Rect(x, y + bar_height + bar_gap, width_self, bar_height)  # Green bar

        # Draw bars
        draw_rect_with_border(screen,global_rect, (255, 165, 0),4)# Orange (global)
        #pygame.draw.rect(screen, (255, 165, 0), global_rect)  # Orange (global)
        #pygame.draw.rect(screen, (0, 128, 0), self_rect)  # Green (self)
        draw_rect_with_border(screen, self_rect, (0, 128, 0),4)

        # Format label text (now passed explicitly)
        label_text_global = f"{weight_value:.2f}"
        label_text_local = f"{weight_value:.2f}"

        # Call label function with the actual rectangle positions
        self.draw_weight_label(screen, label_text_global, global_rect, (255, 165, 0))  # Label for global bar
        self.draw_weight_label(screen, label_text_local, self_rect, (0, 128, 0))  # Label for self bar

    def draw_weight_label(self, screen, text, rect, bar_color):
        """
        Draws a weight label with a background for readability.

        - If the bar is wide enough, places the label inside the bar.
        - If the bar is too small, places the label outside (to the right).
        - Uses a black semi-transparent background to improve contrast.

        Parameters:
            screen     (pygame.Surface): The surface to draw on.
            text       (str): The weight value as a formatted string.
            rect       (pygame.Rect): The bar rectangle (determines placement).
            bar_color  (tuple): The RGB color of the bar (for future use, e.g., dynamic contrast).
        """

        # Define the minimum width required to place the text inside the bar
        min_label_width = 30

        # Create font and render text
        font = pygame.font.Font(None, self.font_size_weight)  # Small font for weight labels
        text_surface = font.render(text, True, (255, 255, 255))  # White text
        text_rect = text_surface.get_rect()

        # Determine label placement: inside if enough space, otherwise outside
        if rect.width >= min_label_width:
            text_rect.center = rect.center  # Center text inside the bar
        else:
            text_rect.midleft = (rect.right + 5, rect.centery)  # Place outside to the right

        # Draw a semi-transparent background behind the text for readability
        bg_rect = text_rect.inflate(4, 2)  # Slight padding
        pygame.draw.rect(screen, (0, 0, 0, 150), bg_rect)  # Dark transparent background

        # Render text onto screen
        screen.blit(text_surface, text_rect)

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


        """ ALTERNATE WEIGHT SQL.  Base scale on bottom 90% to reduce outlier impact
        WITH OrderedWeights AS (
            SELECT ABS(json_each.value) AS abs_weight
            FROM Neuron, json_each(Neuron.weights)
            WHERE model = ? AND nid = ?
            ORDER BY abs_weight DESC
            LIMIT (SELECT COUNT(*) * 0.9 FROM Neuron, json_each(Neuron.weights) WHERE model = ? AND nid = ?)
        )
        SELECT MAX(abs_weight) AS adjusted_max FROM OrderedWeights;
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
