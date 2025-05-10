import pygame
from src.engine.Utils import *
from src.NeuroForge import Const

class DisplayModel__NeuronWeights:
    """
    DisplayModel__NeuronWeights is created by DisplayModel_Neuron.
    It is an instance of visualizer following the  strategy pattern.
    It holds a reference to DisplayModel__Neuron which is where it gets most of it's information

    This class has the following primary purposes:
    1) Initialize - store information that will not change ( Margins, padding, bar height, etc.
    2) Calculate changing information specific to this visualization (i.e. bar width when weight grows)
    3) Draw the "Standard" components of the neuron.  (Body, Banner, and Banner Text)
    4) Invoke the appropriate "Visualizer" to draw the details of the Neuron
    """

    def __init__(self, neuron, ez_printer):
        # Configuration settings
        self.padding_top                = 3
        self.gap_between_bars    = 0
        self.gap_between_weights        = 2
        self.BANNER_HEIGHT              = 29  # 4 pixels above + 26 pixels total height #TODO this should be tied to drawing banner rather than a const
        self.right_margin               = 40  # SET IN ITITALNew: Space reserved for activation visualization
        self.padding_bottom             = 3
        self.bar_border_thickness       = 1

        # Neuron attributes
        self.neuron                     = neuron  # ✅ Store reference to parent neuron
        self.num_weights                = 0
        self.bar_height                 = 0
        self.max_activation             = 0

        # Weight mechanics
        self.ez_printer                 = ez_printer
        self.my_fcking_labels           = [] # WARNING: Do NOT rename. Debugging hell from Python interpreter defects led to this.
        self.need_label_coord           = True #track if we recorded the label positions for the arrows to point from
        self.initialize()

    def initialize(self):
        """
        Anything that only needs to be determined once should be done here
        i.e. doesn't change from iteration to iteration
        """
        self.num_weights = len(self.neuron.weights)
        self.neuron_height = self.neuron.location_height

        if self.num_weights > 0:
            self.bar_height= self.calculate_bar_height(num_weights=self.num_weights, neuron_height=self.neuron_height, padding_top=self.padding_top,padding_bottom=self.padding_bottom, gap_between_bars= self.gap_between_bars,gap_between_weights=self.gap_between_weights)

    def render(self):                   #self.debug_weight_changes()
        self.draw_weight_bars()

        self.draw_activation_bar()

    def draw_activation_bar(self):
        """
        Draws the activation bar inside the right margin of the neuron.
        - Bar height is scaled relative to the **global max activation**.
        - The bar is drawn from the **bottom up** (low values = short bars).
        - Uses `self.right_margin` as the width.
        """
        if self.neuron.max_activation == 0:  # Safety check
            return

        neuron_x = self.neuron.location_left + self.neuron.location_width  # Start at right edge
        neuron_y = self.neuron.location_top  # Top of neuron

        # 🔹 Normalize activation (scaled to fit the neuron)
        activation_magnitude = abs(self.neuron.activation_value)
        bar_height = (activation_magnitude / self.neuron.max_activation) * self.neuron.location_height - self.BANNER_HEIGHT
        if bar_height > self.neuron_height -  self.BANNER_HEIGHT - 5:   # If activation is to large
            bar_height = self.neuron_height -  self.BANNER_HEIGHT - 5  # Trim it

        # 🔹 Define bar position (grows **upward** from bottom)
        bar_rect = pygame.Rect(
            neuron_x-self.right_margin , neuron_y + self.neuron.location_height - bar_height,  # X,Y (bottom-aligned)
            self.right_margin, bar_height  # Width, Height
        )

        # 🔹 Choose color based on activation sign
        bar_color = Const.COLOR_FOR_ACT_POSITIVE if self.neuron.activation_value >= 0 else Const.COLOR_FOR_ACT_NEGATIVE  # Green for positive, Red for negative

        # 🔹 Draw the activation bar
        draw_rect_with_border(self.neuron.screen, bar_rect, bar_color, 2)

        # return surpresses activation and weighted sum
        # 🔹 Writes the raw sum inside the neuron, bottom right corner, with a background for visibility.
        weighted_sum = self.calculate_weighted_sum()
        text = f" Wt\nSum\n{smart_format(weighted_sum)}"  # ✅ Rounded to 2 decimal places  ✅ Convert to string
        draw_text_with_background(self.neuron.screen, text, self.neuron.location_left + self.neuron.location_width-4, self.neuron.location_top + self.neuron.location_height - 55 , Const.FONT_SIZE_WEIGHT, right_align=True, border_color=Const.COLOR_YELLOW_BRIGHT)

        # 🔹 Writes the activation value inside the neuron, centered on the right wall, with a background for visibility.
        draw_text_with_background(self.neuron.screen, self.neuron.activation_value, self.neuron.location_left + self.neuron.location_width-4, self.neuron.location_top + self.neuron.location_height // 2 , Const.FONT_SIZE_WEIGHT+2, right_align=True, border_color=Const.COLOR_YELLOW_BRIGHT)


    def calculate_weighted_sum(self):
        """
        Calculates the weighted sum that is displayed in bottom right and fed to activation function.
        """
        return sum(
            weight * input_value
            for weight, input_value in zip(self.neuron.weights_before, [1] + self.neuron.neuron_inputs)
        )

    def calculate_bar_height(self, num_weights, neuron_height, padding_top, padding_bottom, gap_between_bars, gap_between_weights):
        """
        Calculate the height of each weight bar dynamically based on available space.

        :param num_weights: Number of weights for the neuron
        :param neuron_height: Total height of the neuron
        :param padding_top: Space above the first set of bars
        :param padding_bottom: Space below the last set of bars
        :param gap_between_bars: Gap between the two bars of the same weight
        :param gap_between_weights: Gap between different weights
        :return: The calculated height for each individual bar
        """
        # Calculate available space after removing padding
        available_height = neuron_height - (padding_top + padding_bottom + self.BANNER_HEIGHT)

        # Each weight has two bars, so total bar slots = num_weights * 2
        total_gaps = (num_weights * gap_between_weights) + (num_weights * 2 - 1) * gap_between_bars

        # Ensure the remaining space is distributed across all bars
        if total_gaps >= available_height:
            raise ValueError(f"Not enough space in neuron height to accommodate weights and gaps.\nNeuron Height: {neuron_height}, Available Height: {available_height},\nTotal Gaps: {total_gaps}, Computed Bar Height: {bar_height}" )

        # Compute actual bar height
        total_bar_height = available_height - total_gaps
        bar_height = total_bar_height / (num_weights * 2)

        return bar_height

    def draw_weight_bars(self):
        """
        Draw all weight bars inside the neuron, considering padding, spacing, and bar height.
        This function ensures bars are evenly spaced and positioned inside the neuron.
        """

        # Compute the starting X and Y positions as well as length of bars
        start_x     = self.neuron.location_left + 5  # Small left padding for visibility
        start_y     = self.neuron.location_top + self.padding_top + self.BANNER_HEIGHT # Start from the top padding
        bar_lengths = self.calculate_weight_bar_lengths()   # For example magnitude of Weight

        for i, (bar_self, bar_global) in enumerate(bar_lengths):    # Compute vertical position of this weight's bars
            y_pos = start_y + i * (self.bar_height * 2 + self.gap_between_weights)

            # Call function to draw the two bars for this weight
            self.draw_two_bars_for_one_weight(start_x, y_pos, bar_self, bar_global, self.bar_height, self.gap_between_bars, self.neuron.weights_before[i],i)

        if len(self.my_fcking_labels) > 0:
            self.need_label_coord = False #Only need to record on first pass.

    def calculate_weight_bar_lengths(self):
        """
        Calculates the bar lengths for visualizing weight magnitudes.
        - The first bar represents the weight's magnitude relative to itself (normalized per weight).
        - The second bar represents the weight's magnitude relative to all weights (global normalization).
        - Bars are scaled relative to the neuron's width minus the right margin.
        """

        # Adjust neuron width to account for the right margin
        neuron_width = self.neuron.location_width - self.right_margin
        if neuron_width < 20:  # Prevent bars from being too small
            neuron_width = 20

        bar_lengths = []
        #if self.neuron.nid ==0:            ez_debug(wts=self.neuron.weights, wts_before=self.neuron.weights_before)
        for i, weight in enumerate(self.neuron.weights_before):
            abs_weight = abs(weight)  # Use absolute value for visualization

            # Normalize relative to this weight's historical max
            self_max = self.neuron.max_per_weight[i] if self.neuron.max_per_weight[i] != 0 else 1
            norm_self = abs_weight / self_max  # Scale between 0 and 1

            # Normalize relative to the absolute global max weight   # Scale between 0 and 1
            norm_global = abs_weight / (Const.MAX_WEIGHT if Const.MAX_WEIGHT != 0 else 1)

            # Scale to neuron width (so bars fit inside the neuron)
            bar_self = norm_self * neuron_width
            bar_global = norm_global * neuron_width
            bar_lengths.append((bar_self, bar_global))
        return bar_lengths

    def draw_two_bars_for_one_weight(self, x, y, width_self, width_global, bar_height, bar_gap
            , weight_value, weight_id):
        """
        Draws two horizontal bars for a single weight visualization with labels.

        - Top bar = Global max reference.
        - Bottom Bar = Self max reference.
        - Labels are drawn inside if space allows, or outside if bars are too small.
        """

        # Create rectangles first
        global_rect = pygame.Rect(x, y, width_global, bar_height)  # Orange bar
        self_rect = pygame.Rect(x, y + bar_height + bar_gap, width_global, bar_height)  # Green bar
        #self_rect = pygame.Rect(x, y + bar_height + bar_gap, width_self, bar_height)  # Green bar

        color1 = Const.COLOR_FOR_BAR1_POSITIVE if weight_value >= 0 else Const.COLOR_FOR_BAR1_NEGATIVE
        color2 = Const.COLOR_FOR_BAR2_POSITIVE if weight_value >= 0 else Const.COLOR_FOR_BAR2_NEGATIVE

        # Draw bars with borders
        draw_rect_with_border(self.neuron.screen, global_rect, color1, self.bar_border_thickness)  # Orange (global max)
        draw_rect_with_border(self.neuron.screen, self_rect, color2, self.bar_border_thickness)  # Green (self max)

        # Draw labels dynamically based on available space
        label_rects=[]
        label_space = self.draw_weight_label(weight_value, global_rect)  # Label for global bar
        # THIS IS READY TO GO ======-> label_space = self.draw_weight_label(weight_value, self_rect)  # Label for self bar
        label_rects.append(label_space)
        label_rects.append(label_space)
        self.draw_weight_index_label(weight_id, y+self.bar_height-9, label_rects)

    def draw_weight_label(self, value_to_print, rect):
        """
        Draws a weight label with a background for readability.

        - If the bar is wide enough, places the label inside the bar.
        - If the bar is too small, places the label outside (to the right).
        - Uses a black semi-transparent background to improve contrast.
        """

        # Define the minimum width required to place the text inside the bar
        min_label_width = 30

        # Create font and render text
        font = pygame.font.Font(None, Const.FONT_SIZE_WEIGHT)
        text_surface = font.render(smart_format(value_to_print), True, Const.COLOR_WHITE)  # White text
        text_rect = text_surface.get_rect()

        # Determine label placement: inside if enough space, otherwise outside
        if rect.width >= min_label_width:
            text_rect.center = rect.center  # Center text inside the bar
        else:
            text_rect.midleft = (rect.right + 5, rect.centery)  # Place outside to the right

        # Ensure label doesn't go out of bounds
        if text_rect.right > self.neuron.screen.get_width():
            text_rect.right = self.neuron.screen.get_width() - 5

        # Draw a semi-transparent background behind the text for readability
        bg_rect = text_rect.inflate(4, 2)  # Slight padding
        pygame.draw.rect(self.neuron.screen, (0, 0, 0, 150), bg_rect)  # Dark transparent background

        # Render text onto screen
        self.neuron.screen.blit(text_surface, text_rect)
        return text_rect # to make sure weight index label doesn't collide.

    def draw_weight_index_label(self, weight_index, y_pos,existing_labels_rects):
        """
        Draws a small label with the weight index on the left wall of the neuron,
        positioned in the middle between the two bars.

        :param weight_index: The index of the weight.
        :param y_pos: The y-position of the weight bars.
        :param existing_labels_rects: list of rects for other labels that might collide.
        """
        # Compute label position
        label_x = self.neuron.location_left  + 5 # Slightly left of the neuron
        label_y = y_pos   # Middle of the two bars

        # Format the label text and get text rect
        label_text = f"Wt #{weight_index}"
        if weight_index == 0:
            label_text = "Bias"
        text_rect = get_text_rect(label_text, Const.FONT_SIZE_WEIGHT) #Get rect for index label.
        text_rect.topleft = label_x,label_y

        if self.neuron.layer == 0 and self.neuron.location_left > text_rect.width + 5:
            label_x = self.neuron.location_left - text_rect.width -  3
            draw_text_with_background(self.neuron.screen, label_text, label_x, label_y, Const.FONT_SIZE_WEIGHT, Const.COLOR_WHITE, Const.COLOR_BLUE, border_color=Const.COLOR_BLACK)
            # Record label loc for input arrows to go to.
            if self.need_label_coord:
                self.my_fcking_labels.append((label_x-text_rect.width * 0.2, label_y))
            return
        if self.need_label_coord:
            self.my_fcking_labels.append((label_x,label_y))

        # Check if there is a collision
        if not check_label_collision(text_rect, existing_labels_rects):
            draw_text_with_background(self.neuron.screen, label_text, label_x, label_y, Const.FONT_SIZE_WEIGHT, Const.COLOR_WHITE, Const.COLOR_BLUE, border_color=Const.COLOR_BLACK)
