import pygame
from src.engine.RamDB import RamDB
from src.engine.Utils import draw_rect_with_border, draw_text_with_background


class DisplayModel__NeuronWeights:
    def __init__(self, neuron, ez_printer):
        #Adjustable settings
        self.neuron = neuron  # ✅ Store reference to parent neuron
        self.num_weights = 0
        self.bar_height = 0
        self.neuron_height = 0
        self.max_activation = 0
        self.ez_printer = ez_printer
        self.padding_top = 3
        self.gap_between_weight_bars = 1
        self.gap_between_weights = 2
        self.BANNER_HEIGHT = 29  # 4 pixels above + 26 pixels total height #TODO this should be tied to drawing banner rather than a const
        self.my_fcking_labels=[]
        self.need_label_coord = True #track if we recorded the label positions for the arrows to point from
        self.right_margin = 40  # SET IN ITITALNew: Space reserved for activation visualization
        self.font_size_weight = 24
        self.initialize()
        """
                
        self.padding_bottom = 3        
                
        self.model_id = model_id
        self.min_weight = float('inf')  # Track min/max for scaling
        self.max_weight = float('-inf')
        
        
        self.need_label_coord = True #track if we recorded the label positions for the arrows to point from
        
        
        
        self.previous_weights = None  # Store last weights for comparison
        
        #TODO consolidate below to above
        self.global_max_activation = 0
        """
    def initialize(self):
        self.num_weights = len(self.neuron.weights)
        self.neuron_height = self.neuron.location_height
        #self.max_activation = self.get_max_activation_for_run(self.neuron.db, self.model_id)

        if self.num_weights > 0:
            self.bar_height= self.calculate_bar_height(
                num_weights=self.num_weights,neuron_height=self.neuron_height
                ,padding_top=self.padding_top,padding_bottom=self.padding_bottom
                ,gap_between_weight_bars= self.gap_between_weight_bars,gap_between_weights=self.gap_between_weights
            )
        #print(f"INITIALIZING//////////   Max per weight{self.max_per_weight}\tGlobal max: {self.global_max}")
        #self.debug_bar_placement()

    def render(self, screen):                   #self.debug_weight_changes()
        self.draw_weight_bars(screen)
        #self.draw_activation_bar(screen)
        #self.draw_activation_value(screen)

    def draw_weight_bars(self, screen):
        """
        Draw all weight bars inside the neuron, considering padding, spacing, and bar height.
        This function ensures bars are evenly spaced and positioned inside the neuron.
        """

        # Compute the starting X and Y positions as well as length of bars
        start_x = self.neuron.location_left + 5  # Small left padding for visibility
        start_y = self.neuron.location_top + self.padding_top + self.BANNER_HEIGHT # Start from the top padding
        bar_lengths = self.calculate_weight_bar_lengths()
        #print(f"padding top: {self.padding_top}\tStart_y: {start_y}")
        print(f"going to visualizer barlength{bar_lengths}")
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
        for i, weight in enumerate(self.neuron.weights):
            abs_weight = abs(weight)  # Use absolute value for visualization

            # Normalize relative to this weight's historical max
            self_max = self.neuron.max_per_weight[i] if self.neuron.max_per_weight[i] != 0 else 1
            print (f"self max{self.neuron.max_per_weight}")
            norm_self = abs_weight / self_max  # Scale between 0 and 1

            # Normalize relative to the absolute global max weight
            norm_global = abs_weight / self.neuron.global_weight_max  # Scale between 0 and 1

            # Scale to neuron width (so bars fit inside the neuron)
            bar_self = norm_self * neuron_width
            bar_global = norm_global * neuron_width
            bar_lengths.append((bar_self, bar_global))
        return bar_lengths

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

        # Draw bars with borders
        draw_rect_with_border(screen, global_rect, (255, 165, 0), 4)  # Orange (global max)
        draw_rect_with_border(screen, self_rect, (0, 128, 0), 4)  # Green (self max)

        # Format label text
        label_text_global = f"{weight_value:.2f}"
        label_text_local = f"{weight_value:.2f}"

        # Draw labels dynamically based on available space
        self.draw_weight_label(screen, label_text_global, global_rect, (255, 165, 0))  # Label for global bar
        self.draw_weight_label(screen, label_text_local, self_rect, (0, 128, 0))  # Label for self bar

    def draw_weight_label(self, screen, text, rect, bar_color):
        """
        Draws a weight label with a background for readability.

        - If the bar is wide enough, places the label inside the bar.
        - If the bar is too small, places the label outside (to the right).
        - Uses a black semi-transparent background to improve contrast.
        """

        # Define the minimum width required to place the text inside the bar
        min_label_width = 30

        # Create font and render text
        font = pygame.font.Font(None, 20)  # Adjusted font size for better readability
        text_surface = font.render(text, True, (255, 255, 255))  # White text
        text_rect = text_surface.get_rect()

        # Determine label placement: inside if enough space, otherwise outside
        if rect.width >= min_label_width:
            text_rect.center = rect.center  # Center text inside the bar
        else:
            text_rect.midleft = (rect.right + 5, rect.centery)  # Place outside to the right

        # Ensure label doesn't go out of bounds
        if text_rect.right > screen.get_width():
            text_rect.right = screen.get_width() - 5

        # Draw a semi-transparent background behind the text for readability
        bg_rect = text_rect.inflate(4, 2)  # Slight padding
        pygame.draw.rect(screen, (0, 0, 0, 150), bg_rect)  # Dark transparent background

        # Render text onto screen
        screen.blit(text_surface, text_rect)

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




