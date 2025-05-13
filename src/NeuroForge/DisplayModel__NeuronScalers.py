import pygame
from src.engine.Utils import *
from src.NeuroForge import Const
import json

class DisplayModel__NeuronScalers:
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
        self.padding_top                = 5
        self.gap_between_bars    = 0
        self.gap_between_weights        = 0
        self.BANNER_HEIGHT              = 29  # 4 pixels above + 26 pixels total height #TODO this should be tied to drawing banner rather than a const
        self.right_margin               = 40  # SET IN ITITALNew: Space reserved for activation visualization
        self.padding_bottom             = 3
        self.bar_border_thickness       = 1
        self.max_oval_height            =       40
        self.oval_overhang              =   1.1
        self.font                       = pygame.font.Font(None, Const.FONT_SIZE_WEIGHT)
        self.font_small                 = pygame.font.Font(None, Const.FONT_SIZE_SMALL)
        # Neuron attributes
        self.neuron                     = neuron  # ✅ Store reference to parent neuron
        self.num_weights                = 0
        self.bar_height                 = 0
        self.max_activation             = 0

        # Weight mechanics
        self.ez_printer                 = ez_printer
        self.my_fcking_labels           = [] # WARNING: Do NOT rename. Debugging hell from Python interpreter defects led to this.
        self.label_y_positions          = [] # ffor outgoing arrows
        self.need_label_coord           = True #track if we recorded the label positions for the arrows to point from
        self.num_weights = 0
        self.neuron_height = self.neuron.location_height


    def render(self):                   #self.debug_weight_changes()

        rs = Const.dm.get_model_iteration_data()
        #ez_debug(rs=rs)
        unscaled_inputs = rs.get("inputs", "[]")
        scaled_inputs   = rs.get("inputs_unscaled", "[]")
        self.num_weights = len(scaled_inputs)
        #ez_debug(In_Render=self.neuron.model_id)

        self.draw_scale_oval(scaled_inputs, unscaled_inputs)
        #ez_debug(my_fcking_labels=self.my_fcking_labels)

    def draw_scale_oval(self, scaled_inputs, unscaled_inputs):
        """
        Changed for Input Scaler.
        This function ensures ovals are evenly spaced and positioned inside the neuron.
        """
        rs = Const.dm.get_model_iteration_data()
        unscaled_inputs = json.loads(rs.get("inputs", "[]"))
        scaled_inputs   = json.loads(rs.get("inputs_unscaled", "[]"))
        self.num_weights = len(scaled_inputs)

        # 1) figure out our oval size
        self.bar_height = 1 * self.calculate_bar_height(
            num_weights=self.num_weights,
            neuron_height=self.neuron_height,
            padding_top=self.padding_top,
            padding_bottom=self.padding_bottom,
            gap_between_bars=self.gap_between_bars,
            gap_between_weights=self.gap_between_weights
        )
        oval_width  = self.neuron.location_width
        oval_height = self.bar_height

        # 2) draw the header (same hack you had before)
        start_x = self.neuron.location_left
        start_y =             self.neuron.location_top-5

        self.draw_oval_with_text(
            self.neuron.screen,
            start_x,
            start_y,
            oval_width,

            35,
            False,      #overhang
            "",
            "Scaler",
            "",
            self.font
        )

        # 3) compute the exact y-positions for each input
        scale_methods = self.neuron.config.scaler.get_scaling_names()
        y_positions = DisplayModel__NeuronScalers._compute_oval_y_positions(
            top=self.neuron.location_top,
            total_height=self.neuron_height,
            banner_height=self.BANNER_HEIGHT,
            padding_top=self.padding_top,
            padding_bottom=self.padding_bottom,
            num_items=self.num_weights,
            oval_height=oval_height,
            min_gap=self.gap_between_weights
        )

        # 4) draw each input oval, record labels once
        for i, ((scaled, unscaled), method, y_pos) in enumerate(
            zip(zip(scaled_inputs, unscaled_inputs), scale_methods, y_positions)
        ):
            self.draw_oval_with_text(
                self.neuron.screen,
                start_x,
                y_pos,
                oval_width,
                oval_height,
                True,
                scaled,
                method,
                unscaled,
                self.font
            )
            if self.need_label_coord:
                self.my_fcking_labels.append((start_x, y_pos))
                self.label_y_positions.append(
                    (start_x + self.neuron.location_width + 20, y_pos)
                )

        self.need_label_coord = False




    @staticmethod
    def _compute_oval_y_positions(
        top: float,
        total_height: float,
        banner_height: float,
        padding_top: float,
        padding_bottom: float,
        num_items: int,
        oval_height: float,
        min_gap: float
    ) -> list[float]:
        """
        Return y-coordinates so that the (num_items + 1) gaps
        — above the first, between each, and below the last —
        are all equal (never smaller than min_gap).
        """
        if num_items < 1:
            return []

        # total free space after carving out header, paddings, and ovals
        free = (
            total_height
            - banner_height
            - padding_top
            - padding_bottom
            - num_items * oval_height
        )
        gaps = num_items + 1
        raw_gap = free / gaps
        gap = max(raw_gap, min_gap)

        y = top + padding_top + banner_height + gap
        positions = []
        for _ in range(num_items):
            positions.append(y)
            y += oval_height + gap
        return positions



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

    def draw_pill(self, surface, rect, color):
        """
        Draws a horizontal pill (oval the long way) into rect:
        two half-circles on the ends plus a connecting rectangle.
        """
        x, y, w, h = rect
        radius = h // 2

        # center rectangle
        center_rect = pygame.Rect(x + radius, y, w - 2*radius, h)
        pygame.draw.rect(surface, color, center_rect)

        # end-caps
        pygame.draw.circle(surface, color, (x + radius, y + radius), radius)
        pygame.draw.circle(surface, color, (x + w - radius, y + radius), radius)


    def blit_text_aligned(self, surface, text, font, color, area_rect, align, padding=5):
        """
        Renders text into area_rect with one of three alignments:
          'left', 'center', 'right'.
        """
        text=str(text)
        surf = font.render(text, True, color)
        r = surf.get_rect()
        # vertical center
        r.centery = area_rect.centery

        if align == 'left':
            r.x = area_rect.x + padding
        elif align == 'right':
            r.right = area_rect.right - padding
        else:  # center
            r.centerx = area_rect.centerx

        surface.blit(surf, r)


    def draw_oval_with_text(
        self,
        surface,
        x, y,
        proposed_width, proposed_height, overhang,
        text1, text2, text3,
        font,
        oval_color=(Const.COLOR_BLUE),
        text_color=(Const.COLOR_WHITE),
        padding=8
    ):
        """
        Draws a horizontal oval of size (width×height) at (x,y),
        then left-aligns text1 in the left half-circle,
              center-aligns text2 in the middle,
              right-aligns text3 in the right half-circle.
        """
        # 1) draw the shape
        height = min(proposed_height, self.max_oval_height)
        if overhang:
            width = proposed_width * 1.1
            pill_rect = pygame.Rect(x- proposed_width*.05, y, width, height)
        else:
            width = proposed_width
            pill_rect = pygame.Rect(x, y, width, height)

        self.draw_pill(surface, pill_rect, oval_color)

        # 2) compute the three areas
        radius = height // 2
        # left half-circle bounding box
        left_area   = pygame.Rect(x, y,            height, height)
        # middle rectangle
        middle_area = pygame.Rect(x + radius, y,   width - 2*radius, height)
        # right half-circle bounding box
        right_area  = pygame.Rect(x + width - height, y, height, height)

        # 3) blit the three texts
        self.blit_text_aligned(surface, smart_format(text1), self.font, text_color, left_area,   'left',   padding)
        #self.blit_text_aligned(surface, text2, self.font_small, text_color, middle_area, 'center', padding)
        self.blit_text_aligned(surface, smart_format(text3), self.font, text_color, right_area,  'right',  padding+30)

