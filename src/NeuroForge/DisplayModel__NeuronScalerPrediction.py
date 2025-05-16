import pygame
from src.engine.Utils import *
from src.NeuroForge import Const
import json

class DisplayModel__NeuronScalerPrediction:
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
        self.gap_between_bars           = 0
        self.gap_between_weights        = 0
        self.BANNER_HEIGHT              = 29  # 4 pixels above + 26 pixels total height #TODO this should be tied to drawing banner rather than a const
        self.right_margin               = 40  # SET IN ITITALNew: Space reserved for activation visualization
        self.padding_bottom             = 3
        self.bar_border_thickness       = 1
        self.oval_height                = 24
        self.oval_overhang              =   1.069
        self.font                       = pygame.font.Font(None, Const.FONT_SIZE_WEIGHT)
        self.font_small                 = pygame.font.Font(None, Const.FONT_SIZE_SMALL)
        # Neuron attributes
        self.neuron                     = neuron  # ✅ Store reference to parent neuron
        self.num_weights                = 0
        self.bar_height                 = 20
        self.max_activation             = 0


        # Weight mechanics
        self.ez_printer                 = ez_printer
        self.my_fcking_labels           = [] # WARNING: Do NOT rename. Debugging hell from Python interpreter defects led to this.
        self.label_y_positions          = [] # for outgoing arrows
        self.need_label_coord           = True #track if we recorded the label positions for the arrows to point from
        self.num_weights                = 0
        self.neuron_height              = self.neuron.location_height


    def render(self):                   #self.debug_weight_changes()

        rs = Const.dm.get_model_iteration_data()
        #ez_debug(rs=rs)
        prediction_raw      = rs.get("prediction_raw",  "[]")
        prediction_unscaled = rs.get("prediction_unscaled",  "[]")
        target_raw          = rs.get("target",  "[]")
        target_unscaled     = rs.get("target_unscaled",  "[]")
        error_raw           = target_raw - prediction_raw
        error_unscaled      = target_unscaled - prediction_unscaled

        prediction_unscaled=smart_format(prediction_unscaled)
        #ez_debug(prediction_unscaled=prediction_unscaled)

        oval_width  = self.neuron.location_width
        oval_height = self.bar_height

        # 2) draw the header (same hack you had before)
        start_x = self.neuron.location_left
        start_y = self.neuron.location_top-5

        #Draws the 3d looking oval behind the header to differentiate
        #self.draw_oval_with_text(     start_x,            start_y,            oval_width,            35,            False,      #overhang            "",            "Scaler",            "",            self.font        )

        start_y = 60
        oval1_y = start_y
        oval2_y = start_y + self.oval_height*2
        oval3_y = start_y + self.oval_height*4

        self.output_one_set(self.neuron.location_left- self.oval_overhang
                            , self.neuron.location_top + oval1_y
                            , target_raw,"Target", target_unscaled)

        self.output_one_set(self.neuron.location_left- self.oval_overhang
                            , self.neuron.location_top + oval2_y
                            , prediction_raw,"Prediction", prediction_unscaled)

        self.output_one_set(self.neuron.location_left- self.oval_overhang
                            , self.neuron.location_top + oval3_y
                            , error_raw,"Error", error_unscaled)

    def output_one_set(self, x_pos, y_pos, label, raw_value, unscaled_value):
        self.draw_oval_with_text(
            x_pos,
            y_pos,
            self.neuron.location_width + self.oval_overhang*2,
            True,
            label,
            raw_value,
            unscaled_value,
            oval_color=Const.COLOR_BLUE,
            text_color=Const.COLOR_WHITE,
            padding=8
        )
        #print(f"LABEL1 ={label}")
        if self.need_label_coord and raw_value == "Prediction":
            #print(f"LABEL2 ={label}")
            self.my_fcking_labels.append((self.neuron.location_left- self.oval_overhang, y_pos+ self.oval_height * .5))
            self.label_y_positions.append(
                (x_pos + self.neuron.location_width + 5, y_pos)
            )
            self.need_label_coord = False


    def draw_oval_with_text(
        self,        x, y,
        proposed_width,  overhang,
        text1, text2, text3,
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

        if overhang:
            width = proposed_width * 1.1
            pill_rect = pygame.Rect(x- proposed_width*.05, y, width, self.oval_height)
        else:
            width = proposed_width
            pill_rect = pygame.Rect(x, y, width, self.oval_height)

        self.draw_pill( pill_rect, oval_color)

        # 2) compute the three areas
        radius = self.oval_height // 2

        label_area  = pygame.Rect(x + 1, y- self.oval_height   *  .69,   width - 2*radius-11, self.oval_height)
        left_area   = pygame.Rect(x, y,            self.oval_height, self.oval_height)
        right_area  = pygame.Rect(x + width - self.oval_height, y, self.oval_height, self.oval_height)
        scale_label = "Unscaled"
        if self.neuron.my_model.layer_width < 195:
            scale_label = ""    #not enough room so remove it.
        if self.neuron.my_model.layer_width < 160:
            text1 = ""    #not enough room so remove it.

        # 3) blit the three texts
        self.blit_text_aligned(self.neuron.screen, smart_format(text1), self.font, text_color, left_area,   'left',   padding)
        self.blit_text_aligned(self.neuron.screen, text2, self.font, Const.COLOR_BLACK, label_area, 'left', padding)
        self.blit_text_aligned(self.neuron.screen, scale_label, self.font, Const.COLOR_BLACK, label_area, 'right', padding)
        self.blit_text_aligned(self.neuron.screen, smart_format(text3), self.font, text_color, right_area,  'right',  padding+30)

    def draw_pill(self,  rect, color):
        """
        Draws a horizontal pill (oval the long way) into rect:
        two half-circles on the ends plus a connecting rectangle.
        """
        x, y, w, h = rect
        radius = h // 2

        # center rectangle
        center_rect = pygame.Rect(x + radius, y, w - 2*radius, h)
        #ez_debug(center_rect=center_rect)
        pygame.draw.rect(self.neuron.screen, color, center_rect)

        # end-caps
        pygame.draw.circle(self.neuron.screen, color, (x + radius, y + radius), radius)
        pygame.draw.circle(self.neuron.screen, color, (x + w - radius, y + radius), radius)

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
