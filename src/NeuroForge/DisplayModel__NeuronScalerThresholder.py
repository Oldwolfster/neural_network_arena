import pygame
from src.NNA.engine.Utils import *
from src.NeuroForge import Const
import json

class DisplayModel__NeuronScalerThresholder:
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
        #print("hello from thresh")
        self.banner_height              = 40
        self.oval_height                = 19
        self.oval_vertical_tweak        = 39.699
        self.oval_overhang              =   12.069
        # Neuron attributes
        self.neuron                     = neuron  # ✅ Store reference to parent neuron
        self.font                       = pygame.font.Font(None, Const.FONT_SIZE_WEIGHT)
        self.font_small                 = pygame.font.Font(None, Const.FONT_SIZE_SMALL)
        #self.neuron.location_top        = 249
        above_neuron                    = neuron.model.prediction_scaler_neuron
        self.neuron.location_top        = above_neuron.location_top + above_neuron.location_height+6.9
        self.neuron.location_height     = 96


        # Weight mechanics
        self.ez_printer                 = ez_printer
        self.my_fcking_labels           = [] # WARNING: Do NOT rename. Debugging hell from Python interpreter defects led to this.
        self.label_y_positions          = []
        self.need_label_coord           = True #track if we recorded the label positions for the arrows to point from

    def render(self):                   #self.debug_weight_changes()
        rs = Const.dm.get_model_iteration_data()
        prediction_raw      = rs.get("prediction_raw",  "[]")
        prediction_unscaled = rs.get("prediction_unscaled",  "[]")
        target_raw          = rs.get("target",  "[]")
        target_unscaled     = rs.get("target_unscaled",  "[]")
        error_raw           = rs.get("error",  "[]")
        error_unscaled      = rs.get("error_unscaled",  "[]")

        self.draw_top_plane()

        question = f"if {smart_format(prediction_raw)} is > {self.neuron.TRI.bd_threshold}"
        alpha =  self.neuron.TRI.bd_target_alpha
        beta =   self.neuron.TRI.bd_target_beta
        alpha_unscaled = round(self.neuron.TRI.bd_target_alpha_unscaled)
        beta_unscaled  = round(self.neuron.TRI.bd_target_beta_unscaled)

        if alpha == alpha_unscaled and beta ==  beta_unscaled: # No scaling
            alpha_txt = alpha
            beta_txt  = beta
        else:
            alpha_txt = f"{alpha} / {alpha_unscaled}"
            beta_txt  = f"{beta} / {beta_unscaled}"


        alpha_txt = self.neuron.TRI.training_data.target_labels[0]
        beta_txt = self.neuron.TRI.training_data.target_labels[1]

        self.output_one_set(1, question ,"", "")
        self.output_one_set(2, "YES ->", "",beta_txt)
        self.output_one_set(3,"NO  ->", "",alpha_txt)
        self.need_label_coord = False

    def output_one_set(self, index, label, raw_value, unscaled_value):

        y_pos = (index - 1) * self.oval_height * 1.1369 +self.oval_vertical_tweak + self.neuron.location_top

        x_offset = 20 if index>1 else 0 # indent 2nd two ovals
        self.draw_oval_with_text(
            self.neuron.location_left- self.oval_overhang+x_offset,
            y_pos,
            self.neuron.location_width + self.oval_overhang*2-x_offset,
            True,
            label,
            raw_value,
            unscaled_value,
            text_color=Const.COLOR_WHITE,
            padding=8
        )
        #print(f"LABEL1 ={label}")
        if self.need_label_coord: #  and raw_value == "Prediction":
            self.label_y_positions.append((self.neuron.location_left + self.oval_overhang + self.neuron.location_width, y_pos+ self.oval_height * .5))
            if raw_value == "Prediction":
                self.my_fcking_labels.append((self.neuron.location_left- self.oval_overhang, y_pos+ self.oval_height * .5))


    def draw_oval_with_text(
        self,        x, y,
        proposed_width,  overhang,
        raw_value, label, unscaled_value,
        text_color=(Const.COLOR_WHITE),
        padding=8
    ):
        """
        Draws a horizontal oval of size (width×height) at (x,y),
        then left-aligns text1 in the left half-circle,
              center-aligns text2 in the middle,
              right-aligns unscaled_value in the right half-circle.
        """
        # 1) draw the shape

        if overhang:
            width = proposed_width * 1.1
            pill_rect = pygame.Rect(x- proposed_width*.05, y, width, self.oval_height)
        else:
            width = proposed_width
            pill_rect = pygame.Rect(x, y, width, self.oval_height)

        self.draw_pill(pill_rect)

        # 2) compute the three areas
        radius = self.oval_height // 2
        txt_y_adj = 52
        label_area  = pygame.Rect(x + 12, y- self.oval_height   *  .69 - 3,   width - 2*radius-11, self.oval_height)
        left_area   = pygame.Rect(x, y ,  self.oval_height, self.oval_height)
        right_area  = pygame.Rect(x + width - self.oval_height-30, y , self.oval_height, self.oval_height)


        if self.neuron.model.layer_width < 160:
            text1 = ""    #not enough room so remove it.

        # 3) blit the three texts


        #self.blit_text_aligned(self.neuron.screen, label_area, label, self.font, Const.COLOR_BLACK,  'left', padding)
        #self.blit_text_aligned(self.neuron.screen, pill_rect,  smart_format(raw_value), self.font, text_color,   'left',   padding)
        #self.blit_text_aligned(self.neuron.screen, label_area, label, self.font, Const.COLOR_BLACK,  'left', padding)
        #self.blit_text_aligned(self.neuron.screen, pill_rect,  smart_format(unscaled_value), self.font, text_color,  'right',  padding)
        global_label_area = label_area.move(self.neuron.model.left, self.neuron.model.top)
        txt_y_adj = 2
        global_pill_rect =  pill_rect.move(self.neuron.model.left,self.neuron.model.top+ txt_y_adj)

        Const.dm.schedule_draw(
            self.blit_text_aligned,
            Const.SCREEN,
            global_label_area,
            label,
            self.font,
            Const.COLOR_BLACK,
            'left',
            padding
        )

        Const.dm.schedule_draw(
            self.blit_text_aligned,
            Const.SCREEN,
            global_pill_rect,
            smart_format(raw_value),
            self.font,
            text_color,
            'left',
            padding
        )

        Const.dm.schedule_draw(
            self.blit_text_aligned,
            Const.SCREEN,
            global_label_area,
            label,
            self.font,
            Const.COLOR_BLACK,
            'left',
            padding
        )

        Const.dm.schedule_draw(
            self.blit_text_aligned,
            Const.SCREEN,
            global_pill_rect,
            smart_format(unscaled_value),
            self.font,
            text_color,
            'right',
            padding
        )

    def draw_differentshapeneuron(self):
        # 2) draw the header (same hack you had before) #Draws the 3d looking oval behind the header to differentiate
        top_plan_rect  = pygame.Rect(self.neuron.location_left, self.neuron.location_top, self.neuron.location_width,self.neuron.location_height )
        self.draw_pill(top_plan_rect)

    def draw_top_plane(self):
        # 2) draw the header (same hack you had before) #Draws the 3d looking oval behind the header to differentiate
        top_plan_rect  = pygame.Rect(self.neuron.location_left, self.neuron.location_top-3.69, self.neuron.location_width, self.banner_height* .369 )
        self.draw_pill(top_plan_rect)
        #self.draw_differentshapeneuron()


    def draw_pill(self,  rect, color= Const.COLOR_BLUE):
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

    def blit_text_aligned(self, surface, area_rect, text, font, color,  align, padding=5):
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
