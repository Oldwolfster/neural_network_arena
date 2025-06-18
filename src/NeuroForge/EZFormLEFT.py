from typing import Dict
import pygame
from src.NeuroForge import Const
from src.NeuroForge.EZSurface import EZSurface
from src.NNA.engine.Utils import get_darker_color


class EZForm(EZSurface):
    """Wrapper for rendering dynamic UI forms with auto-positioning and scaling."""

    def __init__(
        self,
        fields: Dict[str, str],
        width_pct: int,
        height_pct: int,
        left_pct: int,
        top_pct: int,
        banner_text="Form",
        banner_color=Const.COLOR_BLUE,
        bg_color=Const.COLOR_FOR_BACKGROUND,
        font_color=Const.COLOR_BLACK,
        shadow_offset_x=6,  # Can be negative for left-side shadows
        #shadow_offset_y=5
    ):
        #Store the simple stuff
        self.child_name = self.__class__.__name__  # Store the subclass name
        self.shadow_offset_x = shadow_offset_x                
        self.need_label_coord = True    # Track first-time label positions for arrows
        self.label_y_positions = []      # Track first-time label positions for arrows
        #self.shadow_offset_y = shadow_offset_y
        self.fields = fields
        self.banner_text = banner_text
        self.banner_color = banner_color
        self.font_color = font_color
        self.banner_font = pygame.font.Font(None, 36)
        self.field_font = pygame.font.Font   (None, 24)
        self.spacing = 10
        self.shadow_color = Const.COLOR_FOR_SHADOW

        # Calculate dimensions and position based on percentages
        self.form_width = int(Const.SCREEN_WIDTH * (width_pct / 100))
        self.form_height = int(Const.SCREEN_HEIGHT * (height_pct / 100))
        form_left = int(Const.SCREEN_WIDTH * (left_pct / 100))
        form_top = int(Const.SCREEN_HEIGHT * (top_pct / 100))

        super().__init__(width_pct, height_pct, left_pct, top_pct, bg_color, False,
                         shadow_offset_x, shadow_offset_x,0,0)

        self.form_rect = pygame.Rect(self.shadow_offset_x, 0, self.form_width, self.form_height)
        self.shadow_rect = pygame.Rect(0, shadow_offset_x, self.form_width, self.form_height)

        #Calculate Banner Rect
        banner_surface = self.banner_font.render(self.banner_text, True, Const.COLOR_WHITE)
        self.banner_height = banner_surface.get_height() + 8
        self.banner_text_rect = banner_surface.get_rect(center=(self.width // 2, self.banner_height // 2))
        self.banner_rect = pygame.Rect(self.shadow_offset_x, 0, self.form_width, self.banner_height)
        self.render() # Necessary to capture the label positions

    def set_colors(self, correct: int):  #Called from DisplayPanelPrediction
        """
        Allows updating colors dynamically.

        Parameters:
        correct: int: 1 for yes or 0 for now
        """
        if correct:
            self.banner_color =  get_darker_color(Const.COLOR_GREEN_FOREST)
        else:
            self.banner_color = get_darker_color(Const.COLOR_FOR_ACT_NEGATIVE)

    def render(self):
        """Render the form with a banner, background, shadow, and dynamic fields."""
        self.clear()

        # 1️⃣ Draw Shadow (offset in the correct direction)
        #shadow_x = self.shadow_offset if self.shadow_offset > 0 else 0
        #shadow_y = abs(self.shadow_offset)
        #shadow_rect = pygame.Rect(shadow_x, shadow_y, self.width, self.height)
        pygame.draw.rect(
            self.surface,
            self.shadow_color,
            self.shadow_rect,
            border_radius= self.shadow_offset_x* 1
            #border_top_left_radius=20,
            #border_top_right_radius=20,
            #border_bottom_left_radius=20,
            #border_bottom_right_radius=23
        )

        # 2️⃣ Fill Background AFTER Shadow (only inside form area)
        form_rect = pygame.Rect(self.shadow_offset_x, 0, self.width, self.height)
        pygame.draw.rect(self.surface, Const.COLOR_FOR_BACKGROUND, self.form_rect, border_radius=4)

        # 3️⃣ Render Banner
        banner_surface = self.banner_font.render(self.banner_text, True, Const.COLOR_WHITE)
        pygame.draw.rect(self.surface, self.banner_color, self.banner_rect, border_radius=4)
        self.surface.blit(banner_surface, self.banner_text_rect)

        # 4️⃣ Draw Outer Box (Form Border)
        pygame.draw.rect(self.surface, self.banner_color, self.form_rect, 3, border_radius=4)

        # 5️⃣ Adjust starting Y position for fields
        field_start_y = self.banner_height + self.spacing
        total_fields = len(self.fields)
        field_spacing = (self.height - field_start_y) // total_fields if total_fields else 30

        for i, (label, value) in enumerate(self.fields.items()):
            y_pos_label = field_start_y + (i * field_spacing)
            y_pos_value = y_pos_label + 20

            # Render Label
            label_surface = self.field_font.render(label, True, self.font_color)
            label_rect = label_surface.get_rect(left=self.spacing+ self.shadow_offset_x, centery=y_pos_label)
            self.surface.blit(label_surface, label_rect)

            # Render Input Box
            box_margin = int(self.width * 0.05)
            value_box_rect = pygame.Rect(box_margin+self.shadow_offset_x, y_pos_value - 15, self.form_width - (2 * box_margin), 30)

            # Track label positions for arrows (convert to global coordinates)
            if self.need_label_coord:
                global_x = self.left + self.width +16 # Right edge of the box
                global_y = self.top + y_pos_value  # Convert local Y to global Y
                self.label_y_positions.append((global_x, global_y))  # Store full (x, y) position


            pygame.draw.rect(self.surface, Const.COLOR_WHITE, value_box_rect)
            pygame.draw.rect(self.surface, self.banner_color, value_box_rect, 2)

            #print(f"Latest value to print on form {value}")
            value_surface = self.field_font.render(value, True, self.font_color)
            value_rect = value_surface.get_rect(center=value_box_rect.center)
            self.surface.blit(value_surface, value_rect)

        if self.label_y_positions:
            self.need_label_coord = False

