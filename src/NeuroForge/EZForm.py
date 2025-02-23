from typing import Dict
import pygame
from src.NeuroForge import Const
from src.NeuroForge.EZSurface import EZSurface

class EZForm(EZSurface):
    """Wrapper for rendering dynamic UI forms with auto-positioning and scaling."""

    def __init__(
        self,
        fields: Dict[str, str],
        width_pct: int,
        height_pct: int,
        left_pct: int,
        top_pct: int,
        banner_text: str = "Form",
        banner_color=Const.COLOR_BLUE,
        bg_color=Const.COLOR_WHITE,
        font_color=Const.COLOR_BLACK
    ):
        super().__init__(Const.SCREEN, width_pct, height_pct, left_pct, top_pct, bg_color)

        self.fields = fields
        self.banner_text = banner_text
        self.banner_color = banner_color
        self.font_color = font_color
        self.banner_font = pygame.font.Font(None, 36)
        self.field_font = pygame.font.Font(None, 24)
        self.spacing = 10

        # Track first-time label positions for arrows
        self.need_label_coord = True
        self.arrow_labels_position = []

    def render(self):
        """Render the form with a banner and dynamic fields."""
        self.clear()

        # Render Banner
        banner_surface = self.banner_font.render(self.banner_text, True, Const.COLOR_WHITE)
        banner_height = banner_surface.get_height() + 8
        banner_rect = pygame.Rect(0, 0, self.width, banner_height)
        pygame.draw.rect(self.surface, self.banner_color, banner_rect, border_radius=4)

        banner_text_rect = banner_surface.get_rect(center=(self.width // 2, banner_height // 2))
        self.surface.blit(banner_surface, banner_text_rect)

        # Draw Outer Box (Form Border)
        border_rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(self.surface, self.banner_color, border_rect, 3, border_radius=4)

        # Adjust starting Y position for fields
        field_start_y = banner_height + self.spacing
        total_fields = len(self.fields)
        field_spacing = (self.height - field_start_y) // total_fields if total_fields else 30

        for i, (label, value) in enumerate(self.fields.items()):
            y_pos_label = field_start_y + (i * field_spacing)
            y_pos_value = y_pos_label + 25

            # Render Label
            label_surface = self.field_font.render(label, True, self.font_color)
            label_rect = label_surface.get_rect(left=self.spacing, centery=y_pos_label)
            self.surface.blit(label_surface, label_rect)

            # Render Input Box
            box_margin = int(self.width * 0.05)
            value_box_rect = pygame.Rect(box_margin, y_pos_value - 15, self.width - (2 * box_margin), 30)

            # Track label positions for arrows
            if self.need_label_coord:
                self.arrow_labels_position.append(y_pos_value)

            pygame.draw.rect(self.surface, Const.COLOR_WHITE, value_box_rect)
            pygame.draw.rect(self.surface, self.banner_color, value_box_rect, 2)

            value_surface = self.field_font.render(value, True, self.font_color)
            value_rect = value_surface.get_rect(center=value_box_rect.center)
            self.surface.blit(value_surface, value_rect)

        if self.arrow_labels_position:
            self.need_label_coord = False
