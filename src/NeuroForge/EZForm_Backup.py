from typing import Dict, Any
import pygame
from src.NeuroForge.EZSurface import EZSurface


class EZForm(EZSurface):
    """Wrapper for rendering dynamic forms."""
    def __init__(self, screen: pygame.Surface, fields: Dict[str, str], width: int, height: int, left: int, top: int, bg_color=(255, 255, 255)):
        super().__init__(screen, width, height, left, top, bg_color)
        self.fields = fields
        self.font = pygame.font.Font(None, 24)  # Default font
        self.spacing = 10  # Space between fields

    def render(self):
        """Render labels and values dynamically with better alignment."""
        self.clear()
        total_fields = len(self.fields)
        field_spacing = self.height // (total_fields + 1)  # Space between fields, vertically

        for i, (label, value) in enumerate(self.fields.items()):
            # Calculate positions for the label and value
            y_pos_label = (i + 1) * field_spacing - 20  # Label above
            y_pos_value = (i + 1) * field_spacing + 10  # Value below

            # Render label
            label_surface = self.font.render(label, True, (0, 0, 0))
            label_rect = label_surface.get_rect(center=(self.width // 2, y_pos_label))
            self.surface.blit(label_surface, label_rect)

            # Render value in a centered text box spanning 90% of the panel's width
            box_margin = int(self.width * 0.05)  # 5% margin on both sides
            value_box_rect = pygame.Rect(
                box_margin, y_pos_value - 10, self.width - (2 * box_margin), 30  # Adjusted width
            )
            pygame.draw.rect(self.surface, (255, 255, 255), value_box_rect)  # White box
            pygame.draw.rect(self.surface, (0, 0, 255), value_box_rect, 2)  # Blue border

            value_surface = self.font.render(value, True, (0, 0, 0))
            value_rect = value_surface.get_rect(center=value_box_rect.center)
            self.surface.blit(value_surface, value_rect)


    def update(self, updates: Dict[str, Any]):
        """Update field values dynamically."""
        self.fields.update(updates)
        print("Updating fields:", updates)

