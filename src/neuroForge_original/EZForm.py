from typing import Dict, Any
import pygame
from src.neuroForge_original.EZSurface import EZSurface


class EZForm(EZSurface):
    """Wrapper for rendering dynamic forms."""
    def __init__(self, screen: pygame.Surface, fields: Dict[str, str], width_pct: int, height_pct: int, left_pct: int, top_pct: int, banner_text: str = "Form", banner_color=(0, 0, 255), bg_color=(255, 255, 255), font_color=(0, 0, 0)):
        super().__init__(screen, width_pct, height_pct, left_pct, top_pct, bg_color)
        self.fields = fields
        self.banner_text = banner_text
        self.banner_color = banner_color
        self.font_color = font_color
        self.bannerfont = pygame.font.Font(None, 36)  # Default font
        self.font = pygame.font.Font(None, 24)  # Default font
        self.spacing = 10  # Space between fields
        self.need_label_coord = True #track if we recorded the label positions for the arrows to point from
        self.arrow_labels_position =[]

    def render(self):
        """Render the form with a banner and dynamic fields."""
        self.clear()
        #print("Rendering EZForm")
        # Render the banner
        label_text = self.bannerfont.render(self.banner_text, True, self.banner_color)
        text_height = label_text.get_height()
        banner_height = text_height + 8  # Add padding
        banner_rect = pygame.Rect(0, 0, self.width, banner_height)
        pygame.draw.rect(self.surface, self.banner_color, banner_rect, border_radius = 4)

        banner_surface = self.bannerfont.render(self.banner_text, True, (255, 255, 255))  # White text
        banner_rect = banner_surface.get_rect(center=(self.width // 2, banner_height // 2))
        self.surface.blit(banner_surface, banner_rect)

        # Set dimensions of panel
        outer_box_rect = pygame.Rect(0,0,self.width,  self.height)
        shadow_offset = -5 if self.left_pct < .5 else 5
        border_radius = 10  # Adjust for roundness
        # Draw shadow
        shadow_color = (30, 30, 100)  # Darker red for depth
        shadow_rect = outer_box_rect.move(shadow_offset, abs( shadow_offset))
        #pygame.draw.rect(self.surface, shadow_color, shadow_rect) #, border_radius=border_radius)
        #pygame.draw.rect(self.surface, self.bg_color, outer_box_rect) #, border_radius=border_radius)
        #draw main box
        pygame.draw.rect(
            self.surface,
            self.banner_color,  # Blue border
            outer_box_rect,
            3  # Border width
            ,border_radius = 4,
        )
        # Adjust starting Y position for fields (below the banner)
        field_start_y = banner_height + self.spacing + 10
        total_fields = len(self.fields)
        field_spacing = (self.height - field_start_y) // (total_fields)  # Space between fields, vertically

        for i, (label, value) in enumerate(self.fields.items()):
            print(f"number of items{len(self.fields.items())}")
            # Calculate positions for the label and value
            y_pos_label = field_start_y + (i * field_spacing)
            y_pos_value = y_pos_label + 25  # Value box below label

            # Render label
            label_surface = self.font.render(label, True, self.font_color)
            label_rect = label_surface.get_rect(left=self.spacing, centery=y_pos_label)
            self.surface.blit(label_surface, label_rect)

            # Render value in a centered text box spanning 90% of the panel's width
            box_margin = int(self.width * 0.05)  # 5% margin on both sides
            value_box_rect = pygame.Rect(
                box_margin, y_pos_value - 15, self.width - (2 * box_margin), 30  # Adjusted height
            )
            if (self.need_label_coord == True):
                self.arrow_labels_position.append(y_pos_value)
            pygame.draw.rect(self.surface, (255, 255, 255), value_box_rect)  # White box
            pygame.draw.rect(self.surface, self.banner_color, value_box_rect, 2)  # Blue border

            value_surface = self.font.render(value, True, self.font_color)
            value_rect = value_surface.get_rect(center=value_box_rect.center)
            self.surface.blit(value_surface, value_rect)
        if len(self.arrow_labels_position)>0:
            self.need_label_coord= False #Only need to record on first pass.
        print(self.arrow_labels_position)