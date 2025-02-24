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
        banner_text="Form",
        banner_color=Const.COLOR_BLUE,
        bg_color=Const.COLOR_FOR_BACKGROUND,
        font_color=Const.COLOR_BLACK,
        shadow_offset_x=5,  # Can be negative for left-side shadows
        #shadow_offset_y=5
    ):
        #Store the simple stuff
        self.child_name = self.__class__.__name__  # Store the subclass name
        self.shadow_offset_x = shadow_offset_x
        #self.shadow_offset_y = shadow_offset_y
        self.fields = fields
        self.banner_text = banner_text
        self.banner_color = banner_color
        self.font_color = font_color
        self.banner_font = pygame.font.Font(None, 36)
        self.field_font = pygame.font.Font(None, 24)
        self.spacing = 10
        self.shadow_color = Const.COLOR_FOR_SHADOW

        # Calculate dimensions and position based on percentages
        self.form_width = int(Const.SCREEN_WIDTH * (width_pct / 100))
        self.form_height = int(Const.SCREEN_HEIGHT * (height_pct / 100))
        form_left = int(Const.SCREEN_WIDTH * (left_pct / 100))
        form_top = int(Const.SCREEN_HEIGHT * (top_pct / 100))


        # Convert shadow offsets from pixels to percentages (rounding up)
        #shadow_x_pct = round((shadow_offset_x / Const.SCREEN_WIDTH) * 100)
        #shadow_y_pct = round((shadow_offset_y / Const.SCREEN_HEIGHT) * 100)

        # Calculate new width, height, left, and top including shadow expansion
        #width_pct_shad = width_pct + shadow_x_pct
        #height_pct_shad = height_pct + shadow_y_pct
        #left_pct_shad = left_pct - shadow_x_pct if shadow_offset_x < 0 else left_pct
        #top_pct_shad = top_pct - shadow_y_pct if shadow_offset_y < 0 else top_pct

        # Calculate final surface dimensions and placement
        #width_pct_surf = max(width_pct, width_pct_shad)
        #height_pct_surf = max(height_pct, height_pct_shad)
        #left_pct_surf = min(left_pct, left_pct_shad)
        #top_pct_surf = min(top_pct, top_pct_shad)

        # Initialize EZSurface with the total surface space
        #super().__init__(width_pct_surf , height_pct_surf, left_pct_surf, top_pct_surf, bg_color)
        super().__init__(width_pct, height_pct, left_pct, top_pct, bg_color, False,
                         shadow_offset_x, shadow_offset_x,0,0)


        # Store main and shadow rectangles for rendering
        #self.form_rect = pygame.Rect(0, 0, self.width, self.height) #these are pulling from EZSurface
        #self.form_rect = pygame.Rect(form_left, form_top, form_width, form_height)
        self.form_rect = pygame.Rect(0, 0, self.form_width, self.form_height)
        #self.shadow_rect = pygame.Rect(abs(shadow_offset_x), abs(shadow_offset_y), self.width, self.height)
        self.shadow_rect = pygame.Rect(shadow_offset_x, shadow_offset_x, self.form_width, self.form_height)
        #self.surface_rect = pygame.Rect(0, 0, self.width, self.height)  # Surface dimensions
        #self.surface_rect = pygame.Rect(left_pct_surf, top_pct_surf,width_pct_surf,height_pct_surf)

        #Calculate Banner Rect
        banner_surface = self.banner_font.render(self.banner_text, True, Const.COLOR_WHITE)
        self.banner_height = banner_surface.get_height() + 8
        self.banner_text_rect = banner_surface.get_rect(center=(self.width // 2, self.banner_height // 2))
        self.banner_rect = pygame.Rect(0, 0, self.form_width, self.banner_height)

        #print("New FORM!!!!")
        #print(f"shadow_offset_x={shadow_offset_x}\tleft_pct={left_pct}\tleft_pct_shad={left_pct_shad} ")
        #print(f"width_pct_surf={width_pct_surf}\theight_pct_surf={height_pct_surf}\tleft_pct_surf={left_pct_surf}\ttop_pct_surf={top_pct_surf}")
        #print(f"self.form_rect={self.form_rect}")
        #print(f"self.shadow_rect={self.shadow_rect}")
        #print(f"shadow_offset_x={shadow_offset_x}")
        #print(f"self.surface_rect={self.surface_rect}")



        # Track first-time label positions for arrows
        self.need_label_coord = True
        self.arrow_labels_position = []

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
        form_rect = pygame.Rect(0, 0, self.width, self.height)
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
            y_pos_value = y_pos_label + 25

            # Render Label
            label_surface = self.field_font.render(label, True, self.font_color)
            label_rect = label_surface.get_rect(left=self.spacing, centery=y_pos_label)
            self.surface.blit(label_surface, label_rect)

            # Render Input Box
            box_margin = int(self.width * 0.05)
            value_box_rect = pygame.Rect(box_margin, y_pos_value - 15, self.form_width - (2 * box_margin), 30)

            # Track label positions for arrows
            if self.need_label_coord:
                self.arrow_labels_position.append(y_pos_value)

            pygame.draw.rect(self.surface, Const.COLOR_WHITE, value_box_rect)
            pygame.draw.rect(self.surface, self.banner_color, value_box_rect, 2)

            #print(f"Latest value to print on form {value}")
            value_surface = self.field_font.render(value, True, self.font_color)
            value_rect = value_surface.get_rect(center=value_box_rect.center)
            self.surface.blit(value_surface, value_rect)

        if self.arrow_labels_position:
            self.need_label_coord = False
