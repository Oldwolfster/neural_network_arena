import pygame
from src.NeuroForge import Const
from src.NeuroForge.ui.GlowText import GlowText
from src.NNA.engine.UtilsPyGame import get_darker_color


class HoloPanel:
    def __init__(
        self,
        parent_surface: pygame.Surface,
        title: str,
        left_pct: int,
        top_pct: int,
        width_pct: int,
        height_pct: int,
        fields=None,
        banner_color=Const.COLOR_BLUE,
    ):
        self.parent = parent_surface
        self.title = title
        self.banner_color = banner_color
        self.fields = fields or []

        # Percent-based positioning relative to parent surface
        pw, ph = self.parent.get_size()
        self.width = int(pw * (width_pct / 100))
        self.height = int(ph * (height_pct / 100))
        self.left = int(pw * (left_pct / 100))
        self.top = int(ph * (top_pct / 100))

        self.panel_rect = pygame.Rect(self.left, self.top, self.width, self.height)
        self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        self.banner_font = pygame.font.Font(None, 32)
        self.label_font = pygame.font.Font(None, 22)
        self.spacing = 10

        self.render()

    def render(self):
        #print(f"In HoloPanel update - {self.title}")
        #self.surface.fill((0, 0, 0, 0))  # Clear with transparency

        # 🔹 Transparent black overlay
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 20))  # Lower alpha = more see-through
        self.surface.blit(overlay, (0, 0))  #TODO this is where it isn't ta

        # 🔹 Optional: A faint inner border
        #pygame.draw.rect(self.surface, Const.COLOR_BLACK, self.surface.get_rect(), 5, border_radius=6)

        # 🔹 Title text
        #banner_surface = self.banner_font.render(self.title, True, Const.COLOR_WHITE)
        #self.surface.blit(banner_surface, (self.spacing, self.spacing))
        title_glow = GlowText(
            text=self.title,
            x=self.spacing,
            y=self.spacing,
            font=self.banner_font,
            surface=self.surface
        )
        title_glow.draw_me()

        # 👇 Always blit to parent
        self.blit_to_parent()

    def get_total_rect(self):
        return self.panel_rect

    def get_label_rect(self, index):
        """Returns a rect on the parent surface for the label at row index."""
        row_height = 40
        y = self.top + 40 + index * row_height
        return pygame.Rect(self.left + 10, y, int(self.width * 0.4), 30)

    def get_input_rect(self, index):
        """Returns a rect on the parent surface for the input box at row index."""
        row_height = 40
        y = self.top + 40 + index * row_height
        return pygame.Rect(self.left + int(self.width * 0.45), y, int(self.width * 0.45), 30)

    def blit_to_parent(self):
        self.parent.blit(self.surface, self.panel_rect)
