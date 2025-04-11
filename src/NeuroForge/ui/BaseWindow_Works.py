import pygame
from src.NeuroForge.EZSurface import EZSurface
from src.NeuroForge import Const
from src.NeuroForge.ui.GlowText import GlowText
import os

class BaseWindow(EZSurface):
    def __init__(
        self,
        width_pct=60,
        height_pct=60,
        left_pct=20,
        top_pct=15,
        banner_color=Const.COLOR_BLACK,
        banner_text=None,
        background_image_path=None
    ):
        super().__init__(width_pct, height_pct, left_pct, top_pct, bg_color=Const.COLOR_BLUE)

        self.form_width = int(Const.SCREEN_WIDTH * (width_pct / 100))
        self.form_height = int(Const.SCREEN_HEIGHT * (height_pct / 100))
        self.form_left = int(Const.SCREEN_WIDTH * (left_pct / 100))
        self.form_top = int(Const.SCREEN_HEIGHT * (top_pct / 100))

        self.banner_text = banner_text
        self.banner_color = banner_color
        self.font = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 69)

        self.background_image = None
        self.load_background(background_image_path)

        # 🔧 Panel management
        self.panels = []
        self.panel_lookup = {}
        self.title = GlowText(
            text="Match Config",
            x=30,
            y=self.form_height - 90,
            font=self.font,
            surface=self.surface

        )
        ############ END OF CONSTRUCTOR ############

    def load_background(self, background_image_path):
        if background_image_path:
            image_path = self.resolve_local_asset(background_image_path)
            self.background_image = pygame.image.load(image_path).convert()
            self.background_image = pygame.transform.scale(self.background_image, (self.form_width, self.form_height))
            overlay = pygame.Surface((self.form_width, self.form_height), flags=pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            self.background_image.blit(overlay, (0, 0))

    def add_panel(self, title, left_pct, top_pct, width_pct, height_pct, panel_id=None):
        from src.NeuroForge.ui.HoloPanel import HoloPanel  # Local import to avoid circular refs
        panel = HoloPanel(
            parent_surface=self.surface,
            title=title,
            left_pct=left_pct,
            top_pct=top_pct,
            width_pct=width_pct,
            height_pct=height_pct
        )
        self.panels.append(panel)
        if panel_id:
            self.panel_lookup[panel_id] = panel
        return panel


        #for panel in self.panels:
        #    panel.draw_me()

    def update_me(self):
        for panel in self.panels:
            if hasattr(panel, "update_me"):
                panel.update_me()

    def render(self):
        margin = 8
        if self.background_image:
            self.surface.blit(self.background_image, (0, 0))

        self.form_rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(self.surface, self.banner_color, self.form_rect, 3, border_radius=4)

        if self.banner_text:
            molten_glow, molten_text = self.render_glowing_text(
                self.banner_text,
                self.font_large,
                (255, 50, 0),
                (255, 150, 50),
                5
            )
            text_x = (self.width - molten_text.get_width()) // 2
            text_y = (self.height - molten_text.get_height()) // 5
            self.surface.blit(molten_glow, (text_x - 5, text_y - 5))
            self.surface.blit(molten_text, (text_x, text_y))

    def render_glowing_text(self, text, font, color, glow_color, glow_strength=5):
        base = font.render(text, True, color)
        glow = font.render(text, True, glow_color)
        for _ in range(glow_strength):
            glow = pygame.transform.smoothscale(glow, (glow.get_width() + 1, glow.get_height() + 1))
        return glow, base

    def resolve_local_asset(self, relative_path: str) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(current_dir, relative_path))
