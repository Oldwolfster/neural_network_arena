import pygame
from src.NeuroForge.EZSurface import EZSurface
from src.NeuroForge import Const
import os
from src.NeuroForge.ui.GlowText import GlowText

class BaseWindow(EZSurface):
    def __init__(
        self,
        width_pct=60,
        height_pct=60,
        left_pct=20,
        top_pct=15,
        banner_color=Const.COLOR_BLACK,
        title_text=None,
        background_image_path=None
    ):
        super().__init__(width_pct, height_pct, left_pct, top_pct, bg_color=Const.COLOR_BLUE)

        self.form_width = int(Const.SCREEN_WIDTH * (width_pct / 100))
        self.form_height = int(Const.SCREEN_HEIGHT * (height_pct / 100))
        self.form_left = int(Const.SCREEN_WIDTH * (left_pct / 100))
        self.form_top = int(Const.SCREEN_HEIGHT * (top_pct / 100))
        self.banner_color = banner_color
        self.font = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 69)

        self.background_image = None
        self.load_background(background_image_path)

        # 🔧 Panel management
        self.children = []
        self.child_lookup = {}
        self.title = GlowText(
            title_text,
            30,
            15,
            font=self.font_large,
            surface=self.surface,
            center=True
        )
        ############ END OF CONSTRUCTOR ############

    def load_background(self, background_image_path):
        if background_image_path:
            image_path = self.resolve_local_asset(background_image_path)
            self.background_image = pygame.image.load(image_path).convert()
            self.background_image = pygame.transform.scale(self.background_image, (self.form_width, self.form_height))
            #Add translucent overlay to background
            overlay = pygame.Surface((self.form_width, self.form_height), flags=pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            self.background_image.blit(overlay, (0, 0))

    def resolve_local_asset(self, relative_path: str) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(current_dir, relative_path))

    def update_me(self):
        for child in self.children:
            if hasattr(child, "update_me"):
                child.update_me()

    def render_standard_window_items(self):
        if self.background_image:
            self.surface.blit(self.background_image, (0, 0))
        form_rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(self.surface, self.banner_color, form_rect, 3, border_radius=4)
        self.title.draw_me()

    def render(self):
        self.render_standard_window_items()
        for child in self.children:
            child.render()