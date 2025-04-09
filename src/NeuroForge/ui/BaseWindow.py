import pygame
from src.NeuroForge.EZSurface import EZSurface
from src.NeuroForge import Const
from src.engine.TrainingData import TrainingData
import os
class BaseWindow(EZSurface):
    def __init__(self,  width_pct=98, height_pct=4.369, left_pct=1, top_pct=0, banner_color =Const.COLOR_BLACK,
        banner_text=None,
        background_image_path=None):
        ######### END OF FUNCTION SIGNATURE #########

        """Creates a banner displaying epoch and iteration information."""
        super().__init__(width_pct, height_pct, left_pct, top_pct, bg_color=Const.COLOR_BLUE)


        # Calculate dimensions and position based on percentages
        self.form_width = int(Const.SCREEN_WIDTH * (width_pct / 100))
        self.form_height = int(Const.SCREEN_HEIGHT * (height_pct / 100))
        self.form_left = int(Const.SCREEN_WIDTH * (left_pct / 100))
        self.form_top = int(Const.SCREEN_HEIGHT * (top_pct / 100))

        self.banner_text = banner_text  # 🔥 Centerpiece text
        self.banner_color = banner_color
        self.font = pygame.font.Font(None, 36)  # Use instance variable for efficiency
        self.font_large = pygame.font.Font(None, 69)  # Larger font for center title

        #Calculate Banner Rect
        #banner_surface = self.banner_font.render(self.banner_text, True, Const.COLOR_WHITE)
        #self.banner_height = banner_surface.get_height() + 8
        #self.banner_text_rect = banner_surface.get_rect(center=(self.width // 2, self.banner_height // 2+4))
        #self.banner_rect = pygame.Rect(0, 0, self.form_width, self.banner_height)
        self.load_background(background_image_path)




    def render(self):
        """Draw the banner with the latest text."""
        self.clear()
        margin = 8

        # Left-aligned text (Epoch & Iteration)
        #label_epoch = self.font.render(self.banner_text_left, True, Const.COLOR_WHITE)
        #label_epoch_rect = pygame.Rect(margin, margin, 100, 100)

        # Right-aligned text (Problem Type)
        #label_problem = self.font.render(self.banner_text_rite, True, Const.COLOR_WHITE)
        #label_problem_rect = pygame.Rect(self.width - label_problem.get_width() - margin, margin, label_problem.get_width(), label_problem.get_height())

        # Centered molten glow effect for "Neuro Forge"
        molten_glow, molten_text = self.render_glowing_text(self.banner_text, self.font_large, (255, 50, 0), (255, 150, 50), 5)

        # Get center position
        text_x = (self.width - molten_text.get_width()) // 2
        text_y = (self.height - molten_text.get_height()) // 2 +3

        # 0️⃣ Draw Background Image with translucent overlay (if present)
        if self.background_image:
            self.surface.blit(self.background_image, (0, 0))

        # 4️⃣ Draw Outer Box (Form Border)
        self.form_rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(self.surface, self.banner_color, self.form_rect, 3, border_radius=4)


        # Blit the glow effect first for depth, then the main text
        self.surface.blit(molten_glow, (text_x - 5, text_y - 5))
        self.surface.blit(molten_text, (text_x, text_y))

    def load_background(self, background_image_path):
        if background_image_path:

            image_path = self.resolve_local_asset(background_image_path)
            self.background_image = pygame.image.load(image_path).convert()
            self.background_image = pygame.transform.scale(self.background_image, (self.form_width, self.form_height))

            # After self.background_image is loaded and scaled
            overlay = pygame.Surface((self.form_width, self.form_height), flags=pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))  # RGBA: black with alpha=140 (tweak this as needed)
            self.background_image.blit(overlay, (0, 0))


        #self.holo_gladiators = HoloPanel(            parent_surface=self.surface,            title="🏛️ Gladiators",            left_pct=5,            top_pct=10,            width_pct=40,            height_pct=60        )


    def render_glowing_text(self, text, font, color, glow_color, glow_strength=5):
        """
        Creates a molten glow effect by layering a glow behind the main text.
        """
        base = font.render(text, True, color)
        glow = font.render(text, True, glow_color)
        for _ in range(glow_strength):
            glow = pygame.transform.smoothscale(glow, (glow.get_width() + 1, glow.get_height() + 1))
        return glow, base

    def resolve_local_asset(self, relative_path: str) -> str:
        """
        Resolves a path relative to the current file (NeuroForge).
        Assumes assets are already under 'NeuroForge/assets'.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(current_dir, relative_path))

    def update_me(self):
        pass
