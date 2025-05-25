import pygame
from src.NeuroForge.EZSurface import EZSurface
from src.NeuroForge import Const
from src.engine.TrainingData import TrainingData

class DisplayBanner(EZSurface):
    def __init__(self, training_data: TrainingData, max_epoch: int, max_iteration: int, width_pct=98, height_pct=4.369, left_pct=1, top_pct=0):
        """Creates a banner displaying epoch and iteration information."""
        super().__init__(width_pct, height_pct, left_pct, top_pct, bg_color=Const.COLOR_BLUE)
        self.child_name = "Top Banner"

        self.banner_text_left = "Loading..."
        self.banner_text_rite = ""
        self.banner_text_mid = "Neuro Forge"  # 🔥 Centerpiece text
        self.max_epoch = max_epoch
        self.max_iteration = max_iteration
        self.problem_type = training_data.problem_type
        self.font = pygame.font.Font(None, 36)  # Use instance variable for efficiency
        self.font_large = pygame.font.Font(None, 69)  # Larger font for center title

    def render(self):
        """Draw the banner with the latest text."""
        self.clear()

        margin = 8

        # Left-aligned text (Epoch & Iteration)
        label_epoch = self.font.render(self.banner_text_left, True, Const.COLOR_WHITE)
        label_epoch_rect = pygame.Rect(margin, margin, 100, 100)

        # Right-aligned text (Problem Type)
        label_problem = self.font.render(self.banner_text_rite, True, Const.COLOR_WHITE)
        label_problem_rect = pygame.Rect(self.width - label_problem.get_width() - margin, margin, label_problem.get_width(), label_problem.get_height())

        # Centered molten glow effect for "Neuro Forge"
        molten_glow, molten_text = self.render_glowing_text(self.banner_text_mid, self.font_large, (255, 50, 0), (255, 150, 50), 5)

        # Get center position
        text_x = (self.width - molten_text.get_width()) // 2
        text_y = (self.height - molten_text.get_height()) // 2 +3

        # Draw the border
        border_rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(self.surface, Const.COLOR_BLACK, border_rect, 4)

        # Blit elements to surface
        self.surface.blit(label_epoch, label_epoch_rect)
        self.surface.blit(label_problem, label_problem_rect)

        # Blit the glow effect first for depth, then the main text
        self.surface.blit(molten_glow, (text_x - 5, text_y - 5))
        self.surface.blit(molten_text, (text_x, text_y))

    def render_glowing_text(self, text, font, color, glow_color, glow_strength=5):
        """
        Creates a molten glow effect by layering a glow behind the main text.
        """
        base = font.render(text, True, color)
        glow = font.render(text, True, glow_color)
        for _ in range(glow_strength):
            glow = pygame.transform.smoothscale(glow, (glow.get_width() + 1, glow.get_height() + 1))
        return glow, base

    def update_me(self):
        """Update the banner text dynamically and trigger a re-render."""
        iteration = Const.vcr.CUR_ITERATION
        epoch = Const.vcr.CUR_EPOCH_MASTER
        self.banner_text_rite = f"{Const.TRIs[0].training_data.arena_name}: {self.problem_type}"

        self.banner_text_left = f"Epoch: {epoch}/{self.max_epoch} Sample: {iteration}/{self.max_iteration}"
