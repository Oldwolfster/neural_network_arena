import pygame
from src.NeuroForge.EZSurface import EZSurface
from src.NeuroForge import Const

class DisplayBanner(EZSurface):
    def __init__(self, problem_type: str, max_epoch: int, max_iteration: int, width_pct=96, height_pct=4, left_pct=2, top_pct=0):
        """Creates a banner displaying epoch and iteration information."""
        super().__init__(width_pct, height_pct, left_pct, top_pct, bg_color=Const.COLOR_BLUE)
        self.child_name = "Top Banner"
        self.banner_text = "Loading..."
        self.max_epoch = max_epoch
        self.max_iteration = max_iteration
        self.problem_type = problem_type

        self.font = pygame.font.Font(None, 36)  # Use instance variable for efficiency

    def render(self):
        """Draw the banner with the latest text."""
        self.clear()

        # Render the banner text
        label = self.font.render(self.banner_text, True, Const.COLOR_WHITE)
        label_rect = label.get_rect(center=(self.width // 2, self.height // 2))

        # Draw the border
        border_rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(self.surface, Const.COLOR_BLACK, border_rect, 2)

        self.surface.blit(label, label_rect)

    def update_me(self ):
        """Update the banner text dynamically and trigger a re-render."""
        iteration = Const.CUR_ITERATION
        epoch = Const.CUR_EPOCH
        self.banner_text = f"{self.problem_type}    Epoch: {epoch}/{self.max_epoch}    Iteration: {iteration}/{self.max_iteration}"

