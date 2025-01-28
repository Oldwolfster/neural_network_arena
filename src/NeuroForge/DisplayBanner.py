import json
from typing import List
import pygame

from src.NeuroForge.DisplayModel import DisplayModel
from src.NeuroForge.EZSurface import EZSurface
from src.engine.RamDB import RamDB

class DisplayBanner(EZSurface):
    def __init__(self, screen : pygame.Surface, problem_type : str, max_epoch : int, max_iteration : int, width_pct  , height_pct, left_pct, top_pct):
        super().__init__    (screen, width_pct, height_pct, left_pct, top_pct, bg_color=(0, 0, 255))
        self.banner_text    = "Loading..."
        self.max_epoch      = max_epoch
        self.max_iteration  = max_iteration
        self.problem_type   = problem_type

    def render(self):
        """Draw inputs on the surface."""
        self.clear()  # Clear surface with background color

        # Set up font and text
        font = pygame.font.Font(None, 36)
        label = font.render(self.banner_text, True, (255, 255, 255))  # White text
        label_rect = label.get_rect(center=(self.width // 2, self.height // 2))  # Center the text

        # Draw the border
        border_rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(self.surface, (0, 0, 0), border_rect, 2)  # Yellow border with 2px thickness

        # Render the text
        self.surface.blit(label, label_rect)


    def update_me(self,  data : dict):
        iteration           = data.get("iteration", "N/A")
        epoch               = data.get("epoch", "N/A")
        self.banner_text    = f"{self.problem_type}    Epoch: {epoch}/{self.max_epoch}    Iteration: {iteration}/{self.max_iteration}"

