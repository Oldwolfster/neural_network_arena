from abc import ABC, abstractmethod
import pygame
from src.NeuroForge import Const

class EZSurface(ABC):
    def __init__(self, width_pct=100, height_pct=100, left_pct=0, top_pct=0, bg_color=Const.COLOR_WHITE, transparent=False):
        """Creates a resizable and positionable surface within the main screen."""
        self.screen_width = Const.SCREEN_WIDTH
        self.screen_height = Const.SCREEN_HEIGHT

        self.left_pct = left_pct
        self.width_pct = width_pct
        self.height_pct = height_pct

        # Calculate dimensions and position based on percentages
        self.width = int(self.screen_width * (width_pct / 100))
        self.height = int(self.screen_height * (height_pct / 100))
        self.left = int(self.screen_width * (left_pct / 100))
        self.top = int(self.screen_height * (top_pct / 100))

        # Create the surface with optional transparency
        self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA if transparent else 0)
        self.bg_color = bg_color
        self.surface.fill(self.bg_color)  # Set initial background color

    @abstractmethod
    def render(self):
        """Abstract method to be implemented by child classes to render custom content."""
        pass

    def draw_me(self):
        """Clears, renders, and blits the surface onto the main screen."""
        self.clear()
        self.render()
        Const.SCREEN.blit(self.surface, (self.left, self.top))

    def clear(self):
        """Clears the surface with the background color or maintains transparency."""
        if self.surface.get_flags() & pygame.SRCALPHA:
            self.surface.fill((0, 0, 0, 0))  # Clear with transparency
        else:
            self.surface.fill(self.bg_color)

    def resize(self, new_width_pct=None, new_height_pct=None):
        """Dynamically resizes the surface while maintaining position."""
        if new_width_pct:
            self.width_pct = new_width_pct
            self.width = int(self.screen_width * (new_width_pct / 100))

        if new_height_pct:
            self.height_pct = new_height_pct
            self.height = int(self.screen_height * (new_height_pct / 100))

        self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA if self.surface.get_flags() & pygame.SRCALPHA else 0)
        self.clear()  # Refresh surface with new size
