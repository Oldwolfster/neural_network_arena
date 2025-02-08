from abc import ABC, abstractmethod
import pygame


class EZSurface(ABC):
    def __init__(self, screen: pygame.Surface, width_pct=100, height_pct=100, left_pct=0, top_pct=0, bg_color=(255, 255, 255)):
        self.screen = screen
        self.screen_width = screen.get_width()
        self.screen_height = screen.get_height()

        self.left_pct = left_pct
        # Calculate dimensions and position based on percentages
        self.width = int(self.screen_width * (width_pct / 100))
        self.height = int(self.screen_height * (height_pct / 100))
        self.left = int(self.screen_width * (left_pct / 100))
        self.top = int(self.screen_height * (top_pct / 100))

        # Create the surface
        self.surface = pygame.Surface((self.width, self.height))
        self.bg_color = bg_color
        self.surface.fill(self.bg_color)  # Set initial background color

    @abstractmethod
    def render(self):
        """Abstract method to be implemented by child classes to render custom content."""
        pass

    def draw_me(self):
        """Default method to clear the surface, invoke rendering, and blit to the screen."""
        self.clear()
        self.render()  # Call the child class's implementation
        self.screen.blit(self.surface, (self.left, self.top))

    def clear(self):
        """Clear the surface with the background color."""
        self.surface.fill(self.bg_color)
