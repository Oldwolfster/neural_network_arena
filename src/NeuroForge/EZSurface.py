from abc import ABC, abstractmethod
import pygame
from src.NeuroForge import Const

class EZSurface(ABC):
    """
    This is an abstract class, in that it is never instantiated directly, rather
    the components Display_Manager manages inherit from it. It provides...
    1)  Standard methods for updating and rendering.
    2)  An independent surface (similar to "camera" in systems like Unity,
        allowing for independent zooming and positioning.
    """
    __slots__ = ("screen_width", "screen_height", "left_pct", "width_pct", "top_pct", "height_pct", "width", "height", "left", "top", "surface", "bg_color")
    def __init__(self, width_pct=100, height_pct=100, left_pct=0, top_pct=0,bg_color=Const.COLOR_WHITE,transparent=False,
            pixel_adjust_width = 0, pixel_adjust_height = 0, pixel_adjust_left = 0, pixel_adjust_top = 0):

        """Creates a resizable and positionable surface within the main screen."""
        self.screen_width = Const.SCREEN_WIDTH  #TODO remove these
        self.screen_height = Const.SCREEN_HEIGHT #TODO remove these
        self.left_pct = left_pct
        self.width_pct = width_pct
        self.top_pct = top_pct
        self.height_pct = height_pct


        # Calculate dimensions and position based on percentages
        self.width = int(self.screen_width * (width_pct / 100)) + pixel_adjust_width  #remove the plus 50
        self.height = int(self.screen_height * (height_pct / 100)) + pixel_adjust_height
        self.left = int(self.screen_width * (left_pct / 100)) + pixel_adjust_left
        self.top = int(self.screen_height * (top_pct / 100)) + pixel_adjust_top

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
        """Clears the surface only if `clear_surface` is enabled."""
#        if not self.clear_surface:
#            print(f"Skipping clear for {self.__class__.__name__}")
#            return  # ✅ Skip clearing if not needed
#
#        print(f"Clearing surface for {self.__class__.__name__}")  # 🔍 Debugging line

        if self.surface.get_flags() & pygame.SRCALPHA:
            self.surface.fill((0, 0, 0, 0))  # ✅ Preserve transparency
        else:
            self.surface.fill(self.bg_color)  # ✅ Standard clearing for non-alpha surfaces



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
