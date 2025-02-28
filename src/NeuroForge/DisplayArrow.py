import math
import pygame

from src.engine.Utils import ez_debug


class DisplayArrow:
    def __init__(self, start_x: int, start_y: int, end_x: int, end_y: int, screen):
        # 🔹 Ensure values are integers (or raise an error)
        if not all(isinstance(val, (int, float)) for val in (start_x, start_y, end_x, end_y)):
            raise TypeError(f"Expected int or float, but got: start_y={start_y}, start_x={start_x}, end_y={end_y}, end_x={end_x}")

        self.screen = screen
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y

        self.arrow_size = 18  # Size of the arrowhead
        self.color = (0, 0, 0)  # Default black
        self.thickness = 4  # Standard thickness

    def draw(self):
        """ Draws the connection line and arrowhead on the screen. """
        # Draw the main connection line
        pygame.draw.line(self.screen, self.color, (self.start_x, self.start_y), (self.end_x, self.end_y), self.thickness)

        # Draw the arrowhead
        arrow_points = self._calculate_arrowhead(self.end_x, self.end_y, self.start_x, self.start_y)
        pygame.draw.polygon(self.screen, self.color, [(self.end_x, self.end_y)] + arrow_points)

    def _calculate_arrowhead(self, end_x, end_y, start_x, start_y):
        """ Calculates the arrowhead coordinates based on the line direction. """
        angle = math.atan2(end_y - start_y, end_x - start_x)
        arrow_point1 = (
            end_x - self.arrow_size * math.cos(angle - math.pi / 10),
            end_y - self.arrow_size * math.sin(angle - math.pi / 10)
        )
        arrow_point2 = (
            end_x - self.arrow_size * math.cos(angle + math.pi / 10),
            end_y - self.arrow_size * math.sin(angle + math.pi / 10)
        )
        return [arrow_point1, arrow_point2]
