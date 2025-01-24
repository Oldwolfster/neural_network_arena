import math

import pygame


class DisplayConnection:
    def __init__(self, from_neuron, to_neuron, weight=0):
        self.from_neuron = from_neuron  # Reference to DisplayNeuron
        self.to_neuron = to_neuron      # Reference to DisplayNeuron
        self.weight = weight
        self.color = (0, 0, 0)          # Default to black, could be dynamic
        self.thickness = 1             # Line thickness, could vary by weight
        self.arrow_size = 10           # Size of the arrowhead

    def draw_me(self, screen):
        # Calculate start and end points
        start_x = self.from_neuron.location_left + self.from_neuron.location_width
        start_y = self.from_neuron.location_top + self.from_neuron.location_height // 2
        end_x = self.to_neuron.location_left
        end_y = self.to_neuron.location_top + self.to_neuron.location_height // 2

        # Draw the main connection line
        pygame.draw.line(screen, self.color, (start_x, start_y), (end_x, end_y), self.thickness)

        # Calculate arrowhead points
        angle = math.atan2(end_y - start_y, end_x - start_x)  # Angle of the connection line
        arrow_point1 = (
            end_x - self.arrow_size * math.cos(angle - math.pi / 12),
            end_y - self.arrow_size * math.sin(angle - math.pi / 12)
        )
        arrow_point2 = (
            end_x - self.arrow_size * math.cos(angle + math.pi / 12),
            end_y - self.arrow_size * math.sin(angle + math.pi / 12)
        )

        # Draw the arrowhead
        pygame.draw.polygon(screen, self.color, [(end_x, end_y), arrow_point1, arrow_point2])
